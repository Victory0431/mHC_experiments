import os
import time
import math
import pickle
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt_model import GPT, GPTConfig
from metrics import LossSpikeTracker, reduce_aux, tokens_per_second


out_dir = "outputs/debug"
eval_interval = 500
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

wandb_log = False
wandb_project = "nanoGPT-mhc-ablation"
wandb_run_name = "debug"
wandb_mode = "online"

dataset = "openwebtext"
data_dir = ""
gradient_accumulation_steps = 2
batch_size = 32
block_size = 256

n_layer = 12
n_head = 6
n_embd = 384
dropout = 0.0
bias = False
residual_mode = "vanilla"
hc_mult = 4
hc_sinkhorn_iters = 10
hc_eps = 1e-6

learning_rate = 6e-4
max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 6e-5

backend = "nccl"
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = True
seed = 1337

config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(__file__), "configurator.py"), encoding="utf-8").read())
config = {k: globals()[k] for k in config_keys}


def log_with_time(*parts):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *parts, flush=True)


def default_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "nanoGPT", "data", dataset))


ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if not data_dir:
    data_dir = default_data_dir()
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    log_with_time("config:", config)
    log_with_time(f"data_dir={data_dir}")
    log_with_time(f"tokens_per_iter={tokens_per_iter:,}")

torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def get_batch(split):
    path = os.path.join(data_dir, "train.bin" if split == "train" else "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


iter_num = 0
best_val_loss = 1e9
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta_vocab_size = pickle.load(f)["vocab_size"]

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size or 50304,
    dropout=dropout,
    residual_mode=residual_mode,
    hc_mult=hc_mult,
    hc_sinkhorn_iters=hc_sinkhorn_iters,
    hc_eps=hc_eps,
)
if init_from != "scratch":
    raise ValueError("This clean experiment runner currently supports init_from='scratch' only.")

model = GPT(GPTConfig(**model_args)).to(device)
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
base_model = model

if compile:
    log_with_time("compiling model")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config, mode=wandb_mode)
    wandb.define_metric("train/iter")
    wandb.define_metric("train/*", step_metric="train/iter")
    wandb.define_metric("eval/iter")
    wandb.define_metric("eval/*", step_metric="eval/iter")

X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
spikes = LossSpikeTracker()
model.train()

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        log_with_time(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "eval/iter": iter_num,
                "eval/train_loss": losses["train"],
                "eval/val_loss": losses["val"],
                "eval/lr": lr,
                "eval/mfu": running_mfu * 100,
            })
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": base_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                log_with_time(f"saved checkpoint to {out_dir}")
    if iter_num == 0 and eval_only:
        break

    last_aux = {}
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            _, loss, aux = model(X, Y, return_aux=True)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        scaler.scale(loss).backward()
        last_aux = aux

    grad_norm = None
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        is_spike, spike_amplitude = spikes.update(lossf)
        if local_iter_num >= 5:
            mfu = base_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        aux_metrics = reduce_aux(last_aux)
        tok_s = tokens_per_second(tokens_per_iter, dt)
        log_parts = [
            f"iter {iter_num}",
            f"loss {lossf:.4f}",
            f"lr {lr:.2e}",
            f"time {dt*1000:.2f}ms",
            f"tok/s {tok_s:.0f}",
            f"mfu {running_mfu*100:.2f}%" if running_mfu >= 0 else "mfu n/a",
        ]
        if grad_norm is not None:
            log_parts.append(f"grad_norm {float(grad_norm):.4f}")
        if "activation_norm" in aux_metrics:
            log_parts.append(f"act_norm {aux_metrics['activation_norm']:.4f}")
        if "stream_std" in aux_metrics:
            log_parts.append(f"stream_std {aux_metrics['stream_std']:.4f}")
        if "mixer_row_error" in aux_metrics:
            log_parts.append(f"row_err {aux_metrics['mixer_row_error']:.2e}")
            log_parts.append(f"col_err {aux_metrics['mixer_col_error']:.2e}")
        if is_spike:
            log_parts.append(f"loss_spike amp={spike_amplitude:.4f}")
        log_with_time(", ".join(log_parts))
        if wandb_log:
            payload = {
                "train/iter": iter_num,
                "train/batch_loss": lossf,
                "train/lr": lr,
                "train/mfu": running_mfu * 100 if running_mfu >= 0 else 0.0,
                "train/tokens_per_second": tok_s,
                "train/loss_spike": int(is_spike),
                "train/loss_spike_count": spikes.count,
                "train/loss_spike_max_amplitude": spikes.max_amplitude,
            }
            if grad_norm is not None:
                payload["train/grad_norm_before_clip"] = float(grad_norm)
            for key, value in aux_metrics.items():
                payload[f"train/{key}"] = value
            wandb.log(payload)

    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
