eval_interval = 500
log_interval = 10
eval_iters = 100
always_save_checkpoint = True
init_from = "scratch"

wandb_log = True
wandb_project = "nanoGPT-mhc-ablation"
wandb_mode = "online"

dataset = "openwebtext"
gradient_accumulation_steps = 1
batch_size = 128
block_size = 512

n_layer = 24
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

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

device = "cuda"
dtype = "bfloat16"
compile = True
seed = 1337

hc_mult = 4
hc_sinkhorn_iters = 10
hc_eps = 1e-6
