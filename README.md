# mHC Residual Ablation

This is a clean nanoGPT-style experiment directory for comparing three residual designs under the same data, model size, and training budget:

- `vanilla`: standard pre-norm Transformer residuals.
- `hc`: naive dynamic Hyper-Connection without doubly stochastic constraints.
- `mhc`: DeepSeek-V4-style dynamic Hyper-Connection with Sinkhorn-constrained combination matrices.

The original DeepSeek-V4 inference `model.py` is kept in this directory as a reference. The trainable GPT experiment code does not import it because the inference model depends on custom kernels and serving-specific components. The mHC implementation here mirrors its key HC idea: keep `hc_mult` residual streams, compute token-wise `pre/post/comb` mixing weights, reduce streams before attention/MLP, then write the update back into multiple streams.

## Structure

```text
mHC_experiments/
  model.py                 # DeepSeek-V4 inference reference, not used by training
  gpt_model.py             # shared GPT body
  residuals.py             # VanillaBlock, HyperConnectionBlock, Sinkhorn mHC
  train.py                 # training entrypoint with timestamped logs
  metrics.py               # loss spike and aux metric helpers
  configurator.py          # nanoGPT-style config override helper
  configs/
    base.py
    vanilla.py
    hc.py
    mhc.py
  scripts/
    run_smoke.sh
    run_all.sh
  reports/
    figures/
  outputs/
```

## Data

By default `train.py` reads the existing nanoGPT OpenWebText memmap files:

```text
/home/qjh/llm_learning/transformer_research/nanoGPT/data/openwebtext/train.bin
/home/qjh/llm_learning/transformer_research/nanoGPT/data/openwebtext/val.bin
```

The prepared dataset is about 9.0B train tokens and 4.4M validation tokens.

To use another location:

```bash
python train.py configs/vanilla.py --data_dir=/path/to/openwebtext
```

The directory must contain `train.bin` and `val.bin` encoded as GPT-2 token ids in `uint16`.

## Smoke Test

From this directory:

```bash
conda activate mhc_env
./scripts/run_smoke.sh
```

This runs all three residual modes on CPU for two iterations each. It is only a code-path check.

## Main Runs

Single run:

```bash
conda activate mhc_env
torchrun --standalone --nproc_per_node=2 train.py configs/vanilla.py
torchrun --standalone --nproc_per_node=2 train.py configs/hc.py
torchrun --standalone --nproc_per_node=2 train.py configs/mhc.py
```

Or run all three sequentially:

```bash
conda activate mhc_env
NPROC_PER_NODE=2 ./scripts/run_all.sh
```

The default main config is:

```text
n_layer=24
n_head=6
n_embd=384
block_size=512
batch_size=64
gradient_accumulation_steps=1
max_iters=10000
hc_mult=4
```

With 2 GPUs this is 131,072 tokens per iteration and about 1.31B tokens for 10k iterations. On 2x H200, prefer this larger micro-batch over extra gradient accumulation because it keeps the GPUs busier while preserving the same global token batch.

In this experiment runner, `gradient_accumulation_steps` means micro-steps per GPU process. Global tokens per optimizer step are:

```text
batch_size * block_size * gradient_accumulation_steps * number_of_gpus
```

## W&B

Do not hard-code API keys in this repository. Configure W&B from the shell:

```bash
wandb login
python train.py configs/mhc.py --wandb_log=True
```

Every training log line printed by `train.py` includes a timestamp. W&B logs use separate `train/iter` and `eval/iter` step metrics.

## Metrics

Training logs include:

- train batch loss
- eval train/val loss
- learning rate
- gradient norm before clipping
- activation norm
- stream standard deviation for HC/mHC
- mHC/HC mixing row and column stochastic errors
- loss spike count and max amplitude
- tokens/sec and MFU estimate

Use these to compare stability, not only final validation loss.

## GitHub

To connect this directory to a new repository:

```bash
git init
git add .
git commit -m "initial mHC residual ablation scaffold"
git branch -M main
git remote add origin https://github.com/Victory0431/mHC_experiments.git
git push -u origin main
```

Review files before pushing, especially `orders.txt`, because it contains external account details.
