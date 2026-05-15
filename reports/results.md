# mHC Residual Ablation Results

This report summarizes the first complete 3-way ablation:

1. `vanilla`: standard pre-norm residual connection.
2. `hc`: naive dynamic Hyper-Connection, row-normalized residual mixing only.
3. `mhc`: mHC-style dynamic Hyper-Connection with Sinkhorn-constrained residual mixing.

All three runs use the same OpenWebText memmap data and the same training budget:

```text
n_layer = 24
n_head = 6
n_embd = 384
block_size = 512
batch_size = 128
gradient_accumulation_steps = 1
n_gpus = 2
max_iters = 10000
tokens_per_iter = 131,072
total_train_tokens = 1.31B
```

The model has about `61.8M` non-embedding parameters for vanilla and about `63.6M` parameters for HC/mHC. The extra parameters come from the dynamic residual mixing modules.

## Final Results

| Run | Residual mode | Params | Final train loss | Final val loss | Best val loss | Avg throughput | Checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| vanilla | standard residual | 61.80M | 3.5001 | 3.5095 | 3.5071 @ 9500 | ~779k tok/s | `outputs/vanilla/ckpt.pt` |
| hc | naive dynamic Hyper-Connection | 63.57M | 3.4433 | 3.4520 | 3.4520 @ 10000 | ~357k tok/s | `outputs/hc/ckpt.pt` |
| mhc | Sinkhorn-constrained Hyper-Connection | 63.57M | 3.4301 | 3.4391 | 3.4391 @ 10000 | ~348k tok/s | `outputs/mhc/ckpt.pt` |

Validation loss deltas:

```text
HC - vanilla  = -0.0575
mHC - vanilla = -0.0704
mHC - HC      = -0.0129
```

In this small GPT-style pre-training setting, both multi-stream residual variants beat vanilla, and mHC is the best of the three.

## Run Details

### Vanilla

- Log: `logs/vanilla_20260514_162536.log`
- W&B: https://wandb.ai/jiahongqin1-ucas-hias/nanoGPT-mhc-ablation/runs/0cvwcxtr
- Final eval at step 10000: train loss `3.5001`, val loss `3.5095`.
- Best eval: val loss `3.5071` at step 9500.
- Training was stable. One tiny automatic loss-spike event was logged near the end, with amplitude `0.0107`; this was not operationally meaningful.

### HC

- Log: `logs/hc_20260515_104304.log`
- W&B: https://wandb.ai/jiahongqin1-ucas-hias/nanoGPT-mhc-ablation/runs/59yuroq1
- Final eval at step 10000: train loss `3.4433`, val loss `3.4520`.
- Best eval: val loss `3.4520` at step 10000.
- HC used the same data, optimizer, context length, batch size, LR schedule, and training token budget as vanilla.
- HC was slower than vanilla because each block carries the additional dynamic multi-stream residual mixing path.
- HC row error was near zero because its combination matrix is row-normalized by softmax. Column error remained high, as expected, because naive HC does not constrain column sums.

### mHC

- Log: `logs/mhc_20260515_120507.log`
- W&B: https://wandb.ai/jiahongqin1-ucas-hias/nanoGPT-mhc-ablation/runs/8o87xpty
- Final eval at step 10000: train loss `3.4301`, val loss `3.4391`.
- Best eval: val loss `3.4391` at step 10000.
- mHC used the same data, optimizer, context length, batch size, LR schedule, and training token budget as vanilla and HC.
- mHC was slightly slower than HC, which is expected because Sinkhorn normalization adds work.
- mHC column error stayed near `1e-6`, showing that the Sinkhorn projection strongly enforced one side of the stochastic constraint. Row error remained around `0.35` near the end, so our current 10-iteration Sinkhorn setting is not a perfect doubly stochastic projection.

## Reference: DeepSeek-V4 mHC

The local paper reference is `reports/DeepSeek_V4.pdf`.

The DeepSeek-V4 paper frames mHC as one of the key architecture upgrades, alongside hybrid attention and Muon. In Section 2.2, it says the core idea is to constrain the residual mapping matrix onto the manifold of doubly stochastic matrices, the Birkhoff polytope, to stabilize signal propagation across layers while preserving expressivity. The paper also states that this makes the residual transformation non-expansive because the spectral norm is bounded by 1.

The paper's standard HC formulation expands the residual stream by a factor `n_hc`, then uses three mappings:

```text
A_l: input mixing from multiple streams to the layer input
B_l: residual stream transformation
C_l: output/update mixing back to multiple streams
```

DeepSeek-V4 applies Sigmoid constraints to `A_l` and `C_l`, and uses Sinkhorn-Knopp to project `B_l` toward the doubly stochastic manifold. In the model setup section, DeepSeek-V4 uses:

```text
n_hc = 4
Sinkhorn iterations t_max = 20
```

The paper also notes that mHC increases activation memory and pipeline communication, and that their production implementation relies on fused kernels, recomputation, and pipeline scheduling optimizations to keep overhead low.

## How Close Is This Reproduction?

This experiment is a **minimal controlled reproduction of the residual-connection idea**, not a full DeepSeek-V4 reproduction.

What we reproduced:

- Multi-stream residual state with `hc_mult = 4`.
- Dynamic token-conditioned residual mixing inspired by the released DeepSeek-V4 inference code.
- Three controlled variants: vanilla residual, naive HC, and Sinkhorn-constrained mHC.
- Same dataset, model width/depth, batch size, optimizer, LR schedule, and training token budget across runs.
- Stability-oriented metrics: gradient norm, activation norm, stream spread, row/column stochastic errors, loss spikes, throughput.

What we did not reproduce:

- DeepSeek-V4's MoE architecture.
- CSA/HCA hybrid attention.
- Muon optimizer.
- production mHC fused kernels and recomputation strategy.
- `t_max = 20`; this run used `hc_sinkhorn_iters = 10`.
- trillion/284B scale behavior.

Because of these differences, the right claim is not "we reproduced DeepSeek-V4 mHC performance." The right claim is:

> In a controlled nanoGPT-style pre-training setting, a DeepSeek-V4-inspired mHC residual mixing variant improved validation loss over both vanilla residual and naive HC under the same token budget, while showing the expected stochastic-mixing behavior and expected throughput overhead.

## Experimental Takeaways

### 1. Multi-stream residuals helped.

HC improved final validation loss from `3.5095` to `3.4520`. This suggests that widening the residual stream and dynamically routing information through residual channels was already beneficial in this setting.

### 2. The manifold constraint added a smaller but real gain.

mHC improved final validation loss from HC's `3.4520` to `3.4391`. The gain is smaller than the vanilla-to-HC gain, but it is in the direction predicted by the paper: constrained residual mixing appears to help beyond naive multi-stream residuals.

### 3. The stability story is partially supported, not fully proven.

The runs did not exhibit severe instability in any condition. Vanilla had one tiny logged spike; HC and mHC logged none. Gradient norms were controlled for all three runs. Therefore, this experiment supports the paper's design motivation only weakly on the stability axis; the setup may not be deep or unstable enough to make the stability difference dramatic.

The stronger observed signal is convergence/loss: mHC ended with the lowest validation loss under the same training token budget.

### 4. Our Sinkhorn constraint is visibly active but imperfect.

For HC, row error was near zero and column error was high, matching row-softmax behavior. For mHC, column error was near `1e-6`, while row error stayed around `0.35`. This means the current implementation enforces a strong stochastic constraint, but with `10` Sinkhorn iterations it is not fully doubly stochastic by the metric we log.

For a closer paper match, rerun mHC with:

```text
hc_sinkhorn_iters = 20
```

and compare whether row error decreases and whether validation loss changes.

### 5. mHC has a real throughput cost in this implementation.

Vanilla ran at about `779k tok/s`, while HC and mHC ran around `357k` and `348k tok/s`. DeepSeek-V4 explicitly addresses this overhead with fused kernels and recomputation; our plain PyTorch implementation does not. This is expected and should be described as an engineering limitation rather than a conceptual failure.

## Bottom Line

The experiment achieved its purpose as a small research reproduction:

- controlled data/model/budget,
- three-way residual ablation,
- mHC-style Sinkhorn constraint,
- stability and throughput metrics,
- clean comparison against vanilla and naive HC.

The result is encouraging:

```text
vanilla val loss: 3.5095
HC val loss:      3.4520
mHC val loss:     3.4391
```

This supports the hypothesis that DeepSeek-V4-style constrained residual mixing can improve small-scale GPT pre-training behavior. The reproduction is directional rather than definitive: to get closer to the paper, the next run should use `hc_sinkhorn_iters = 20`, possibly a deeper model, and ideally a slightly more unstable training regime where the stability benefits are easier to separate from ordinary loss improvements.

