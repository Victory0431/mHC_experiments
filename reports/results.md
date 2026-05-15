# mHC Residual Ablation Results

All runs use the same OpenWebText memmap data and the same training budget unless noted:

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

## Summary

| Run | Residual mode | Params | Final train loss | Final val loss | Throughput | Checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| vanilla | standard residual | 61.80M | 3.5001 | 3.5095 | ~0.78M-0.83M tok/s | `outputs/vanilla/ckpt.pt` |
| hc | naive dynamic Hyper-Connection | 63.57M | 3.4433 | 3.4520 | ~0.35M-0.37M tok/s | `outputs/hc/ckpt.pt` |

## Vanilla

- Log: `logs/vanilla_20260514_162536.log`
- W&B: https://wandb.ai/jiahongqin1-ucas-hias/nanoGPT-mhc-ablation/runs/0cvwcxtr
- Final eval at step 10000: train loss `3.5001`, val loss `3.5095`.
- Training was stable. Gradient norm stayed controlled; one tiny automatic loss-spike event was logged near the end and was not operationally meaningful.

## HC

- Log: `logs/hc_20260515_104304.log`
- W&B: https://wandb.ai/jiahongqin1-ucas-hias/nanoGPT-mhc-ablation/runs/59yuroq1
- Final eval at step 10000: train loss `3.4433`, val loss `3.4520`.
- HC used the same data, optimizer, context length, batch size, and training token budget as vanilla.
- HC adds dynamic multi-stream residual mixing, increasing parameters from `61.80M` to `63.57M` and lowering throughput relative to vanilla.
- HC finished with lower validation loss than vanilla in this run, but it also had higher activation norm scale and lower throughput. This should be interpreted as an ablation signal, not a final claim about large-scale mHC.

## Notes

- The HC/mHC metric path was patched to avoid scalar `.item()` graph breaks inside `torch.compile`. This does not change the model or training objective; it only keeps metric collection from disrupting compilation.
- `row_err` for HC is near zero because the naive HC combination matrix is row-normalized with softmax. `col_err` is not constrained and remains high, which is expected for the naive HC condition.
- mHC is the next required run. It should use the same base config and only switch `residual_mode = "mhc"`.
