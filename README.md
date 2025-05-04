# Triton Flash Attention

This repository implements the Flash Attention algorithm using the Triton programming language for efficient GPU computation.

## Overview

Flash Attention is a technique introduced in the paper ["FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135) by Dao et al. It achieves both memory efficiency (O(N) memory usage) and computational efficiency for attention mechanisms in transformer models.

This implementation leverages Triton, an open-source language and compiler designed for writing highly efficient GPU kernels. The goal is to provide:

- A memory-efficient implementation of attention mechanisms
- Performance comparable to or better than CUDA implementations
- A clean, readable codebase that can be easily extended

## Features

- ✅ O(N) memory complexity instead of O(N²) in standard attention
- ✅ Support for causal attention masks (used in decoder-only models like GPT)
- ✅ Support for different sequence lengths and batch sizes
- ✅ Numerically stable implementation with proper handling of softmax
- ✅ Performance benchmarks against other implementations


## How It Works

Flash Attention achieves its efficiency through block-wise operations that optimize memory access patterns on GPUs:

1. **Tiling**: The algorithm processes the attention matrix in tiles/blocks, reducing memory requirements
2. **IO-Awareness**: Carefully manages data movement between high-bandwidth memory (HBM) and on-chip SRAM
3. **Softmax Recomputation**: Avoids storing the full attention matrix by recomputing parts as needed

The implementation uses Triton's block-level parallelism to efficiently map these operations to GPU hardware.

## Performance

Performance varies by hardware, but you can expect:

- 2-4x speedup over PyTorch's native attention for long sequences
- Constant memory usage regardless of sequence length
- Better scaling for longer sequences (where standard attention often runs out of memory)


## Citation

If you use this implementation in your work, please cite:

```bibtex
@article{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and Ré, Christopher},
  journal={arXiv preprint arXiv:2205.14135},
  year={2022}
}
```


## Acknowledgments

- The [original Flash Attention paper](https://arxiv.org/abs/2205.14135) by Dao et al.
- [OpenAI Triton](https://github.com/openai/triton) for the programming framework
