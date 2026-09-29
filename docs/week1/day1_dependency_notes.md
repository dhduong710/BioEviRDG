# Week 1 Day 1 - Dependency Notes

## Public GraphOracle dependency conflict

The public GraphOracle `requirements.txt` pins:

- `datasets==3.6.0`
- `multiprocess==0.70.18`

However, `datasets==3.6.0` requires `multiprocess<0.70.17`.

For the Week 1 reproducible environment, we therefore use:

- `datasets==3.6.0`
- `multiprocess==0.70.16`

The original public `requirements.txt` is kept unchanged.

PyTorch CUDA packages are installed separately:

- `torch==2.5.1+cu121`
- `torch_scatter==2.1.2+pt25cu121`
