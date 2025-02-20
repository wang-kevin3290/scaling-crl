# Clean-JaxGCRL

JaxGCRL code ported to [cleanrl](https://github.com/vwxyzjn/cleanrl) style implementations. To optimize for both speed and ease of understanding. Comes at cost of duplicate / redundant code.

## Installation

```bash
CONDA_OVERRIDE_CUDA="12.0" conda create --name expl-env python=3.10 numpy==1.26.4 jax==0.4.23 "jaxlib==0.4.23=cuda120*" flax==0.7.4 -c conda-forge -c nvidia -y
pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0
```