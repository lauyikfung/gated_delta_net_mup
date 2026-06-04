# Unlocking Feature Learning in Gated Delta Networks at Scale

[![arXiv](https://img.shields.io/badge/arXiv-2606.04048-b31b1b.svg)](https://arxiv.org/abs/2606.04048)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg) 

Training and scaling Large Language Models demand enormous computational resources, motivating both efficient sub-quadratic architectures and principled hyperparameter tuning methods. While the Maximal Update Parametrization ($\mu$P) has enabled zero-shot hyperparameter transfer for standard Transformers, its extension to linear models, particularly those with structured state transitions and complicated architectures, remains largely unexplored. By rigorously propagating coordinate-size estimates through the forward pass, gating mechanisms, and recurrent state dynamics, we derive the scaling rules for Gated Delta Network. Experiments on language-model pre-training confirm that our configurations enable stable learning-rate transfer across model widths under both AdamW and SGD, whereas standard parametrization fails to transfer, validating the correctness and practical utility of our analysis. This repository implements the paper "[Unlocking Feature Learning in Gated Delta Networks at Scale](https://arxiv.org/abs/2606.04048)".

- Authors: [Yifeng Liu](https://lauyikfung.github.io), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

[[Huggingface](https://huggingface.co/papers/2606.04048)]

## Installation

This repo uses `uv` for virtual environments and dependency management. Ensure you have **Python 3.12**
installed (PyTorch 2.5.x does not publish Python 3.13 wheels for macOS/Windows).

- **Create and Activate a Virtual Environment (Python 3.12)**

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
```

*If Python 3.12 isn't on your PATH, pass an absolute path to the Python 3.12 binary instead of `3.12`.*

Sanity check: `which python` should point to `.venv/bin/python` after activation.

- **Install Required Packages**

```bash
# Recommended (installs from pyproject.toml)
uv sync

# Alternatively (legacy requirements.txt)
# uv pip install -r requirements.txt
```

## Development

## Data Preparation

Fineweb-Edu-100B is a large-scale educational dataset hosted on Hugging Face.

1. **Run the Data Preparation Script**

   ```bash
   python data/fineweb-edu/fineweb-edu.py
   ```

## Pretraining

Pretrain the GPT and other models using the prepared datasets. The provided scripts support distributed training across multiple GPUs.

1. **Using the Provided Bash Script**

   Execute the pretraining script, which handles the training process.

   ```bash
   bash pretrain.sh
   ```
2. **Manual Execution with `torchrun`**

   For more control or customization, use `torchrun` to initiate training. Replace `train_gpt_mha_rope_medium_adam_80g8.py` with your desired configuration file.

   ```bash
   python train_adam_finewebedu.py \
       config/train_gdn_small_adam_10BT_ctx1024_80g1.py \
       --model=gdn
   ```

   - `--nproc_per_node=8` specifies the number of processes (typically matching the number of GPUs).
   - Optional: enable ZeRO-1 optimizer state sharding (still uses DDP) with `--zero_stage=1`.
   - For SGD experiments:

     ```
     python train_sgd_finewebedu.py \
         config/train_gdn_small_sgd_10BT_ctx1024_80g1.py \
         --model=gdn
     ```

Weights & Biases logging is enabled by default. Disable it with `--wandb_log=False` or by setting
`wandb_log=False` in the config file.

## Support of Amazon Trainium Chips

See [lauyikfung/Amazon_Trainium_Optimizer/gdn_mup_code](https://github.com/lauyikfung/Amazon_Trainium_Optimizer/tree/main/gdn_mup_code) for code implemented with torch-neuronx.

## Citation

If you use SDPG in your research or application, please consider citing it!

```bibtex
@article{liu2026unlocking,
      title={Unlocking Feature Learning in Gated Delta Networks at Scale}, 
      author={Liu, Yifeng and Gu, Quanquan},
      journal={arXiv preprint arXiv:2606.04048},
      year={2026}
}
```
