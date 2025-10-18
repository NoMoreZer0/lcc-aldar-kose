## Aldar Köse Storyboard Generator (ML Stack)

This module contains the machine learning components that power the Aldar Köse Storyboard Generator. It is optimized for rapid hackathon iteration while remaining production-ready for GPU deployments (A100/H100) and Colab workflows.

### Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt

mkdir -p /dev/shm/hf_cache/hub
mkdir -p /dev/shm/hf_cache/transformers
chmod -R 777 /dev/shm/hf_cache


export HF_HOME=/dev/shm/hf_cache
export HF_HUB_CACHE=/dev/shm/hf_cache/hub
export TRANSFORMERS_CACHE=/dev/shm/hf_cache/transformers

python -m ml.src.cli.generate_storyboard --logline "Aldar Köse tricks a greedy merchant in the Kazakh steppe." --frames 8
```

Outputs are written to `ml/outputs/<run_id>/frame_XX.png` along with an `index.json` manifest.

### Colab

Run the complete pipeline with the provided notebook:

[📓 Colab Inference Notebook](./notebooks/colab_inference.ipynb)

### One-command demo

After installing root dependencies (see repository README), execute:

```bash
make demo
```

### Layout

- `configs/` – YAML configs (default generation options, thresholds).
- `assets/identity/` – Optional Aldar Köse reference imagery to bootstrap identity adapters.
- `src/` – Python package containing CLI entrypoints, diffusion pipelines, evaluation, and LLM shot planning.
- `tests/` – Pytest-based smoke and unit tests for configuration and metric utilities.
- `notebooks/` – Reproducible Colab notebook for end-to-end storyboard generation.
