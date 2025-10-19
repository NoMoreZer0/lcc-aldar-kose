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
export OPENAI_API_KEY="<OPENAPI_KEY>"

python -m ml.src.cli.generate_storyboard --logline "Aldar Köse tricks a greedy merchant in the Kazakh steppe." --frames 8
```

Outputs are written to `ml/outputs/<run_id>/frame_XX.png` along with an `index.json` manifest.

