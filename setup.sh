#!/bin/bash
python3 -m venv mb_env
source mb_env/bin/activate
pip install uv
cd verl
uv pip install -e ".[vllm,geo]"
uv pip install wheel
uv pip install flash-attn --no-build-isolation
uv pip install scipy scikit-learn matplotlib umap-learn hdbscan kneed tqdm ipdb wordcloud math-verify lmoments3 scikit-dimension
uv pip install --upgrade "huggingface_hub[cli]"
uv pip install huggingface-hub==0.36.0
uv pip install transformers==4.57.6
uv pip install hf_transfer