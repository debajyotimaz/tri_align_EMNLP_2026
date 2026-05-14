export HF_HOME=/hf_cache
export HUGGINGFACE_HUB_CACHE=/hf_cache
export VLLM_CACHE_ROOT=/vllm_cache


python run_inference_hate.py --config english.yaml
python run_inference_hate.py --config hindi.yaml