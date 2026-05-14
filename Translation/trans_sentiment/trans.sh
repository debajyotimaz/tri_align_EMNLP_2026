export HF_HOME=/hf_cache
export HUGGINGFACE_HUB_CACHE=/hf_cache
export VLLM_CACHE_ROOT=/vllm_cache

# English
python run_inference.py --config english.yaml
python run_inference.py --config english_val.yaml
python run_inference.py --config english_test.yaml

# Hindi
python run_inference.py --config hindi.yaml
python run_inference.py --config hindi_val.yaml
python run_inference.py --config hindi_test.yaml

