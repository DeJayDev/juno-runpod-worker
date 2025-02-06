import os

def load():
    os.environ["HF_TOKEN_PATH"] = "/workspace/.hf-token"
    os.environ["HF_ASSETS_CACHE"] = "/workspace/.cache/huggingface"
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/workspace/.cache/huggingface"
    #os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["VLLM_CACHE_ROOT"] = "/workspace/.cache/vllm/"
    os.environ["VLLM_USE_V1"] = "0"

