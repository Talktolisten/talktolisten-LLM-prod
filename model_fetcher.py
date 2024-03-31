'''
RunPod | Transformer | Model Fetcher
'''

import argparse

import torch
from transformers import (AutoTokenizer, GPTQConfig)
from peft import AutoPeftModelForCausalLM


def download_model(model_name):
    quantization_config = GPTQConfig(bits = 4, use_exllama=False)

    # --------------------------------- hieunguyenminh/v3 --------------------------------- #
    AutoTokenizer.from_pretrained(model_name)
    AutoPeftModelForCausalLM.from_pretrained(model_name, 
                                            low_cpu_mem_usage=True,
                                            return_dict=True,
                                            torch_dtype=torch.float16,
                                            quantization_config=quantization_config
                                            )

# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_name", type=str,
                    default="hieunguyenminh/v3", help="HF repo of the model to download.")

if __name__ == "__main__":
    args = parser.parse_args()
    download_model(args.model_name)