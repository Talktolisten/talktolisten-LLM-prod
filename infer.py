'''
RunPod | Transformer | Handler
'''
import argparse

import torch
import runpod
from runpod.serverless.utils.rp_validator import validate
from transformers import (AutoTokenizer, GenerationConfig)
from peft import AutoPeftModelForCausalLM
import time

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SCHEMA = {
    'system': {
        'type': str,
        'required': True
    },
    'prompt': {
        'type': str,
        'required': True
    },
    'max_length': {
        'type': int,
        'required': False,
        'default': 128
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0.9
    }
}


def generator(job):
    '''
    Run the job input to generate text output.
    '''
    # Validate the input
    st_time = time.time()

    val_input = validate(job['input'], INPUT_SCHEMA)
    if 'errors' in val_input:
        return {"error": val_input['errors']}
    val_input = val_input['validated_input']

    prompt = "[INST]" + val_input['system'] +'\n[USER-QUESTION]' + val_input['prompt'] + '[/INST]'

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=val_input['temperature'],
    max_new_tokens=val_input['max_length'],
    pad_token_id=tokenizer.eos_token_id
    )

    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        outputs = model.generate(**inputs, generation_config=generation_config)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    end_time = time.time()

    return {"output": result, "time": end_time - st_time}


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model_name", type=str,
                    default="hieunguyenminh/v3", help="URL of the model to download.")


if __name__ == "__main__":
    args = parser.parse_args()

    # --------------------------------- hieunguyeminh/v3 --------------------------------- #    
    tokenizer= AutoTokenizer.from_pretrained(args.model_name)
    model = AutoPeftModelForCausalLM.from_pretrained(args.model_name, 
                                            low_cpu_mem_usage=True,
                                            return_dict=True,
                                            torch_dtype=torch.float16,
                                            device_map=device,
                                            local_files_only=True).to(device)
    
    runpod.serverless.start({"handler": generator})