import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"
from transformers import AutoTokenizer
from src.modeling_olmoe import OlmoeForCausalLM
import time
import torch
import numpy as np
import random 
import json

# 启用 4 张 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("./models/OLMoE-1B-7B-0924")
model = OlmoeForCausalLM.from_pretrained("./models/OLMoE-1B-7B-0924", torch_dtype=torch.float16).to(DEVICE) #, num_experts_per_tok=1

# 多卡包装
model = torch.nn.DataParallel(model)

with open("./dataset/sonnet.txt", "r") as file:
    prompts = [line.strip() for line in file.readlines() if line.strip()]

prompt_sizes = [8, 16, 32, 64, 128, 256, 512, 1024] 

output_dir = "./parallel_output/OLMoE/sonnet"
os.makedirs(output_dir, exist_ok=True)

batch_size = 8
all_inference_time = {}

for size in prompt_sizes:
    print(f"Prompt size:{size}")

    sampled_prompts = random.sample(prompts, size)
    
    start_time = time.time()
    decoded_outputs = []

    for i in range(0, size, batch_size):
        batch_prompts = sampled_prompts[i:i + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            batch_outputs = model.generate(**inputs, max_new_tokens=50)

        decoded = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        decoded_outputs.extend(decoded)

    inference_time = time.time() - start_time
    all_inference_time[str(size)] = round(inference_time, 4)

    print(f"Completed size {size} in {inference_time:.2f} seconds")

    output_file_path = os.path.join(output_dir, f"output_{size}.txt")
    with open(output_file_path, "w") as output_file:
        output_file.write(f"Prompt: {sampled_prompts}\nGenerated Summary: {decoded_outputs}\n")
    
    print(f"End of inference.({size} prompts)")
    
timing_file_path = os.path.join(output_dir, "all_inference_time.json")
with open(timing_file_path, "w") as f:
    json.dump(all_inference_time, f, indent=4)
print(f"All inference time saved to : {timing_file_path}")