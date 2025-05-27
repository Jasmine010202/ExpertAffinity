import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoTokenizer
from src.modeling_switch_transformers import SwitchTransformersForConditionalGeneration
#from plot import plot_expert_selection_distribution as draw

import os
import numpy as np
import random 
import json
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("/data/shared_workspace/LLM_weights/switch-base-64")
model = SwitchTransformersForConditionalGeneration.from_pretrained("/data/shared_workspace/LLM_weights/switch-base-64").to(DEVICE)

# with open("./dataset/code/mbpp_8.txt", "r") as file:
#     prompts = [line.strip() for line in file.readlines() if line.strip()]

prompt_sizes = [8] # 8, 16, 32, 64, 128, 256, 512, 1024
batch_size = 64

output_dir = "./different_tasks/output/generate"
os.makedirs(output_dir, exist_ok=True)

routing_data_dir = "./different_tasks/expert_trace/Switch_Transformer/generate/sonnet"
os.makedirs(routing_data_dir, exist_ok=True)

for size in prompt_sizes:
    # with open(f"./dataset/math/GSM8K_test_{size}.jsonl", "r", encoding="utf-8") as f:
    #     prompts = [json.loads(line)["question"] for line in f if line.strip()] # mbpp-"text"  GSM8K-"question"

    with open(f"./dataset/sonnet/sonnet_{size}.txt", "r") as file:
        prompts = [line.strip() for line in file.readlines() if line.strip()]

    #print(prompts)

    output_file_path =  f"{output_dir}/output_{size}.txt"
    
    model.encode_routing_trace = []  
    model.decode_routing_trace = []

    for i in range(0, size, batch_size):
        #for i in range(num_batches):
            batch_prompts = prompts[i:i+batch_size]
            #print("prompt:",prompt)
            #input_ids = tokenizer(batch_prompts, return_tensors="pt").input_ids  # Batch size 1
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(DEVICE) 
            outputs = model.generate(input_ids, max_new_tokens=50)
            #decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            #output_file.write(f"Prompt: {prompt}\nGenerated: {decoded_text}\n")
    
    print(f"End of inference.({size} prompts)")

    # 保存路由信息
    with open(output_file_path, "a") as output_file:
        output_file.write(f"encode:\n{model.encode_routing_trace}\ndecode:\n{model.decode_routing_trace}\n")

    if model.encode_routing_trace:
        #print(model.encode_routing_trace)
        encode_trace = np.concatenate(model.encode_routing_trace, axis=0)
        np.save(f"{routing_data_dir}/encode_routing_trace_{size}.npy", encode_trace)
        print(f"Saved: {routing_data_dir}/encode_routing_trace_{size}.npy")

    if model.decode_routing_trace:
        #print(model.decode_routing_trace.shape)
        decode_trace = np.concatenate(model.decode_routing_trace, axis=0)
        np.save(f"{routing_data_dir}/decode_routing_trace_{size}.npy", decode_trace)
        print(f"Saved: {routing_data_dir}/decode_routing_trace_{size}.npy")



