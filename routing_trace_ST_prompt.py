import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

prompt_sizes = [512] # 8, 16, 32, 64, 128, 256, 512, 1024 974

output_dir = "./different_tasks/trace_by_prompt/output/generate"
os.makedirs(output_dir, exist_ok=True)

routing_data_dir = "./different_tasks/trace_by_prompt/expert_trace/Switch_Transformer/generate/sonnet"
os.makedirs(routing_data_dir, exist_ok=True)

for size in prompt_sizes:
    # with open(f"./dataset/math/GSM8K_test_{size}.jsonl", "r", encoding="utf-8") as f:
    #     prompts = [json.loads(line)["question"] for line in f if line.strip()]  # mbpp-"text"  GSM8K-"question"
    with open(f"./dataset/sonnet/sonnet_{size}_back.txt", "r") as file:
        prompts = [line.strip() for line in file.readlines() if line.strip()]

    output_file_path =  f"{output_dir}/output_{size}.txt"

    encode_trace = []
    decode_trace = []
    full_trace_prompt = []

    
    with open(output_file_path, "w") as output_file:
        #for prompt in prompts:
        for prompt_id, prompt in enumerate(prompts):

            model.encode_routing_trace = []  
            model.decode_routing_trace = []
            # model.routing_trace = []

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(DEVICE) 
            outputs = model.generate(input_ids, max_new_tokens=50)
            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            en_trace = model.encode_routing_trace[0] # shape: [tokens_encode, 6]
            num_encode_tokens = en_trace.shape[0]

            de_trace = np.concatenate(model.decode_routing_trace, axis=0)  # shape: [tokens_decode, 6]
            num_decode_tokens = de_trace.shape[0]

            # 记录这个prompt的完整路由
            # 补齐12层
            encode_pad = np.full((num_encode_tokens, 6), -1, dtype=int) 
            decode_pad = np.full((num_decode_tokens, 6), -1, dtype=int)
           
            encode_padded = np.concatenate([en_trace, encode_pad], axis=1)  # [tokens_encode, 12]
            decode_padded = np.concatenate([decode_pad, de_trace], axis=1)  # [tokens_decode, 12]

            full_trace = np.concatenate([encode_padded, decode_padded], axis=0)
            
            output_file.write(f"{prompt_id}\nPrompt: {prompt}\nGenerated: {decoded_text}\n")
            output_file.write(f"model.encode_routing_trace:\n{model.encode_routing_trace}\nmodel.decode_routing_trace:\n{model.decode_routing_trace}\n")
            output_file.write(f"encode:\n{en_trace}\ndecode:\n{de_trace}\n")
            output_file.write(f"full_trace:\n{full_trace}\n")

            encode_trace.append(en_trace)
            decode_trace.append(de_trace)
            #full_trace_prompt.append(full_trace)

            full_trace_prompt.append({
                "prompt_id": prompt_id,
                "trace": full_trace.tolist()  # shape: [tokens_total, 12]
            })

    print(f"End of inference.({size} prompts)")

    # # 保存路由信息
    # with open(output_file_path, "a") as output_file:
    #     output_file.write(f"encode:\n{model.encode_routing_trace}\ndecode:\n{model.decode_routing_trace}\n")
    all_encode_trace = np.concatenate(encode_trace, axis=0)
    np.save(f"{routing_data_dir}/encode_routing_trace_{size}_back.npy", all_encode_trace)
    print(f"Saved: {routing_data_dir}/encode_routing_trace_{size}.npy")
    
    all_decode_trace = np.concatenate(decode_trace, axis=0)
    np.save(f"{routing_data_dir}/decode_routing_trace_{size}_back.npy", all_decode_trace)
    print(f"Saved: {routing_data_dir}/decode_routing_trace_{size}.npy")

    with open(f"{routing_data_dir}/routing_trace_{size}_back.jsonl", "w") as f:
        for item in full_trace_prompt:
            json.dump(item, f)
            f.write("\n")
    # with open(f"{routing_data_dir}/routing_trace_{size}.json", "w") as f:
    #     json.dump(full_trace_prompt, f, indent=4, separators=(',', ': '))
    # print(f"Saved: {routing_data_dir}/routing_trace_{size}.json")