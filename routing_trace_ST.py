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

prompt_sizes = [8] # 8, 16, 32, 64, 128, 256, 512, 1024 974


# output_dir = "./different_tasks/0516/output/code"
# os.makedirs(output_dir, exist_ok=True)

# routing_data_dir = "./different_tasks/0516/expert_trace/Switch_Transformer/code/mbpp"
# os.makedirs(routing_data_dir, exist_ok=True)

for size in prompt_sizes:
    with open(f"./dataset/code/mbpp_{size}.jsonl", "r", encoding="utf-8") as f:
        prompts = [json.loads(line)["text"] for line in f if line.strip()]  # mbpp-"text"  GSM8K-"question"
    # with open(f"./dataset/sonnet/sonnet_{size}.txt", "r") as file:
    #     prompts = [line.strip() for line in file.readlines() if line.strip()]
    prompts=prompts[:1]
    #print(prompts)

    #output_file_path =  f"{output_dir}/output_{size}.txt"
    
    model.encode_routing_trace = []  
    model.decode_routing_trace = []

    
    for prompt in prompts:
            #print("prompt:",prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(DEVICE) 
            #input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)  # Batch size 1
        outputs = model.generate(input_ids, max_new_tokens=50)
        decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            #print("decoded_text:",decoded_text)

        #output_file.write(f"Prompt: {prompt}\nGenerated: {decoded_text}\n")
    
    print(f"End of inference.({size} prompts)")

    # # 保存路由信息
    # with open(output_file_path, "a") as output_file:
    #     output_file.write(f"encode:\n{model.encode_routing_trace}\ndecode:\n{model.decode_routing_trace}\n")

    # if model.encode_routing_trace:
    #     encode_trace = np.concatenate(model.encode_routing_trace, axis=0)
    #     np.save(f"{routing_data_dir}/encode_routing_trace_{size}.npy", encode_trace)
    #     print(f"Saved: {routing_data_dir}/encode_routing_trace_{size}.npy")

    # if model.decode_routing_trace:
    #     decode_trace = np.concatenate(model.decode_routing_trace, axis=0)
    #     np.save(f"{routing_data_dir}/decode_routing_trace_{size}.npy", decode_trace)
    #     print(f"Saved: {routing_data_dir}/decode_routing_trace_{size}.npy")


# for prompt in prompts:
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # Batch size 1
#     outputs = model.generate(input_ids, max_new_tokens=50)
#     decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     with open(f"{output_dir}/output.txt", "a") as output_file:
#         output_file.write(f"Prompt: {prompt}\nGenerated Summary: {decoded_text}\n")

# print("End of inference")        

# with open(f"{output_dir}/output.txt", "a") as output_file:
#     output_file.write(f"encode:\n{model.encode_routing_trace}\ndecode:\n{model.decode_routing_trace}\n")


# if model.encode_routing_trace:
#     encode_trace = np.concatenate(model.encode_routing_trace, axis=0)
#     np.save(f"{routing_data_dir}/encode_routing_trace.npy", encode_trace)
#     print(f"Saved:{routing_data_dir}/encode_routing_trace.npy")

# if model.decode_routing_trace:
#     decode_trace = np.concatenate(model.decode_routing_trace, axis=0)
#     np.save(f"{routing_data_dir}/decode_routing_trace.npy", decode_trace)
#     print(f"Saved:{routing_data_dir}/decode_routing_trace.npy")


#检查数据格式：
# encode_trace_data = np.load(f"{routing_data_dir}/encode_routing_trace.npy")
# decode_trace_data = np.load(f"{routing_data_dir}/decode_routing_trace.npy")

# print("Encode Trace Shape:", encode_trace_data.shape)  # (num_tokens, num_layers)
# print(encode_trace_data[:15])
# print("Decode Trace Shape:", decode_trace_data.shape)  # (num_tokens, num_layers)
# print(decode_trace_data[:15])


# # 改批处理
# batch_size = 8 
# with open(f"{output_dir}/output.txt", "w") as output_file:
#     for i in range(0, len(prompts), batch_size):
#         batch_prompts = prompts[i:i + batch_size]  # 获取当前 batch
#         batch_inputs = tokenizer(batch_prompts, padding=True, truncation=True, return_tensors="pt")
#         batch_outputs = model.generate(batch_inputs.input_ids, max_new_tokens=50)
#         decoded_texts = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

#         for prompt, summary in zip(batch_prompts, decoded_texts):
#             output_file.write(f"Prompt: {prompt}\nGenerated Summary: {summary}\n\n")
