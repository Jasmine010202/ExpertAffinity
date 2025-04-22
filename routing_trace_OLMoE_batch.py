import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoTokenizer
from src.modeling_olmoe import OlmoeForCausalLM

import torch
# import os
import numpy as np
import random 
import time
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load different ckpts via passing e.g. `revision=step10000-tokens41B`

tokenizer = AutoTokenizer.from_pretrained("./models/OLMoE-1B-7B-0924")
tokenizer.padding_side = "left"
model = OlmoeForCausalLM.from_pretrained("./models/OLMoE-1B-7B-0924", torch_dtype=torch.float16).to(DEVICE) #, num_experts_per_tok=1

# with open("./dataset/sonnet.txt", "r") as file:
#     prompts = [line.strip() for line in file.readlines() if line.strip()]

prompt_sizes = [8, 16, 32, 64, 128, 256, 512, 1024] 
batch_size = 64

output_dir = "./output/OLMoE/sonnet/top8/bs64"
os.makedirs(output_dir, exist_ok=True)

routing_data_dir = "./expert_trace/OLMoE/sonnet/top8/bs64"
os.makedirs(routing_data_dir, exist_ok=True)

inference_time_dir = "./inference_time/OLMoE/sonnet/top8/bs64"
os.makedirs(inference_time_dir, exist_ok=True)

all_inference_time = {}

for size in prompt_sizes:
    # sampled_prompts = random.sample(prompts, size)

    # 加载 prompts
    with open(f"./dataset/sonnet/sonnet_{size}.txt", "r") as file:
        prompts = [line.strip() for line in file.readlines() if line.strip()]
    # output_file_path =  f"{output_dir}/output_{size}.txt"
    
    decoded_outputs = []
    # 整体
    model.experts_routing_trace = [] 
    start_time = time.time()

    #num_batches = size // batch_size
    # num_batches = (size + batch_size - 1) // batch_size

    for i in range(0, size, batch_size):
    #for i in range(num_batches):
        batch_prompts = prompts[i:i+batch_size]
        #batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
            
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 分batch记路由
        # model.experts_routing_trace = [] 
        outputs = model.generate(**inputs, max_new_tokens=50)
    
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_outputs.extend(decoded)

        # if model.experts_routing_trace:
        #     batch_routing_trace_data = torch.cat(model.experts_routing_trace, dim=0).cpu().numpy()
        #     np.save(f"{routing_data_dir}/decode_routing_trace_{size}_batch{i}.npy", batch_routing_trace_data)
        #     print(f"Saved: {routing_data_dir}/decode_routing_trace_{size}_batch{i}.npy")

    inference_time = time.time() - start_time
    all_inference_time[str(size)] = round(inference_time, 4)

    print(f"Completed size {size} in {inference_time:.4f} seconds")

    # 写入输出
    output_file_path = os.path.join(output_dir, f"output_{size}.txt")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        #for prompt, result in zip(sampled_prompts, decoded_outputs):
        for prompt, result in zip(prompts, decoded_outputs):
            output_file.write(f"Prompt: {prompt}\nGenerated: {result}\n\n")

    print(f"End of inference. ({size} prompts)")

    
    # 保存路由信息
    # with open(output_file_path, "a") as output_file:
    #     output_file.write(f"routing_trace:\n{model.experts_routing_trace}\n")

    # 整体记路由
    if model.experts_routing_trace:
        routing_trace_data = torch.cat(model.experts_routing_trace, dim=0).cpu().numpy()
        np.save(f"{routing_data_dir}/decode_routing_trace_{size}.npy", routing_trace_data)
        print(f"Saved: {routing_data_dir}/decode_routing_trace_{size}.npy")

timing_file_path = os.path.join(inference_time_dir, "all_inference_time.json")
with open(timing_file_path, "w") as f:
    json.dump(all_inference_time, f, indent=4)