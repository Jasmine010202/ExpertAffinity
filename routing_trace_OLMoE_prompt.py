import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from transformers import AutoTokenizer
from src.modeling_olmoe import OlmoeForCausalLM

import torch
# import os
import numpy as np
import random 
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load different ckpts via passing e.g. `revision=step10000-tokens41B`

tokenizer = AutoTokenizer.from_pretrained("./models/OLMoE-1B-7B-0924")
model = OlmoeForCausalLM.from_pretrained("./models/OLMoE-1B-7B-0924", torch_dtype=torch.float16).to(DEVICE) #, num_experts_per_tok=1

# with open("./dataset/sonnet/sonnet.txt", "r") as file:
#     prompts = [line.strip() for line in file.readlines() if line.strip()]
dataset_name = "conala"   #sonnet GSM8K conala gigaword
prompt_sizes = [512] # 8, 16, 32, 64, 128, 256, 512, 1024

# output_dir = "./Occult_test/output/OLMoE/conala/top8"
# os.makedirs(output_dir, exist_ok=True)

#routing_data_dir = f"./Occult_test/expert_trace/New/used_for_occult/by_prompt/OLMoE_{dataset_name}_top8"
routing_data_dir = f"./Occult_test/expert_trace/New/traffic_test/by_prompt/OLMoE_{dataset_name}_top8"
os.makedirs(routing_data_dir, exist_ok=True)

for size in prompt_sizes:
    # sampled_prompts = random.sample(prompts, size)
    # with open(f"./dataset/{dataset_name}/used_for_occult/{dataset_name}_{size}.txt", "r") as file:
    # with open(f"./dataset/{dataset_name}/used_for_occult/traffic_test/{dataset_name}_{size}.txt", "r") as file:
    #     sampled_prompts = [line.strip() for line in file.readlines() if line.strip()]
    
    # with open(f"./dataset/math/{dataset_name}/used_for_occult/{dataset_name}_test_{size}.jsonl", "r", encoding="utf-8") as f:
    # with open(f"./dataset/math/{dataset_name}/used_for_occult/traffic_test/{dataset_name}_{size}.jsonl", "r", encoding="utf-8") as f:
    #     sampled_prompts = [json.loads(line)["question"] for line in f if line.strip()]  # mbpp-"text"  GSM8K-"question" conala-"intent"

    #with open(f"./dataset/code/{dataset_name}/used_for_occult/{dataset_name}_{size}.jsonl", "r", encoding="utf-8") as f:
    with open(f"./dataset/code/{dataset_name}/used_for_occult/traffic_test/{dataset_name}_{size}.jsonl", "r", encoding="utf-8") as f:
        sampled_prompts = [json.loads(line)["intent"] for line in f if line.strip()]

    #routing_trace = []
    trace_by_prompt = []
    full_trace_list = []

    # for prompt in sampled_prompts:
    for prompt_id , prompt in enumerate(sampled_prompts):

        model.experts_routing_trace = []  

        print(f"prompt_id:[{prompt_id }]")  

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        # outputs = model.generate(**inputs, max_length=64)
        outputs = model.generate(**inputs, max_new_tokens=50)

        #print(model.experts_routing_trace)
        # print("每个 token 的 routing trace shape：")
        # for i, t in enumerate(model.experts_routing_trace):
        #     print(f"  Token {i}: {t.shape}")

        trace_tensor = torch.cat(model.experts_routing_trace, dim=0).cpu().numpy()
        # trace_tensor = torch.stack(model.experts_routing_trace, dim=0).cpu().numpy()
        
        # print(trace_tensor.shape)

        full_trace_list.append(trace_tensor)

        trace_by_prompt.append({
                "prompt_id": prompt_id,
                "trace": trace_tensor.tolist()  
            })
    
    print(f"End of inference.({size} prompts)")

    # 保存路由信息
    # with open(output_file_path, "a") as output_file:
    #     output_file.write(f"routing_trace:\n{model.experts_routing_trace}\n")
    with open(f"{routing_data_dir}/routing_trace_{size}.jsonl", "w") as f:
        for item in trace_by_prompt:
            json.dump(item, f)
            f.write("\n")

    full_trace_array = np.concatenate(full_trace_list, axis=0)
    print(full_trace_array.shape)
    np.save(f"{routing_data_dir}/decode_routing_trace_{size}.npy", full_trace_array)
    print(f"Saved: {routing_data_dir}/decode_routing_trace_{size}.npy")

    # if model.experts_routing_trace:
    #     routing_trace_data = torch.cat(model.experts_routing_trace, dim=0).cpu().numpy()
    #     #routing_trace_data = np.concatenate(model.experts_routing_trace, axis=0)
    #     #torch.cat(model.experts_routing_trace, dim=0).cpu().numpy()
    #     #encode_trace = np.concatenate(model.encode_routing_trace, axis=0)
    #     np.save(f"{routing_data_dir}/decode_routing_trace_{size}.npy", routing_trace_data)
    #     print(f"Saved: {routing_data_dir}/decode_routing_trace_{size}.npy")






