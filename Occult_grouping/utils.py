import json
import numpy as np


def extract_routing_trace(file_path):
    routing_trace = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            prompt_id = entry["prompt_id"]
            trace = entry["trace"]  # shape: [num_tokens, num_layers]
            routing_trace.append({
                "prompt_id": prompt_id,
                "trace": trace
            })
    return routing_trace


# Map prompt_id to GPU
# 模拟DP+EP batch_size=64 prompt_num = 512 单节点4张卡
def prompt_to_gpu(prompt_id):
    if prompt_id < 64 or (256 <= prompt_id < 320):
        return 0
    elif 64 <= prompt_id < 128 or (320 <= prompt_id < 384):
        return 1
    elif 128 <= prompt_id < 192 or (384 <= prompt_id < 448):
        return 2
    else:
        return 3


def extract_expert_placement(num_layers, num_experts_per_layer, file_path):
    with open(file_path,'r') as f:
        placement_dict = json.load(f)
    
    # 转成np.array  [layers,experts]
    expert_placement = np.full((num_layers, num_experts_per_layer), -1, dtype=int)
    for layer_str, gpu_experts_lists in placement_dict.items():
        layer_id = int(layer_str)
        for gpu_id, expert_list in enumerate(gpu_experts_lists):
            for expert_id in expert_list:
                expert_placement[layer_id, expert_id] = gpu_id

    return expert_placement


def extract_replicated_experts(num_layers, file_path):
    with open(file_path,'r') as f:
        replicated_dict = json.load(f)
    
    replicated_experts_list = []
    for layer_id in range(num_layers):
        layer_str = str(layer_id)
        replicated_experts = replicated_dict.get(layer_str, [])
        replicated_experts_list.append(replicated_experts)

    return replicated_experts_list


