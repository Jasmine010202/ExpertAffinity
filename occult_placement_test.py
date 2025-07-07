import sys
sys.path.append("/data/workspace/hanyu2")

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
import json
import copy

from occult_1.moe_co.grouping import group_experts_on_collaboration, group_experts_on_collaboration_heterogeneous_group


model_name = "OLMoE"#Switch_Transformer OLMoE
input_name = "conala"   #GSM8K、sonnet、mbpp、conala
#phrase_mode = "decode" #decode
#prompt_nums = [512] # 8, 16, 32, 64, 128, 256, 512, 1024
top_k = 8 # ST:1,OL:8
num_of_prompts = 512#
num_of_experts_per_layer = 64

collaboration_dir = f"Occult_test/expert_collaboration"
os.makedirs(collaboration_dir, exist_ok=True)

placement_dir = f"Occult_test/expert_placement"
os.makedirs(placement_dir, exist_ok=True)


def generate_collaboration_matrix(routing_data):
    num_tokens, num_layers, _ = routing_data.shape
    experts_collaboration_matrix = np.zeros((num_layers, num_of_experts_per_layer, num_of_experts_per_layer))

    for layer in range(num_layers):
        for token in range(num_tokens):
            experts_intra_layer = routing_data[token,layer]
            for i in range(top_k):
                for j in range (i+1, top_k):
                    expert_i, expert_j = experts_intra_layer[i], experts_intra_layer[j]
                    experts_collaboration_matrix[layer, expert_i, expert_j] += 1
                    experts_collaboration_matrix[layer, expert_j, expert_i] += 1 # 对称矩阵
        
    np.save(f"{collaboration_dir}/{model_name}_Expert_Collaboration_{input_name}_{num_of_prompts}.npy", experts_collaboration_matrix)
    return experts_collaboration_matrix



if __name__ == "__main__":
    
    
    routing_data = np.load(f"Occult_test/expert_trace/{model_name}/{input_name}/top{top_k}/decode_routing_trace_{num_of_prompts}.npy")

    #routing_data = np.load(f"expert_trace/OLMoE/gigaword/top8/decode_routing_trace_512.npy")
        
    experts_collaboration_matrix = generate_collaboration_matrix(routing_data)

    # print(experts_collaboration_matrix.shape)
    # print(experts_collaboration_matrix)
        
    num_layers = experts_collaboration_matrix.shape[0]
    all_layers_placement = {}   #存所有层的专家放置结果
    all_layers_placement_heterogeneous = {}   #存所有层的专家放置结果


    for layer_id in range(num_layers):
        # print(f"layer_id:{layer_id}")

        # # 检查是不是有专家没有被激活过
        # matrix = torch.tensor(experts_collaboration_matrix[layer_id])
        # unused_experts = (matrix.sum(dim=0) == 0).nonzero()
        # print(f"Layer {layer_id}: unused experts: {unused_experts.flatten().tolist()}")

        collaboration_per_layer = experts_collaboration_matrix[layer_id] #每一层的专家协作矩阵
        collaboration_per_layer = torch.from_numpy(collaboration_per_layer) # numpy转tensor

        collab1 = copy.deepcopy(collaboration_per_layer)
        collab2 = copy.deepcopy(collaboration_per_layer)
        
        placement_per_layer = group_experts_on_collaboration(expert_collab_counts = collab1, num_groups = 4, even_groups = True)
        placement_per_layer_heterogeneous = group_experts_on_collaboration_heterogeneous_group(expert_collab_counts = collab2, num_groups = 4, num_fast_groups = 2 ,even_groups = True)

        all_layers_placement[f"{layer_id}"] = placement_per_layer
        all_layers_placement_heterogeneous[f"{layer_id}"] = placement_per_layer_heterogeneous

    placement_file_path = os.path.join(placement_dir, f"{model_name}_{input_name}_placement_{num_of_prompts}.json")
    with open(placement_file_path,"w") as f1:
        json.dump(all_layers_placement, f1, indent=2)

    placement_heterogeneous_file_path = os.path.join(placement_dir, f"{model_name}_{input_name}_placement_{num_of_prompts}_heterogeneous.json")
    with open(placement_heterogeneous_file_path,"w") as f2:
        json.dump(all_layers_placement_heterogeneous, f2, indent=2)

    # ####################
    # collab = experts_collaboration_matrix[7] 
    # collab = torch.from_numpy(collab) 

    # # print(collab.shape)

    # # collab1 = copy.deepcopy(collaboration_per_layer)
    # # collab2 = copy.deepcopy(collaboration_per_layer)
        
    # #placement_per_layer = group_experts_on_collaboration(expert_collab_counts = collab1, num_groups = 4, even_groups = True)
    # placement_heterogeneous = group_experts_on_collaboration_heterogeneous_group(expert_collab_counts = collab, num_groups = 4, num_fast_groups = 2 ,even_groups = True)
    # print(placement_heterogeneous)
