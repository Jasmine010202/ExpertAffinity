import numpy as np
import os
import torch
import json
import copy

from Occult_grouping import group_experts_on_collaboration


model_name = "OLMoE"#Switch_Transformer OLMoE
input_name = "sonnet"   #GSM8K、sonnet、mbpp、conala
top_k = 8 # ST:1,OL:8
num_of_prompts = 512#
num_of_experts_per_layer = 64

# 节点数、GPU数
num_of_nodes = 2
num_of_gpus_per_node = 2
num_of_gpus = num_of_nodes * num_of_gpus_per_node

#enable_copy = True
num_replicated_experts = 4 

collaboration_dir = f"./Occult_test/expert_collaboration"
os.makedirs(collaboration_dir, exist_ok=True)

placement_dir = f"./Occult_test/expert_placement/MultiNodes_MultiGPUs"
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
    # 路由
    # 生成放置方案，用used_for_occult
    routing_data = np.load(f"./Occult_test/expert_trace/used_for_occult/{model_name}/{input_name}/top{top_k}/decode_routing_trace_{num_of_prompts}.npy")
    
    # 各层协作矩阵
    # experts_collaboration_matrix = generate_collaboration_matrix(routing_data)
    experts_collaboration_matrix = np.load(f"./Occult_test/expert_collaboration/{model_name}_Expert_Collaboration_{input_name}_{num_of_prompts}.npy")
    
    # print(experts_collaboration_matrix.shape)
    # print(experts_collaboration_matrix)
        
    num_layers = experts_collaboration_matrix.shape[0]
    all_layers_placement = {}   #存所有层的专家放置结果


    for layer_id in range(num_layers):
        group_list = [] # 记录本层专家分组

        collaboration_per_layer = experts_collaboration_matrix[layer_id] #每一层的专家协作矩阵
        collaboration_per_layer = torch.from_numpy(collaboration_per_layer) # numpy转tensor

        collab_copy = copy.deepcopy(collaboration_per_layer) # 代码会修改协作性矩阵，置为0，为什么？
       
        node_groups = group_experts_on_collaboration(expert_collab_counts = collab_copy, num_groups = num_of_nodes, even_groups = True)
        #print(node_groups)
        
        for node_id, node_experts in enumerate(node_groups):
            node_experts_index_tensor = torch.tensor(node_experts, dtype=torch.long)
            #print(node_experts_index_tensor)
            node_sub_collab_matrix = collaboration_per_layer[node_experts_index_tensor][:, node_experts_index_tensor]
            #print(node_sub_collab_matrix.shape)
            
            # 节点内卡间再次分组 [!返回的是局部索引]
            gpu_groups_intra_node = group_experts_on_collaboration(expert_collab_counts = node_sub_collab_matrix, num_groups = num_of_gpus_per_node, even_groups = True)

            # print("node_experts", node_experts)
            # print("gpu_groups_intra_node", gpu_groups_intra_node)

            # 转回全局专家id
            for gpu_experts in gpu_groups_intra_node:
                global_experts_index = [node_experts[i] for i in gpu_experts]
                group_list.append(global_experts_index)
                # print("global_experts_index", global_experts_index)

        all_layers_placement[f"{layer_id}"] = group_list

    placement_file_name = f"{model_name}_{input_name}_{num_of_prompts}_nodes{num_of_nodes}_gpus{num_of_nodes*num_of_gpus_per_node}.json"
    placement_file_path = os.path.join(placement_dir, placement_file_name)
    with open(placement_file_path,"w") as f:
        json.dump(all_layers_placement, f, indent=2)

   
    