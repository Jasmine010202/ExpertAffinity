import numpy as np
import os
import torch
import json
import copy

from Occult_grouping import group_experts_on_collaboration
from Spectral_grouping_multi import spectral_cluster_on_collaboration_even, spectral_cluster_on_collaboration_uneven


model_name = "OLMoE"#Switch_Transformer OLMoE
input_name = "sonnet"   #GSM8K、sonnet、mbpp、conala
top_k = 8 # ST:1,OL:8
num_of_prompts = 512#

num_layers = 16
num_of_experts_per_layer = 64

method = "spectral" # occult spectral
even_groups = True
# 节点数、GPU数
num_of_nodes = 2
num_of_gpus_per_node = 2
num_of_gpus = num_of_nodes * num_of_gpus_per_node

enable_replicated = True
num_replicated_experts = 4
replicated_type = "Activation"      # Collaboration Activation

collaboration_dir = f"./Occult_test/expert_collaboration"
os.makedirs(collaboration_dir, exist_ok=True)


re_dir = f"/Duplicate/{replicated_type}" if enable_replicated else ""
placement_dir = f"./Occult_test/expert_placement/{method}/MultiNodes_MultiGPUs{re_dir}"
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



#  根据协作矩阵，先按节点分组，再按各节点上的GPU划分
def grouping_multi_nodes_gpus(collab_matrix: torch.Tensor, experts_global_index_list: list):

    experts_group_list = [] # 记录本层专家分组

    if method == "occult":
        #collab_matrix = torch.from_numpy(collab_matrix) # occult输入要求是tensor
        collab_matrix_copy = copy.deepcopy(collab_matrix) # 代码会修改协作性矩阵，置为0

        node_groups = group_experts_on_collaboration(expert_collab_counts = collab_matrix_copy, num_groups = num_of_nodes, even_groups = True)
        #print(node_groups)
    else:
        #print("spectral")
        collab_matrix = collab_matrix.numpy()   # spectral输入是numpy.array
        if even_groups:
            node_groups = spectral_cluster_on_collaboration_even(collab_matrix, num_of_nodes)
        else:
            node_groups = spectral_cluster_on_collaboration_uneven(collab_matrix, num_of_nodes)
        
    for node_id, node_experts in enumerate(node_groups):
        node_experts_index_tensor = torch.tensor(node_experts, dtype=torch.long)
        # print(node_experts_index_tensor)
        node_sub_collab_matrix = collab_matrix[node_experts_index_tensor][:, node_experts_index_tensor]
        #print(node_sub_collab_matrix.shape)
            
        # 节点内卡间再次分组 [!返回的是局部索引]
        if method == "occult":
            gpu_groups_intra_node = group_experts_on_collaboration(expert_collab_counts = node_sub_collab_matrix, num_groups = num_of_gpus_per_node, even_groups = True)
        else:
            #print("spectral")
            if even_groups:
                gpu_groups_intra_node = spectral_cluster_on_collaboration_even(node_sub_collab_matrix, num_of_gpus_per_node)
            else:
                gpu_groups_intra_node = spectral_cluster_on_collaboration_uneven(node_sub_collab_matrix, num_of_gpus_per_node)

        #print("node_experts", node_experts)
        #print("gpu_groups_intra_node", gpu_groups_intra_node)

        # 转回全局专家id
        for gpu_experts in gpu_groups_intra_node:
            global_experts_index = [experts_global_index_list[node_experts[i]] for i in gpu_experts]
            experts_group_list.append(global_experts_index)
            #print("global_experts_index", global_experts_index)
        
    return experts_group_list



def compute_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)



def experts_activations_count(routing_trace):
    experts_activation_stats = np.zeros((num_layers, num_of_experts_per_layer), dtype = int) 

    num_tokens = routing_trace.shape[0]

    for token_id in range(num_tokens):
        for layer_id in range(num_layers):
            for expert_id in routing_trace[token_id, layer_id]:
                experts_activation_stats[layer_id, expert_id]  += 1

    return experts_activation_stats



if __name__ == "__main__":
    # 路由
    # 生成放置方案，用used_for_occult
    # 
    routing_trace = np.load(f"/data/shared_workspace/hanyu/Occult_test/expert_trace/used_for_occult/by_prompt/{model_name}_sonnet_top{top_k}/decode_routing_trace_{num_of_prompts}.npy")
    
    # 各层协作矩阵
    # experts_collaboration_matrix = generate_collaboration_matrix(routing_trace)
    experts_collaboration_matrix = np.load(f"./Occult_test/expert_collaboration/{model_name}_Expert_Collaboration_{input_name}_{num_of_prompts}.npy")
    
    experts_collaboration_matrix = torch.from_numpy(experts_collaboration_matrix)
    # print(experts_collaboration_matrix.shape)
    # print(experts_collaboration_matrix)
        
    #num_layers = experts_collaboration_matrix.shape[0]
    #print(experts_collaboration_matrix.shape)

    experts_activation_matrix = torch.from_numpy(experts_activations_count(routing_trace))
    
    all_layers_placement = {}   #存所有层的专家放置结果
    all_layers_replicated_experts = {} #存每一层复制的专家

    # print(experts_collaboration_matrix.sum())
    # print(experts_activation_matrix.sum())

    for layer_id in range(num_layers):
        #group_list = []

        collaboration_per_layer = experts_collaboration_matrix[layer_id] #每一层的专家协作矩阵
        # collaboration_per_layer = collaboration_per_layer.to(torch.int) # 改整数
        #collaboration_per_layer = torch.from_numpy(collaboration_per_layer) # numpy转tensor


        if enable_replicated:
            
            replicated_experts = []

            if replicated_type == "Activation":
                # 方案一：挑激活次数最高的4个
                #experts_activations = collaboration_per_layer.sum(dim=1)
                experts_activation_count = experts_activation_matrix[layer_id]
                #print(experts_activation_count)
                replicated_experts = torch.topk(experts_activation_count, num_replicated_experts).indices.tolist()
                #print(replicated_experts)
            else:
                # 方案二：挑协作范围最广的前4个
                experts_collab_entropy = compute_entropy(collaboration_per_layer)  # (num_experts,)
                #print(experts_collab_entropy)
                replicated_experts = torch.topk(experts_collab_entropy, num_replicated_experts).indices.tolist()
                #print(replicated_experts)
                # sorted_expert_on_entropy = torch.argsort(expert_collab_entropy, descending=False)
                # fast_experts = sorted_expert_on_entropy[-num_fast_groups * group_capacity:]
                # slow_experts = sorted_expert_on_entropy[:num_slow_groups * group_capacity]


            # 取剩余专家协作矩阵
            remaining_experts = [i for i in range(num_of_experts_per_layer) if i not in replicated_experts]
            remaining_experts_index_tensor = torch.tensor(remaining_experts, dtype=torch.long)
            #print(remaining_experts_index_tensor)
            remaining_sub_collab_matrix = collaboration_per_layer[remaining_experts_index_tensor][:, remaining_experts_index_tensor ]

            # 剩余专家分组
            remaining_experts_groups = grouping_multi_nodes_gpus(remaining_sub_collab_matrix, remaining_experts)

            # # 第一组是复制的专家
            # group_list.append(replicated_experts)

            # for group in remaining_experts_groups:
            #     group_list.append(group)
                
                # final_group = replicated_experts + group
                # group_list.append(final_group)
            
            #print(remaining_experts_groups)
            all_layers_placement[f"{layer_id}"] = remaining_experts_groups
            all_layers_replicated_experts[f"{layer_id}"] = replicated_experts

        else:
            # 直接分组
            experts_global_index_list = list(range(num_of_experts_per_layer))
            experts_group_list = grouping_multi_nodes_gpus(collaboration_per_layer, experts_global_index_list)

            #print(group_list)
        
            all_layers_placement[f"{layer_id}"] = experts_group_list

        # break

    
    
    prefix = f"{model_name}_{input_name}_{num_of_prompts}"
    even_prefix = (
        "_even" if method == "spectral" and even_groups else
        "_uneven" if method == "spectral" and not even_groups else
        ""
    )
    suffix = f"_re{num_replicated_experts}" if enable_replicated else ""

    placement_file_name = f"{prefix}{even_prefix}_nodes{num_of_nodes}_gpus{num_of_nodes*num_of_gpus_per_node}{suffix}.json"
    placement_file_path = os.path.join(placement_dir, placement_file_name)
    with open(placement_file_path,"w") as f1:
        json.dump(all_layers_placement, f1, indent=2)

    if enable_replicated:
        replicated_experts_file_name = f"{prefix}{even_prefix}_nodes{num_of_nodes}_gpus{num_of_nodes*num_of_gpus_per_node}{suffix}_replicated_experts.json"
        replicated_file_path = os.path.join(placement_dir, replicated_experts_file_name)
        with open(replicated_file_path,"w") as f2:
            json.dump(all_layers_replicated_experts, f2, indent=2)

   
    