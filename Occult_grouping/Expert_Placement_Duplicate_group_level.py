import numpy as np
import os
import torch
import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns

from Occult_grouping import group_experts_on_collaboration
from Spectral_grouping_multi import spectral_cluster_on_collaboration_even, spectral_cluster_on_collaboration_uneven, spectral_cluster_on_collaboration_semi_even
from utils import extract_routing_trace, extract_expert_placement, extract_replicated_experts, prompt_to_gpu, extract_expert_placement_multi_copies


model_name = "OLMoE"#Switch_Transformer OLMoE
input_name = "sonnet"   #GSM8K、sonnet、mbpp、conala
top_k = 8 # ST:1,OL:8
num_of_prompts = 512#

num_layers = 16
num_experts_per_layer = 64

# 节点数、GPU数
num_nodes = 2
num_gpus_per_node = 2
num_gpus = num_nodes * num_gpus_per_node

# 分组方案
method = "spectral" # occult spectral
even_groups = False
# gpu_node_mapping = np.array([0, 0, 1, 1]) 

# 文件路径
collaboration_dir = f"./Occult_test/expert_collaboration"
os.makedirs(collaboration_dir, exist_ok=True)

placement_dir = f"./Occult_test/expert_placement/Duplicate_Group_Level/MultiNodes_MultiGPU/One_Replica_of_Highest_Load_Group"
os.makedirs(placement_dir, exist_ok=True)


# enable_replicated = True

def generate_collaboration_matrix(routing_data):
    num_tokens, num_layers, _ = routing_data.shape
    experts_collaboration_matrix = np.zeros((num_layers, num_experts_per_layer, num_experts_per_layer))

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
        collab_matrix_copy = copy.deepcopy(collab_matrix) # 代码会修改协作性矩阵，置为0
        node_groups = group_experts_on_collaboration(expert_collab_counts = collab_matrix_copy, num_groups = num_nodes, even_groups = True)
    else:
        collab_matrix = collab_matrix.numpy()   # spectral输入是numpy.array
        if even_groups:
            node_groups = spectral_cluster_on_collaboration_even(collab_matrix, num_nodes)
        else:
            node_groups = spectral_cluster_on_collaboration_uneven(collab_matrix, num_nodes)
        
    for node_id, node_experts in enumerate(node_groups):
        node_experts_index_tensor = torch.tensor(node_experts, dtype=torch.long)
        node_sub_collab_matrix = collab_matrix[node_experts_index_tensor][:, node_experts_index_tensor]
            
        # 节点内卡间再次分组 [!返回的是局部索引]
        if method == "occult":
            gpu_groups_intra_node = group_experts_on_collaboration(expert_collab_counts = node_sub_collab_matrix, num_groups = num_gpus_per_node, even_groups = True)
        else:
            if even_groups:
                gpu_groups_intra_node = spectral_cluster_on_collaboration_even(node_sub_collab_matrix, num_gpus_per_node)
            else:
                gpu_groups_intra_node = spectral_cluster_on_collaboration_uneven(node_sub_collab_matrix, num_gpus_per_node)

        # 转回全局专家id
        for gpu_experts in gpu_groups_intra_node:
            global_experts_index = [experts_global_index_list[node_experts[i]] for i in gpu_experts]
            experts_group_list.append(global_experts_index)
        
    return experts_group_list


# def experts_activations_count(routing_trace):
#     experts_activation_stats = np.zeros((num_layers, num_experts_per_layer), dtype = int) 

#     num_tokens = routing_trace.shape[0]

#     for token_id in range(num_tokens):
#         for layer_id in range(num_layers):
#             for expert_id in routing_trace[token_id, layer_id]:
#                 experts_activation_stats[layer_id, expert_id]  += 1

#     return experts_activation_stats


# 初次分组之后、复制专家之前，计算每个GPU的负载
def compute_gpu_claculation_load_per_layer(routing_trace, expert_placement):
    # gpus_token_load_per_layer = [{gpu_id:0 for gpu_id in range(num_gpus)} for layer in range(num_layers)]
    gpus_token_load_per_layer = np.full((num_layers, num_gpus), 0, dtype=int)
    
    num_tokens = routing_trace.shape[0]

    for token_id in range(num_tokens):
        for layer_id in range(num_layers):
            expert_ids = routing_trace[token_id, layer_id] # 这一层激活的专家
            for expert_id in expert_ids:
                expert_gpu_id = expert_placement[layer_id, expert_id]
                gpus_token_load_per_layer[layer_id][expert_gpu_id] += 1

    return gpus_token_load_per_layer


def replicate_heavy_experts_group(gpus_loads_per_layer_initial, all_layers_placement_dict_initial):
    replicated_placement_dict = copy.deepcopy(all_layers_placement_dict_initial)

    update_gpu_loads = copy.deepcopy(gpus_loads_per_layer_initial)

    # Format: {layer_id: {"replicated_group": list_of_expert_ids, "original_gpu": int, "replica_gpu": int}}
    replication_info_per_layer = {} 


    for layer_id in range(num_layers):
        gpu_loads = gpus_loads_per_layer_initial[layer_id]   #当前层GPU负载

        # 找到最大和最小负载的GPU
        heavy_gpu_idx = np.argmax(gpu_loads) 
        light_gpu_idx = np.argmin(gpu_loads)

        if heavy_gpu_idx == light_gpu_idx:
            replication_info_per_layer[layer_id] = None 
            continue  # 没有负载差异，跳过

        heavy_group = all_layers_placement_dict_initial[str(layer_id)][heavy_gpu_idx]

        # 记录复制信息
        replication_info_per_layer[layer_id] = {
            "replicated_group": heavy_group,
            "original_gpu": int(heavy_gpu_idx),
            "target_gpu": int(light_gpu_idx)
        }

        # 更新放置后的dict 
        replicated_placement_dict[str(layer_id)][light_gpu_idx].extend(heavy_group)

        # 更新复制后GPU权重（简易版:现在只有一组专家被复制和一个副本，就负载减半然后转移）
        # load_delta = gpu_loads[heavy_gpu_idx] / 2.0
        load_delta = gpu_loads[heavy_gpu_idx] // 2
        update_gpu_loads[layer_id][heavy_gpu_idx] -= load_delta
        update_gpu_loads[layer_id][light_gpu_idx] += load_delta

    return replicated_placement_dict, update_gpu_loads, replication_info_per_layer


# 根据刚更新后的负载计算轮询的权重
def calculate_routing_polling_weights(gpus_load_per_layer_updated, replication_info_per_layer):

    polling_weights = {}
    
    for layer_id in range(num_layers):
        weights_for_current_layer = {}

        replication_info = replication_info_per_layer[layer_id]

        if replication_info is not None: 
            replicated_group = replication_info["replicated_group"]
            original_gpu = replication_info["original_gpu"]
            target_gpu = replication_info["target_gpu"]

            # 更新之后专家组原先所在GPU和副本所在GPU的负载
            original_gpu_load = gpus_load_per_layer_updated[layer_id][original_gpu]
            target_gpu_load = gpus_load_per_layer_updated[layer_id][target_gpu]

            epsilon = 1e-6  # 避免除零

            # 路由轮询权重和负载的倒数成正比，负载越低，权重越高
            original_weight = 1.0 / (original_gpu_load + epsilon)
            target_weight = 1.0 / (target_gpu_load + epsilon)

            # 归一化
            total_weight = original_weight + target_weight
            original_weight_normalized = original_weight / total_weight
            target_weight_normalized = target_weight / total_weight

            weights_info = {
                int(original_gpu) : float(original_weight_normalized),
                int(target_gpu) : float(target_weight_normalized)
                }

            weights_for_current_layer[str(replicated_group)] = weights_info
        
        polling_weights[str(layer_id)] = weights_for_current_layer

    return polling_weights


if __name__ == "__main__":
    # 路由
    # 生成放置方案，用used_for_occult
    routing_trace = np.load(f"/data/shared_workspace/hanyu/Occult_test/expert_trace/used_for_occult/by_prompt/{model_name}_sonnet_top{top_k}/decode_routing_trace_{num_of_prompts}.npy")
    # print(routing_trace.shape)
    # 各层协作矩阵
    # experts_collaboration_matrix = generate_collaboration_matrix(routing_trace)
    experts_collaboration_matrix = np.load(f"./Occult_test/expert_collaboration/{model_name}_Expert_Collaboration_{input_name}_{num_of_prompts}.npy")
    experts_collaboration_matrix = torch.from_numpy(experts_collaboration_matrix)
    
    # experts_activation_matrix = torch.from_numpy(experts_activations_count(routing_trace))
    
    # all_layers_replicated_experts = {} #存每一层复制的专家

    # 第一步，生成初始放置方案
    all_layers_placement_dict_initial = {}   #存所有层的专家放置结果

    for layer_id in range(num_layers):
        collaboration_per_layer = experts_collaboration_matrix[layer_id] #每一层的专家协作矩阵
        experts_global_index_list = list(range(num_experts_per_layer))
        
        experts_group_list = grouping_multi_nodes_gpus(collaboration_per_layer, experts_global_index_list)
        all_layers_placement_dict_initial[f"{layer_id}"] = experts_group_list
    
    # 放置方案转成matrix [num_layers, num_expert_per_layer]
    experts_placement_matrix_initial = extract_expert_placement(num_layers, num_experts_per_layer, None, all_layers_placement_dict_initial)
    # print(experts_placement_matrix_initial.shape)

    # 第二步，计算每一层每个GPU的负载
    gpus_load_per_layer_initial = compute_gpu_claculation_load_per_layer(routing_trace, experts_placement_matrix_initial)
    # print(gpus_load_per_layer_initial)

    # 第三步，找到承载负载最大和最小专家组的GPU，把负载大的专家组复制到小的GPU上去，顺便根据副本数量预估平衡后的负载
    placement_dict_after_replicated, gpus_load_per_layer_updated, replication_info_per_layer = replicate_heavy_experts_group(gpus_load_per_layer_initial, all_layers_placement_dict_initial)
    # print(gpus_load_per_layer_updated)
    # print(replication_info_per_layer)
    # 生成有副本的专家放置matrix
    experts_placement_matrix_updated = extract_expert_placement_multi_copies(num_layers, num_experts_per_layer, None, placement_dict_after_replicated)

    # 第四步，根据更新之后预估的GPU负载，计算路由轮询的权重
    polling_weights_replicated_experts = calculate_routing_polling_weights(gpus_load_per_layer_updated, replication_info_per_layer)
    print(polling_weights_replicated_experts)
    
    '''
    wrr_weights = calculate_wrr_weights(gpus_load_per_layer_updated, replication_info_per_layer)
   

    # --- 第五步：模拟实际路由，计算最终每层GPU负载 ---
    print("\n--- Step 5: Simulating final GPU loads with WRR routing ---")
    final_gpus_load_per_layer = compute_final_load_with_wrr_routing(
        routing_trace,
        experts_placement_matrix_initial, # 路由时仍然需要原始放置来确定未复制专家的GPU
        replication_info_per_layer,
        wrr_weights
    )
    print("Final GPU loads simulated.")

    '''

    # 各项文件保存    
    prefix = f"{model_name}_{input_name}_{num_of_prompts}"
    if method == "spectral":
        if even_groups:
            even_prefix = "_even"
        else:
            even_prefix = "_uneven"
    else:
        even_prefix = ""

    # 初始放置方案
    initial_file_name = f"{prefix}{even_prefix}_nodes{num_nodes}_gpus{num_nodes*num_gpus_per_node}_initial.json"
    initial_file_path = os.path.join(placement_dir, initial_file_name)
    with open(initial_file_path,"w") as f1:
        json.dump(all_layers_placement_dict_initial, f1, indent=2)

    # 复制后的放置方案
    replicated_file_name = f"{prefix}{even_prefix}_nodes{num_nodes}_gpus{num_nodes*num_gpus_per_node}_replicated.json"
    replicated_file_path = os.path.join(placement_dir, replicated_file_name)
    with open(replicated_file_path,"w") as f2:
        json.dump(placement_dict_after_replicated, f2, indent=2)

    # 复制信息
    replicated_info_file_name = f"{prefix}{even_prefix}_nodes{num_nodes}_gpus{num_nodes*num_gpus_per_node}_replicated_info.json"
    replicated_info_file_path = os.path.join(placement_dir, replicated_info_file_name)
    with open(replicated_info_file_path,"w") as f3:
        json.dump(replication_info_per_layer, f3, indent=2)

    # 被复制专家组的轮询权重
    polling_weights_file_name = f"{prefix}{even_prefix}_nodes{num_nodes}_gpus{num_nodes*num_gpus_per_node}_polling_weights.json"
    polling_weights_file_path = os.path.join(placement_dir, polling_weights_file_name)
    with open(polling_weights_file_path,"w") as f4:
        json.dump(polling_weights_replicated_experts, f4, indent=2)