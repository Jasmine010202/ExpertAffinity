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

placement_dir = f"./Occult_test/expert_placement/Duplicate_Group_Level/MultiNodes_MultiGPU/Several_Replicas_of_Highest_Load_Group"
os.makedirs(placement_dir, exist_ok=True)


# enable_replicated = True

def generate_collaboration_matrix(routing_data):
    num_tokens, num_layers, _ = routing_data.shape
    experts_collaboration_matrix = np.zeros((num_layers, num_experts_per_layer, num_experts_per_layer))

    for layer_id in range(num_layers):
        for token_id in range(num_tokens):
            experts_intra_layer = routing_data[token_id,layer_id]
            for i in range(top_k):
                for j in range (i+1, top_k):
                    expert_i, expert_j = experts_intra_layer[i], experts_intra_layer[j]
                    experts_collaboration_matrix[layer_id, expert_i, expert_j] += 1
                    experts_collaboration_matrix[layer_id, expert_j, expert_i] += 1 # 对称矩阵
        
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


# 初次分组之后、复制专家之前，计算每个GPU的负载
def compute_gpu_claculation_load_per_layer(routing_trace, expert_placement):
    # gpus_token_load_per_layer = [{gpu_id:0 for gpu_id in range(num_gpus)} for layer in range(num_layers)]
    gpus_token_load_per_layer = np.full((num_layers, num_gpus), 0, dtype = int)
    
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

    '''
    Format: {layer_id: 
                    {"replicated_group": list_of_expert_ids, 
                     "original_gpu": int, 
                     "replica_gpu": [int, ...],
                     "num_replicas": int}}
    '''
    
    replication_info_per_layer = {} 


    for layer_id in range(num_layers):
        gpu_loads = gpus_loads_per_layer_initial[layer_id]   #当前层GPU负载

        # 根据计算负载的不均衡程度，决定复制数量 [计算负载偏斜系数]   
        max_load = np.max(gpu_loads)
        mean_load = np.mean(gpu_loads)      

        if mean_load == 0:
            skew_factor = 0     # 
        else:
            skew_factor = max_load / mean_load

        # num_replicas = 0
        # if skew_factor > 2:
        #     num_replicas = min(2, num_gpus // 2)
        # elif 1.5 <= skew_factor <= 2:
        #     num_replicas = 1
        # else: 
        #     replication_info_per_layer[layer_id] = None     # 偏斜程度<1.5,偏斜不大，可以不复制
        #     continue 

        num_replicas = 1
        if skew_factor > 2:
            num_replicas = min(2, num_gpus // 2)
        
        
        # print("layer_id", layer_id)
        # print("num_replicas", num_replicas)

        # 找到负载最大的GPU对应的专家组
        heavy_gpu_idx = np.argmax(gpu_loads) 
        heavy_group = all_layers_placement_dict_initial[str(layer_id)][heavy_gpu_idx]

        # 负载从小到大的 num_replicas 个GPU作为负载目标
        light_gpu_list = []
        sorted_gpu_indices = np.argsort(gpu_loads)
        light_gpu_list = sorted_gpu_indices[:num_replicas].tolist()

        # 如果没有找到合适的目标GPU，可能:GPU数量不够或负载都差不多
        if not light_gpu_list:
            replication_info_per_layer[layer_id] = None 
            continue  

        # 记录复制信息
        replication_info_per_layer[layer_id] = {
            "replicated_group": heavy_group,
            "original_gpu": int(heavy_gpu_idx),
            "target_gpus": light_gpu_list,
            "num_replicas": num_replicas
        }

        # 更新放置后的dict 
        for light_gpu_idx in light_gpu_list:
            replicated_placement_dict[str(layer_id)][light_gpu_idx].extend(heavy_group)

        # 更新复制后GPU权重
        num_total_instances = 1 + num_replicas
        load_per_instance = gpu_loads[heavy_gpu_idx] // num_total_instances

        # 原始GPU更新为分摊之后的一部分
        update_gpu_loads[layer_id][heavy_gpu_idx] = load_per_instance
        # 目标GPU，在原有专家组负载的基础上加上分摊的负载
        for light_gpu_idx in light_gpu_list:
            update_gpu_loads[layer_id][light_gpu_idx] += load_per_instance

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
            target_gpus = replication_info["target_gpus"]

            # 更新之后专家组原先所在GPU和副本所在GPU的负载
            replica_gpus_list = [original_gpu] + target_gpus
            replica_loads = [gpus_load_per_layer_updated[layer_id][gpu_id] for gpu_id in replica_gpus_list]

            epsilon = 1e-6  # 避免除零

            # 路由轮询权重和负载的倒数成正比，负载越低，权重越高
            instance_weights = [1.0 / (load + epsilon) for load in replica_loads]

            # 归一化后保存权重信息
            total_weight = sum(instance_weights)
            weights_info = {}

            for i, gpu_id in enumerate(replica_gpus_list):
                normalized_weight = instance_weights[i] / total_weight
                weights_info[int(gpu_id)] = float(normalized_weight)

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
    # experts_placement_matrix_updated = extract_expert_placement_multi_copies(num_layers, num_experts_per_layer, None, placement_dict_after_replicated)

    # 第四步，根据更新之后预估的GPU负载，计算路由轮询的权重
    polling_weights_replicated_experts = calculate_routing_polling_weights(gpus_load_per_layer_updated, replication_info_per_layer)
    # print(polling_weights_replicated_experts)
    

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