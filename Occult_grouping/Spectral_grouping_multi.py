import numpy as np
import os
import torch
from sklearn.cluster import SpectralClustering
import json


__all__ = ["spectral_cluster_on_collaboration_uneven", "spectral_cluster_on_collaboration_even"]

model_name = "OLMoE"#Switch_Transformer OLMoE
input_name = "sonnet"   #GSM8K、sonnet、mbpp、conala
top_k = 8 # ST:1,OL:8
num_of_prompts = 512#
num_of_experts_per_layer = 64

num_of_nodes = 2
num_of_gpus_per_node = 2

collaboration_dir = f"./Occult_test/expert_collaboration"
os.makedirs(collaboration_dir, exist_ok=True)

placement_dir = f"./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs"
os.makedirs(placement_dir, exist_ok=True)


def spectral_cluster_on_collaboration_uneven(experts_collaboration_matrix, num_clusters):
    collaboration_matrix = experts_collaboration_matrix.astype(np.float32)   # 默认要求浮点数
    # spectral 谱聚类
    s_clustering = SpectralClustering(n_clusters = num_clusters, affinity = 'precomputed', assign_labels = 'kmeans', random_state = 42)
    cluster_results = s_clustering.fit_predict(collaboration_matrix)

    cluster_list = [[] for _ in range(num_clusters)]
    for expert_id, cluster_id in enumerate(cluster_results):
        cluster_list[cluster_id].append(int(expert_id)) #cluster_id -> list of experts_id

    return cluster_list


def collab_score_intra_cluster(collaboration_matrix, temp_cluster):
    if len(temp_cluster) <= 1:
        return 0
    sub_matrix = collaboration_matrix[np.ix_(temp_cluster, temp_cluster)] # 亲和性子矩阵
    return sub_matrix.sum()


def spectral_cluster_on_collaboration_even(experts_collaboration_matrix, num_clusters):
    num_experts = experts_collaboration_matrix.shape[0]
    max_per_cluster = num_experts // num_clusters  # 每个GPU的专家数量

    collaboration_matrix = experts_collaboration_matrix.astype(np.float32)   # 默认要求浮点数

    # 谱聚类 初步
    initial_clusters = spectral_cluster_on_collaboration_uneven(collaboration_matrix, num_clusters)

    # 聚类调整
    final_cluster_list = [[] for _ in range(num_clusters)]

    overflow_experts = [] # 存超过16个专家
    for cluster_id, experts in enumerate(initial_clusters):
        if len(experts) > max_per_cluster:
            sub_collab_matrix = collaboration_matrix[np.ix_(experts, experts)] # 取这一组专家对应的亲和性矩阵
            collab_scores = sub_collab_matrix.sum(axis=1) # 对每一行求和-这个专家和cluster里所有专家的亲和性之和
            sorted_indices = np.argsort(-collab_scores) # 对亲和性降序排序 -：降序，默认升序
            selected = [experts[i] for i in sorted_indices[:max_per_cluster]] # 保留亲和性最高的前16个
            overflow = [experts[i] for i in sorted_indices[max_per_cluster:]] # 超过16个，后面重新分组
            final_cluster_list[cluster_id] = selected
            overflow_experts.extend(overflow)
        else:
            final_cluster_list[cluster_id] = experts.copy()

    #对于初次聚类多出来的专家找一个亲和性较高的聚类
    for expert in overflow_experts:
        best_cluster = None
        best_score = -np.inf

        for cluster_id, experts in enumerate(final_cluster_list):
            if len(experts) >= max_per_cluster:
                continue
            temp_cluster = experts + [expert] # 把这个专家加入当前聚类，测亲和性
            score = collab_score_intra_cluster(collaboration_matrix, temp_cluster)
            if score > best_score:
                best_score = score
                best_cluster = cluster_id

        if best_cluster is not None:
            final_cluster_list[best_cluster].append(expert)
        else:
            raise RuntimeError(f"Cann't find cluster for expert{expert}")

    for cluster_id, experts in enumerate(final_cluster_list):
        assert len(experts) == max_per_cluster, f"Cluster {cluster_id} has {len(experts)} experts!"

    return final_cluster_list



if __name__ == "__main__":

    # 路由数据，1024
    routing_data = np.load(f'./Occult_test/expert_trace/used_for_occult/{model_name}/{input_name}/top{top_k}/decode_routing_trace_{num_of_prompts}.npy')
  
    #experts_collaboration_matrix = generate_collaboration_matrix(routing_data)
    experts_collaboration_matrix = np.load(f"./Occult_test/expert_collaboration/{model_name}_Expert_Collaboration_{input_name}_{num_of_prompts}.npy")
    
    num_layers = experts_collaboration_matrix.shape[0]
    all_layers_placement = {}   #存所有层的专家放置结果

    even_group = True

    
    for layer_id in range(num_layers):
        group_list = [] # 记录本层专家分组

        collaboration_per_layer = experts_collaboration_matrix[layer_id] #每一层的专家协作矩阵
        
        if even_group:
            node_groups = spectral_cluster_on_collaboration_even(collaboration_per_layer, num_of_nodes)
        else:
            node_groups = spectral_cluster_on_collaboration_uneven(collaboration_per_layer, num_of_nodes)
        # print(node_groups)
        
        for node_id, node_experts in enumerate(node_groups):
            node_experts_index_tensor = torch.tensor(node_experts, dtype=torch.long)
            #print(node_experts_index_tensor)
            node_sub_collab_matrix = collaboration_per_layer[node_experts_index_tensor][:, node_experts_index_tensor]
            # print(node_sub_collab_matrix.shape)
            
            # 节点内卡间再次分组 [!返回的是局部索引]
            if even_group:
                gpu_groups_intra_node = spectral_cluster_on_collaboration_even(node_sub_collab_matrix, num_of_gpus_per_node)
            else:
                gpu_groups_intra_node = spectral_cluster_on_collaboration_uneven(node_sub_collab_matrix, num_of_gpus_per_node)

            # print("node_experts", node_experts)
            # print("gpu_groups_intra_node", gpu_groups_intra_node)

            # 转回全局专家id
            for gpu_experts in gpu_groups_intra_node:
                global_experts_index = [node_experts[i] for i in gpu_experts]
                group_list.append(global_experts_index)
                # print("global_experts_index", global_experts_index)

        all_layers_placement[f"{layer_id}"] = group_list
        # break

    # print(all_layers_placement)

    placement_file_name = f"{model_name}_spectral_even_{input_name}_{num_of_prompts}_nodes{num_of_nodes}_gpus{num_of_nodes*num_of_gpus_per_node}.json"
    placement_file_path = os.path.join(placement_dir, placement_file_name)
    with open(placement_file_path,"w") as f:
        json.dump(all_layers_placement, f, indent=2)