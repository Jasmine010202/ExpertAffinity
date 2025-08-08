import numpy as np
import os
import torch
from sklearn.cluster import SpectralClustering
import json


__all__ = ["spectral_cluster_on_collaboration_uneven", "spectral_cluster_on_collaboration_even", "spectral_cluster_on_collaboration_semi_even"]




# 不均衡
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

# 均衡
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


# 半均衡
def spectral_cluster_on_collaboration_semi_even(experts_collaboration_matrix, num_clusters, even_rate):
    num_experts = experts_collaboration_matrix.shape[0]
    expected_per_cluster = num_experts // num_clusters  # 每个GPU的专家数量的理想值（平均）

    # # 根据比例计算每组专家数目的上下界
    # delta = int(round(expected_per_cluster * even_rate))
    # min_per_cluster = expected_per_cluster - delta
    # max_per_cluster = expected_per_cluster + delta

    if even_rate == 0:
        min_per_cluster = expected_per_cluster
        max_per_cluster = expected_per_cluster
    else:
        delta = max(1, int(round(expected_per_cluster * even_rate)))
        min_per_cluster = max(1, expected_per_cluster - delta)
        max_per_cluster = expected_per_cluster + delta

    min_per_cluster = max(1, min_per_cluster)   # 避免rate>1出现负数或者有的组没有专家的问题

    # print(f"Expert range per cluster: [{min_per_cluster}, {max_per_cluster}]")

    collaboration_matrix = experts_collaboration_matrix.astype(np.float32)   # 默认要求浮点数
    final_cluster_list = [[] for _ in range(num_clusters)]   # 最终聚类结果

    # step1: 初步谱聚类，结果是不均衡的
    initial_clusters = spectral_cluster_on_collaboration_uneven(collaboration_matrix, num_clusters)

    # 聚类调整
    # step2: 处理专家数超过上界的聚类
    overflow_experts = [] # 存超过上界的专家
    for cluster_id, experts in enumerate(initial_clusters):
        if len(experts) > max_per_cluster:
            sub_collab_matrix = collaboration_matrix[np.ix_(experts, experts)] # 取这一组专家对应的亲和性矩阵
            collab_scores = sub_collab_matrix.sum(axis=1) # 对每一行求和-这个专家和cluster里所有专家的亲和性之和
            sorted_indices = np.argsort(-collab_scores) # 对亲和性降序排序 -：降序，默认升序
            
            selected = [experts[i] for i in sorted_indices[:max_per_cluster]] # 保留亲和性最高的前max_per_cluster个
            overflow = [experts[i] for i in sorted_indices[max_per_cluster:]] # 超过max_per_cluster个，后面重新分组
            
            final_cluster_list[cluster_id] = selected
            overflow_experts.extend(overflow)
        else:
            final_cluster_list[cluster_id] = experts.copy()

    # print("final_cluster_list", final_cluster_list)
    # print("overflow_experts", overflow_experts)
    
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


    # step3: 处理专家数少于下界的聚类，从数目多的分组里选和当前组亲和性比较高的专家，迁移过来
    # 目标：尽量保住已有分组的亲和性结构不被破坏，并去补全数目不足的聚类，让负载尽量均衡，
    # 而且可以知道每个不足的聚类到底缺多少个专家，
    # 那么从整体上，所有可以提供迁移专家的聚类里，选择对应数目的边缘专家（在原先聚类里亲和度最低），去放到对他们来说亲和性较高的聚类去补全数目
    
    # 专家数不足下界的聚类，和需要的专家数目
    undersized_clusters = {cluster_id : min_per_cluster - len(final_cluster_list[cluster_id]) 
                           for cluster_id in range(num_clusters) if len(final_cluster_list[cluster_id]) < min_per_cluster}
    # print("undersized_clusters", undersized_clusters)
    free_slots = sum(undersized_clusters.values())
    # print("free_slots", free_slots)

    if free_slots > 0:
        donor_experts_pool = [] # (expert_id, donor_cluster_id, score) 可迁移的专家id，所属聚类，该专家和聚类内其他专家的总体亲和性

        # 找可以提供专家的聚类和可以迁移的专家
        for donor_cluster_id, experts in enumerate(final_cluster_list):
            donor_capacity = len(experts) - min_per_cluster     # 可供迁移的专家数量
            if donor_capacity <= 0:     # 必须保证迁出专家之后，提供可迁移专家的cluster不会低于min_per_cluster
                continue

            sub_collab_matrix = collaboration_matrix[np.ix_(experts, experts)] # 取这一组专家对应的亲和性矩阵
            collab_scores = sub_collab_matrix.sum(axis=1) # 对每一行求和-这个专家和cluster里所有专家的亲和性之和
            normalized_scores = collab_scores / (len(experts) - 1 + 1e-6)   # 每个组专家数量不一样，归一化，让来自各个聚类的候选专家的组内亲和性有可比性 -1是为了去掉自己
            sorted_indices = np.argsort(normalized_scores) # 默认升序排序 

            # 取出可供迁移的专家,是和组内其他专家亲和性得分很低的一批
            for index in sorted_indices[:donor_capacity]:
                expert_id = experts[index]
                donor_experts_pool.append((expert_id, donor_cluster_id, normalized_scores[index]))

        # print("donor_experts_pool", donor_experts_pool)

        # 按照归一化的组内亲和性整体排序，选出刚好满足空余位置数目的备选专家
        donor_experts_pool.sort(key = lambda x: x[2])   # 升序
        # print("donor_experts_pool", donor_experts_pool)
        selected_experts = donor_experts_pool[:free_slots]  #取出刚好满足需求的最边缘专家（归一下亲和性最弱）


        # 给被选中的专家找到合适的聚类（亲和性高的）
        for expert_id, donor_cluster_id, _ in selected_experts:
            best_cluster = None
            best_score = -np.inf
            for cluster_id, num_slots in undersized_clusters.items():
                if num_slots <= 0:
                    continue
                temp_cluster = final_cluster_list[cluster_id] + [expert_id]
                score = collab_score_intra_cluster(collaboration_matrix, temp_cluster)

                if score > best_score:
                    best_score = score
                    best_cluster = cluster_id
                
            if best_cluster is not None:
                final_cluster_list[best_cluster].append(expert_id)
                final_cluster_list[donor_cluster_id].remove(expert_id)
                undersized_clusters[best_cluster] -= 1
            else:
                raise RuntimeError(f"Cann't find cluster for expert{expert}")
        
        
    # step4: 最终校验每个组的专家数
    for cluster_id, experts in enumerate(final_cluster_list):
        if not (min_per_cluster <= len(experts) <= max_per_cluster):
            raise RuntimeError(f"Cluster {cluster_id} has {len(experts)} experts, out of range [{min_per_cluster}, {max_per_cluster}]")
        # assert len(experts) == expected_per_cluster, f"Cluster {cluster_id} has {len(experts)} experts!"

    return final_cluster_list



if __name__ == "__main__":

    model_name = "OLMoE"#Switch_Transformer OLMoE
    input_name = "sonnet"   #GSM8K、sonnet、mbpp、conala
    top_k = 8 # ST:1,OL:8
    num_of_prompts = 512#
    num_of_experts_per_layer = 64

    num_of_nodes = 2
    num_of_gpus_per_node = 2

    collaboration_dir = f"./Occult_test/expert_collaboration"
    os.makedirs(collaboration_dir, exist_ok=True)

    placement_dir = f"./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/even_Node_semi_even_GPU"
    os.makedirs(placement_dir, exist_ok=True)
    # f"./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs" 
    

    # 路由数据，1024
    routing_data = np.load(f'/data/shared_workspace/hanyu/Occult_test/expert_trace/used_for_occult/{model_name}/{input_name}/top{top_k}/decode_routing_trace_{num_of_prompts}.npy')
  
    #experts_collaboration_matrix = generate_collaboration_matrix(routing_data)
    experts_collaboration_matrix = np.load(f"./Occult_test/expert_collaboration/{model_name}_Expert_Collaboration_{input_name}_{num_of_prompts}.npy")
    
    num_layers = experts_collaboration_matrix.shape[0]
    all_layers_placement = {}   #存所有层的专家放置结果

    even_group = True
    enable_semi_even = True
    even_rate = 0.9  # 0 <= rate <= 1 rate = 0就是均衡方案

    
    for layer_id in range(num_layers):
        group_list = [] # 记录本层专家分组

        collaboration_per_layer = experts_collaboration_matrix[layer_id] #每一层的专家协作矩阵

        # 单机多卡
        # group_list = spectral_cluster_on_collaboration_semi_even(collaboration_per_layer, num_of_nodes * num_of_gpus_per_node, even_rate)
        
        # 多机多卡
        if even_group:
            if enable_semi_even:
                node_groups = spectral_cluster_on_collaboration_semi_even(collaboration_per_layer, num_of_nodes, even_rate)
            else:
                node_groups = spectral_cluster_on_collaboration_even(collaboration_per_layer, num_of_nodes)
            # node_groups = spectral_cluster_on_collaboration_even(collaboration_per_layer, num_of_nodes)
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
                if enable_semi_even:
                    gpu_groups_intra_node = spectral_cluster_on_collaboration_semi_even(node_sub_collab_matrix, num_of_gpus_per_node, even_rate)
                else:
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

    placement_file_name = f"{model_name}_spectral_semi_even_{input_name}_{num_of_prompts}_nodes{num_of_nodes}_gpus{num_of_nodes*num_of_gpus_per_node}_rate{even_rate}.json"
    # f"OLMoE_sonnet_spectral_semi_even_placement_512_rate{even_rate}.json"

    placement_file_path = os.path.join(placement_dir, placement_file_name)
    with open(placement_file_path,"w") as f:
        json.dump(all_layers_placement, f, indent=2)