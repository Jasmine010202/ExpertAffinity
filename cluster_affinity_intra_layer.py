import numpy as np
import os
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import json
import seaborn as sns
import matplotlib.patches as patches
import networkx as nx


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="OLMoE", help='Name of the model')
parser.add_argument('--input_name', type=str, default="sonnet", help='Name of the dataset')
parser.add_argument('--phrase_mode', type=str, default="decode", help='encode or decode')
parser.add_argument('--top_k', type=int, default=8,help='Value of top_k')
parser.add_argument('--num_of_experts_pre_layer', type=int, default=64,help='Number of experts per layer')
args = parser.parse_args()


def affinity_intra_layer(routing_data, num_of_experts_pre_layer):
    num_tokens, num_layers, top_k = routing_data.shape
    affinity_matrix_all_layers = np.zeros((num_layers, num_of_experts_pre_layer, num_of_experts_pre_layer))

    for layer in range(num_layers):
        for token in range(num_tokens):
            experts_intra_layer = routing_data[token,layer]
            for i in range(top_k):
                for j in range (i+1, top_k):
                    expert_i, expert_j = experts_intra_layer[i], experts_intra_layer[j]
                    affinity_matrix_all_layers[layer, expert_i, expert_j] += 1
                    affinity_matrix_all_layers[layer, expert_j, expert_i] += 1 # 对称矩阵
    return affinity_matrix_all_layers


def affinity_score_intra_cluster(affinity_matrix, temp_cluster):
    if len(temp_cluster) <= 1:
        return 0
    sub_matrix = affinity_matrix[np.ix_(temp_cluster, temp_cluster)] # 亲和性子矩阵
    return sub_matrix.sum()


def balanced_clusters_for_all_layers(affinity_matrix_all_layers, num_clusters, result_path=None):
    num_layers, num_experts, _ = affinity_matrix_all_layers.shape
    max_per_cluster = num_experts // num_clusters  # 每个GPU的专家数量
    all_layers_clusters = {}

    for layer_id in range(num_layers):
        affinity_matrix = affinity_matrix_all_layers[layer_id].astype(np.float32)   # 默认要求浮点数

        # 谱聚类 初步
        clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
        labels = clustering.fit_predict(affinity_matrix)

        initial_clusters = defaultdict(list)
        for expert_id, cluster_id in enumerate(labels):
            initial_clusters[int(cluster_id)].append(int(expert_id))  #cluster_id -> list of experts_id
        
        # 聚类调整，每个16个
        final_clusters = defaultdict(list)

        overflow_experts = [] # 超过16个专家
        for cluster_id, experts in initial_clusters.items():
            if len(experts) > max_per_cluster:
                sub_affinity_matrix = affinity_matrix[np.ix_(experts, experts)] # 取这一组专家对应的亲和性矩阵
                affinity_scores = sub_affinity_matrix.sum(axis=1) # 对每一行求和-这个专家和cluster里所有专家的亲和性之和
                sorted_indices = np.argsort(-affinity_scores) # 对亲和性降序排序 -：降序，默认升序
                selected = [experts[i] for i in sorted_indices[:max_per_cluster]] # 保留亲和性最高的前16个
                overflow = [experts[i] for i in sorted_indices[max_per_cluster:]] # 超过16个，后面重新分组
                final_clusters[cluster_id] = selected
                overflow_experts.extend(overflow)
            else:
                final_clusters[cluster_id] = experts.copy()
        
        #对于初次聚类多出来的专家找一个亲和性较高的聚类
        for expert in overflow_experts:
            best_cluster = None
            best_score = -np.inf

            for cluster_id, experts in final_clusters.items():
                if len(experts) >= max_per_cluster:
                    continue
                temp_cluster = experts + [expert] # 把这个专家加入当前聚类，测亲和性
                score = affinity_score_intra_cluster(affinity_matrix, temp_cluster)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_id

            if best_cluster is not None:
                final_clusters[best_cluster].append(expert)
            else:
                raise RuntimeError(f"Cann't find cluster for expert{expert}")
            
        for cluster_id, experts in final_clusters.items():
            assert len(experts) == max_per_cluster, f"Cluster {cluster_id} has {len(experts)} experts!"

        all_layers_clusters[f"layer_{layer_id}"] = final_clusters

    if result_path:
        with open(result_path, 'w') as result_file:
            json.dump(all_layers_clusters, result_file, indent=4)
        print(f"Cluster results saved to : {result_path}")

    return all_layers_clusters

def clusters_for_all_layers(affinity_matrix_all_layers, num_clusters, result_path=None):
    num_layers = affinity_matrix_all_layers.shape[0]
    all_layers_clusters = {}

    for layer_id in range(num_layers):
        affinity_matrix = affinity_matrix_all_layers[layer_id].astype(np.float32)   # 默认要求浮点数

        # spectral 谱聚类
        clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)
        labels = clustering.fit_predict(affinity_matrix)
        # print(labels)

        layer_clusters = defaultdict(list)
        for expert_id, cluster_id in enumerate(labels):
            layer_clusters[int(cluster_id)].append(int(expert_id))  #cluster_id -> list of experts_id

        all_layers_clusters[f"layer_{layer_id}"] = layer_clusters

    if result_path:
        with open(result_path, 'w') as result_file:
            json.dump(all_layers_clusters, result_file, indent=4)
        print(f"Cluster results saved to : {result_path}")

    return all_layers_clusters

# 聚类后的层内亲和性图
def plot_after_clustering(affinity_matrix, labels, layer_id, fig_dir):
    filename = os.path.join(fig_dir, f"layer{layer_id}_affinity_after_cluster.png")

    num_experts = len(labels)
    
    sorted_indices = np.argsort(labels) # 按聚类顺序重排id
    # print(sorted_indices)
    sorted_labels = labels[sorted_indices] # 聚类编号
    # print(sorted_labels)
    sorted_expert_ids = [str(i) for i in sorted_indices]    # 转str
    # print(sorted_expert_ids)

    reordered_matrix = affinity_matrix[sorted_indices][:, sorted_indices]   # 重排亲和矩阵
    # print(reordered_matrix)

    # 红色分界线
    group_counts = [np.sum(sorted_labels == g) for g in sorted(set(labels))]
    boundaries = np.cumsum(group_counts)

    fig, ax = plt.subplots(figsize=(12,10))
    #plt.figure(figsize=(12, 10))

    sns.heatmap(reordered_matrix, annot=False, 
        cmap="YlGnBu", linewidths=0.5, square=True, 
        cbar_kws={"shrink": 0.5}, ax=ax)
    
    ax.set_xticks(np.arange(num_experts) + 0.5)
    ax.set_yticks(np.arange(num_experts) + 0.5)
    ax.set_xticklabels(sorted_expert_ids, rotation=90, fontsize=7)
    ax.set_yticklabels(sorted_expert_ids, rotation=0, fontsize=7)

    # 分界线
    for b in boundaries[:-1]:
        ax.axhline(b, color='red', linestyle='--', linewidth=1.2)
        ax.axvline(b, color='red', linestyle='--', linewidth=1.2)

    plt.title(f"Affinity Matrix (After cluster) — Layer {layer_id}")
    plt.xlabel("Expert ID")
    plt.ylabel("Expert ID")
    
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Picture for layer_{layer_id} saved to : {filename}")



if __name__ == "__main__":

    model_name = args.model_name
    input_name = args.input_name
    phrase_mode = args.phrase_mode
    top_k = args.top_k
    num_of_experts_pre_layer = args.num_of_experts_pre_layer

    fig_dir = f"cluster_result/spectral/balance/figs"
    os.makedirs(fig_dir, exist_ok=True)
    result_path = f"cluster_result/spectral/balance/clusters_result_all_layers.json"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    # 路由数据，1024
    routing_data = np.load(f'expert_trace/{model_name}/{input_name}/top{top_k}/{phrase_mode}_routing_trace_1024.npy')
  
    affinity_matrix_all_layers = affinity_intra_layer(routing_data, num_of_experts_pre_layer)
    
    #all_clusters = clusters_for_all_layers(affinity_matrix_all_layers, num_clusters=4, result_path=result_path)
    all_clusters = balanced_clusters_for_all_layers(affinity_matrix_all_layers, num_clusters=4, result_path=result_path)
    

    # 聚类后的亲和性图
    for layer_id in range(len(affinity_matrix_all_layers)):
        labels = np.zeros(num_of_experts_pre_layer, dtype=int)
        layer_clusters = all_clusters[f"layer_{layer_id}"]

        for group_id, expert_list in layer_clusters.items():
            for expert_id in expert_list:
                labels[expert_id] = int(group_id)
        
        plot_after_clustering(affinity_matrix_all_layers[layer_id], labels, layer_id,fig_dir=fig_dir) 