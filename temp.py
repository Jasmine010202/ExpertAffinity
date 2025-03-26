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


# def cluster_experts_per_layer(affinity_matrix_all_layers):
#     num_layers, num_experts, _ = affinity_matrix_all_layers.shape
    
#     placement = []
#     for co_mat in co_matrices:
#         # 降维后聚类
#         pca = PCA(n_components=10)
#         reduced = pca.fit_transform(co_mat)
#         kmeans = KMeans(n_clusters=num_gpus, random_state=0).fit(reduced)
#         groups = defaultdict(list)
#         for expert_id, cluster_id in enumerate(kmeans.labels_):
#             groups[cluster_id].append(expert_id)
#         placement.append(groups)
#     return placement  # List[Dict[int, List[int]]]

# def cluster_and_plot(co_matrix, n_clusters=4, use_pca=False, pca_dims=10, title=''):
#     # 每个专家的共现向量是共现矩阵的一行
#     features = co_matrix.astype(np.float32)
    
#     if use_pca:
#         pca = PCA(n_components=pca_dims)
#         reduced = pca.fit_transform(features)
#         features_used = reduced
#     else:
#         features_used = features

#     # 聚类
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     labels = kmeans.fit_predict(features_used)

#     # 2D 可视化（降到2维画图）
#     vis_features = PCA(n_components=2).fit_transform(features_used)

#     plt.figure(figsize=(5, 5))
#     plt.title(title)
#     plt.scatter(vis_features[:, 0], vis_features[:, 1], c=labels, cmap='tab10', s=60)
#     for i in range(len(labels)):
#         plt.text(vis_features[i, 0], vis_features[i, 1], str(i), fontsize=8, ha='center', va='center')
#     plt.xlabel("PCA1")
#     plt.ylabel("PCA2")
#     plt.grid(True)
#     #plt.show()
#     plt.savefig(title, dpi=300, bbox_inches="tight")
#     plt.close()

#     return labels

# def compare_cluster_labels(labels1, labels2, num_groups=4, verbose=True):
#     assert len(labels1) == len(labels2), "标签长度不一致"
#     num_items = len(labels1)

#     # 记录不一致的专家编号
#     disagreement = []
#     for i in range(num_items):
#         if labels1[i] != labels2[i]:
#             disagreement.append(i)

#     # 计算一致率
#     agreement_rate = 1 - len(disagreement) / num_items

#     if verbose:
#         print(f"总专家数: {num_items}")
#         print(f"分组不一致的专家数: {len(disagreement)}")
#         print(f"一致率: {agreement_rate:.2%}")
#         if disagreement:
#             print("不同分组的专家编号:", disagreement)

#     return {
#         "total": num_items,
#         "diff_count": len(disagreement),
#         "agreement_rate": agreement_rate,
#         "disagreement_indices": disagreement
#     }



# def cluster_experts_per_layer(co_matrix_all_layers, num_groups=4):
#     """
#     :param co_matrix_all_layers: shape (num_layers, 64, 64)
#     :return: List[Dict[group_id -> List[expert_ids]]]
#     """
#     num_layers, num_experts, _ = co_matrix_all_layers.shape
#     placement = []

#     for layer in range(num_layers):
#         co_matrix = co_matrix_all_layers[layer]

#         # 将每个专家的共现向量作为特征（64维）
#         features = co_matrix.astype(np.float32)

#         # 可选降维（有时能让聚类更稳定）
#         pca = PCA(n_components=10)
#         reduced = pca.fit_transform(features)

#         # KMeans 聚类
#         kmeans = KMeans(n_clusters=num_groups, random_state=0)
#         labels = kmeans.fit_predict(reduced)

#         # 构建 group -> expert_id 的映射
#         group_dict = {i: [] for i in range(num_groups)}
#         for expert_id, group_id in enumerate(labels):
#             group_dict[group_id].append(expert_id)

#         placement.append(group_dict)

#     return placement  # List[Dict[group_id -> List[expert_ids]]]

# def spectral_cluster_and_visualize(co_matrix, num_groups=4, layer_id=0, save_path=None):
#     """
#     对某层共现矩阵做谱聚类并保存可视化图像

#     :param co_matrix: shape (64, 64), 共现矩阵
#     :param num_groups: 聚类组数（通常等于GPU数）
#     :param layer_id: 当前层编号，用于命名保存文件
#     :param save_path: 可选，保存图片的路径，默认保存为 spectral_clustering_layer{layer_id}.png
#     :return: 聚类标签，shape (64,)
#     """
#     co_matrix = co_matrix.astype(np.float32)

#     # ---------------- Spectral Clustering ----------------
#     clustering = SpectralClustering(
#         n_clusters=num_groups,
#         affinity='precomputed',   # 直接使用你提供的亲和矩阵
#         assign_labels='kmeans',   # 或 'discretize' 也可以试试
#         random_state=42
#     )
#     labels = clustering.fit_predict(co_matrix)

#     # ---------------- Visualization ----------------
#     # 用 PCA 降到二维，做可视化
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(co_matrix)

#     plt.figure(figsize=(6, 6))
#     plt.title(f"Spectral Clustering - Layer {layer_id}")
#     plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=60)

#     # 标出每个点的专家编号
#     for i in range(len(labels)):
#         plt.text(reduced[i, 0], reduced[i, 1], str(i), fontsize=8, ha='center', va='center')

#     plt.xlabel("PCA1")
#     plt.ylabel("PCA2")
#     plt.grid(True)

#     filename = os.path.join(save_path, f"spectral_clustering_layer{layer_id}.png")

    
#     plt.savefig(filename)
#     plt.close()
#     print(f"图像已保存到: {filename}")

#     return labels


# def cluster_experts_per_layer(affinity_matrix, num_cluster, layer_id, save_path):
#     # 谱聚类
#     model = SpectralClustering(
#         n_clusters=num_cluster,
#         affinity='precomputed',
#         random_state=42,
#         assign_labels='discretize'
#     )
    
#     labels = model.fit_predict(affinity_matrix)
    
#     silhouette = silhouette_score(affinity_matrix, labels, metric='precomputed') #质量评估
#     print(f"Layer {layer_id} - Silhouette Score: {silhouette:.3f}")
    
#     # 保存聚类结果到JSON文件
#     cluster_dict = defaultdict(list)
#     for expert_id, cluster_id in enumerate(labels):
#         cluster_dict[int(cluster_id)].append(int(expert_id))  
    
#     result_dir = os.path.join(save_path, "cluster_results")
#     os.makedirs(result_dir, exist_ok=True)
#     with open(os.path.join(result_dir, f"layer_{layer_id}_clusters.json"), 'w') as f:
#         json.dump(cluster_dict, f, indent=2)
    

#     # 可视化并保存图片
#     plt.figure(figsize=(12, 10))
#     # 根据聚类标签重新排序矩阵
#     sorted_indices = np.argsort(labels)
#     sorted_matrix = affinity_matrix[sorted_indices][:, sorted_indices]
    
#     # 绘制热力图
#     ax = sns.heatmap(
#         sorted_matrix,
#         cmap="YlGnBu",
#         square=True,
#         cbar_kws={"shrink": 0.8},
#         xticklabels=sorted_indices,
#         yticklabels=sorted_indices
#     )
    
#     # 添加分隔线
#     unique_labels = np.unique(labels)
#     current_pos = 0
#     for label in unique_labels:
#         count = np.sum(labels == label)
#         ax.axhline(current_pos + count, color='red', linewidth=2)
#         ax.axvline(current_pos + count, color='red', linewidth=2)
#         current_pos += count
    
#     plt.title(f"Layer {layer_id} Expert Affinity (Clustered)\nSilhouette Score: {silhouette:.3f}")
#     plt.xlabel("Expert ID")
#     plt.ylabel("Expert ID")
#     plt.tight_layout()
    
#     # 保存图片
#     img_path = os.path.join(save_path, f"layer_{layer_id}_clusters.png")
#     plt.savefig(img_path, dpi=150, bbox_inches='tight')
#     plt.close()
    
#     return labels

def spectral_clustering_all_layers(affinity_matrix_all_layers, num_groups=4, save_json_path=None):
    num_layers = affinity_matrix_all_layers.shape[0]
    all_layer_groupings = {}

    for layer_id in range(num_layers):
        affinity_matrix = affinity_matrix_all_layers[layer_id].astype(np.float32)

        # 谱聚类（直接用亲和矩阵）
        clustering = SpectralClustering(
            n_clusters=num_groups,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
        labels = clustering.fit_predict(affinity_matrix)

        # 构建 group_id -> list of expert_id 的字典
        layer_grouping = defaultdict(list)
        for expert_id, group_id in enumerate(labels):
            layer_grouping[int(group_id)].append(int(expert_id))  # 转为 int 方便存储

        all_layer_groupings[f"layer_{layer_id}"] = layer_grouping

    # 保存为 JSON
    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        with open(save_json_path, 'w') as f:
            json.dump(all_layer_groupings, f, indent=4)
        print(f"聚类结果已保存到: {save_json_path}")

    return all_layer_groupings

def draw_spectral_cluster_map(affinity_matrix, labels, layer_id, save_dir):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(affinity_matrix)

    plt.figure(figsize=(6, 6))
    plt.title(f"Spectral Clustering - Layer {layer_id}")
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=60)

    for i in range(len(labels)):
        plt.text(reduced[i, 0], reduced[i, 1], str(i), fontsize=8, ha='center', va='center')

    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"layer{layer_id}_spectral.png")
    plt.savefig(save_path)
    plt.close()
    print(f"图像已保存: {save_path}")

def visualize_reordered_affinity(affinity_matrix, labels, layer_id, save_dir):
    """
    根据聚类标签对亲和矩阵进行排序，并画出热力图
    """
    os.makedirs(save_dir, exist_ok=True)

    # 将专家按照聚类顺序重新排序
    sorted_indices = np.argsort(labels)
    reordered_matrix = affinity_matrix[sorted_indices][:, sorted_indices]

    # 用于画红线的分界点
    group_sizes = [np.sum(labels == g) for g in sorted(set(labels))]
    boundaries = np.cumsum(group_sizes)

    plt.figure(figsize=(10, 9))
    ax = sns.heatmap(reordered_matrix, cmap='YlGnBu', square=True, cbar=True)

    # 画红色分界线（跳过最后一个，因为那是边界外）
    for b in boundaries[:-1]:
        ax.axhline(b, color='red', linestyle='--', linewidth=1.2)
        ax.axvline(b, color='red', linestyle='--', linewidth=1.2)

    plt.title(f"Experts Affinity Matrix (Cluster Ordered) — Layer {layer_id}")
    plt.xlabel("Expert ID (cluster sorted)")
    plt.ylabel("Expert ID (cluster sorted)")

    save_path = os.path.join(save_dir, f"layer{layer_id}_affinity_clustered.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f" 聚类排序的亲和图已保存: {save_path}")

def visualize_affinity_with_clustered_ids(affinity_matrix, labels, layer_id, save_dir):
    """
    对亲和矩阵进行聚类排序，并显示排序后的原始专家ID，同时画出聚类边界线
    """
    os.makedirs(save_dir, exist_ok=True)

    num_experts = len(labels)

    # ① 对专家按照聚类顺序排序（先按 group_id，再按 expert_id 排内部）
    sorted_indices = np.argsort(labels)
    sorted_labels = labels[sorted_indices]
    sorted_expert_ids = [str(i) for i in sorted_indices]

    # ② 重排亲和矩阵
    reordered_matrix = affinity_matrix[sorted_indices][:, sorted_indices]

    # ③ 计算红线分组边界
    group_counts = [np.sum(sorted_labels == g) for g in sorted(set(labels))]
    boundaries = np.cumsum(group_counts)

    # ④ 绘制热力图
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(reordered_matrix, cmap='YlGnBu', square=True, cbar=True)

    # 设置 tick 显示为 排序后的原始专家 ID
    ax.set_xticks(np.arange(num_experts) + 0.5)
    ax.set_yticks(np.arange(num_experts) + 0.5)
    ax.set_xticklabels(sorted_expert_ids, rotation=90, fontsize=7)
    ax.set_yticklabels(sorted_expert_ids, rotation=0, fontsize=7)

    # ⑤ 画红色虚线
    for b in boundaries[:-1]:
        ax.axhline(b, color='red', linestyle='--', linewidth=1.2)
        ax.axvline(b, color='red', linestyle='--', linewidth=1.2)

    plt.title(f"Affinity Matrix (Cluster Sorted + Expert IDs) — Layer {layer_id}")
    plt.xlabel("Expert ID (original, cluster sorted)")
    plt.ylabel("Expert ID (original, cluster sorted)")

    save_path = os.path.join(save_dir, f"layer{layer_id}_affinity_cluster_sorted_ids.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"聚类专家编号图已保存: {save_path}")


def plot_expert_affinity_graph(affinity_matrix, layer_id, save_dir, threshold=5000):
    """
    将专家亲和性矩阵转换为图结构并可视化
    :param affinity_matrix: shape (64, 64), 共现矩阵
    :param layer_id: 当前层编号
    :param save_dir: 图片保存目录
    :param threshold: 边权阈值，小于此值不画（防止太密）
    """
    os.makedirs(save_dir, exist_ok=True)

    G = nx.Graph()
    num_experts = affinity_matrix.shape[0]

    # 添加节点
    for i in range(num_experts):
        G.add_node(i)

    # 添加边（仅保留高于阈值的）
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            weight = affinity_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    # 生成布局（spring 布局模拟物理斥力效果）
    pos = nx.spring_layout(G, seed=42)

    # 提取边权重用于设置线宽
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for _, _, d in edges]
    edge_colors = [min(w, 10000) for w in edge_weights]  # 裁剪极端值

    # 绘图
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, font_size=8)

    # 边宽度和颜色映射权重
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        width=[w / 5000 for w in edge_weights],  # 缩放粗细
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues
    )

    plt.title(f"Expert Affinity Graph — Layer {layer_id} (threshold>{threshold})")
    plt.axis('off')

    save_path = os.path.join(save_dir, f"layer{layer_id}_affinity_graph.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"🌐 专家图已保存: {save_path}")


def plot_expert_subgraph(affinity_matrix, center_expert_id, layer_id, save_dir, threshold=1000):
    """
    只画某个中心专家的子图（它自己 + 所有连接的专家）
    """
    os.makedirs(save_dir, exist_ok=True)
    num_experts = affinity_matrix.shape[0]

    G = nx.Graph()

    # 添加所有节点（可选：也可以只添加有连接的）
    for i in range(num_experts):
        G.add_node(i)

    # 添加满足阈值的边
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            weight = affinity_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    # 检查中心专家是否存在连接
    if center_expert_id not in G:
        print(f"⚠️ Expert {center_expert_id} has no edges above threshold {threshold}")
        return

    # 提取子图：中心专家 + 它连接的节点
    neighbors = list(G.neighbors(center_expert_id))
    subgraph_nodes = neighbors + [center_expert_id]
    subgraph = G.subgraph(subgraph_nodes)

    # 布局：spring layout 更美观
    pos = nx.spring_layout(subgraph, seed=42)

    edge_weights = [d['weight'] for _, _, d in subgraph.edges(data=True)]

    # 绘图
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(subgraph, pos, node_size=600, node_color='skyblue')
    nx.draw_networkx_labels(subgraph, pos, font_size=9)
    nx.draw_networkx_edges(
        subgraph, pos,
        width=[w / 500 for w in edge_weights],
        edge_color=edge_weights,
        edge_cmap=plt.cm.Blues
    )

    plt.title(f"Expert Subgraph — Center: {center_expert_id} (Layer {layer_id})")
    plt.axis('off')

    save_path = os.path.join(save_dir, f"layer{layer_id}_subgraph_expert{center_expert_id}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"🔍 专家子图已保存: {save_path}")


if __name__ == "__main__":

    # model_name = "OLMoE"#Switch_Transformer OLMoE
    # input_name = "sonnet"
    # phrase_mode = "decode" #decode
    # top_k = 8 # ST:1,OL:8
    # num_of_experts_pre_layer = 64

    # fig_dir = f"cluster/spectral/test"
    # os.makedirs(fig_dir, exist_ok=True)

    model_name = args.model_name
    input_name = args.input_name
    phrase_mode = args.phrase_mode
    top_k = args.top_k
    num_of_experts_pre_layer = args.num_of_experts_pre_layer
    
    routing_data = np.load(f'expert_trace/{model_name}/{input_name}/top{top_k}/{phrase_mode}_routing_trace_1024.npy')
    
    affinity_matrix_all_layers = affinity_intra_layer(routing_data, num_of_experts_pre_layer)
    
    # for i in range(len(affinity_matirx_all_layers)):
    #     # test_matrix = affinity_matirx_all_layers[i]

    #     # labels_pca = cluster_and_plot(test_matrix, use_pca=True, pca_dims=10, title=f"layer{i}_With PCA (10 dims)")
    #     # labels_raw = cluster_and_plot(test_matrix, use_pca=False, title=f"layer{i}_Without PCA (Raw 64 dims)")

    #     # diff_result = compare_cluster_labels(labels_pca, labels_raw)
        
    #     co_matrix = affinity_matirx_all_layers[i]  # 第5层

    #     labels = cluster_experts_per_layer(co_matrix, 4, layer_id=i, save_path=fig_dir)

    # fig_dir = f"sp_cluster_test/New/figs"
    # result_json_path = f"sp_cluster_test/New/spectral/grouping_result.json"
    # os.makedirs(fig_dir, exist_ok=True)
    
    #     # 聚类 + 保存分组结果
    # all_groupings = spectral_clustering_all_layers(
    #     affinity_matrix_all_layers,
    #     num_groups=4,
    #     save_json_path=result_json_path
    # )

    # # 逐层绘图
    # for layer_id in range(len(affinity_matrix_all_layers)):
    #     labels = np.zeros(num_of_experts_pre_layer, dtype=int)
    #     # 从 JSON 分组结果还原 label
    #     layer_groups = all_groupings[f"layer_{layer_id}"]
    #     for group_id_str, expert_list in layer_groups.items():
    #         for expert_id in expert_list:
    #             labels[expert_id] = int(group_id_str)

    #     draw_spectral_cluster_map(
    #         affinity_matrix_all_layers[layer_id],
    #         labels,
    #         layer_id,
    #         save_dir=fig_dir
    #     )

    #     visualize_reordered_affinity(
    #         affinity_matrix_all_layers[layer_id],
    #         labels,
    #         layer_id,
    #         save_dir=fig_dir  # 同一个文件夹保存
    #     )
        
    #     visualize_affinity_with_clustered_ids(
    #         affinity_matrix_all_layers[layer_id],
    #         labels,
    #         layer_id,
    #         save_dir=fig_dir
    #     )
    

    co_matrix = affinity_matrix_all_layers[14]  # 第 X 层

    # plot_expert_affinity_graph(
    #     affinity_matrix=co_matrix,
    #     layer_id=14,
    #     save_dir="graph",
    #     threshold=1000  # 控制显示哪些边
    # )
    plot_expert_subgraph(
        affinity_matrix=affinity_matrix_all_layers[14],
        center_expert_id=4,
        layer_id=14,
        save_dir="graph",
        threshold=1000  # 较低一点让子图有边
    )