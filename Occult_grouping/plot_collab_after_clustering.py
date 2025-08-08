import numpy as np
import os
import torch
import json
import copy
import matplotlib.pyplot as plt
import seaborn as sns

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
    model_name = "OLMoE"#Switch_Transformer OLMoE
    input_name = "sonnet"   #GSM8K、sonnet、mbpp、conala
    top_k = 8 # ST:1,OL:8
    num_of_prompts = 512#

    num_layers = 16
    num_of_experts_per_layer = 64


    experts_collaboration_matrix = np.load(f"./Occult_test/expert_collaboration/{model_name}_Expert_Collaboration_{input_name}_{num_of_prompts}.npy")
        

    with open("Occult_test/expert_placement/Duplicate_Group_Level/MultiNodes_MultiGPU/Several_Replicas_of_Highest_Load_Group/semi_even_rate/node0.1/gpu0.2/OLMoE_sonnet_512_semi_even_nodes2_gpus4_initial.json", "r") as f:
        all_layers_placement = json.load(f)

    fig_dir = f"Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/collaboration_figs_after_clustering/semi_even/node0.1_gpu0.2"
    os.makedirs(fig_dir, exist_ok=True)
        
    for layer_id in range(len(experts_collaboration_matrix)):
        labels = np.zeros(num_of_experts_per_layer, dtype=int)
        layer_clusters = all_layers_placement[f"{layer_id}"]

        for group_id, expert_list in enumerate(layer_clusters):
            for expert_id in expert_list:
                labels[expert_id] = int(group_id)
            
        plot_after_clustering(experts_collaboration_matrix[layer_id], labels, layer_id,fig_dir=fig_dir) 