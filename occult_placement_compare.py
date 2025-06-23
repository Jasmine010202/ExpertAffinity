import json
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np
import os


# 逐层把专家放置方案转成聚类标签列表
def mapping_placement_to_group_labels(placement_all_layers):
    num_layers = len(placement_all_layers)
    labels_all_layers = []
    for layer_id in range(num_layers):
        placement_per_layer = placement_all_layers[str(layer_id)] # List[List[int]]
        label = [-1] * sum(len(group) for group in placement_per_layer) # 长度-专家总数,-1表示未分组，后续转成所属Group的编号
        
        for group_id, group in enumerate(placement_per_layer):
            for expert_id in group:
                label[expert_id] = group_id

        labels_all_layers.append(label) # 逐层存聚类标签

    return labels_all_layers


def plot_score_per_Layer(Score_per_layer, num_layers, result_dir, Metrics, suffix):
    
    layers = np.arange(num_layers)
    scores = np.array(Score_per_layer)
    width = 0.3

    # colors = ['#a6d9c7', '#7fcdbb', '#41b6c4', '#1d91c0']
    colors = ['#7fcdbb', '#41b6c4', '#1d91c0']

    plt.figure(figsize=(12, 6))

    plt.bar(layers - width, scores[:, 0], width=width, color=colors[0], label='sonnet vs GSM8K')
    # plt.bar(layers, nmi_score_per_layer[:, 1], width=width, color=colors[1], label='sonnet vs mbpp')
    # plt.bar(layers + width, nmi_score_per_layer[:, 2], width=width, color=colors[2], label='GSM8K vs mbpp')
    plt.bar(layers, scores[:, 1], width=width, color=colors[1], label='sonnet vs conala')
    plt.bar(layers + width, scores[:, 2], width=width, color=colors[2], label='GSM8K vs conala')

     # 柱顶加数值标签（横向）
    for i, val in enumerate(scores[:, 0]):
        plt.text(layers[i] - width, val + 0.005, f"{val:.2f}", ha='center', fontsize=6, rotation=0)

    for i, val in enumerate(scores[:, 1]):
        plt.text(layers[i], val + 0.005, f"{val:.2f}", ha='center', fontsize=6, rotation=0)

    for i, val in enumerate(scores[:, 2]):
        plt.text(layers[i] + width, val + 0.005, f"{val:.2f}", ha='center', fontsize=6, rotation=0)

    plt.xlabel('Layer_ID')
    plt.ylabel(f'{Metrics} Score')
    plt.title(f'{Metrics} Score Comparison (per layer)')
    plt.xticks(ticks=layers, labels=[str(i) for i in layers])  # 显示完整的层编号
    plt.ylim(0, 0.40) # 1.05
    plt.legend()
    plt.tight_layout()

    plt_path = os.path.join(result_dir, f"layerwise_{Metrics}_comparison{suffix}.png")
    plt.savefig(plt_path)
    plt.close()


def plot_average_NMI_ARI(avg_NMI, avg_ARI, result_dir, suffix):
    combinations = ['sonnet vs GSM8K', 'sonnet vs conala', 'GSM8K vs conala']
    colors = ['#7fcdbb', '#41b6c4', '#1d91c0']
    
    x = np.arange(len(combinations))
    width = 0.35

    plt.figure(figsize=(10, 6))

    # NMI bars (left)
    bars_NMI = plt.bar(x - width/2, avg_NMI, width=width, color=colors[0], label='NMI')

    # ARI bars (right)
    bars_ARI = plt.bar(x + width/2, avg_ARI, width=width, color=colors[1], label='ARI')
    
    # 添加数值标签
    for i, bar in enumerate(bars_NMI):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{avg_NMI[i]:.4f}", ha='center', va='bottom', fontsize=9)

    for i, bar in enumerate(bars_ARI):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{avg_ARI[i]:.4f}", ha='center', va='bottom', fontsize=9)

    plt.xticks(x, combinations)
    plt.ylim(0, 0.35)   #1.05
    plt.ylabel(f'Average Score')
    plt.title(f'Average NMI and ARI Score Comparison')
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(result_dir, f'average_NMI_ARI_comparison{suffix}.png')
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":

    base_dir = f"Occult_test/expert_placement"
    result_dir = f"Occult_test/placement_compare/sonnet_GSM8K_conala"
    os.makedirs(result_dir,exist_ok=True)

    enable_heterogeneous = True
    suffix = "_heterogeneous_4_1" if enable_heterogeneous else ""

    with open(f"{base_dir}/OLMoE_sonnet_placement_512{suffix}.json", 'r') as f1:
        sonnet_placement = json.load(f1)
    with open(f"{base_dir}/OLMoE_GSM8K_placement_512{suffix}.json", 'r') as f2:
        GSM8K_placement = json.load(f2)
    with open(f"{base_dir}/OLMoE_conala_placement_512{suffix}.json", 'r') as f3:
        conala_placement = json.load(f3)
    # with open(f"{base_dir}/OLMoE_gigaword_placement_512{suffix}.json", 'r') as f4:
    #     gigaword_placement = json.load(f4)

    # 放置方案转成聚类标签
    sonnet_labels = mapping_placement_to_group_labels(sonnet_placement)
    GSM8K_labels = mapping_placement_to_group_labels(GSM8K_placement)
    conala_labels = mapping_placement_to_group_labels(conala_placement)
    # gigaword_labels = mapping_placement_to_group_labels(gigaword_placement)
    
    # 计算每一层三种放置两两之间的NMI
    num_layers = 16
    NMI_per_layer = []
    ARI_per_layer = []

    for layer_id in range(num_layers):
        s_layer = sonnet_labels[layer_id]
        G_layer = GSM8K_labels[layer_id]
        c_layer = conala_labels[layer_id]
        # d_layer = gigaword_labels[layer_id]   #gigaword

        NMI_s_G = normalized_mutual_info_score(s_layer, G_layer)
        NMI_s_c = normalized_mutual_info_score(s_layer, c_layer)
        NMI_G_c = normalized_mutual_info_score(G_layer, c_layer)
        # NMI_s_d = normalized_mutual_info_score(s_layer, d_layer)
        # NMI_G_d = normalized_mutual_info_score(G_layer, d_layer)

        ARI_s_G = adjusted_rand_score(s_layer, G_layer)
        ARI_s_c = adjusted_rand_score(s_layer, c_layer)
        ARI_G_c = adjusted_rand_score(G_layer, c_layer)
        # ARI_s_d = adjusted_rand_score(s_layer, d_layer)
        # ARI_G_d = adjusted_rand_score(G_layer, d_layer)

        # 逐层记录三种放置的NMI值
        # NMI_per_layer.append([NMI_s_G, NMI_s_d, NMI_G_d])
        # ARI_per_layer.append([ARI_s_G, ARI_s_d, ARI_G_d])
        NMI_per_layer.append([NMI_s_G, NMI_s_c, NMI_G_c])
        ARI_per_layer.append([ARI_s_G, ARI_s_c, ARI_G_c])

    #print(NMI_per_layer)

    # 每层3种对比的柱状图
    plot_score_per_Layer(NMI_per_layer, num_layers, result_dir,"NMI",suffix)

    plot_score_per_Layer(ARI_per_layer, num_layers, result_dir,"ARI",suffix)

    # 每一组数据集组合的平均 NMI（按列）
    avg_NMI_per_combination = np.mean(NMI_per_layer, axis=0)
    #print(avg_NMI_per_combination)
    avg_ARI_per_combination = np.mean(ARI_per_layer, axis=0)

    # 所有层、所有组合的整体平均 NMI
    avg_NMI_overall = np.mean(NMI_per_layer)
    #print(avg_NMI_overall)
    avg_ARI_overall = np.mean(ARI_per_layer)

    plot_average_NMI_ARI(avg_NMI_per_combination, avg_ARI_per_combination, result_dir, suffix)
   
    NMI_file_path = os.path.join(result_dir, f"NMI_per_layer_dataset_combination{suffix}.npy")
    np.save(NMI_file_path, NMI_per_layer)

    ARI_file_path = os.path.join(result_dir, f"ARI_per_layer_dataset_combination{suffix}.npy")
    np.save(ARI_file_path, ARI_per_layer)

    print(f"Average NMI across layers:\n"
        f"\tsonnet vs GSM8K: {avg_NMI_per_combination[0]:.4f}\n"
        f"\tGSM8K vs conala: {avg_NMI_per_combination[1]:.4f}\n"
        f"\tGSM8K vs conala: {avg_NMI_per_combination[2]:.4f}\n"
        # f"\tGSM8K vs gigaword: {avg_NMI_per_combination[1]:.4f}\n"
        # f"\tGSM8K vs gigaword: {avg_NMI_per_combination[2]:.4f}\n"
        f"\toverall: {avg_NMI_overall:.4f}")

    print(f"Average ARI across layers:\n"
        f"\tsonnet vs GSM8K: {avg_ARI_per_combination[0]:.4f}\n"
        f"\tGSM8K vs conala: {avg_ARI_per_combination[1]:.4f}\n"
        f"\tGSM8K vs conala: {avg_ARI_per_combination[2]:.4f}\n"
        # f"\tGSM8K vs gigaword: {avg_ARI_per_combination[1]:.4f}\n"
        # f"\tGSM8K vs gigaword: {avg_ARI_per_combination[2]:.4f}\n"
        f"\toverall: {avg_ARI_overall:.4f}")


