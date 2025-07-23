import json
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_combined_copies_and_load_stats(num_of_token_copies, stats_all_layers_avg_scheme,
                                        placement_schemes, labels, fig_width, fig_path, type):
    if type == "GPU":
        device = "gpu"
    else:
        device = "node"
    
    dataset = ["sonnet", "GSM8K", "conala"]

    x = np.arange(len(placement_schemes))
    width = 0.35

    inter_gpu_values = [num_of_token_copies[dataset[0]][p]["num_inter_gpu"] for p in placement_schemes]
    inter_node_values = [num_of_token_copies[dataset[0]][p]["num_inter_node"] for p in placement_schemes]

    std_vals = [stats_all_layers_avg_scheme[p][device]["std"] for p in placement_schemes]
    max_vals = [stats_all_layers_avg_scheme[p][device]["max"] for p in placement_schemes]
    min_vals = [stats_all_layers_avg_scheme[p][device]["min"] for p in placement_schemes]

    fig, ax1 = plt.subplots(figsize=(fig_width, 8))
    colors = ["#7fcdbb", "#41b6c4", "#225ea8", "#1d91c0", "#1d6fc0"]

    # ===== 左轴：柱状图（通信开销） =====
    bar1 = ax1.bar(x - width/2, inter_gpu_values, width, label="Inter-GPU Copies", color=colors[0])
    bar2 = ax1.bar(x + width/2, inter_node_values, width, label="Inter-Node Copies", color=colors[1])
    ax1.set_ylabel("Num of Token Copies")
    ax1.tick_params(axis='y')

    # 添加柱状图标注
    offset = max(inter_gpu_values + inter_node_values) * 0.015
    for i in range(len(x)):
        ax1.text(x[i] - width/2, inter_gpu_values[i] + offset, str(inter_gpu_values[i]), ha='center', fontsize=8)
        ax1.text(x[i] + width/2, inter_node_values[i] + offset, str(inter_node_values[i]), ha='center', fontsize=8)

    # ===== 右轴：折线图（负载统计） =====
    ax2 = ax1.twinx()
    ax2.plot(x, std_vals, marker='o', linestyle='-', color=colors[2], linewidth=2, label="Avg. load Std")
    ax2.plot(x, max_vals, marker='^', linestyle='--', color=colors[3], linewidth=2, label="Avg. load Max")
    ax2.plot(x, min_vals, marker='s', linestyle='--', color=colors[4], linewidth=2, label="Avg. load Min")
    ax2.set_ylabel(f"{type} Load (Avg. Std / Max / Min)")
    ax2.tick_params(axis='y')

    ax2.set_ylim(0, max(std_vals + max_vals) * 1.1)

    offset_line = 3000
    # 标注折线图点值
    for i in range(len(x)):
        ax2.text(i, std_vals[i] - offset_line, f"{std_vals[i]:.2f}", ha='center', va='top', fontsize=8)
        ax2.text(i, max_vals[i] + offset_line, f"{max_vals[i]:.0f}", ha='center', va='bottom',fontsize=8)
        ax2.text(i, min_vals[i] - offset_line, f"{min_vals[i]:.0f}", ha='center', va='top',fontsize=8)
    
    # ===== 合并图例 =====
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels_combined = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels_combined, loc='upper left', fontsize=10)

    # x轴
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    plt.title(f"Num of Token Copies & {type} Load Statistics")
    plt.tight_layout()

    plt.savefig(fig_path)
    plt.close()


def plot_combined_copies_and_load_stats_gpu_node(num_of_token_copies, stats_all_layers_avg_scheme,
                                                 placement_schemes, labels, fig_width, fig_path):
    dataset = ["sonnet", "GSM8K", "conala"]

    x = np.arange(len(placement_schemes))
    width = 0.35

    inter_gpu_values = [num_of_token_copies[dataset[0]][p]["num_inter_gpu"] for p in placement_schemes]
    inter_node_values = [num_of_token_copies[dataset[0]][p]["num_inter_node"] for p in placement_schemes]

    gpu_std_vals = [stats_all_layers_avg_scheme[p]["gpu"]["std"] for p in placement_schemes]
    gpu_max_vals = [stats_all_layers_avg_scheme[p]["gpu"]["max"] for p in placement_schemes]
    # gpu_min_vals = [stats_all_layers_avg_scheme[p]["gpu"]["min"] for p in placement_schemes]

    node_std_vals = [stats_all_layers_avg_scheme[p]["node"]["std"] for p in placement_schemes]
    node_max_vals = [stats_all_layers_avg_scheme[p]["node"]["max"] for p in placement_schemes]
    # node_min_vals = [stats_all_layers_avg_scheme[p]["node"]["min"] for p in placement_schemes]

    fig, ax1 = plt.subplots(figsize=(fig_width, 8))
    colors = ["#7fcdbb", "#41b6c4", "#225ea8", "#1d91c0", "#1d6fc0"]

    # ===== 左轴：柱状图（通信开销） =====
    bar1 = ax1.bar(x - width/2, inter_gpu_values, width, label="Inter-GPU Copies", color=colors[0])
    bar2 = ax1.bar(x + width/2, inter_node_values, width, label="Inter-Node Copies", color=colors[1])
    ax1.set_ylabel("Num of Token Copies")
    ax1.tick_params(axis='y')

    # 添加柱状图标注
    offset = max(inter_gpu_values + inter_node_values) * 0.015
    for i in range(len(x)):
        ax1.text(x[i] - width/2, inter_gpu_values[i] + offset, str(inter_gpu_values[i]), ha='center', fontsize=8)
        ax1.text(x[i] + width/2, inter_node_values[i] + offset, str(inter_node_values[i]), ha='center', fontsize=8)

    # ===== 右轴：折线图（负载统计） =====
    ax2 = ax1.twinx()
    ax2.plot(x, gpu_std_vals, marker='o', linestyle='-', color=colors[2], linewidth=2, label="Avg. GPU load Std")
    # ax2.plot(x, gpu_max_vals, marker='^', linestyle='--', color=colors[3], linewidth=2, label="Avg. GPU load Max")
    # ax2.plot(x, gpu_min_vals, marker='*', linestyle='--', color=colors[4], linewidth=2, label="Avg. GPU load Min")

    ax2.plot(x, node_std_vals, marker='s', linestyle='--', color=colors[4], linewidth=2, label="Avg. Node load Std")
    # ax2.plot(x, node_max_vals, marker='+', linestyle='--', color=colors[3], linewidth=2, label="Avg. Node load Max")
    # ax2.plot(x, node_min_vals, marker='x', linestyle='--', color=colors[4], linewidth=2, label="Avg. Node load Min")

    ax2.set_ylabel(f"GPU / Node Load (Avg. Std / Max / Min)")
    ax2.tick_params(axis='y')

    ax2.set_ylim(0, max(gpu_std_vals + gpu_max_vals + node_std_vals + node_max_vals) * 1.1)
    # ax2.set_ylim(0, max(gpu_std_vals + node_std_vals) * 1.1)

    offset_line = 3000
    # 标注折线图点值
    for i in range(len(x)):
        ax2.text(i, gpu_std_vals[i] - offset_line, f"{gpu_std_vals[i]:.2f}", ha='center', va='top', fontsize=8)
        # ax2.text(i, gpu_max_vals[i] + offset_line, f"{gpu_max_vals[i]:.0f}", ha='center', va='bottom',fontsize=8)
        # ax2.text(i, gpu_min_vals[i] + offset_line, f"{gpu_min_vals[i]:.0f}", ha='center', va='bottom',fontsize=8)

        ax2.text(i, node_std_vals[i] + offset_line, f"{node_std_vals[i]:.2f}", ha='center', va='bottom', fontsize=8)
        # ax2.text(i, node_max_vals[i] + offset_line, f"{node_max_vals[i]:.0f}", ha='center', va='bottom',fontsize=8)
        # ax2.text(i, node_min_vals[i] + offset_line, f"{node_min_vals[i]:.0f}", ha='center', va='bottom',fontsize=8)
    
    # ===== 合并图例 =====
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels_combined = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels_combined, loc='upper left', fontsize=10)

    # x轴
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    plt.title(f"Num of Token Copies & GPU/Node Loads Statistics")
    plt.tight_layout()

    plt.savefig(fig_path)
    plt.close()



if __name__ == "__main__":
    model_name = "OLMoE"
    # input_name = "sonnet"   
    prompt_nums = [512] # 8, 16, 32, 64, 128, 256, 512, 1024
    top_k = 8 

    num_replicated_experts = 20

    fig_dir = f"Calculation_Load/sonnet/{model_name}_top{top_k}/num_of_experts_copies_compare/figs"   
    os.makedirs(fig_dir, exist_ok=True)

    result_dir = f"Calculation_Load/sonnet/{model_name}_top{top_k}/num_of_experts_copies_compare/data" 
    os.makedirs(result_dir, exist_ok=True)

    # 每层负载统计均值
    with open(f"Calculation_Load/sonnet/OLMoE_top8/num_of_token_loads_duplicate_activation.json", "r") as f:
        loads_stats_all_layers_avg = json.load(f)

    # 通信量
    with open(f"Token_Copies_Compare_sim/Duplicate/sonnet/OLMoE_top8/Activation/num_of_token_copies_duplicate_activation.json",'r') as f:
        num_of_token_copies = json.load(f)


    placement_schemes = ["vanilla_placement",
                         "sonnet_occult_multi_repli_act_4",
                         "sonnet_occult_multi_repli_act_8",
                         "sonnet_occult_multi_repli_act_12",
                         "sonnet_occult_multi_repli_act_16",
                         "sonnet_occult_multi_repli_act_20",
                         "sonnet_spectral_even_multi_repli_act_4", 
                         "sonnet_spectral_even_multi_repli_act_8", 
                         "sonnet_spectral_even_multi_repli_act_12", 
                         "sonnet_spectral_even_multi_repli_act_16", 
                         "sonnet_spectral_even_multi_repli_act_20", 
                         "sonnet_spectral_uneven_multi_repli_act_4",
                         "sonnet_spectral_uneven_multi_repli_act_8",
                         "sonnet_spectral_uneven_multi_repli_act_12",
                         "sonnet_spectral_uneven_multi_repli_act_16",
                         "sonnet_spectral_uneven_multi_repli_act_20"
                        ]
    labels = ["Vanilla", 
              "Occult_Repli_Act_4", 
              "Occult_Repli_Act_8",
              "Occult_Repli_Act_12",
              "Occult_Repli_Act_16",
              "Occult_Repli_Act_20",
              "Spectral_Even_Repli_Act_4", 
              "Spectral_Even_Repli_Act_8", 
              "Spectral_Even_Repli_Act_12", 
              "Spectral_Even_Repli_Act_16", 
              "Spectral_Even_Repli_Act_20", 
              "Spectral_Uneven_Repli_Act_4",
              "Spectral_Uneven_Repli_Act_8",
              "Spectral_Uneven_Repli_Act_12",
              "Spectral_Uneven_Repli_Act_16",
              "Spectral_Uneven_Repli_Act_20"
              ]
    
    fig_path = os.path.join(f"{fig_dir}",f"communication_computing_repli_compare_gpu.svg")
    plot_combined_copies_and_load_stats(num_of_token_copies, loads_stats_all_layers_avg,
                                        placement_schemes, labels, 34, fig_path, "GPU") 
        
    fig_path = os.path.join(f"{fig_dir}",f"communication_computing_repli_compare_node.svg")
    plot_combined_copies_and_load_stats(num_of_token_copies, loads_stats_all_layers_avg,
                                        placement_schemes, labels, 34, fig_path, "Node")                # 13-34 10-28
        
    fig_path = os.path.join(f"{fig_dir}",f"communication_computing_repli_compare_gpu_node.svg")
    plot_combined_copies_and_load_stats_gpu_node(num_of_token_copies, loads_stats_all_layers_avg,
                                        placement_schemes, labels, 34, fig_path) 
        