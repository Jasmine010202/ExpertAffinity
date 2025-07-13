import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json

from utils import extract_routing_trace, extract_expert_placement, extract_replicated_experts, prompt_to_gpu


num_layers = 16
num_experts_per_layer = 64

# GPU 0和1映射到node 0；2和3映射到node 1
gpu_node_mapping = np.array([0, 0, 1, 1]) 

# 节点数、GPU数
num_nodes = 2
num_gpus_per_node = 2
num_gpus = num_nodes * num_gpus_per_node


def experts_activations_count(routing_trace):
    experts_activation_stats = np.zeros((num_layers, num_experts_per_layer), dtype = int) 

    for prompt in routing_trace:
        token_traces = np.array(prompt["trace"])
        num_tokens = token_traces.shape[0]

        for token_id in range(num_tokens):
            for layer_id in range(num_layers):
                for expert_id in token_traces[token_id, layer_id]:
                    experts_activation_stats[layer_id, expert_id] += 1

    #np.save(f"./Occult_test/expert_activation/traffic_test_by_prompt/OLMoE_top8_sonnet_512.npy", experts_activation_stats)
    return experts_activation_stats


def compute_claculation_load_total(routing_trace, expert_placement, enable_replicated = False, replicated_experts = None):
    # num_layers, num_experts = experts_activation_stats.shape
    # num_gpus = num_of_nodes * num_of_gpus_per_node
    
    gpus_token_load = {gpu_id:0 for gpu_id in range(num_gpus)}
    nodes_token_load = {node_id:0 for node_id in range(num_nodes)}

    for prompt in routing_trace:
        # token所在GPU和节点
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        token_node_id = gpu_node_mapping[token_gpu_id]

        token_routing_traces = np.array(prompt["trace"])
        num_tokens = token_routing_traces.shape[0]

        for token_id in range(num_tokens):
            for layer_id in range(num_layers):
                expert_ids = token_routing_traces[token_id, layer_id] # 这一层激活的专家

                # 如果有专家复制
                if enable_replicated and replicated_experts is not None:
                    replicated_experts_this_layer = set(replicated_experts[layer_id])

                    # 区分被复制和不被复制的专家
                    replicated_expert_ids = [expert_id for expert_id in expert_ids if expert_id in replicated_experts_this_layer]
                    not_replicated_expert_ids = [expert_id for expert_id in expert_ids if expert_id not in replicated_experts_this_layer]
        
                    # 被复制的专家, 负载数记录到token所在GPU
                    for expert_id in replicated_expert_ids:
                        gpus_token_load[token_gpu_id] +=1

                    # 未被复制的专家
                    for expert_id in not_replicated_expert_ids:
                        expert_gpu_id = expert_placement[layer_id, expert_id]
                        gpus_token_load[expert_gpu_id] +=1

                # 没有专家复制
                else:
                    for expert_id in expert_ids:
                        expert_gpu_id = expert_placement[layer_id, expert_id]
                        gpus_token_load[expert_gpu_id] +=1


    # 每个节点的总负载
    for gpu_id, gpu_load in gpus_token_load.items():
        node_id = gpu_node_mapping[gpu_id]
        nodes_token_load[node_id] += gpu_load

    result={"gpus":gpus_token_load, "nodes":nodes_token_load}
    
    return result


def compute_claculation_load_per_layer(routing_trace, expert_placement, enable_replicated = False, replicated_experts = None):
    # num_layers, num_experts = experts_activation_stats.shape
    # num_gpus = num_of_nodes * num_of_gpus_per_node

    # gpus_token_load = {gpu_id:0 for gpu_id in range(num_gpus)}
    # nodes_token_load = {node_id:0 for node_id in range(num_nodes)}

    gpus_token_load_per_layer = [{gpu_id:0 for gpu_id in range(num_gpus)} for layer in range(num_layers)]
    nodes_token_load_per_layer = [{node_id:0 for node_id in range(num_nodes)} for layer in range(num_layers)]

    for prompt in routing_trace:
        # token所在GPU和节点
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        token_node_id = gpu_node_mapping[token_gpu_id]

        token_routing_traces = np.array(prompt["trace"])
        num_tokens = token_routing_traces.shape[0]

        for token_id in range(num_tokens):
            for layer_id in range(num_layers):
                expert_ids = token_routing_traces[token_id, layer_id] # 这一层激活的专家

                # 如果有专家复制
                if enable_replicated and replicated_experts is not None:
                    replicated_experts_this_layer = set(replicated_experts[layer_id])

                    # 区分被复制和不被复制的专家
                    replicated_expert_ids = [expert_id for expert_id in expert_ids if expert_id in replicated_experts_this_layer]
                    not_replicated_expert_ids = [expert_id for expert_id in expert_ids if expert_id not in replicated_experts_this_layer]
        
                    # 被复制的专家, 负载数记录到token所在GPU
                    for expert_id in replicated_expert_ids:
                        gpus_token_load_per_layer[layer_id][token_gpu_id] += 1   
                        # gpus_token_load[token_gpu_id] +=1

                    # 未被复制的专家
                    for expert_id in not_replicated_expert_ids:
                        expert_gpu_id = expert_placement[layer_id, expert_id]
                        
                        # gpus_token_load[expert_gpu_id] +=1
                        gpus_token_load_per_layer[layer_id][expert_gpu_id] += 1

                # 没有专家复制
                else:
                    for expert_id in expert_ids:
                        expert_gpu_id = expert_placement[layer_id, expert_id]
                        # gpus_token_load[expert_gpu_id] +=1
                        gpus_token_load_per_layer[layer_id][expert_gpu_id] += 1


    # 每个节点的总负载
    # for gpu_id, gpu_load in gpus_token_load.items():
    #     node_id = gpu_node_mapping[gpu_id]
    #     nodes_token_load[node_id] += gpu_load
    
    # 每层每个节点的负载
    for layer_id in range(num_layers):
        for gpu_id, gpu_load in gpus_token_load_per_layer[layer_id].items():
            node_id = gpu_node_mapping[gpu_id]
            nodes_token_load_per_layer[layer_id][node_id] += gpu_load

    # result_total={"gpus":gpus_token_load, "nodes":nodes_token_load}
    result_per_layer={"gpus":gpus_token_load_per_layer, "nodes":nodes_token_load_per_layer}
    
    # return result_total, result_per_layer
    return result_per_layer

def calculate_load_stats_per_layer(token_loads):
    gpu_stats_per_layer = {"mean":[], "var":[], "std":[], "max":[], "min":[]}

    gpu_loads_per_layer = token_loads["gpus"] 

    for layer_gpu_loads_dict in gpu_loads_per_layer:
        gpu_loads = list(layer_gpu_loads_dict.values())
        gpu_stats_per_layer["mean"].append(int(np.mean(gpu_loads)))
        gpu_stats_per_layer["var"].append(float(np.var(gpu_loads)))
        gpu_stats_per_layer["std"].append(float(np.std(gpu_loads)))
        gpu_stats_per_layer["max"].append(int(np.max(gpu_loads)))
        gpu_stats_per_layer["min"].append(int(np.min(gpu_loads)))

    gpu_stats_all_layers_avg = {
        "mean": float(np.mean(gpu_stats_per_layer["mean"])),
        "var": float(np.mean(gpu_stats_per_layer["var"])),
        "std": float(np.mean(gpu_stats_per_layer["std"])),
        "max": float(np.mean(gpu_stats_per_layer["max"])),
        "min": float(np.mean(gpu_stats_per_layer["min"]))
    }

    return gpu_stats_per_layer, gpu_stats_all_layers_avg

# 整体负载加和
def plot_num_of_loads_compare(num_of_token_loads, grouping_scheme, placement_schemes, labels, fig_path_prefix):
    dataset = ["sonnet", "GSM8K", "conala"]

    width = 0.18

    colors = ["#a6d9c7", "#7fcdbb", "#41b6c4","#269ec0", "#1d91c0"]

    gpu_values = []
    node_values = []
    for p in placement_schemes:
        gpu_loads = num_of_token_loads[dataset[0]][p]["gpus"]
        node_loads = num_of_token_loads[dataset[0]][p]["nodes"]

        gpu_values.append([gpu_loads[str(i)] for i in range(num_gpus)])
        node_values.append([node_loads[str(i)] for i in range(num_nodes)])

    gpu_values = np.array(gpu_values)
    node_values = np.array(node_values)

    # print(gpu_values)
    # print(node_values)
    
    # ========== GPU负载 ==========
    x_gpus = np.arange(num_gpus) 
    
    plt.figure(figsize=(16, 6))
    for scheme_idx in range(len(placement_schemes)):
        plt.bar(x_gpus + width * scheme_idx - width * (len(placement_schemes)-1)/2, gpu_values[scheme_idx, :], width, label=labels[scheme_idx], color=colors[scheme_idx])

    max_gpu_value = gpu_values.max()
    plt.ylim(0, max_gpu_value * 1.1)

    offset = max_gpu_value * 0.01
    for gpu in range(num_gpus):
        for scheme_idx in range(len(placement_schemes)):
            val = gpu_values[scheme_idx, gpu]
            plt.text(x_gpus[gpu] + width * scheme_idx - width * (len(placement_schemes)-1)/2, val + offset, f"{val:,}", ha="center", fontsize=8)

    plt.xticks(x_gpus, [f"GPU {i}" for i in range(num_gpus)])
    plt.ylabel("Num of Tokens")
    plt.title(f"Comparison of GPU Token Loads-{grouping_scheme}")     # ({dataset[0]})
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    plt.savefig(f"{fig_path_prefix}_gpu_loads.svg")
    plt.close()

    # ========== Node 负载 ==========
    x_nodes = np.arange(num_nodes)

    plt.figure(figsize=(10, 6))
    for scheme_idx in range(len(placement_schemes)):
        plt.bar(x_nodes + width * scheme_idx - width * (len(placement_schemes)-1)/2, node_values[scheme_idx, :], width, label=labels[scheme_idx], color=colors[scheme_idx])

    max_node_value = node_values.max()
    plt.ylim(0, max_node_value * 1.1)

    offset = max_node_value * 0.01
    for node in range(num_nodes):
        for scheme_idx in range(len(placement_schemes)):
            val = node_values[scheme_idx, node]
            plt.text(x_nodes[node] + width * scheme_idx - width * (len(placement_schemes)-1)/2, val + offset, f"{val:,}", ha="center",fontsize=8)

    plt.xticks(x_nodes, [f"Node {i}" for i in range(num_nodes)])
    plt.ylabel("Num of Tokens")
    plt.title(f"Comparison of Node Token Loads-{grouping_scheme}")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()
    plt.savefig(f"{fig_path_prefix}_node_loads.svg")
    plt.close()


def plot_load_std_line(num_of_token_loads, placement_schemes, labels, fig_path):
    dataset = ["sonnet", "GSM8K", "conala"]
    
    gpu_load_std = []
    node_load_std = []

    # intra_node0_gpu_std = []
    # intra_node1_gpu_std = []
    
    for scheme in placement_schemes:
        gpu_loads_dict = num_of_token_loads[dataset[0]][scheme]["gpus"]
        node_loads_dict = num_of_token_loads[dataset[0]][scheme]["nodes"]

        gpu_loads = [gpu_loads_dict[str(i)] for i in range(len(gpu_loads_dict))]
        node_loads = [node_loads_dict[str(i)] for i in range(len(node_loads_dict))]

        # print(f"Scheme: {scheme} -> GPU loads: {gpu_loads}")
        # print(f"Scheme: {scheme} -> node_loads: {node_loads}")


        gpu_std = np.std(gpu_loads)
        node_std = np.std(node_loads)

        gpu_load_std.append(gpu_std)
        node_load_std.append(node_std)

        # # 构建每个节点内 GPU 的负载
        # node_gpu_loads = {0: [], 1: []}
        # for gpu_id_str, load in gpu_loads_dict.items():
        #     gpu_id = int(gpu_id_str)
        #     node_id = gpu_node_mapping[gpu_id]
        #     node_gpu_loads[node_id].append(load)

        # # 分别记录两个节点内的 GPU 标准差
        # for nid in [0, 1]:
        #     if len(node_gpu_loads[nid]) > 1:
        #         std_val = np.std(node_gpu_loads[nid])
        #     else:
        #         std_val = 0.0
        #     if nid == 0:
        #         intra_node0_gpu_std.append(std_val)
        #     else:
        #         intra_node1_gpu_std.append(std_val)
    
    # print(gpu_load_std)
    # print(node_load_std)

    x = np.arange(len(placement_schemes))
    
    plt.figure(figsize=(12, 6))
    
    # GPU方差折线
    plt.plot(x, gpu_load_std, marker='o', linestyle='-', color="#41b6c4", linewidth=2, label="GPU Loads")
    
    # Node方差折线
    plt.plot(x, node_load_std, marker='s', linestyle='-', color="#1d91c0", linewidth=2, label="Node Loads")

    # # 节点内曲线（两条线）
    # plt.plot(x, intra_node0_gpu_std, marker='^', linestyle='--', color="#7fcdbb", linewidth=2, label="Intra-Node0 GPU Std")
    # plt.plot(x, intra_node1_gpu_std, marker='v', linestyle='--', color="#45b8c9", linewidth=2, label="Intra-Node1 GPU Std")

    
    # 在每个点上标注
    for i, v in enumerate(gpu_load_std):
        plt.text(i, v * 1.02, f"{v:,.0f}", ha='center', fontsize=8)
    for i, v in enumerate(node_load_std):
        plt.text(i, v * 1.02, f"{v:,.0f}", ha='center', fontsize=8)

    # for i, v in enumerate(intra_node0_gpu_std):
    #     plt.text(i, v * 1.02, f"{v:,.0f}", ha='center', fontsize=8)

    # for i, v in enumerate(intra_node1_gpu_std):
    #     plt.text(i, v * 1.02, f"{v:,.0f}", ha='center', fontsize=8)
    
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel("Standard Deviation of Token Loads")
    plt.title("Standard Deviation of Token Loads Across GPUs and Nodes")
    
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def plot_per_scheme_layerwise_bar_line(stats_per_layer_scheme, labels, fig_path_prefix):
    for scheme_key, stats_per_layer in stats_per_layer_scheme.items():
        # num_layers = len(stats_per_layer)
        std_vals = stats_per_layer["std"]
        max_vals = stats_per_layer["max"]
        min_vals = stats_per_layer["min"]

        x = np.arange(num_layers)

        fig, ax1 = plt.subplots(figsize=(24, 6))

        # 柱状图：标准差
        bar = ax1.bar(x, std_vals, width=0.5, color="#a6d9c7", label="Std")
        ax1.set_ylabel("Standard Deviation of GPU Loads")
        ax1.tick_params(axis='y')
        ax1.set_ylim(0, max(std_vals) * 1.15)


        # 在柱状图上标注 std 值
        for i, val in enumerate(std_vals):
            ax1.text(x[i], val + max(std_vals) * 0.005, f"{val:.0f}", ha='center', va='bottom', fontsize=8)

        # 折线图：最大最小值
        ax2 = ax1.twinx()
        ax2.plot(x, max_vals, linestyle='-', marker='o', color="#41b6c4", label="Max")
        ax2.plot(x, min_vals, linestyle='--', marker='s', color="#1d91c0", label="Min")
        ax2.set_ylabel("Max / Min GPU Load per layer")
        ax2.tick_params(axis='y')
        ax2.set_ylim(0, max(max_vals) * 1.1)  # 比最大值高15%


        # 在折线图上标注 max 值
        for i, val in enumerate(max_vals):
            ax2.text(x[i], val + max(max_vals) * 0.03, f"{val:.0f}", ha='center', va='bottom', fontsize=8)

        # 在折线图上标注 min 值
        for i, val in enumerate(min_vals):
            ax2.text(x[i], val - max(max_vals) * 0.025, f"{val:.0f}", ha='center', va='bottom', fontsize=8)

        # 图例处理
        lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
        lines, labels_ = [sum(lol, []) for lol in zip(*lines_labels)]
        ax1.legend(lines, labels_, loc='upper left')

        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Layer {i}" for i in range(num_layers)])
        title_label = labels[scheme_key] if scheme_key in labels else scheme_key
        ax1.set_title(f"Layerwise GPU Load Stats - {title_label}")


        fig.tight_layout()
        os.makedirs(os.path.dirname(fig_path_prefix), exist_ok=True)
        fig.savefig(f"{fig_path_prefix}/{scheme_key}.svg")
        plt.close()



def plot_avg_load_stats(stats_all_layers_avg_scheme, placement_schemes, labels, fig_path_prefix):
    std_vals = [stats_all_layers_avg_scheme[p]["std"] for p in placement_schemes]
    max_vals = [stats_all_layers_avg_scheme[p]["max"] for p in placement_schemes]
    min_vals = [stats_all_layers_avg_scheme[p]["min"] for p in placement_schemes]

    x = np.arange(len(placement_schemes))

    plt.figure(figsize=(14, 6))

    # 标准差线
    plt.plot(x, std_vals, marker='o', linestyle='-', color="#7fcdbb", linewidth=2, label="Avg. Std")

    # 最大最小值线
    plt.plot(x, max_vals, marker='^', linestyle='--', color="#1d91c0", linewidth=2, label="Avg. Max")
    plt.plot(x, min_vals, marker='s', linestyle='--', color="#225ea8", linewidth=2, label="Avg. Min")

    max_load_value = max(max_vals)
    plt.ylim(0, max_load_value * 1.1)

    # 标注每个点数值
    for i, val in enumerate(std_vals):
        plt.text(i, val * 1.02, f"{val:.0f}", ha='center', fontsize=8)
    for i, val in enumerate(max_vals):
        plt.text(i, val * 1.02, f"{val:.0f}", ha='center', fontsize=8)
    for i, val in enumerate(min_vals):
        plt.text(i, val * 0.98, f"{val:.0f}", ha='center', fontsize=8)

    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel("Avg. Std of GPU Loads & Avg. Max / Min GPU Load")
    plt.title("Overall GPU Load Statistics")
    plt.legend()
    plt.tight_layout()
    
    fig_path = os.path.join(fig_path_prefix,"all_layers_avg.svg")
    plt.savefig(fig_path)
    plt.close()


def plot_combined_copies_and_load_stats(num_of_token_copies, stats_all_layers_avg_scheme,
                                        placement_schemes, labels, fig_width, fig_path):
    dataset = ["sonnet", "GSM8K", "conala"]

    x = np.arange(len(placement_schemes))
    width = 0.35

    inter_gpu_values = [num_of_token_copies[dataset[0]][p]["num_inter_gpu"] for p in placement_schemes]
    inter_node_values = [num_of_token_copies[dataset[0]][p]["num_inter_node"] for p in placement_schemes]
    std_vals = [stats_all_layers_avg_scheme[p]["std"] for p in placement_schemes]
    max_vals = [stats_all_layers_avg_scheme[p]["max"] for p in placement_schemes]
    min_vals = [stats_all_layers_avg_scheme[p]["min"] for p in placement_schemes]

    fig, ax1 = plt.subplots(figsize=(fig_width, 7))

    # ===== 左轴：柱状图（通信开销） =====
    color_gpu = "#7fcdbb"
    color_node = "#41b6c4"
    bars1 = ax1.bar(x - width/2, inter_gpu_values, width, label="Inter-GPU Copies", color=color_gpu)
    bars2 = ax1.bar(x + width/2, inter_node_values, width, label="Inter-Node Copies", color=color_node)
    ax1.set_ylabel("Num of Token Copies")
    ax1.tick_params(axis='y')

    # 添加柱状图标注
    offset = max(inter_gpu_values + inter_node_values) * 0.015
    for i in range(len(x)):
        ax1.text(x[i] - width/2, inter_gpu_values[i] + offset, str(inter_gpu_values[i]), ha='center', fontsize=8)
        ax1.text(x[i] + width/2, inter_node_values[i] + offset, str(inter_node_values[i]), ha='center', fontsize=8)

    # ===== 右轴：折线图（负载统计） =====
    ax2 = ax1.twinx()
    ax2.plot(x, std_vals, marker='o', linestyle='-', color="#225ea8", linewidth=2, label="Avg. Std")
    ax2.plot(x, max_vals, marker='^', linestyle='--', color="#1d91c0", linewidth=2, label="Avg. Max")
    ax2.plot(x, min_vals, marker='s', linestyle='--', color="#2d8365", linewidth=2, label="Avg. Min")
    ax2.set_ylabel("GPU Load (Avg. Std / Max / Min)")
    ax2.tick_params(axis='y')

    # 标注折线图点值
    for i, v in enumerate(std_vals):
        ax2.text(i, v * 1.02, f"{v:.0f}", ha='center', fontsize=8)
    for i, v in enumerate(max_vals):
        ax2.text(i, v * 1.02, f"{v:.0f}", ha='center', fontsize=8)
    for i, v in enumerate(min_vals):
        ax2.text(i, v * 0.98, f"{v:.0f}", ha='center', fontsize=8)

    # ===== 合并图例 =====
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels_combined = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels_combined, loc='upper left', fontsize=10)

    # x轴
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    plt.title("Token Communication vs. Overall GPU Load Statistics")
    plt.tight_layout()

    plt.savefig(fig_path)
    plt.close()


if __name__ == "__main__":

    model_name = "OLMoE"
    # input_name = "sonnet"   
    prompt_nums = [512] # 8, 16, 32, 64, 128, 256, 512, 1024
    top_k = 8 

    fig_dir = f"Calculation_Load/sonnet/{model_name}_top{top_k}/per_layer/figs"   
    os.makedirs(fig_dir, exist_ok=True)

    result_dir = f"Calculation_Load/sonnet/{model_name}_top{top_k}/per_layer/data" 
    os.makedirs(result_dir, exist_ok=True)


    for num in prompt_nums:
        # # 路由
        # sonnet_routing_trace = extract_routing_trace(f"./Occult_test/expert_trace/traffic_test/by_prompt/{model_name}_sonnet_top{top_k}/routing_trace_{num}.jsonl")

        # # 专家激活次数 -> 计算负载（token个数）
        # # experts_activations = experts_activations_count(sonnet_routing_trace)
        # # experts_activations = np.load("./Occult_test/expert_activation/traffic_test_by_prompt/OLMoE_top8_sonnet_512.npy")

        # ###############################################Placement##################################################
        # vanilla_placement = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/OLMoE_vanilla_placement.json")

        # sonnet_occult_placement = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/occult/OLMoE_sonnet_placement_512.json")
        # sonnet_occult_multi_placement = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/OLMoE_sonnet_512_nodes2_gpus4.json")
        # sonnet_spectral_even_placement = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/spectral/OLMoE_sonnet_spectral_even_placement_512.json")
        # sonnet_spectral_uneven_placement = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/spectral/OLMoE_sonnet_spectral_uneven_placement_512.json")
        # sonnet_spectral_even_multi_placement = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/OLMoE_spectral_even_sonnet_512_nodes2_gpus4.json")
        # sonnet_spectral_uneven_multi_placement = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/OLMoE_spectral_uneven_sonnet_512_nodes2_gpus4.json")

        # ####duplicate
        # act_replicated_experts_list = extract_replicated_experts(num_layers, "./Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_nodes2_gpus4_re4_replicated_experts.json")
        # sonnet_occult_multi_repli_act = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_nodes2_gpus4_re4.json")
        # sonnet_spectral_even_multi_repli_act = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_even_nodes2_gpus4_re4.json")
        # sonnet_spectral_uneven_multi_repli_act = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_uneven_nodes2_gpus4_re4.json")

        # collab_replicated_experts_list = extract_replicated_experts(num_layers, "./Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/Duplicate/Collaboration/OLMoE_sonnet_512_nodes2_gpus4_re4_replicated_experts.json")
        # sonnet_occult_multi_repli_collab = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/Duplicate/Collaboration/OLMoE_sonnet_512_nodes2_gpus4_re4.json")
        # sonnet_spectral_even_multi_repli_collab = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/Duplicate/Collaboration/OLMoE_sonnet_512_even_nodes2_gpus4_re4.json")
        # sonnet_spectral_uneven_multi_repli_collab = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/Duplicate/Collaboration/OLMoE_sonnet_512_uneven_nodes2_gpus4_re4.json")


        # ###############################################整体负载加和##################################################
        # '''
        # sonnet_vanilla_loads = compute_claculation_load_total(sonnet_routing_trace, vanilla_placement)
        
        # # Occult
        # sonnet_s_occult_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_occult_placement)
        # sonnet_occult_multi_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_occult_multi_placement)
        # #duplicate Activation
        # sonnet_occult_multi_repli_act_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_occult_multi_repli_act, True, act_replicated_experts_list)
        # #duplicate Collaboration
        # sonnet_occult_multi_repli_collab_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_occult_multi_repli_collab, True, collab_replicated_experts_list)
        
        # # Spectral Even
        # sonnet_spectral_even_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_spectral_even_placement)
        # sonnet_spectral_even_multi_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_spectral_even_multi_placement)
        # #duplicate Activation
        # sonnet_spectral_even_multi_repli_act_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_spectral_even_multi_repli_act, True, act_replicated_experts_list)
        # #duplicate Collaboration
        # sonnet_spectral_even_multi_repli_collab_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_spectral_even_multi_repli_collab, True, collab_replicated_experts_list)

        # # Spectral Uneven
        # sonnet_spectral_uneven_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_spectral_uneven_placement)
        # sonnet_spectral_uneven_multi_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_spectral_uneven_multi_placement)
        # #duplicate Activation
        # sonnet_spectral_uneven_multi_repli_act_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_spectral_uneven_multi_repli_act, True, act_replicated_experts_list)
        # #duplicate Collaboration
        # sonnet_spectral_uneven_multi_repli_collab_loads = compute_claculation_load_total(sonnet_routing_trace, sonnet_spectral_uneven_multi_repli_collab, True, collab_replicated_experts_list)
        # '''

        # sonnet_vanilla_loads = compute_claculation_load_per_layer(sonnet_routing_trace, vanilla_placement)
        
        # # Occult
        # sonnet_s_occult_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_occult_placement)
        # sonnet_occult_multi_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_occult_multi_placement)
        # #duplicate Activation
        # sonnet_occult_multi_repli_act_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_occult_multi_repli_act, True, act_replicated_experts_list)
        # #duplicate Collaboration
        # sonnet_occult_multi_repli_collab_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_occult_multi_repli_collab, True, collab_replicated_experts_list)
        
        # # Spectral Even
        # sonnet_spectral_even_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_spectral_even_placement)
        # sonnet_spectral_even_multi_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_spectral_even_multi_placement)
        # #duplicate Activation
        # sonnet_spectral_even_multi_repli_act_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_spectral_even_multi_repli_act, True, act_replicated_experts_list)
        # #duplicate Collaboration
        # sonnet_spectral_even_multi_repli_collab_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_spectral_even_multi_repli_collab, True, collab_replicated_experts_list)

        # # Spectral Uneven
        # sonnet_spectral_uneven_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_spectral_uneven_placement)
        # sonnet_spectral_uneven_multi_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_spectral_uneven_multi_placement)
        # #duplicate Activation
        # sonnet_spectral_uneven_multi_repli_act_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_spectral_uneven_multi_repli_act, True, act_replicated_experts_list)
        # #duplicate Collaboration
        # sonnet_spectral_uneven_multi_repli_collab_loads = compute_claculation_load_per_layer(sonnet_routing_trace, sonnet_spectral_uneven_multi_repli_collab, True, collab_replicated_experts_list)

        # num_of_token_loads = {}
        # num_of_token_loads["sonnet"] = {}

        # num_of_token_loads["sonnet"]["vanilla_placement"] = sonnet_vanilla_loads

        # num_of_token_loads["sonnet"]["sonnet_occult_placement"] = sonnet_s_occult_loads
        # num_of_token_loads["sonnet"]["sonnet_occult_multi_placement"] = sonnet_occult_multi_loads
        # num_of_token_loads["sonnet"]["sonnet_occult_multi_repli_act_placement"] = sonnet_occult_multi_repli_act_loads
        # num_of_token_loads["sonnet"]["sonnet_occult_multi_repli_collab_placement"] = sonnet_occult_multi_repli_collab_loads
        
        # num_of_token_loads["sonnet"]["sonnet_spectral_even_placement"] = sonnet_spectral_even_loads
        # num_of_token_loads["sonnet"]["sonnet_spectral_even_multi_placement"] = sonnet_spectral_even_multi_loads
        # num_of_token_loads["sonnet"]["sonnet_spectral_even_multi_repli_act_placement"] = sonnet_spectral_even_multi_repli_act_loads
        # num_of_token_loads["sonnet"]["sonnet_spectral_even_multi_repli_collab_placement"] = sonnet_spectral_even_multi_repli_collab_loads

        # num_of_token_loads["sonnet"]["sonnet_spectral_uneven_placement"] = sonnet_spectral_uneven_loads
        # num_of_token_loads["sonnet"]["sonnet_spectral_uneven_multi_placement"] = sonnet_spectral_uneven_multi_loads
        # num_of_token_loads["sonnet"]["sonnet_spectral_uneven_multi_repli_act_placement"] = sonnet_spectral_uneven_multi_repli_act_loads
        # num_of_token_loads["sonnet"]["sonnet_spectral_uneven_multi_repli_collab_placement"] = sonnet_spectral_uneven_multi_repli_collab_loads


        # filename = os.path.join(result_dir, f"num_of_token_loads_original_multi_duplicate_activation_collaboration.json")
        # with open (filename, "w") as f:
        #     json.dump(num_of_token_loads, f, indent=2)

        # with open("Calculation_Load/sonnet/OLMoE_top8/per_layer/data/num_of_token_loads_original_multi_duplicate_activation_collaboration.json", "r") as f:
        #     num_of_token_loads = json.load(f)

        # stats_per_layer_scheme = {}
        # stats_all_layers_avg_scheme = {}
        # for scheme, token_loads in num_of_token_loads["sonnet"].items():
        #     stats_per_layer_scheme[scheme], stats_all_layers_avg_scheme[scheme] = calculate_load_stats_per_layer(token_loads)

        # filename = os.path.join(result_dir, f"stats_of_token_loads_per_layer.json")
        # with open (filename, "w") as f:
        #     json.dump(stats_per_layer_scheme, f, indent=2)

        # filename = os.path.join(result_dir, f"stats_of_token_loads_all_layers_avg.json")
        # with open (filename, "w") as f:
        #     json.dump(stats_all_layers_avg_scheme, f, indent=2)

        with open("Calculation_Load/sonnet/OLMoE_top8/per_layer/data/stats_of_token_loads_per_layer.json", "r") as f:
            stats_per_layer_scheme = json.load(f)

        with open("Calculation_Load/sonnet/OLMoE_top8/per_layer/data/stats_of_token_loads_all_layers_avg.json", "r") as f:
            stats_all_layers_avg_scheme = json.load(f)


        placement_schemes = ["vanilla_placement", 
                             "sonnet_occult_placement",
                             "sonnet_occult_multi_placement", 
                             "sonnet_occult_multi_repli_act_placement",
                            #  "sonnet_occult_multi_repli_collab_placement",
                             "sonnet_spectral_even_placement",
                             "sonnet_spectral_even_multi_placement", 
                             "sonnet_spectral_even_multi_repli_act_placement", 
                            #  "sonnet_spectral_even_multi_repli_collab_placement",
                             "sonnet_spectral_uneven_placement",
                             "sonnet_spectral_uneven_multi_placement", 
                             "sonnet_spectral_uneven_multi_repli_act_placement",
                            #  "sonnet_spectral_uneven_multi_repli_collab_placement"
                             ]
        labels = ["Vanilla", 
                  "Occult",
                  "Occult_Multi", 
                  "Occult_Multi_Repli_Act", 
                #   "Occult_Multi_Repli_Collab", 
                  "Spectral_Even",
                  "Spectral_Even_Multi", 
                  "Spectral_Even_Multi_Repli_Act", 
                #   "Spectral_Even_Multi_Repli_Collab", 
                  "Spectral_Uneven",
                  "Spectral_Uneven_Multi", 
                  "Spectral_Uneven_Multi_Repli_Act",
                #   "Spectral_Uneven_Multi_Repli_Collab"
                  ]
        
        # plot_per_scheme_layerwise_bar_line(stats_per_layer_scheme, labels, f"{fig_dir}/load_stats_per_layer")

        # plot_avg_load_stats(stats_all_layers_avg_scheme,  placement_schemes, labels, fig_dir)

        with open(f"Token_Copies_Compare_sim/Duplicate/sonnet/OLMoE_top8/Activation_Collaboration/re4/data/num_of_token_copies_duplicate_multi_original_activation_collaboration.json",'r') as f:
            num_of_token_copies = json.load(f)

        fig_path = os.path.join(fig_dir,f"communication_computing_10.svg")
        plot_combined_copies_and_load_stats(num_of_token_copies, stats_all_layers_avg_scheme,
                                        placement_schemes, labels, 25, fig_path)


        # 整体负载加和
        '''
        plot_num_of_loads_compare(num_of_token_loads, "Occult", placement_schemes = [placement_schemes[0]]+placement_schemes[1:5], 
                                  labels = [labels[0]]+labels[1:5], fig_path_prefix=f"{fig_dir}/Occult_Multi_Repli")
        
        plot_num_of_loads_compare(num_of_token_loads, "Spectral_Even", placement_schemes = [placement_schemes[0]]+placement_schemes[5:9], 
                                  labels = [labels[0]]+labels[5:9], fig_path_prefix=f"{fig_dir}/Spectral_Even_Multi_Repli")
        
        plot_num_of_loads_compare(num_of_token_loads, "Spectral_Uneven", placement_schemes = [placement_schemes[0]]+placement_schemes[9:13], 
                                  labels = [labels[0]]+labels[9:13], fig_path_prefix=f"{fig_dir}/Spectral_Uneven_Multi_Repli")

        # plot_num_of_loads_compare(num_of_token_loads, "Original", placement_schemes = placement_schemes[0:2]+[placement_schemes[5]]+[placement_schemes[9]], 
        #                           labels = labels[0:2]+[labels[5]]+[labels[9]], fig_path_prefix=f"{fig_dir}/Original")
        
        # plot_num_of_loads_compare(num_of_token_loads, "Multi", placement_schemes = [placement_schemes[0]]+[placement_schemes[2]]+[placement_schemes[6]]+[placement_schemes[10]], 
        #                           labels = [labels[0]]+[labels[2]]+[labels[6]]+[labels[10]], fig_path_prefix=f"{fig_dir}/Multi")
        
        # plot_num_of_loads_compare(num_of_token_loads, "Repli_Act", placement_schemes = [placement_schemes[0]]+[placement_schemes[3]]+[placement_schemes[7]]+[placement_schemes[11]], 
        #                           labels = [labels[0]]+[labels[3]]+[labels[7]]+[labels[11]], fig_path_prefix=f"{fig_dir}/Repli_Act")
        
        plot_num_of_loads_compare(num_of_token_loads, "Repli_Collab", placement_schemes = [placement_schemes[0]]+[placement_schemes[4]]+[placement_schemes[8]]+[placement_schemes[12]], 
                                  labels = [labels[0]]+[labels[4]]+[labels[8]]+[labels[12]], fig_path_prefix=f"{fig_dir}/Repli_Collab")
        
        
        fig_path = os.path.join(fig_dir,f"Load_Standard_Deviation_original_multi_duplicate.svg")
        plot_load_std_line(num_of_token_loads, placement_schemes, labels, fig_path)

        # fig_path = os.path.join(fig_dir,f"Load_Standard_Deviation_original.svg")
        # plot_load_std_line(num_of_token_loads, placement_schemes[0:2]+[placement_schemes[5]]+[placement_schemes[9]], labels[0:2]+[labels[5]]+[labels[9]], fig_path)

        # fig_path = os.path.join(fig_dir,f"Load_Standard_Deviation_multi.svg")
        # plot_load_std_line(num_of_token_loads, [placement_schemes[0]]+[placement_schemes[2]]+[placement_schemes[6]]+[placement_schemes[10]], [labels[0]]+[labels[2]]+[labels[6]]+[labels[10]], fig_path)

        # fig_path = os.path.join(fig_dir,f"Load_Standar_Deviation_duplicate.svg")
        # plot_load_std_line(num_of_token_loads, [placement_schemes[0]]+placement_schemes[3:5]+placement_schemes[7:9]+placement_schemes[11:13], [labels[0]]+labels[3:5]+labels[7:9]+labels[11:13], fig_path)
        
        # fig_path = os.path.join(fig_dir,f"Load_Standard_Deviation_Occult.svg")
        # plot_load_std_line(num_of_token_loads, [placement_schemes[0]]+placement_schemes[1:5], [labels[0]]+labels[1:5], fig_path)

        # fig_path = os.path.join(fig_dir,f"Load_Standard_Deviation_Spectral_even.svg")
        # plot_load_std_line(num_of_token_loads, [placement_schemes[0]]+placement_schemes[5:9], [labels[0]]+labels[5:9], fig_path)

        # fig_path = os.path.join(fig_dir,f"Load_Standar_Deviation_Spectral_uneven.svg")
        # plot_load_std_line(num_of_token_loads, [placement_schemes[0]]+placement_schemes[9:13], [labels[0]]+labels[9:13], fig_path)
        '''