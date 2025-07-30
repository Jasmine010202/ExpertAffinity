import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
import random

from utils import extract_routing_trace, extract_expert_placement, extract_replicated_experts, prompt_to_gpu, extract_expert_placement_multi_copies

num_layers = 16
num_experts_per_layer = 64
gpu_node_mapping = np.array([0, 0, 1, 1]) # GPU 0和1映射到node 0；2和3映射到node 1

# 节点数、GPU数
num_nodes = 2
num_gpus_per_node = 2
num_gpus = num_nodes * num_gpus_per_node


# 通信量和计算负载合并处理：
# 无复制和复制激活次数高的专家
def communication_traffic_calculation_load_analyze(routing_trace, expert_placement, replicated_experts = None):
    # 转发的token副本数
    num_intra_gpu = 0  # token 路由到自己所在 GPU 上的副本数
    num_inter_gpu = 0  # 路由到同节点其它 GPU 的副本数
    num_inter_node = 0  # 路由到不同节点上 GPU 的副本数

    # 计算负载：
    gpus_token_load_per_layer = [{gpu_id:0 for gpu_id in range(num_gpus)} for layer in range(num_layers)]
    nodes_token_load_per_layer = [{node_id:0 for node_id in range(num_nodes)} for layer in range(num_layers)]


    communication_result = {}
    for prompt in routing_trace:
        # token 所在GPU、Node
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        token_node_id = gpu_node_mapping[token_gpu_id]

        token_routing_traces = np.array(prompt["trace"])
        num_tokens = token_routing_traces.shape[0]

        for token_id in range(num_tokens):
            for layer_id in range(num_layers):  
                expert_ids = token_routing_traces[token_id, layer_id]
                
                if replicated_experts is None:
                    # 没有专家复制
                    # 通信量
                    expert_gpu_ids = expert_placement[layer_id, expert_ids]
                    # print(f"expert_gpu_ids:{expert_gpu_ids}")
                    unique_target_gpu_ids = np.unique(expert_gpu_ids)  
                    # print(f"unique_gpu_ids:{unique_target_gpu_ids}")
                   
                    for target_gpu_id in unique_target_gpu_ids:
                        if target_gpu_id == token_gpu_id:
                            num_intra_gpu += 1
                        elif gpu_node_mapping[target_gpu_id] == token_node_id:
                            num_inter_gpu += 1
                        else:
                            num_inter_node += 1

                    # 计算负载
                    for expert_id in expert_ids:
                        expert_gpu_id = expert_placement[layer_id, expert_id]
                        # gpus_token_load[expert_gpu_id] +=1
                        gpus_token_load_per_layer[layer_id][expert_gpu_id] += 1
                else:
                    # 复制激活次数高的个别专家
                    replicated_experts_this_layer = set(replicated_experts[layer_id])   # 本层复制专家
                    
                    # 区分被复制和不被复制的专家
                    replicated_expert_ids = [e for e in expert_ids if e in replicated_experts_this_layer]
                    not_replicated_expert_ids = [e for e in expert_ids if e not in replicated_experts_this_layer]
                    
                    # 通信量
                    # 激活了复制的专家
                    has_replicated = bool(replicated_expert_ids)
                    if has_replicated:
                        num_intra_gpu += 1

                    # 处理其他非复制专家
                    if not_replicated_expert_ids:
                        expert_gpu_ids = expert_placement[layer_id, not_replicated_expert_ids]
                        # print(f"expert_gpu_ids:{expert_gpu_ids}")
                        
                        unique_target_gpu_ids = set(np.unique(expert_gpu_ids))
                        # print(f"unique_gpu_ids:{unique_target_gpu_ids}")

                        # 如果已有intra_gpu，排除掉token所在GPU
                        if has_replicated and token_gpu_id in unique_target_gpu_ids:
                            unique_target_gpu_ids.remove(token_gpu_id)

                        # print(f"unique_gpu_ids:{unique_target_gpu_ids}")

                        for target_gpu_id in unique_target_gpu_ids:
                            if target_gpu_id == token_gpu_id:
                                num_intra_gpu += 1
                            elif gpu_node_mapping[target_gpu_id] == token_node_id:
                                num_inter_gpu += 1
                            else:
                                num_inter_node += 1
                    
                    # 计算负载
                    # 被复制的专家, 负载数记录到token所在GPU
                    for expert_id in replicated_expert_ids:
                        gpus_token_load_per_layer[layer_id][token_gpu_id] += 1   
                        # gpus_token_load[token_gpu_id] +=1

                    # 未被复制的专家
                    for expert_id in not_replicated_expert_ids:
                        expert_gpu_id = expert_placement[layer_id, expert_id]
                        gpus_token_load_per_layer[layer_id][expert_gpu_id] += 1

            #     break
            # break

    # 通信结果汇总
    print(f"GPU内token副本数:\t{num_intra_gpu}")
    print(f"跨GPUtoken副本数:\t{num_inter_gpu}")
    print(f"跨节点token副本数:\t{num_inter_node}")
 
    communication_result["num_intra_gpu"] = num_intra_gpu
    communication_result["num_inter_gpu"] = num_inter_gpu
    communication_result["num_inter_node"] = num_inter_node

    # 计算负载汇总
    # 每层每个节点的负载
    for layer_id in range(num_layers):
        for gpu_id, gpu_load in gpus_token_load_per_layer[layer_id].items():
            node_id = gpu_node_mapping[gpu_id]
            nodes_token_load_per_layer[layer_id][node_id] += gpu_load

    # result_total={"gpus":gpus_token_load, "nodes":nodes_token_load}
    # print(f"GPU:\t{gpus_token_load_per_layer}")
    # print(f"Node:\t{nodes_token_load_per_layer}")

    calculation_result = {"gpus":gpus_token_load_per_layer, "nodes":nodes_token_load_per_layer}

    return communication_result, calculation_result

'''
# 无复制和复制激活次数高的专家
def calculate_num_of_token_copies(routing_trace, expert_placement, replicated_experts = None):
    # 转发的token副本数
    num_intra_gpu = 0  # token 路由到自己所在 GPU 上的副本数
    num_inter_gpu = 0  # 路由到同节点其它 GPU 的副本数
    num_inter_node = 0  # 路由到不同节点上 GPU 的副本数

    result = {}
    for prompt in routing_trace:
        # token 所在GPU、Node
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        token_node_id = gpu_node_mapping[token_gpu_id]

        token_routing_traces = np.array(prompt["trace"])
        num_tokens = token_routing_traces.shape[0]

        for token_id in range(num_tokens):
            for layer_id in range(num_layers):  
                expert_ids = token_routing_traces[token_id, layer_id]
                
                if replicated_experts is None:
                    expert_gpu_ids = expert_placement[layer_id, expert_ids]
                    # print(f"expert_gpu_ids:{expert_gpu_ids}")

                    unique_target_gpu_ids = np.unique(expert_gpu_ids)  
                    # print(f"unique_gpu_ids:{unique_target_gpu_ids}")
                   
                    for target_gpu_id in unique_target_gpu_ids:
                        if target_gpu_id == token_gpu_id:
                            num_intra_gpu += 1
                        elif gpu_node_mapping[target_gpu_id] == token_node_id:
                            num_inter_gpu += 1
                        else:
                            num_inter_node += 1
                else:
                    replicated_experts_this_layer = set(replicated_experts[layer_id])   # 本层复制专家
                    # 分成两类
                    replicated_expert_ids = [e for e in expert_ids if e in replicated_experts_this_layer]
                    non_replicated_expert_ids = [e for e in expert_ids if e not in replicated_experts_this_layer]
                    
                    # 激活了复制的专家
                    has_replicated = bool(replicated_expert_ids)
                    if has_replicated:
                        num_intra_gpu += 1

                    # 处理其他非复制专家
                    if non_replicated_expert_ids:
                        expert_gpu_ids = expert_placement[layer_id, non_replicated_expert_ids]
                        # print(f"expert_gpu_ids:{expert_gpu_ids}")
                        
                        unique_target_gpu_ids = set(np.unique(expert_gpu_ids))
                        # print(f"unique_gpu_ids:{unique_target_gpu_ids}")

                        # 如果已有intra_gpu，排除掉token所在GPU
                        if has_replicated and token_gpu_id in unique_target_gpu_ids:
                            unique_target_gpu_ids.remove(token_gpu_id)

                        # print(f"unique_gpu_ids:{unique_target_gpu_ids}")

                        for target_gpu_id in unique_target_gpu_ids:
                            if target_gpu_id == token_gpu_id:
                                num_intra_gpu += 1
                            elif gpu_node_mapping[target_gpu_id] == token_node_id:
                                num_inter_gpu += 1
                            else:
                                num_inter_node += 1

            #     break
            # break

    print(f"GPU内token副本数:\t{num_intra_gpu}")
    print(f"跨GPUtoken副本数:\t{num_inter_gpu}")
    print(f"跨节点token副本数:\t{num_inter_node}")
 
    result["num_intra_gpu"] = num_intra_gpu
    result["num_inter_gpu"] = num_inter_gpu
    result["num_inter_node"] = num_inter_node

    return result
'''

# 根据历史负载预测的复制后负载情况，加权轮询
def choose_gpu_for_replica_by_polling_weight(polling_weights):
    gpus = [int(gpu_id) for gpu_id in polling_weights.keys()]
    weights = [weight for weight in polling_weights.values()]
   
    # # random.choices 返回一个列表，所以取第一个元素
    selected_gpu_idx = random.choices(gpus, weights=weights, k=1)[0]
    
    return selected_gpu_idx

# 拓扑感知优先，激活的专家属于被复制的专家组的时候，优先考虑和token同GPU/节点的，如果不行，就加权概率随机选择一个副本，权重是为了倾向于把副本交给预估负载较低的GPU
def choose_gpu_for_replica_by_topology(token_gpu_id, token_node_id, replica_belong_to_gpus, polling_weights_for_replicated_group):

    # original_gpu = replication_info["original_gpu"]
    # target_gpu = replication_info["target_gpu"]

    # replica_belong_to_gpus = list(set([original_gpu, target_gpu])) # 确保唯一性
    # print("replica_belong_to_gpus", replica_belong_to_gpus)

    # 优先查找和token同GPU或节点的副本
    local_gpu_replicas = []
    local_node_replicas = []
    for gpu_id in replica_belong_to_gpus:
        if gpu_id == token_gpu_id:
            local_gpu_replicas.append(gpu_id) # 同GPU的副本
        elif gpu_node_mapping[gpu_id] == token_node_id:
            local_node_replicas.append(gpu_id) 

    if local_gpu_replicas:
        # print("local_gpu_replicas",local_gpu_replicas)
        return random.choice(local_gpu_replicas)

    if local_node_replicas:
        # print("local_node_replicas",local_node_replicas)
        return random.choice(local_node_replicas)

    # print("by_polling_weight")
    # 没有同GPU或者同节点的副本，就用权重随机轮询一个
    return choose_gpu_for_replica_by_polling_weight(polling_weights_for_replicated_group)


def communication_traffic_calculation_load_analyze_group_level(routing_trace, expert_placement, replication_info_per_layer = None, polling_weights_replicated_experts = None, routing_policy = "weight"):
    # 转发的token副本数
    num_intra_gpu = 0  # token 路由到自己所在 GPU 上的副本数
    num_inter_gpu = 0  # 路由到同节点其它 GPU 的副本数
    num_inter_node = 0  # 路由到不同节点上 GPU 的副本数

    # 计算负载：
    gpus_token_load_per_layer = [{gpu_id:0 for gpu_id in range(num_gpus)} for layer in range(num_layers)]
    nodes_token_load_per_layer = [{node_id:0 for node_id in range(num_nodes)} for layer in range(num_layers)]


    # print(replication_info_per_layer)
    communication_result = {}
    for prompt in routing_trace:
        # token 所在GPU、Node
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        token_node_id = gpu_node_mapping[token_gpu_id]

        token_routing_traces = np.array(prompt["trace"])
        num_tokens = token_routing_traces.shape[0]

        for token_id in range(num_tokens):
            for layer_id in range(num_layers):  
                activated_expert_ids = token_routing_traces[token_id, layer_id]
                # print("layer_id",layer_id)
                # print(f"activated_expert_ids:{activated_expert_ids}")

                # 本层所有需要访问的目标GPU ID
                unique_target_gpu_ids = set()

                # 本层被复制的专家组
                replication_info = replication_info_per_layer.get(str(layer_id)) 
                replicated_group = []
                replica_belong_to_gpus = []
                # print("replication_info",replication_info)
                
                if replication_info is not None:
                    replicated_group = replication_info["replicated_group"]
                    original_gpu = replication_info["original_gpu"]
                    target_gpu = replication_info["target_gpu"]

                    replica_belong_to_gpus = list(set([original_gpu, target_gpu])) # 确保唯一性

                # print("replicated_group", replicated_group)
                # print("replica_belong_to_gpus", replica_belong_to_gpus)
                    
                # 逐个检查这个token在当前层激活的每一个专家
                for expert_id in activated_expert_ids:
                    # is_replicated = False
                    # belong_to_replicated_group_key = None
                    selected_gpu_id = -1

                    # print("expert_id", expert_id)

                    if expert_id in replicated_group:
                            # 这个专家属于被复制的专家组
                            # print("in replicated_group")
                            belong_to_replicated_group_key = str(replicated_group) # 转换为字符串列表作为键
                            
                            # 加权轮询选择原先的GPU或者被复制到的GPU
                            polling_weights_for_group = polling_weights_replicated_experts.get(str(layer_id), {}).get(belong_to_replicated_group_key) 

                            if routing_policy == "weight":
                                selected_gpu_id = choose_gpu_for_replica_by_polling_weight(polling_weights_for_group)
                            elif routing_policy == "topology":
                                # 优先本地副本
                                selected_gpu_id = choose_gpu_for_replica_by_topology(token_gpu_id, token_node_id, replica_belong_to_gpus, polling_weights_for_group)
                            else:
                                raise ValueError(f"Unknown Routing Policy: {routing_policy}")
                    else:
                        # 不属于被复制的专家，直接找GPUid
                        selected_gpu_id = expert_placement[layer_id, expert_id][0]  # 复制专家组的方案对应的是GPUid列表

                    # print("select_gpu_id", selected_gpu_id)

                    unique_target_gpu_ids.add(selected_gpu_id)

                    # 计算负载
                    gpus_token_load_per_layer[layer_id][selected_gpu_id] += 1

                # print("unique_target_gpu_ids", unique_target_gpu_ids)
                # print(gpus_token_load_per_layer[layer_id])

                # 计算通信量
                for target_gpu_id in unique_target_gpu_ids:
                    if target_gpu_id == token_gpu_id:
                        num_intra_gpu += 1
                    elif gpu_node_mapping[target_gpu_id] == token_node_id:
                        num_inter_gpu += 1
                    else:
                        num_inter_node += 1

        #     break
        # break

    print(f"GPU内token副本数:\t{num_intra_gpu}")
    print(f"跨GPUtoken副本数:\t{num_inter_gpu}")
    print(f"跨节点token副本数:\t{num_inter_node}")
 
    communication_result["num_intra_gpu"] = num_intra_gpu
    communication_result["num_inter_gpu"] = num_inter_gpu
    communication_result["num_inter_node"] = num_inter_node

    # 计算负载汇总
    # 每层每个节点的负载
    for layer_id in range(num_layers):
        for gpu_id, gpu_load in gpus_token_load_per_layer[layer_id].items():
            node_id = gpu_node_mapping[gpu_id]
            nodes_token_load_per_layer[layer_id][node_id] += gpu_load

    
    print(f"GPU:\t{gpus_token_load_per_layer}")
    print(f"Node:\t{nodes_token_load_per_layer}")

    calculation_result = {"gpus":gpus_token_load_per_layer, "nodes":nodes_token_load_per_layer}

    return communication_result, calculation_result


def calculate_load_stats_per_layer(token_loads):
    gpu_stats_per_layer = {"mean":[], "var":[], "std":[], "max":[], "min":[]}
    node_stats_per_layer = {"mean":[], "var":[], "std":[], "max":[], "min":[]}

    gpu_loads_per_layer = token_loads["gpus"] 
    node_loads_per_layer = token_loads["nodes"] 

    # 每一层内的GPU、Node负载的均值、方差、标准差、最大/最小值
    for layer_gpu_loads_dict, layer_node_loads_dict in zip(gpu_loads_per_layer, node_loads_per_layer):
        gpu_loads = list(layer_gpu_loads_dict.values())
        node_loads = list(layer_node_loads_dict.values())

        gpu_stats_per_layer["mean"].append(int(np.mean(gpu_loads)))
        gpu_stats_per_layer["var"].append(float(np.var(gpu_loads)))
        gpu_stats_per_layer["std"].append(float(np.std(gpu_loads)))
        gpu_stats_per_layer["max"].append(int(np.max(gpu_loads)))
        gpu_stats_per_layer["min"].append(int(np.min(gpu_loads)))

        node_stats_per_layer["mean"].append(int(np.mean(node_loads)))
        node_stats_per_layer["var"].append(float(np.var(node_loads)))
        node_stats_per_layer["std"].append(float(np.std(node_loads)))
        node_stats_per_layer["max"].append(int(np.max(node_loads)))
        node_stats_per_layer["min"].append(int(np.min(node_loads)))

    #所有层的均值
    gpu_stats_all_layers_avg = {
        "mean": float(np.mean(gpu_stats_per_layer["mean"])),
        "var": float(np.mean(gpu_stats_per_layer["var"])),
        "std": float(np.mean(gpu_stats_per_layer["std"])),
        "max": float(np.mean(gpu_stats_per_layer["max"])),
        "min": float(np.mean(gpu_stats_per_layer["min"]))
    }

    node_stats_all_layers_avg = {
        "mean": float(np.mean(node_stats_per_layer["mean"])),
        "var": float(np.mean(node_stats_per_layer["var"])),
        "std": float(np.mean(node_stats_per_layer["std"])),
        "max": float(np.mean(node_stats_per_layer["max"])),
        "min": float(np.mean(node_stats_per_layer["min"]))
    }

    stats_per_layer = {"gpu": gpu_stats_per_layer, "node": node_stats_per_layer}
    stats_all_layers_avg = {"gpu": gpu_stats_all_layers_avg, "node": node_stats_all_layers_avg}

    return stats_per_layer, stats_all_layers_avg


def plot_combined_copies_and_load_stats_gpu_node(num_of_token_copies, stats_all_layers_avg_scheme,
                                                 placement_schemes, labels, fig_width, fig_path):
    dataset = ["sonnet", "GSM8K", "conala"]

    x = np.arange(len(placement_schemes))
    width = 0.35

    inter_gpu_values = [num_of_token_copies[dataset[0]][p]["num_inter_gpu"] for p in placement_schemes]
    inter_node_values = [num_of_token_copies[dataset[0]][p]["num_inter_node"] for p in placement_schemes]

    gpu_std_vals = [stats_all_layers_avg_scheme[p]["gpu"]["std"] for p in placement_schemes]
    gpu_max_vals = [stats_all_layers_avg_scheme[p]["gpu"]["max"] for p in placement_schemes]
    gpu_min_vals = [stats_all_layers_avg_scheme[p]["gpu"]["min"] for p in placement_schemes]

    node_std_vals = [stats_all_layers_avg_scheme[p]["node"]["std"] for p in placement_schemes]
    node_max_vals = [stats_all_layers_avg_scheme[p]["node"]["max"] for p in placement_schemes]
    node_min_vals = [stats_all_layers_avg_scheme[p]["node"]["min"] for p in placement_schemes]

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
    # ax2.plot(x, gpu_min_vals, marker='s', linestyle='--', color=colors[4], linewidth=2, label="Avg. GPU load Min")

    ax2.plot(x, node_std_vals, marker='s', linestyle='--', color=colors[4], linewidth=2, label="Avg. Node load Std")
    # ax2.plot(x, node_max_vals, marker='+', linestyle='--', color=colors[3], linewidth=2, label="Avg. Node load Max")
    # ax2.plot(x, node_min_vals, marker='x', linestyle='--', color=colors[4], linewidth=2, label="Avg. Node load Min")

    ax2.set_ylabel(f"GPU / Node Load (Avg. Std / Max / Min)")
    ax2.tick_params(axis='y')

    ax2.set_ylim(0, max(gpu_std_vals + gpu_max_vals + node_std_vals + node_max_vals) * 1.1)

    offset_line = 2000
    # 标注折线图点值
    for i in range(len(x)):
        ax2.text(i, gpu_std_vals[i] + offset_line, f"{gpu_std_vals[i]:.2f}", ha='center', va='bottom', fontsize=8)
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
    ax1.set_xticklabels(labels, rotation=0, ha='center')

    plt.title(f"Num of Token Copies & Overall GPU/Node Loads Statistics —— Spectral Clustering")
    plt.tight_layout()

    plt.savefig(fig_path)
    plt.close()



if __name__ == "__main__":

    model_name = "OLMoE"
    # input_name = "sonnet"   
    prompt_nums = [512] # 8, 16, 32, 64, 128, 256, 512, 1024
    top_k = 8 

    num_replicated_experts = 4

    fig_dir = f"Group_level_Copies_Loads_Compare_sim/MultiNodes_MultiGPU/One_Replica_of_Highest_Load_Group/sonnet_{model_name}_top{top_k}/figs"   
    os.makedirs(fig_dir, exist_ok=True)

    result_dir = f"Group_level_Copies_Loads_Compare_sim/MultiNodes_MultiGPU/One_Replica_of_Highest_Load_Group/sonnet_{model_name}_top{top_k}/data" 
    os.makedirs(result_dir, exist_ok=True)


    for num in prompt_nums:
        sonnet_routing_trace = extract_routing_trace(f"./Occult_test/expert_trace/traffic_test/by_prompt/{model_name}_sonnet_top{top_k}/routing_trace_{num}.jsonl")
       
        # ###############################################Placement##################################################
        vanilla_placement = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/OLMoE_vanilla_placement.json")
        sonnet_spectral_even_multi_placement = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/OLMoE_spectral_even_sonnet_512_nodes2_gpus4.json")
        sonnet_spectral_uneven_multi_placement = extract_expert_placement(num_layers, num_experts_per_layer, "./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/OLMoE_spectral_uneven_sonnet_512_nodes2_gpus4.json")
        ####duplicate
        # Activation
        act_replicated_experts_list = extract_replicated_experts(num_layers, f"./Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_nodes2_gpus4_re{num_replicated_experts}_replicated_experts.json")
        sonnet_spectral_even_multi_repli_act = extract_expert_placement(num_layers, num_experts_per_layer, f"./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_even_nodes2_gpus4_re{num_replicated_experts}.json")
        sonnet_spectral_uneven_multi_repli_act = extract_expert_placement(num_layers, num_experts_per_layer, f"./Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_uneven_nodes2_gpus4_re{num_replicated_experts}.json")
        # group_level
        sonnet_spectral_uneven_multi_repli_group = extract_expert_placement_multi_copies(num_layers, num_experts_per_layer, f"./Occult_test/expert_placement/Duplicate_Group_Level/MultiNodes_MultiGPU/One_Replica_of_Highest_Load_Group/OLMoE_sonnet_512_uneven_nodes2_gpus4_replicated.json")
        with open("./Occult_test/expert_placement/Duplicate_Group_Level/MultiNodes_MultiGPU/One_Replica_of_Highest_Load_Group/OLMoE_sonnet_512_uneven_nodes2_gpus4_replicated_info.json","r") as f1:
            ssumrg_replication_info = json.load(f1)
        with open("./Occult_test/expert_placement/Duplicate_Group_Level/MultiNodes_MultiGPU/One_Replica_of_Highest_Load_Group/OLMoE_sonnet_512_uneven_nodes2_gpus4_polling_weights.json","r") as f2:
            ssumrg_polling_weights = json.load(f2)

        # # print(ssumrg_replication_info)
        # # print(ssumrg_polling_weights)
        # ###############################################Calculate_Num_of_Token_Copies##################################################
        sonnet_vanilla_copies, sonnet_vanilla_loads = communication_traffic_calculation_load_analyze(sonnet_routing_trace, vanilla_placement)
        sonnet_spectral_even_multi_copies, sonnet_spectral_even_multi_loads = communication_traffic_calculation_load_analyze(sonnet_routing_trace, sonnet_spectral_even_multi_placement)
        sonnet_spectral_uneven_multi_copies, sonnet_spectral_uneven_multi_loads = communication_traffic_calculation_load_analyze(sonnet_routing_trace, sonnet_spectral_uneven_multi_placement)
        
        # ####duplicate
        # # Activation
        sonnet_spectral_even_multi_repli_act_copies, sonnet_spectral_even_multi_repli_act_loads = communication_traffic_calculation_load_analyze(sonnet_routing_trace, sonnet_spectral_even_multi_repli_act, act_replicated_experts_list)
        sonnet_spectral_uneven_multi_repli_act_copies, sonnet_spectral_uneven_multi_repli_act_loads = communication_traffic_calculation_load_analyze(sonnet_routing_trace, sonnet_spectral_uneven_multi_repli_act, act_replicated_experts_list)
        
        # # group_level
        # # 加权轮询
        sonnet_spectral_uneven_multi_repli_group_weight_copies, sonnet_spectral_uneven_multi_repli_group_weight_loads = communication_traffic_calculation_load_analyze_group_level(sonnet_routing_trace, sonnet_spectral_uneven_multi_repli_group, ssumrg_replication_info, ssumrg_polling_weights, "weight")
        # 优先本地副本
        sonnet_spectral_uneven_multi_repli_group_topology_copies, sonnet_spectral_uneven_multi_repli_group_topology_loads = communication_traffic_calculation_load_analyze_group_level(sonnet_routing_trace, sonnet_spectral_uneven_multi_repli_group, ssumrg_replication_info, ssumrg_polling_weights, "topology")

        # ###############################################File##################################################
        num_of_token_copies = {}
        num_of_token_copies["sonnet"] = {}
        num_of_token_copies["sonnet"]["vanilla_placement"] = sonnet_vanilla_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_even_multi_placement"] = sonnet_spectral_even_multi_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_uneven_multi_placement"] = sonnet_spectral_uneven_multi_copies
        ####duplicate
        #  Activation
        num_of_token_copies["sonnet"]["sonnet_spectral_even_multi_repli_act_placement"] = sonnet_spectral_even_multi_repli_act_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_uneven_multi_repli_act_placement"] = sonnet_spectral_uneven_multi_repli_act_copies
        # group_level
        num_of_token_copies["sonnet"]["sonnet_spectral_uneven_multi_repli_group_weight_placement"] = sonnet_spectral_uneven_multi_repli_group_weight_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_uneven_multi_repli_group_topology_placement"] = sonnet_spectral_uneven_multi_repli_group_topology_copies


        copies_filename = os.path.join(result_dir, f"num_of_token_copies_spectral_multi_repli_act_group.json")
        with open (copies_filename, "w") as f1:
            json.dump(num_of_token_copies ,f1,indent=2)

        num_of_token_loads = {}
        num_of_token_loads["sonnet"] = {}
        num_of_token_loads["sonnet"]["vanilla_placement"] = sonnet_vanilla_loads
        num_of_token_loads["sonnet"]["sonnet_spectral_even_multi_placement"] = sonnet_spectral_even_multi_loads
        num_of_token_loads["sonnet"]["sonnet_spectral_uneven_multi_placement"] = sonnet_spectral_uneven_multi_loads
        ####duplicate
        #  Activation
        num_of_token_loads["sonnet"]["sonnet_spectral_even_multi_repli_act_placement"] = sonnet_spectral_even_multi_repli_act_loads
        num_of_token_loads["sonnet"]["sonnet_spectral_uneven_multi_repli_act_placement"] = sonnet_spectral_uneven_multi_repli_act_loads
        # group_level
        num_of_token_loads["sonnet"]["sonnet_spectral_uneven_multi_repli_group_weight_placement"] = sonnet_spectral_uneven_multi_repli_group_weight_loads
        num_of_token_loads["sonnet"]["sonnet_spectral_uneven_multi_repli_group_topology_placement"] = sonnet_spectral_uneven_multi_repli_group_topology_loads

        loads_filename = os.path.join(result_dir, f"num_of_token_loads_spectral_multi_repli_act_group.json")
        with open (loads_filename, "w") as f2:
            json.dump(num_of_token_loads, f2, indent=2)
        
        ######################################################## 负载统计数据计算 ########################################################

        stats_per_layer_scheme = {}
        stats_all_layers_avg_scheme = {}
        for scheme, token_loads in num_of_token_loads["sonnet"].items():
            stats_per_layer_scheme[scheme], stats_all_layers_avg_scheme[scheme] = calculate_load_stats_per_layer(token_loads)

        load_stats_per_layer_filename = os.path.join(result_dir, f"stats_of_token_loads_per_layer.json")
        with open (load_stats_per_layer_filename, "w") as f3:
            json.dump(stats_per_layer_scheme, f3, indent=2)

        load_stats_avg_filename = os.path.join(result_dir, f"stats_of_token_loads_all_layers_avg.json")
        with open (load_stats_avg_filename, "w") as f4:
            json.dump(stats_all_layers_avg_scheme, f4, indent=2)

        ####################################################### 画图 ########################################################
        
        file_path_prefix = "./Group_level_Copies_Loads_Compare_sim/MultiNodes_MultiGPU/One_Replica_of_Highest_Load_Group/sonnet_OLMoE_top8/data"
        with open(f"{file_path_prefix}/num_of_token_copies_spectral_multi_repli_act_group.json", "r") as f1:
            num_of_token_copies = json.load(f1)

        with open(f"{file_path_prefix}/stats_of_token_loads_all_layers_avg.json", "r") as f2:
            stats_all_layers_avg_scheme = json.load(f2)
        
        placement_schemes = ["vanilla_placement", 
                             "sonnet_spectral_even_multi_placement", 
                             "sonnet_spectral_even_multi_repli_act_placement", 
                             "sonnet_spectral_uneven_multi_placement", 
                             "sonnet_spectral_uneven_multi_repli_act_placement",
                             "sonnet_spectral_uneven_multi_repli_group_weight_placement",
                             "sonnet_spectral_uneven_multi_repli_group_topology_placement"
                             ]
        labels = ["Vanilla", 
                  "Even_Multi", 
                  "Even_Multi_Repli_Act", 
                  "Uneven_Multi", 
                  "Uneven_Multi_Repli_Act",
                  "Uneven_Multi_Repli_Group_Weight",
                  "Uneven_Multi_Repli_Group_Topology"
                  ]
        
        fig_path = os.path.join(fig_dir,f"communication_computing_compare.svg")
        plot_combined_copies_and_load_stats_gpu_node(num_of_token_copies, stats_all_layers_avg_scheme,
                                        placement_schemes, labels, 24, fig_path)    