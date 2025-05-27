# 模拟专家并行，2个节点，每个节点2块GPU
# 专家放置：GPU0:0~15,GPU1:16~31;GPU2:32~47;GPU3:48~63

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

model_name = "OLMoE"
input_name = "sonnet"
phrase_mode = "decode" #decode
prompt_nums = [8, 16, 32, 64, 128, 256, 512, 1024] # 8, 16, 32, 64, 128, 256, 512, 1024
top_k = 8 
bs = 64
num_of_experts_per_layer = 64
num_expert_pre_GPU = 16

num_of_nodes = 2
num_of_gpus_pre_node = 2
gpu_node_mapping = {0: 0, 1: 0, 2: 1, 3: 1}  # GPU 0和1映射到node 0；2和3映射到node 1
expert_gpu_mapping = {i: i // 16 for i in range(num_of_experts_per_layer)}
#print(expert_gpu_mapping)

fig_dir = f"DCtest/{model_name}/{input_name}/top{top_k}/{phrase_mode}/figs"
os.makedirs(fig_dir, exist_ok=True)
data_dir = f"DCtest/{model_name}/{input_name}/top{top_k}/{phrase_mode}/data"
os.makedirs(data_dir, exist_ok=True)


def communication_rate(routing, inference_time):
    intra_gpu_num = 0  # GPU 内通信
    inter_gpu_num = 0  # GPU 间通信(节点内)
    inter_node_num = 0  # 节点间通信
    total_comm_num =0  # 总通信次数

    num_tokens, num_layers, num_topk = routing.shape

    for token_id in range(num_tokens):
        token_gpu_id = np.random.randint(0, num_of_nodes*num_of_gpus_pre_node)  # 随机选择当前 token 所属 GPU
        token_node_id = gpu_node_mapping[token_gpu_id]  # 获取该 GPU 所属的节点

        for layer_id in range(num_layers):  
            expert_ids = routing[token_id, layer_id, :]
            map_to_gpu = np.vectorize(lambda expert: expert_gpu_mapping[expert]) 
            expert_gpu_ids = map_to_gpu(expert_ids)
            #expert_gpu_ids = expert_gpu_mapping[expert_ids]

            map_to_node = np.vectorize(lambda gpu: gpu_node_mapping[gpu])  # 改映射，map_to_node是dict
            expert_node_ids = map_to_node(expert_gpu_ids)
            #expert_node_ids = gpu_node_mapping[expert_gpu_ids]

            intra_gpu_num += int(2 * np.sum(expert_gpu_ids == token_gpu_id))  # GPU 内通信次数
            inter_gpu_num += int(2 * np.sum((expert_node_ids == token_node_id) & (expert_gpu_ids != token_gpu_id)))  # 同节点跨 GPU 通信
            inter_node_num += int(2 * np.sum(expert_node_ids != token_node_id))  # 跨节点通信
        
    num_and_rate_of_comm = {}
    
    total_comm_num += intra_gpu_num + inter_gpu_num + inter_node_num  

    intra_gpu_rate = intra_gpu_num / total_comm_num *100
    inter_gpu_rate = inter_gpu_num / total_comm_num *100
    inter_node_rate = inter_node_num / total_comm_num *100

    total_comm_per_ms = total_comm_num / (inference_time * 1000)
    intra_gpu_per_ms = intra_gpu_num / (inference_time * 1000)
    inter_gpu_per_ms = inter_gpu_num / (inference_time * 1000)
    inter_node_per_ms = inter_node_num / (inference_time * 1000)

    total_comm_per_s = total_comm_num / (inference_time)
    intra_gpu_per_s = intra_gpu_num / (inference_time)
    inter_gpu_per_s = inter_gpu_num / (inference_time)
    inter_node_per_s = inter_node_num / (inference_time)

    num_and_rate_of_comm ['total_comm_num'] = total_comm_num
    num_and_rate_of_comm ['intra_gpu_num'] = intra_gpu_num
    num_and_rate_of_comm ['inter_gpu_num'] = inter_gpu_num
    num_and_rate_of_comm ['inter_node_num'] = inter_node_num

    num_and_rate_of_comm ['intra_gpu_rate'] = round(intra_gpu_rate, 4)
    num_and_rate_of_comm ['inter_gpu_rate'] = round(inter_gpu_rate, 4)
    num_and_rate_of_comm ['inter_node_rate'] = round(inter_node_rate, 4)

    num_and_rate_of_comm ['total_comm_per_ms'] = round(total_comm_per_ms, 4)
    num_and_rate_of_comm ['intra_gpu_per_ms'] = round(intra_gpu_per_ms, 4)
    num_and_rate_of_comm ['inter_gpu_per_ms'] = round(inter_gpu_per_ms, 4)
    num_and_rate_of_comm ['inter_node_per_ms'] = round(inter_node_per_ms, 4)

    num_and_rate_of_comm ['total_comm_per_s'] = round(total_comm_per_s, 4)
    num_and_rate_of_comm ['intra_gpu_per_s'] = round(intra_gpu_per_s, 4)
    num_and_rate_of_comm ['inter_gpu_per_s'] = round(inter_gpu_per_s, 4)
    num_and_rate_of_comm ['inter_node_per_s'] = round(inter_node_per_s, 4)
    
    # 输出统计结果
    print(f"总通信次数:\t{total_comm_num}\t{total_comm_per_ms:.2f}\t{total_comm_per_s:.2f}")
    print(f"GPU内通信次数:\t{intra_gpu_num}\t{intra_gpu_rate:.2f}%\t{intra_gpu_per_ms:.2f}\t{intra_gpu_per_s:.2f}")
    print(f"跨GPU通信次数:\t{inter_gpu_num}\t{inter_gpu_rate:.2f}%\t{inter_gpu_per_ms:.2f}\t{inter_gpu_per_s:.2f}")
    print(f"跨节点通信次数:\t{inter_node_num}\t{inter_node_rate:.2f}%\t{inter_node_per_ms:.2f}\t{inter_node_per_s:.2f}")
    
    return num_and_rate_of_comm

#batch/all dataset
def experts_selection_inter_layer(routing_data):
    num_tokens, num_layers, top_k = routing_data.shape
    experts_selection_stats = np.zeros((num_layers - 1, num_of_experts_per_layer, num_of_experts_per_layer)) # 记录层间的专家选择关系
    experts_selection_edge_sets = [set() for _ in range(num_layers - 1)]

    for layer in range(num_layers-1):
        for token in range(num_tokens):
            experts_curr_layer = routing_data[token,layer]      #当前层专家选择
            experts_next_layer = routing_data[token,layer+1]    #下一层专家选择
                
            for i in range(top_k):
                for j in range (top_k):
                    expert_i, expert_j = experts_curr_layer[i], experts_next_layer[j]
                    experts_selection_stats[layer, expert_i, expert_j] += 1
                    experts_selection_edge_sets[layer].add((expert_i, expert_j))
    
    return experts_selection_stats, experts_selection_edge_sets


def cosine_similarity_of_experts_selection(experts_selection_stats):
    num_layer_pairs = experts_selection_stats.shape[0]
    cosine_sim_list = []

    for i in range(num_layer_pairs - 1):
        curr_experts_selection = experts_selection_stats[i].flatten().astype(float)
        next_experts_selection = experts_selection_stats[i + 1].flatten().astype(float)

        sim = cosine_similarity(curr_experts_selection.reshape(1, -1), next_experts_selection.reshape(1, -1))[0, 0]
        cosine_sim_list.append(sim)

    average_cosine_sim = np.mean(cosine_sim_list)
    return cosine_sim_list, average_cosine_sim


# 每两层的专家选择关系在不同batch中的jaccard相似度
# def jaccard_similarity_of_experts_selection(selection_edge_sets_per_batch):
#     num_batches = len(selection_edge_sets_per_batch)
#     num_layer_pairs = len(selection_edge_sets_per_batch[0])
   
#     average_jaccard_per_layer_pairs = []

#     for layer_pair_id in range(num_layer_pairs):
#         similarity_between_batches=[]
#         for i, j in combinations(range(num_batches), 2): # 所有batch两两组合 C(x,2)
#             A = selection_edge_sets_per_batch[i][layer_pair_id]
#             B = selection_edge_sets_per_batch[j][layer_pair_id]
#             intersection = len(A & B)
#             union = len(A | B)
#             if union > 0:
#                 sim = intersection / union
#             else:
#                 sim = 1.0
#             #print(sim)
#             similarity_between_batches.append(sim)
#         average_jaccard_per_layer_pairs.append(np.mean(similarity_between_batches))

#     return average_jaccard_per_layer_pairs

# 相邻层之间专家选择关系的Jaccard相似度
def jaccard_similarity_of_experts_selection(selection_edge_sets):
    num_layer_pairs = len(selection_edge_sets)
    jaccard_sim_list = []
    jaccard_sim_matrix = np.zeros((num_layer_pairs, num_layer_pairs))

    for layer_pair_id in range(num_layer_pairs - 1):
        
        A = selection_edge_sets[layer_pair_id]
        B = selection_edge_sets[layer_pair_id + 1]
        intersection = len(A & B)
        union = len(A | B)
        if union > 0:
            sim = intersection / union
        else:
            sim = 1.0
            #print(sim)
        jaccard_sim_list.append(sim)
        jaccard_sim_matrix[layer_pair_id,layer_pair_id + 1] = sim   # 对称
        jaccard_sim_matrix[layer_pair_id + 1, layer_pair_id] = sim
        
    average_jaccard_sim = np.mean(jaccard_sim_list)
    return jaccard_sim_list, average_jaccard_sim, jaccard_sim_matrix

def experts_activations_count(routing_data):
    num_tokens, num_layers, top_k = routing_data.shape
    experts_activation_stats = np.zeros((num_layers, num_of_experts_per_layer)) 

    for token in range(num_tokens):
        for layer in range(num_layers):
            for expert in routing_data[token,layer]:
                experts_activation_stats[layer,expert]  += 1
    return experts_activation_stats


def plot_experts_selection_inter_layer(experts_selection_stats, fig_dir, num_of_prompts):
    save_dir = os.path.join(fig_dir, num_of_prompts)
    os.makedirs(save_dir, exist_ok=True)

    num_layer_pairs, _, _ = experts_selection_stats.shape

    for layer_pair_id in range(num_layer_pairs):
        filename = os.path.join(save_dir, f"experts_selection_layer{layer_pair_id}_to_{layer_pair_id + 1}.png")
        fig, ax = plt.subplots(figsize=(12,10))
        sns.heatmap(
            experts_selection_stats[layer_pair_id], annot=False, 
            cmap="YlGnBu", linewidths=0.5, square=True, 
            cbar_kws={"shrink": 0.5}, ax=ax
        )

        ax.set_xticks(np.arange(num_of_experts_per_layer))
        ax.set_xticklabels(np.arange(num_of_experts_per_layer), rotation=90, fontsize=7)  # 旋转90°防止重叠
        ax.set_yticks(np.arange(num_of_experts_per_layer))
        ax.set_yticklabels(np.arange(num_of_experts_per_layer), fontsize=7)

        plt.title(f"Experts Selection — Layer {layer_pair_id} → {layer_pair_id + 1}", fontsize=14)
        plt.xlabel(f"Experts ID - Layer{layer_pair_id + 1}", fontsize=12)
        plt.ylabel(f"Experts ID - Layer{layer_pair_id}", fontsize=12)

        plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


# def comm_plot(intra_gpu_data, inter_gpu_data, inter_node_data, ylabel, fig_path, show_values=False):
#     x_positions = np.linspace(0, len(prompt_nums) - 1, len(prompt_nums))
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(x_positions, intra_gpu_data, marker='o', label="Intra GPU", markersize=8)
#     plt.plot(x_positions, inter_gpu_data, marker='s', label="Inter GPU", markersize=8)
#     plt.plot(x_positions, inter_node_data, marker='^', label="Inter Node", markersize=8)

#     if show_values:
#         for i, (x, y) in enumerate(zip(x_positions, intra_gpu_data)):
#             plt.text(x, y, f"{y:.1f}%", ha='right', va='bottom', fontsize=10)
#         for i, (x, y) in enumerate(zip(x_positions, inter_gpu_data)):
#             plt.text(x, y, f"{y:.1f}%", ha='right', va='bottom', fontsize=10)
#         for i, (x, y) in enumerate(zip(x_positions, inter_node_data)):
#             plt.text(x, y, f"{y:.1f}%", ha='right', va='bottom', fontsize=10)

#     plt.xlabel("Num of Prompts")
#     plt.ylabel(ylabel)
#     plt.title(f"{model_name}   top{top_k}  batch_size={bs}")
#     plt.legend()
#     plt.grid(True)

#     plt.xticks(x_positions, labels=[str(x) for x in prompt_nums], fontsize=12)
#     plt.yticks(np.arange(0, 101, 10), fontsize=12)  
    
#     plt.savefig(fig_path)
#     plt.close()

def plot_comm_ratio_pie(comm_data, num_of_prompts, fig_path):
    labels = ['Intra-GPU', 'Inter-GPU', 'Inter-Node']
    data = [
        comm_data['intra_gpu_rate'],
        comm_data['inter_gpu_rate'],
        comm_data['inter_node_rate']
    ]

    colors = ['#80E1FF', '#78C0FF', '#7AFFDC'] # '#66c2a5', '#91d1c2', '#bfd3e6'
    plt.figure(figsize=(6,6))
    plt.pie(data, labels=labels, autopct='%1.2f%%', startangle=140, colors=colors)
    plt.axis('equal')
    plt.title("Rate of Communication")

    info = f"{model_name}       top{top_k}      batch_size={bs}     num_of_prompts={num_of_prompts}"
    plt.figtext(0.5, 0.01, info, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def plot_comm_num_per_ms(comm_data, fig_path):
    labels = [str(p) for p in prompt_nums]

    total_num = [comm_data[str(p)]['total_comm_per_ms'] for p in prompt_nums]
    intra_gpu = [comm_data[str(p)]['intra_gpu_per_ms'] for p in prompt_nums]
    inter_gpu = [comm_data[str(p)]['inter_gpu_per_ms'] for p in prompt_nums]
    inter_node = [comm_data[str(p)]['inter_node_per_ms'] for p in prompt_nums]

    x = np.arange(len(prompt_nums))  # prompt 数对应的 x 位置
    width = 0.18  # 每根柱子的宽度

    plt.figure(figsize=(12, 6))
    bars_total = plt.bar(x - 1.5 * width, total_num, width=width, label='Total', color='#264653')
    bars_intra_gpu = plt.bar(x - 0.5 * width, intra_gpu, width=width, label='Intra-GPU', color='#7fcdbb')
    bars_inter_gpu = plt.bar(x + 0.5 * width, inter_gpu, width=width, label='Inter-GPU', color='#41b6c4')
    bars_inter_node = plt.bar(x + 1.5 * width, inter_node, width=width, label='Inter-Node', color='#1d91c0')

    plt.xticks(x, labels)
    plt.xlabel("Num of Prompts", fontsize=10)
    plt.ylabel("Communications / ms", fontsize=10)
    plt.title("Num of Communication per Millisecond", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 预留顶部空间，防止数值被截断
    all_heights = total_num + intra_gpu + inter_gpu + inter_node
    max_height = max(all_heights)
    plt.ylim(0, max_height * 1.15)

    for bars in [bars_total, bars_intra_gpu, bars_inter_gpu, bars_inter_node]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,height + 0.2,f"{height:.2f}",ha='center',va='bottom',fontsize=8,rotation=90)

    info = f"{model_name}       top{top_k}      batch_size={bs}"
    #plt.figtext(0.5, 0.01, info, ha='center', fontsize=10)
    plt.figtext(0.99, 0.01, info, ha='right', fontsize=10)

    plt.tight_layout()
    
    plt.savefig(fig_path)
    plt.close()
       
    
def plot_expert_activations(activations, num_of_prompts, fig_dir, show_values=False):
    fig_path = os.path.join(fig_dir, num_of_prompts)
    os.makedirs(fig_path, exist_ok=True)

    num_layers, _ = activations.shape
    
    for layer in range(num_layers):
        plt.figure(figsize=(10, 4))
        bars = plt.bar(np.arange(num_of_experts_per_layer), activations[layer], color='#4682B4')
        plt.title(f"Expert Activations Distribution - Layer {layer}", fontsize=14)
        plt.xlabel("Expert ID", fontsize=10)
        plt.ylabel("Expert Activations", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        plt.xticks(np.arange(num_of_experts_per_layer), fontsize=7, rotation=90)

        # 预留顶部空间，防止数值被截断
        max_height = np.max(activations[layer])
        plt.ylim(0, max_height * 1.15)

        if show_values:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{int(height)}', ha='center', va='bottom', fontsize=8, rotation=90)

        info = f"{model_name}       top{top_k}      batch_size={bs}     num_of_prompts={num_of_prompts}"
        #plt.figtext(0.5, 0.01, info, ha='center', fontsize=10)
        plt.figtext(0.99, 0.01, info, ha='right', fontsize=10)

        plt.tight_layout()

        plt.savefig(os.path.join(fig_path, f"layer_{layer}_expert_activations.png"), dpi=300)
        plt.close()


# def plot_jaccard_matrix(jaccard_matrix, fig_path):
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(jaccard_matrix, cmap="YlGnBu", annot=False, square=True,
#                 cbar_kws={"label": "Jaccard Similarity"}, mask=np.isnan(jaccard_matrix))

#     num_layer_pairs = jaccard_matrix.shape[0]
#     layer_labels = [f"{i}→{i+1}" for i in range(num_layer_pairs)]
#     plt.xticks(np.arange(num_layer_pairs) + 0.5, layer_labels, rotation=90)
#     plt.yticks(np.arange(num_layer_pairs) + 0.5, layer_labels, rotation=0)

#     plt.title("Jaccard Similarity Between Layer Pairs")
#     plt.xlabel("Layer Pair")
#     plt.ylabel("Layer Pair")
#     plt.tight_layout()

#     plt.savefig(fig_path)
#     plt.close()
       

# def process_comm_rate(routing_arrays):
#     intra_gpu_rates = []
#     inter_gpu_rates = []
#     inter_node_rates = []

#     for routing in routing_arrays:
#         intra_gpu_rate, inter_gpu_rate, inter_node_rate = communication_relationship(routing)
#         intra_gpu_rates.append(intra_gpu_rate)
#         inter_gpu_rates.append(inter_gpu_rate)
#         inter_node_rates.append(inter_node_rate)

    
    # plot(intra_gpu_rates, inter_gpu_rates, inter_node_rates,
    #     os.path.join(fig_dir, "communication_analysis_no_values.png"), show_values=False)
    
    # plot(intra_gpu_rates, inter_gpu_rates, inter_node_rates,
    #     os.path.join(fig_dir, "communication_analysis_with_values.png"), show_values=True)
    


if __name__ == "__main__":
    with open(f"inference_time/{model_name}/{input_name}/top{top_k}/bs{bs}/all_inference_time.json", 'r') as f:
        all_inference_time = json.load(f)
    
    all_comm_data = {}

    #routing_arrays = []
    selection_fig_dir = os.path.join(fig_dir, "expert_selection")
    os.makedirs(selection_fig_dir, exist_ok=True)
    activation_fig_dir = os.path.join(fig_dir, "expert_activation")
    os.makedirs(activation_fig_dir, exist_ok=True)
    communication_fig_dir = os.path.join(fig_dir, "communication")
    os.makedirs(communication_fig_dir, exist_ok=True)
    # jaccard_fig_dir = os.path.join(fig_dir, "jaccard")
    # os.makedirs(jaccard_fig_dir, exist_ok=True)

    for num in prompt_nums:
        routing_array = np.load(f'expert_trace/{model_name}/{input_name}/top{top_k}/bs64/{phrase_mode}_routing_trace_{num}.npy')
        inference_time = all_inference_time[str(num)]
        #routing_arrays.append(routing_array)
        
        # GPU内、GPU间、节点间通信比例、毫秒级通信次数
        comm_data = communication_rate(routing_array, inference_time)
        all_comm_data[str(num)] = comm_data

        # 层与层之间专家通信的关系
        experts_selection_stats, _ = experts_selection_inter_layer(routing_array)
        plot_experts_selection_inter_layer(experts_selection_stats, selection_fig_dir, str(num))

        # 每一层专家激活分布
        experts_activation_stats = experts_activations_count(routing_array)
        plot_expert_activations(experts_activation_stats,str(num),activation_fig_dir,True)
    
    # with open(f"{data_dir}/all_communication_data.json", 'r') as f:
    #     all_comm_data = json.load(f)

    batch_routing_data = np.load(f'expert_trace/{model_name}/{input_name}/top{top_k}/bs64/{phrase_mode}_routing_trace_64.npy')
    # selection_stats, selection_edge_set = experts_selection_inter_layer(routing_array)
    # jaccard_list, avg_jaccard, jaccard_matrix = jaccard_similarity_of_experts_selection(selection_edge_set)
    
    # all_comm_data['jaccard_similarity'] = jaccard_list
    # all_comm_data['avg_jaccard'] = avg_jaccard
    #plot_jaccard_matrix(jaccard_matrix, os.path.join(jaccard_fig_dir, "jaccard_matrix.png"))

    selection_stats, _ = experts_selection_inter_layer(batch_routing_data)
    cosine_list, avg_cosine = cosine_similarity_of_experts_selection(selection_stats)
    all_comm_data['cosine_similarity'] = cosine_list
    all_comm_data['avg_cosine'] = avg_cosine

    # GPU内、GPU间、节点间通信概率 64 和 1024各一个
    plot_comm_ratio_pie(all_comm_data['64'], 64, os.path.join(communication_fig_dir, "communication_rate_64.png"))
    plot_comm_ratio_pie(all_comm_data['1024'], 1024, os.path.join(communication_fig_dir, "communication_rate_1024.png"))
    # ms级通信次数
    plot_comm_num_per_ms(all_comm_data, os.path.join(communication_fig_dir, "num_of_communication_per_ms.png"))

    comm_data_path = os.path.join(data_dir, "all_communication_data.json")
    with open(comm_data_path, "w") as f:
        json.dump(all_comm_data, f, indent=4)


    # batch_nums = 16
    # selection_edge_sets_per_batch = []

    # for batch in range(batch_nums):
    #     routing_array_per_batch = np.load(f'expert_trace/{model_name}/{input_name}/top{top_k}/bs64/1024/{phrase_mode}_routing_trace_1024_batch{batch}.npy')
    #     stats, edge_set = experts_selection_inter_layer(routing_array_per_batch)
    #     selection_edge_sets_per_batch.append(edge_set)

    # average_jaccard_per_layer_pairs = jaccard_similarity_of_experts_selection(selection_edge_sets_per_batch)
    # print("jaccard")
    # print(average_jaccard_per_layer_pairs)
    
    
    #process_comm_rate(routing_arrays, expert_placements)

