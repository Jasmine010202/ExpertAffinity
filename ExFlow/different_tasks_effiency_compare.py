# 没改完

import numpy as np
import matplotlib.pyplot as plt
import os

model_name = "Switch_Transformer" #Switch_Transformer OLMoE
task_name = "math" # generate math code
input_name = "GSM8K" # sonnet mbpp GSM8K
phrase_mode = "encode" #encode  decode

use_bipart = False  #True False
prompt_nums = [8, 16, 32] # 8, 16, 32, 64, 128, 256, 512, 1024
top_k = 8 # ST:1,OL:8

num_of_nodes = 2
num_of_gpus_pre_node = 2
gpu_node_mapping = {0: 0, 1: 0, 2: 1, 3: 1}  # GPU 0和1映射到node 0；2和3映射到node 1

fig_dir = f"figs/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/top{top_k}/{phrase_mode}/method3"
os.makedirs(fig_dir, exist_ok=True)

def rate_calculate(routing, placement):
    num_intra_gpu = 0  # GPU 内通信
    num_inter_gpu = 0  # GPU 间通信(节点内)
    num_inter_node = 0  # 节点间通信
    num_total_comm =0  # 总通信次数

    if top_k == 1:  # 问题，没有考虑第一层的通信和聚合的
        num_tokens, num_layers = routing.shape
        
        # method3
        for token_id in range(num_tokens):
            token_gpu_id = np.random.randint(0, num_of_nodes*num_of_gpus_pre_node)  # 随机给当前token指定GPU
            token_node_id = gpu_node_mapping[token_gpu_id]  # GPU所属的节点

            for layer_id in range(num_layers):  
                expert_ids = routing[token_id, layer_id]
                expert_gpu_ids = placement[layer_id, expert_ids]

                map_to_node = np.vectorize(lambda gpu: gpu_node_mapping[gpu])  # 改映射，map_to_node是dict
                expert_node_ids = map_to_node(expert_gpu_ids)
                #expert_node_ids = gpu_node_mapping[expert_gpu_ids]

                num_intra_gpu += 2 * np.sum(expert_gpu_ids == token_gpu_id)  # GPU 内通信次数
                num_inter_gpu += 2 * np.sum((expert_node_ids == token_node_id) & (expert_gpu_ids != token_gpu_id))  # 同节点跨 GPU 通信
                num_inter_node += 2 * np.sum(expert_node_ids != token_node_id)  # 跨节点通信
        
        num_total_comm += num_intra_gpu + num_inter_gpu + num_inter_node  

        # #原方案
        # # 遍历每个 token 的路由路径
        # for token_id in range(num_tokens):
        #     for layer_id in range(num_layers - 1):  # 计算当前层和下一层之间的通信情况
        #         expert_curr = routing[token_id, layer_id]  # 当前层选择的专家ID
        #         expert_next = routing[token_id, layer_id + 1]  # 下一层选择的专家ID

        #         gpu_curr = placement[layer_id, expert_curr]  # 当前层专家所在 GPU
        #         gpu_next = placement[layer_id + 1, expert_next]  # 下一层专家所在 GPU

        #         num_total_comm += 1

        #         if gpu_curr == gpu_next:
        #             num_intra_gpu += 1  # GPU 内
        #         elif gpu_node_mapping[gpu_curr] == gpu_node_mapping[gpu_next]:
        #             num_inter_gpu += 1  # 跨GPU
        #         else:
        #             num_inter_node += 1  # 跨节点

    else:
        num_tokens, num_layers, num_topk = routing.shape

        # method3
        for token_id in range(num_tokens):
            token_gpu_id = np.random.randint(0, num_of_nodes*num_of_gpus_pre_node)  # 随机选择当前 token 所属 GPU
            token_node_id = gpu_node_mapping[token_gpu_id]  # 获取该 GPU 所属的节点

            for layer_id in range(num_layers):  
                expert_ids = routing[token_id, layer_id, :]
                expert_gpu_ids = placement[layer_id, expert_ids]

                map_to_node = np.vectorize(lambda gpu: gpu_node_mapping[gpu])  # 改映射，map_to_node是dict
                expert_node_ids = map_to_node(expert_gpu_ids)
                #expert_node_ids = gpu_node_mapping[expert_gpu_ids]

                num_intra_gpu += 2 * np.sum(expert_gpu_ids == token_gpu_id)  # GPU 内通信次数
                num_inter_gpu += 2 * np.sum((expert_node_ids == token_node_id) & (expert_gpu_ids != token_gpu_id))  # 同节点跨 GPU 通信
                num_inter_node += 2 * np.sum(expert_node_ids != token_node_id)  # 跨节点通信
        
        num_total_comm += num_intra_gpu + num_inter_gpu + num_inter_node  

        # # method2
        # #print(num_of_nodes*num_of_gpus_pre_node)
        # for token_id in range(num_tokens):
        #     token_gpu_id = np.random.randint(0, num_of_nodes*num_of_gpus_pre_node)  # 随机选择当前 token 所属 GPU
        #     token_node_id = gpu_node_mapping[token_gpu_id]  # 获取该 GPU 所属的节点

        #     for layer_id in range(num_layers):  

        #         for expert_rank in range(num_topk):  
        #             expert_id = routing[token_id, layer_id, expert_rank] # 专家id
        #             expert_gpu_id = placement[layer_id, expert_id] # 专家所在GPU id
        #             expert_node_id = gpu_node_mapping[expert_gpu_id]
                    
        #             #num_total_comm += 1

        #             if expert_gpu_id == token_gpu_id:
        #                 num_intra_gpu += 2  # GPU 内
        #             elif expert_node_id == token_node_id:
        #                 num_inter_gpu += 2  # 跨GPU
        #             else:
        #                 num_inter_node += 2  # 跨节点
        
        # num_total_comm += num_intra_gpu + num_inter_gpu + num_inter_node      
        # #print(f"{num_total_comm}\n{num_intra_gpu}\n{num_inter_gpu}\n{num_inter_node}")      

        # # method1
        # # 双重循环，不考虑聚合
        # for token_id in range(num_tokens):
        #     for layer_id in range(num_layers - 1):  # 计算当前层和下一层之间的通信情况
        #         experts_curr = routing[token_id, layer_id]  # 当前层选择的topk个专家ID
        #         experts_next = routing[token_id, layer_id + 1]  # 下一层选择的topk个专家ID
        #         for e_curr in experts_curr:  # 遍历当前层的 k 个专家
        #             for e_next in experts_next:  # 遍历下一层的 k 个专家
        #                 gpu_curr = placement[layer_id, e_curr]  # 当前层专家所在 GPU
        #                 gpu_next = placement[layer_id + 1, e_next]  # 下一层专家所在 GPU
        #                 num_total_comm += 1
        #                 if gpu_curr == gpu_next:
        #                     num_intra_gpu += 1  # GPU 内
        #                 elif gpu_node_mapping[gpu_curr] == gpu_node_mapping[gpu_next]:
        #                     num_inter_gpu += 1  # 跨GPU
        #                 else:
        #                     num_inter_node += 1  # 跨节点

    intra_gpu_rate = num_intra_gpu / num_total_comm *100
    inter_gpu_rate = num_inter_gpu / num_total_comm *100
    inter_node_rate = num_inter_node / num_total_comm *100

    # 输出统计结果
    print(f"总通信次数:\t{num_total_comm}")
    print(f"GPU内通信次数:\t{num_intra_gpu}\t{intra_gpu_rate:.2f}%")
    print(f"跨GPU通信次数:\t{num_inter_gpu}\t{inter_gpu_rate:.2f}%")
    print(f"跨节点通信次数:\t{num_inter_node}\t{inter_node_rate:.2f}%")

    return intra_gpu_rate, inter_gpu_rate, inter_node_rate

def plot(intra_gpu_rates, inter_gpu_rates, inter_node_rates, fig_path, show_values=False):
    
    x_positions = np.linspace(0, len(prompt_nums) - 1, len(prompt_nums))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, intra_gpu_rates, marker='o', label="Intra GPU", markersize=8)
    plt.plot(x_positions, inter_gpu_rates, marker='s', label="Inter GPU", markersize=8)
    plt.plot(x_positions, inter_node_rates, marker='^', label="Inter Node", markersize=8)

    if show_values:
        for i, (x, y) in enumerate(zip(x_positions, intra_gpu_rates)):
            plt.text(x, y, f"{y:.1f}%", ha='right', va='bottom', fontsize=10)
        for i, (x, y) in enumerate(zip(x_positions, inter_gpu_rates)):
            plt.text(x, y, f"{y:.1f}%", ha='right', va='bottom', fontsize=10)
        for i, (x, y) in enumerate(zip(x_positions, inter_node_rates)):
            plt.text(x, y, f"{y:.1f}%", ha='right', va='bottom', fontsize=10)

    plt.xlabel("Num of Prompts")
    plt.ylabel("Rate of Communication (%)")
    plt.title(f"{model_name}-top{top_k}")
    plt.legend()
    plt.grid(True)

    plt.xticks(x_positions, labels=[str(x) for x in prompt_nums], fontsize=12)
    plt.yticks(np.arange(0, 101, 10), fontsize=12)  
    
    plt.savefig(fig_path)
    plt.close()


def process_comm_rate(routing_arrays, expert_placements):
    intra_gpu_rates = []
    inter_gpu_rates = []
    inter_node_rates = []

    for routing, placement in zip(routing_arrays, expert_placements):
        intra_gpu_rate, inter_gpu_rate, inter_node_rate = rate_calculate(routing, placement)
        intra_gpu_rates.append(intra_gpu_rate)
        inter_gpu_rates.append(inter_gpu_rate)
        inter_node_rates.append(inter_node_rate)

    plot(intra_gpu_rates, inter_gpu_rates, inter_node_rates,
        os.path.join(fig_dir, "communication_analysis_no_values.png"), show_values=False)
    
    plot(intra_gpu_rates, inter_gpu_rates, inter_node_rates,
        os.path.join(fig_dir, "communication_analysis_with_values.png"), show_values=True)
    


if __name__ == "__main__":

    routing_arrays = []
    expert_placements = []

    for num in prompt_nums:
        routing_array = np.load(f'expert_trace/{model_name}/{input_name}/top{top_k}/{phrase_mode}_routing_trace_{num}.npy')
        expert_placement = np.load(f"expert_placement/{model_name}/{input_name}/{'use_bipart' if use_bipart else 'not_use_bipart'}/{phrase_mode}/top{top_k}/{num}/intra2_inter2.npy")
        # routing_array = np.load(f'expert_trace/OLMoE/test/decode_routing_trace_top8.npy')
        # expert_placement = np.load(f"expert_placement/OLMoE/test/use_bipart/decode/top8/top8_test/intra2_inter2.npy")

        routing_arrays.append(routing_array)
        expert_placements.append(expert_placement)
        
    process_comm_rate(routing_arrays, expert_placements)

    # routing_data= np.load(f"expert_trace/{model_name}/{input_name}/{phrase_mode}_routing_trace.npy")  # (num_tokens, num_layers)
    # expert_placement = np.load(f"expert_placement/{model_name}/100_input/{phrase_mode}/intra2_inter2.npy")  # (num_layers, num_experts_per_layer)
    # vanilla_placement = np.load(f"expert_placement/{model_name}/{input_name}/{phrase_mode}/intra2_inter2_vanilla.npy") #均匀放置

    # # print("routing_data:", routing_data.shape)  # (num_tokens, num_layers)
    # # print(routing_data[:15])
    # # print("expert_placement:", expert_placement.shape)  # (num_tokens, num_layers)
    # # print(expert_placement)

    # print("Vanilla Placement:")
    # rate_calculate(routing_data,vanilla_placement)

    # print("Affinity Placement:")
    # rate_calculate(routing_data,expert_placement)