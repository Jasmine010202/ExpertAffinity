import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json

from utils import extract_routing_trace, extract_expert_placement, extract_replicated_experts, prompt_to_gpu

# 补全多机多卡模拟

num_layers = 16
num_of_experts_per_layer = 64
gpu_node_mapping = np.array([0, 0, 1, 1]) # GPU 0和1映射到node 0；2和3映射到node 1


def calculate_num_of_token_copies(routing_trace, expert_placement):
    # 转发的token副本数
    num_intra_gpu = 0  # token 路由到自己所在 GPU 上的副本数
    num_inter_gpu = 0  # 路由到同节点其它 GPU 的副本数
    num_inter_node = 0  # 路由到不同节点上 GPU 的副本数

    #num_layers = expert_placement.shape[0]
    result = {}

    for prompt in routing_trace:
        # token 所在GPU
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        # token 所在node
        token_node_id = gpu_node_mapping[token_gpu_id]

        token_routing_traces = np.array(prompt["trace"])
        num_tokens = token_routing_traces.shape[0]

        for token_id in range(num_tokens):
            for layer_id in range(num_layers):  
                expert_ids = token_routing_traces[token_id, layer_id]
                expert_gpu_ids = expert_placement[layer_id, expert_ids]
                # expert_node_ids = gpu_node_mapping[expert_gpu_ids]
                # print(f"expert_gpu_ids:{expert_gpu_ids}")
                
                ##################################################################################
                # V1.0 每个 GPU 接收一份副本，节点不合并
                unique_target_gpu_ids = np.unique(expert_gpu_ids)  
                #print(f"unique_gpu_ids:{unique_taregt_gpu_ids}")
                
                for target_gpu_id in unique_target_gpu_ids:
                    if target_gpu_id == token_gpu_id:
                        num_intra_gpu += 1
                    elif gpu_node_mapping[target_gpu_id] == token_node_id:
                        num_inter_gpu += 1
                    else:
                        num_inter_node += 1
                ##################################################################################

                # # V2.0 合并节点
                # unique_target_gpu_ids = np.unique(expert_gpu_ids)
                # # 统计跨节点目标节点（去重）
                # target_node_ids = gpu_node_mapping[unique_target_gpu_ids]
                # unique_target_node_ids = np.unique(target_node_ids)
              
                # #print(f"unique_target_gpu_ids:{unique_target_gpu_ids}")
                # # print(f"unique_target_node_ids:{unique_target_node_ids}")

                # # 统计 token 要发往的 GPU（按 GPU 粒度判断 intra/inter_gpu）
                # for gpu_id in unique_target_gpu_ids:
                #     if gpu_id == token_gpu_id:
                #         num_intra_gpu += 1
                #     elif gpu_node_mapping[gpu_id] == token_node_id:
                #         num_inter_gpu += 1
                #     # 注意：跨节点的副本在这里不计数，避免重复计入

                # # 单独统计跨节点目标数（去重）
                # for node_id in unique_target_node_ids:
                #     if node_id != token_node_id:
                #         num_inter_node += 1
                ##################################################################################

        #     break
        # break

    print(f"GPU内token副本数:\t{num_intra_gpu}")
    print(f"跨GPUtoken副本数:\t{num_inter_gpu}")
    print(f"跨节点token副本数:\t{num_inter_node}")
 
    result["num_intra_gpu"] = num_intra_gpu
    result["num_inter_gpu"] = num_inter_gpu
    result["num_inter_node"] = num_inter_node

    return result


def duplicate_calculate_num_of_token_copies(routing_trace, expert_placement, replicated_experts):
    # 转发的token副本数
    num_intra_gpu = 0  # token 路由到自己所在 GPU 上的副本数
    num_inter_gpu = 0  # 路由到同节点其它 GPU 的副本数
    num_inter_node = 0  # 路由到不同节点上 GPU 的副本数

    #num_layers = expert_placement.shape[0]
    result = {}

    for prompt in routing_trace:
        # token 所在GPU
        token_gpu_id = prompt_to_gpu(prompt["prompt_id"])
        # token 所在node
        token_node_id = gpu_node_mapping[token_gpu_id]

        token_routing_traces = np.array(prompt["trace"])
        num_tokens = token_routing_traces.shape[0]

        for token_id in range(num_tokens):
            for layer_id in range(num_layers):  
                expert_ids = token_routing_traces[token_id, layer_id]

                # 本层复制专家
                replicated_experts_this_layer = set(replicated_experts[layer_id])
                
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
                    # expert_node_ids = gpu_node_mapping[expert_gpu_ids]
                    # print(f"expert_gpu_ids:{expert_gpu_ids}")
                    
                    ##################################################################################
                    # V1.0 每个 GPU 接收一份副本，节点不合并
                    unique_target_gpu_ids = set(np.unique(expert_gpu_ids))
                    #print(f"unique_gpu_ids:{unique_target_gpu_ids}")

                    # 如果已有intra_gpu，排除掉token所在GPU
                    if has_replicated and token_gpu_id in unique_target_gpu_ids:
                        unique_target_gpu_ids.remove(token_gpu_id)
                    #print(f"unique_gpu_ids:{unique_target_gpu_ids}")

                    for target_gpu_id in unique_target_gpu_ids:
                        if target_gpu_id == token_gpu_id:
                            num_intra_gpu += 1
                        elif gpu_node_mapping[target_gpu_id] == token_node_id:
                            num_inter_gpu += 1
                        else:
                            num_inter_node += 1
                    ##################################################################################

                    # # V2.0 合并节点
                    # unique_target_gpu_ids = np.unique(expert_gpu_ids)
                    # # 统计跨节点目标节点（去重）
                    # target_node_ids = gpu_node_mapping[unique_target_gpu_ids]
                    # unique_target_node_ids = np.unique(target_node_ids)
                
                    # #print(f"unique_target_gpu_ids:{unique_target_gpu_ids}")
                    # # print(f"unique_target_node_ids:{unique_target_node_ids}")

                    # # 统计 token 要发往的 GPU（按 GPU 粒度判断 intra/inter_gpu）
                    # for gpu_id in unique_target_gpu_ids:
                    #     if gpu_id == token_gpu_id:
                    #         num_intra_gpu += 1
                    #     elif gpu_node_mapping[gpu_id] == token_node_id:
                    #         num_inter_gpu += 1
                    #     # 注意：跨节点的副本在这里不计数，避免重复计入

                    # # 单独统计跨节点目标数（去重）
                    # for node_id in unique_target_node_ids:
                    #     if node_id != token_node_id:
                    #         num_inter_node += 1
                    ##################################################################################

        #     break
        # break

    print(f"GPU内token副本数:\t{num_intra_gpu}")
    print(f"跨GPUtoken副本数:\t{num_inter_gpu}")
    print(f"跨节点token副本数:\t{num_inter_node}")
 
    result["num_intra_gpu"] = num_intra_gpu
    result["num_inter_gpu"] = num_inter_gpu
    result["num_inter_node"] = num_inter_node

    return result


def plot_num_of_copies_compare(num_of_token_copies, placement_schemes, labels, fig_width, fig_path):
    dataset = ["sonnet", "GSM8K", "conala"]
    
    x = np.arange(len(placement_schemes)) 
    width = 0.35

    # 取3个数据集在4种放置下的跨GPU通信副本数
    inter_gpu_values = [num_of_token_copies[dataset[0]][p]["num_inter_gpu"] for p in placement_schemes]
    inter_node_values = [num_of_token_copies[dataset[0]][p]["num_inter_node"] for p in placement_schemes]

    # 颜色
    colors = ["#a6d9c7", "#7fcdbb", "#41b6c4", "#1d91c0"]

    # 创建图形
    plt.figure(figsize=(fig_width, 7))
    plt.bar(x - width/2, inter_gpu_values, width, label="Inter-GPU", color=colors[1])
    plt.bar(x + width/2, inter_node_values, width, label="Inter-Node", color=colors[2])

    max_value = max(inter_gpu_values + inter_node_values)
    plt.ylim(0, max_value * 1.1)  
    
    offset = 10000
    for i in range(len(x)):
        plt.text(x[i] - width/2, inter_gpu_values[i] + offset, str(inter_gpu_values[i]), ha='center', fontsize=9)
        plt.text(x[i] + width/2, inter_node_values[i] + offset, str(inter_node_values[i]), ha='center', fontsize=9)

    plt.xticks(x, labels)
    plt.ylabel("Num of Token Copies")
    plt.title("Comparison of Token Copies Transferred Across GPUs & Nodes")
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) 

    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def plot_num_of_copies_compare_3(num_of_token_copies, placement_schemes, labels, fig_width, fig_path):
    dataset = ["sonnet", "GSM8K", "conala"]
    
    x = np.arange(len(placement_schemes)) 
    width = 0.3

    # 取3个数据集在4种放置下的跨GPU通信副本数
    intra_gpu_copies = [num_of_token_copies[dataset[0]][p]["num_intra_gpu"] for p in placement_schemes]
    inter_gpu_copies = [num_of_token_copies[dataset[0]][p]["num_inter_gpu"] for p in placement_schemes]
    inter_node_copies = [num_of_token_copies[dataset[0]][p]["num_inter_node"] for p in placement_schemes]

    # 颜色
    colors = ["#a6d9c7", "#7fcdbb", "#41b6c4", "#1d91c0"]

    # 创建图形
    plt.figure(figsize=(fig_width, 7))
    plt.bar(x - width, intra_gpu_copies, width, label="Intra-GPU", color=colors[0])
    plt.bar(x, inter_gpu_copies, width, label="Inter-GPU", color=colors[1])
    plt.bar(x + width, inter_node_copies, width, label="Inter-Node", color=colors[2])

    max_value = max(intra_gpu_copies + inter_gpu_copies + inter_node_copies)
    plt.ylim(0, max_value * 1.1) 
    
    offset = 10000
    for i in range(len(x)):
        plt.text(x[i] - width, intra_gpu_copies[i] + offset, str(intra_gpu_copies[i]), ha='center', fontsize=9)
        plt.text(x[i], inter_gpu_copies[i] + offset, str(inter_gpu_copies[i]), ha='center', fontsize=9)
        plt.text(x[i] + width, inter_node_copies[i] + offset, str(inter_node_copies[i]), ha='center', fontsize=9)

    plt.xticks(x, labels)
    plt.ylabel("Num of Token Copies")
    plt.title("Comparison of Token Copies Transferred Across GPUs & Nodes")
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


if __name__ == "__main__":

    model_name = "OLMoE"
    # input_name = "sonnet"   
    prompt_nums = [512] # 8, 16, 32, 64, 128, 256, 512, 1024
    top_k = 8 

    fig_dir = f"Token_Copies_Compare_sim/Duplicate/sonnet/{model_name}_top{top_k}/Activation_Collaboration/figs"    # Duplicate/    Activation_Collaboration/   Activation/
    os.makedirs(fig_dir, exist_ok=True)

    result_dir = f"Token_Copies_Compare_sim/Duplicate/sonnet/{model_name}_top{top_k}/Activation_Collaboration/data" # Activation_Collaboration/   Activation/
    os.makedirs(result_dir, exist_ok=True)


    for num in prompt_nums:
        sonnet_routing_trace = extract_routing_trace(f"../Occult_test/expert_trace/traffic_test/by_prompt/{model_name}_sonnet_top{top_k}/routing_trace_{num}.jsonl")
        # GSM8K_routing_trace = extract_routing_trace(f"Occult_test/expert_trace/traffic_test/by_prompt/{model_name}_GSM8K_top{top_k}/routing_trace_{num}.jsonl")
        # conala_routing_trace = extract_routing_trace(f"Occult_test/expert_trace/traffic_test/by_prompt/{model_name}_conala_top{top_k}/routing_trace_{num}.jsonl")


        ###############################################Placement##################################################
        vanilla_placement = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/OLMoE_vanilla_placement.json")

        sonnet_occult_placement = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/occult/OLMoE_sonnet_placement_512.json")
        ## GSM8K_placement = extract_expert_placement("Occult_test/expert_placement/OLMoE_GSM8K_placement_512.json")
        ## conala_placement = extract_expert_placement("Occult_test/expert_placement/OLMoE_conala_placement_512.json")
        sonnet_occult_multi_placement = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/OLMoE_sonnet_512_nodes2_gpus4.json")
        sonnet_spectral_even_placement = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/spectral/OLMoE_sonnet_spectral_even_placement_512.json")
        sonnet_spectral_uneven_placement = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/spectral/OLMoE_sonnet_spectral_uneven_placement_512.json")
        sonnet_spectral_even_multi_placement = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/OLMoE_spectral_even_sonnet_512_nodes2_gpus4.json")
        sonnet_spectral_uneven_multi_placement = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/OLMoE_spectral_uneven_sonnet_512_nodes2_gpus4.json")

        ####duplicate
        act_replicated_experts_list = extract_replicated_experts(num_layers, "../Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_nodes2_gpus4_re4_replicated_experts.json")
        sonnet_occult_multi_repli_act = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_nodes2_gpus4_re4.json")
        sonnet_spectral_even_multi_repli_act = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_even_nodes2_gpus4_re4.json")
        sonnet_spectral_uneven_multi_repli_act = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/Duplicate/Activation/OLMoE_sonnet_512_uneven_nodes2_gpus4_re4.json")

        collab_replicated_experts_list = extract_replicated_experts(num_layers, "../Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/Duplicate/Collaboration/OLMoE_sonnet_512_nodes2_gpus4_re4_replicated_experts.json")
        sonnet_occult_multi_repli_collab = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/occult/MultiNodes_MultiGPUs/Duplicate/Collaboration/OLMoE_sonnet_512_nodes2_gpus4_re4.json")
        sonnet_spectral_even_multi_repli_collab = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/Duplicate/Collaboration/OLMoE_sonnet_512_even_nodes2_gpus4_re4.json")
        sonnet_spectral_uneven_multi_repli_collab = extract_expert_placement(num_layers, num_of_experts_per_layer, "../Occult_test/expert_placement/spectral/MultiNodes_MultiGPUs/Duplicate/Collaboration/OLMoE_sonnet_512_uneven_nodes2_gpus4_re4.json")
       
        
        ###############################################Calculate_Num_of_Token_Copies##################################################
        sonnet_vanilla_copies = calculate_num_of_token_copies(sonnet_routing_trace, vanilla_placement)
        sonnet_s_occult_copies = calculate_num_of_token_copies(sonnet_routing_trace, sonnet_occult_placement)
        ## sonnet_G_occult_copies = calculate_num_of_token_copies(sonnet_routing_trace, GSM8K_placement)
        ## sonnet_c_occult_copies = calculate_num_of_token_copies(sonnet_routing_trace, conala_placement)
        sonnet_occult_multi_copies = calculate_num_of_token_copies(sonnet_routing_trace, sonnet_occult_multi_placement)
        sonnet_spectral_even_copies = calculate_num_of_token_copies(sonnet_routing_trace, sonnet_spectral_even_placement)
        sonnet_spectral_uneven_copies = calculate_num_of_token_copies(sonnet_routing_trace, sonnet_spectral_uneven_placement)
        sonnet_spectral_even_multi_copies = calculate_num_of_token_copies(sonnet_routing_trace, sonnet_spectral_even_multi_placement)
        sonnet_spectral_uneven_multi_copies = calculate_num_of_token_copies(sonnet_routing_trace, sonnet_spectral_uneven_multi_placement)

        ####duplicate
        # Activation
        sonnet_occult_multi_repli_act_copies = duplicate_calculate_num_of_token_copies(sonnet_routing_trace, sonnet_occult_multi_repli_act, act_replicated_experts_list)
        sonnet_spectral_even_multi_repli_act_copies = duplicate_calculate_num_of_token_copies(sonnet_routing_trace, sonnet_spectral_even_multi_repli_act, act_replicated_experts_list)
        sonnet_spectral_uneven_multi_repli_act_copies = duplicate_calculate_num_of_token_copies(sonnet_routing_trace, sonnet_spectral_uneven_multi_repli_act, act_replicated_experts_list)
        
        # Collaboration
        sonnet_occult_multi_repli_collab_copies = duplicate_calculate_num_of_token_copies(sonnet_routing_trace, sonnet_occult_multi_repli_collab, collab_replicated_experts_list)
        sonnet_spectral_even_multi_repli_collab_copies = duplicate_calculate_num_of_token_copies(sonnet_routing_trace, sonnet_spectral_even_multi_repli_collab, collab_replicated_experts_list)
        sonnet_spectral_uneven_multi_repli_collab_copies = duplicate_calculate_num_of_token_copies(sonnet_routing_trace, sonnet_spectral_uneven_multi_repli_collab, collab_replicated_experts_list)
   

        # GSM8K_vanilla_copies = calculate_num_of_token_copies(GSM8K_routing_trace, vanilla_placement)
        # GSM8K_s_occult_copies = calculate_num_of_token_copies(GSM8K_routing_trace, sonnet_placement)
        # GSM8K_G_occult_copies = calculate_num_of_token_copies(GSM8K_routing_trace, GSM8K_placement)
        # GSM8K_c_occult_copies = calculate_num_of_token_copies(GSM8K_routing_trace, conala_placement)

        # conala_vanilla_copies = calculate_num_of_token_copies(conala_routing_trace, vanilla_placement)
        # conala_s_occult_copies = calculate_num_of_token_copies(conala_routing_trace, sonnet_placement)
        # conala_G_occult_copies = calculate_num_of_token_copies(conala_routing_trace, GSM8K_placement)
        # conala_c_occult_copies = calculate_num_of_token_copies(conala_routing_trace, conala_placement)


        ###############################################File##################################################
        num_of_token_copies = {}
        num_of_token_copies["sonnet"] = {}
        num_of_token_copies["sonnet"]["vanilla_placement"] = sonnet_vanilla_copies
        num_of_token_copies["sonnet"]["sonnet_occult_placement"] = sonnet_s_occult_copies
        num_of_token_copies["sonnet"]["sonnet_occult_multi_placement"] = sonnet_occult_multi_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_even_placement"] = sonnet_spectral_even_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_uneven_placement"] = sonnet_spectral_uneven_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_even_multi_placement"] = sonnet_spectral_even_multi_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_uneven_multi_placement"] = sonnet_spectral_uneven_multi_copies

        ####duplicate
        num_of_token_copies["sonnet"]["sonnet_occult_multi_repli_act_placement"] = sonnet_occult_multi_repli_act_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_even_multi_repli_act_placement"] = sonnet_spectral_even_multi_repli_act_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_uneven_multi_repli_act_placement"] = sonnet_spectral_uneven_multi_repli_act_copies

        num_of_token_copies["sonnet"]["sonnet_occult_multi_repli_collab_placement"] = sonnet_occult_multi_repli_collab_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_even_multi_repli_collab_placement"] = sonnet_spectral_even_multi_repli_collab_copies
        num_of_token_copies["sonnet"]["sonnet_spectral_uneven_multi_repli_collab_placement"] = sonnet_spectral_uneven_multi_repli_collab_copies

        
        # num_of_token_copies["sonnet"]["GSM8K_occult_placement"] = sonnet_G_occult_copies
        # num_of_token_copies["sonnet"]["conala_occult_placement"] = sonnet_c_occult_copies

        # num_of_token_copies["GSM8K"] = {}
        # num_of_token_copies["GSM8K"]["vanilla_placement"] = GSM8K_vanilla_copies
        # num_of_token_copies["GSM8K"]["sonnet_occult_placement"] = GSM8K_s_occult_copies
        # num_of_token_copies["GSM8K"]["GSM8K_occult_placement"] = GSM8K_G_occult_copies
        # num_of_token_copies["GSM8K"]["conala_occult_placement"] = GSM8K_c_occult_copies

        # num_of_token_copies["conala"] = {}
        # num_of_token_copies["conala"]["vanilla_placement"] = conala_vanilla_copies
        # num_of_token_copies["conala"]["sonnet_occult_placement"] = conala_s_occult_copies
        # num_of_token_copies["conala"]["GSM8K_occult_placement"] = conala_G_occult_copies
        # num_of_token_copies["conala"]["conala_occult_placement"] = conala_c_occult_copies

        filename = os.path.join(result_dir, f"num_of_token_copies_duplicate_multi_activation_collaboration.json")
        with open (filename, "w") as f:
            json.dump(num_of_token_copies ,f,indent=2)

         
        # with open(f"Token_Copies_Compare_sim/Duplicate/sonnet/OLMoE_top8/Activation_Collaboration/data/num_of_token_copies_duplicate_multi_original_activation_collaboration.json",'r') as f:
        #     num_of_token_copies = json.load(f)


        placement_schemes = ["vanilla_placement", 
                             "sonnet_occult_placement",
                             "sonnet_occult_multi_placement", 
                             "sonnet_occult_multi_repli_act_placement",
                             "sonnet_occult_multi_repli_collab_placement",
                             "sonnet_spectral_even_placement",
                             "sonnet_spectral_even_multi_placement", 
                             "sonnet_spectral_even_multi_repli_act_placement", 
                             "sonnet_spectral_even_multi_repli_collab_placement",
                             "sonnet_spectral_uneven_placement",
                             "sonnet_spectral_uneven_multi_placement", 
                             "sonnet_spectral_uneven_multi_repli_act_placement",
                             "sonnet_spectral_uneven_multi_repli_collab_placement"
                             ]
        labels = ["Vanilla", 
                  "Occult",
                  "Occult_Multi", 
                  "Occult_Multi_Repli_Act", 
                  "Occult_Multi_Repli_Collab", 
                  "Spectral_Even",
                  "Spectral_Even_Multi", 
                  "Spectral_Even_Multi_Repli_Act", 
                  "Spectral_Even_Multi_Repli_Collab", 
                  "Spectral_Uneven",
                  "Spectral_Uneven_Multi", 
                  "Spectral_Uneven_Multi_Repli_Act",
                  "Spectral_Uneven_Multi_Repli_Collab"
                  ]

        fig_path = os.path.join(fig_dir,f"num_of_token_copies_compare_duplicate_multi_original_activation_collaboration_2.svg")
        plot_num_of_copies_compare(num_of_token_copies, placement_schemes, labels, 36, fig_path)    # 3-10  4-12   5-14   7-20 16   10-24   13-36

        fig_3_path = os.path.join(fig_dir,f"num_of_token_copies_compare_duplicate_multi_original_activation_collaboration_3.svg")
        plot_num_of_copies_compare_3(num_of_token_copies, placement_schemes, labels, 38, fig_3_path)    # 3-12  4-14   5-16   7-22 18  10-26   13-38
       