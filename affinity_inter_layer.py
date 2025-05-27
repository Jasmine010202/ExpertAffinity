import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

    
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





if __name__ == "__main__":

    model_name = "OLMoE"#Switch_Transformer OLMoE
    input_name = "sonnet"
    phrase_mode = "decode" #decode
    prompt_nums = [512] # 8, 16, 32, 64, 128, 256, 512, 1024
    top_k = 8 # ST:1,OL:8
    num_of_experts_per_layer = 64

    result_dir = f"latency_traffic_load/inter_layer_selection/{model_name}/{input_name}/top{top_k}/{phrase_mode}/data"
    os.makedirs(result_dir, exist_ok=True)

    fig_dir = f"latency_traffic_load/inter_layer_selection/{model_name}/{input_name}/top{top_k}/{phrase_mode}/figs"
    os.makedirs(fig_dir, exist_ok=True)

    for num in prompt_nums:
        routing_data = np.load(f'expert_trace/{model_name}/{input_name}/top{top_k}/{phrase_mode}_routing_trace_{num}.npy')
        
        num_tokens, num_layers, top_k = routing_data.shape
        experts_selection_stats = np.zeros((num_layers - 1, num_of_experts_per_layer, num_of_experts_per_layer)) # 记录层间的专家选择关系

        for layer in range(num_layers-1):
            for token in range(num_tokens):
                experts_curr_layer = routing_data[token,layer]      #当前层专家选择
                experts_next_layer = routing_data[token,layer+1]    #下一层专家选择
                for i in range(top_k):
                    for j in range (top_k):
                        expert_i, expert_j = experts_curr_layer[i], experts_next_layer[j]
                        experts_selection_stats[layer, expert_i, expert_j] += 1
                        
        plot_experts_selection_inter_layer(experts_selection_stats,fig_dir, str(num))
        np.save(f"{result_dir}/{model_name}_Inter_Layer_Selection_{input_name}_{num}.npy", experts_selection_stats)


