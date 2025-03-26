import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

    
def plot_experts_affinity(experts_selection_stats, fig_dir, num_of_prompts):
    save_dir = os.path.join(fig_dir, num_of_prompts)
    os.makedirs(save_dir, exist_ok=True)

    num_layers, num_experts_pre_layer, _ = experts_selection_stats.shape

    for layer in range(num_layers):
        filename = os.path.join(save_dir, f"experts_affinity_of_layer{layer}.png")

        fig, ax = plt.subplots(figsize=(12,10))
        #plt.figure(figsize=(16, 14)) #10 8
        # sns.heatmap(
        #     experts_selection_stats[layer], annot=True, xticklabels=range(num_experts_pre_layer), yticklabels=range(num_experts_pre_layer), 
        #     cmap="YlGnBu", fmt=".2f"
        # )

        sns.heatmap(
            experts_selection_stats[layer], annot=False, 
            cmap="YlGnBu", linewidths=0.5, square=True, 
            cbar_kws={"shrink": 0.5}, ax=ax
        )

        ax.set_xticks(np.arange(num_experts_pre_layer))
        ax.set_xticklabels(np.arange(num_experts_pre_layer), rotation=90, fontsize=7)  # 旋转90°防止重叠
        ax.set_yticks(np.arange(num_experts_pre_layer))
        ax.set_yticklabels(np.arange(num_experts_pre_layer), fontsize=7)

        plt.title(f"Experts Affinity Intra layer — layer{layer}", fontsize=14)
        plt.xlabel(f"Experts ID", fontsize=12)
        plt.ylabel(f"Experts ID", fontsize=12)

        plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":

    model_name = "OLMoE"#Switch_Transformer OLMoE
    input_name = "sonnet"
    phrase_mode = "decode" #decode
    prompt_nums = [8] # 8, 16, 32, 64, 128, 256, 512, 1024
    top_k = 8 # ST:1,OL:8
    num_of_experts_pre_layer = 64

    fig_dir = f"affinity_figs/intra_layer/{model_name}/{input_name}/top{top_k}/{phrase_mode}/test"
    os.makedirs(fig_dir, exist_ok=True)

    for num in prompt_nums:
        routing_data = np.load(f'expert_trace/{model_name}/{input_name}/top{top_k}/{phrase_mode}_routing_trace_{num}.npy')
        
        num_tokens, num_layers, _ = routing_data.shape
        experts_selection_stats = np.zeros((num_layers, num_of_experts_pre_layer, num_of_experts_pre_layer))

        for layer in range(num_layers):
            for token in range(num_tokens):
                experts_intra_layer = routing_data[token,layer]
                for i in range(top_k):
                    for j in range (i+1, top_k):
                        expert_i, expert_j = experts_intra_layer[i], experts_intra_layer[j]
                        experts_selection_stats[layer, expert_i, expert_j] += 1
                        experts_selection_stats[layer, expert_j, expert_i] += 1 # 对称矩阵
        
        plot_experts_affinity(experts_selection_stats, fig_dir, str(num))