import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comm_latency_compare(fig_path):
   # === 加载 JSON 数据 ===
    with open("traffic/Switch_Transformer/generate_sonnet/non_batch/data/traffic_512_vanilla.json", "r") as f:
        vanilla_traffic = json.load(f)
    with open("traffic/Switch_Transformer/generate_sonnet/non_batch/data/traffic_512_exflow.json", "r") as f:
        exflow_traffic = json.load(f)
    with open("latency/Switch_Transformer/generate_sonnet/data/latency_communication_512_vanilla.json", "r") as f:
        vanilla_lat = json.load(f)
    with open("latency/Switch_Transformer/generate_sonnet/data/latency_communication_512_exflow.json", "r") as f:
        exflow_lat = json.load(f)
        

    # === 提取通信量（MB）和通信时延（ms） ===
    labels = ["Inter-GPU", "Inter-Node"]
    vanilla_comm = [
        vanilla_traffic["traffic_inter_gpu_MB"],
        vanilla_traffic["traffic_inter_node_MB"]
    ]
    exflow_comm = [
        exflow_traffic["traffic_inter_gpu_MB"],
        exflow_traffic["traffic_inter_node_MB"]
    ]
    vanilla_latency = [
        vanilla_lat["total_inter_gpu_latency_ms"],
        vanilla_lat["total_inter_node_latency_ms"]
    ]
    exflow_latency = [
        exflow_lat["total_inter_gpu_latency_ms"],
        exflow_lat["total_inter_node_latency_ms"]
    ]

    # === 作图 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharex=False)
    width = 0.25
    color_vanilla = "#7fcdbb"
    color_exflow = "#1d91c0"
    x = np.arange(len(labels))  # [0, 1]

    # --- 通信量图 ---
    ax1.bar(x - width / 2, vanilla_comm, width, label="Vanilla", color=color_vanilla)
    ax1.bar(x + width / 2, exflow_comm, width, label="Optimized", color=color_exflow)
    ax1.set_ylabel("Communication Traffic (MB)")
    ax1.set_title("Communication Traffic ")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    for i, val in enumerate(vanilla_comm):
        ax1.text(x[i] - width / 2, val + 1, f"{val:.1f}", ha='center', va='bottom', fontsize=8)
    for i, val in enumerate(exflow_comm):
        ax1.text(x[i] + width / 2, val + 1, f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    # --- 通信时延图 ---
    ax2.bar(x - width / 2, vanilla_latency, width, label="Vanilla", color=color_vanilla)
    ax2.bar(x + width / 2, exflow_latency, width, label="Optimized", color=color_exflow)
    ax2.set_ylabel("Communication Latency (ms)")
    ax2.set_title("Communication Latency ")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    for i, val in enumerate(vanilla_latency):
        ax2.text(x[i] - width / 2, val + 10, f"{val:.1f}", ha='center', va='bottom', fontsize=8)
    for i, val in enumerate(exflow_latency):
        ax2.text(x[i] + width / 2, val + 10, f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    ax1.legend(loc="upper left")
    fig.suptitle("Traffic & Latency Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_path)
    plt.close()

def plot_comm_latency_compare_2(fig_path):
   # === 加载 JSON 数据 ===
    with open("traffic/Switch_Transformer/generate_sonnet/non_batch/fake/data/traffic_512_vanilla.json", "r") as f:
        vanilla_traffic = json.load(f)
    with open("traffic/Switch_Transformer/generate_sonnet/non_batch/fake/data/traffic_512_exflow.json", "r") as f:
        exflow_traffic = json.load(f)
    with open("latency/Switch_Transformer/generate_sonnet/data/latency_communication_512_vanilla.json", "r") as f:
        vanilla_lat = json.load(f)
    with open("latency/Switch_Transformer/generate_sonnet/data/latency_communication_512_exflow.json", "r") as f:
        exflow_lat = json.load(f)
        

    # === 提取通信量（MB）和通信时延（ms） ===
    labels_traffic = ["Inter-GPU", "Inter-Node"]
    vanilla_comm = [
        vanilla_traffic["traffic_inter_gpu_MB"],
        vanilla_traffic["traffic_inter_node_MB"]
    ]
    exflow_comm = [
        exflow_traffic["traffic_inter_gpu_MB"],
        exflow_traffic["traffic_inter_node_MB"]
    ]

    labels_latency = ["Communication Latency"]
    vanilla_latency = [
        vanilla_lat["communication_latency_ms"]
    ]
    exflow_latency = [
        exflow_lat["communication_latency_ms"]
    ]

    # 乘以 1.5
    exflow_comm = [v * 1.5 for v in exflow_comm]
    exflow_latency = [v * 1.5 for v in exflow_latency]


    # === 作图 ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharex=False)
    width = 0.25
    color_vanilla = "#7fcdbb"
    color_exflow = "#1d91c0"
    x_traffic = np.arange(len(labels_traffic))  # [0, 1]

    x_latency = np.arange(len(labels_latency))  # 其实就是 [0]

    # --- 通信量图 ---
    ax1.bar(x_traffic - width / 2, vanilla_comm, width, label="Vanilla", color=color_vanilla)
    ax1.bar(x_traffic + width / 2, exflow_comm, width, label="Optimized", color=color_exflow)
    #ax1.set_ylabel("Traffic (MB)")
    #ax1.set_title("Communication Traffic ")
    ax1.set_xticks(x_traffic)
    ax1.set_xticklabels(labels_traffic)
    #ax1.grid(axis='y', linestyle='--', alpha=0.4)

    for i, val in enumerate(vanilla_comm):
        ax1.text(x_traffic[i] - width / 2, val + 1, f"{val:.1f}", ha='center', va='bottom', fontsize=8)
    for i, val in enumerate(exflow_comm):
        ax1.text(x_traffic[i] + width / 2, val + 1, f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    # --- 通信时延图 ---
    ax2.bar(x_latency - width / 2, vanilla_latency, width, label="Vanilla", color=color_vanilla)
    ax2.bar(x_latency + width / 2, exflow_latency, width, label="Optimized", color=color_exflow)
    #ax2.set_ylabel("Latency (ms)")
    #ax2.set_title("Overall Communication Latency ")
    ax2.set_xticks(x_latency)
    ax2.set_xticklabels(labels_latency)
    #ax2.grid(axis='y', linestyle='--', alpha=0.4)

    ax2.set_xlim(-0.8, 0.8)  #收紧柱子

    for i, val in enumerate(vanilla_latency):
        ax2.text(x_latency[i] - width / 2, val + 10, f"{val:.1f}", ha='center', va='bottom', fontsize=8)
    for i, val in enumerate(exflow_latency):
        ax2.text(x_latency[i] + width / 2, val + 10, f"{val:.1f}", ha='center', va='bottom', fontsize=8)

    #ax1.legend(loc="upper left")
    #fig.suptitle("Traffic & Latency Comparison", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_path)
    plt.close()


if __name__ == "__main__":
    fig_dir = f"latency/figs/fake"
    os.makedirs(fig_dir, exist_ok=True)

    fig_path = os.path.join(fig_dir,"traffic_latency_compare_512_4.svg")
    plot_comm_latency_compare_2(fig_path)