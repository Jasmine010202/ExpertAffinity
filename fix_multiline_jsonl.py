import json
import os

dataset_name = "GSM8K"
input_file = f"Occult_test/expert_trace/traffic_test/by_prompt/OLMoE_{dataset_name}_top8/routing_trace_512.jsonl"   # 源文件路径（就是你保存时带 indent 的）
output_dir = f"Occult_test/expert_trace/traffic_test/by_prompt/fixed/OLMoE_{dataset_name}_top8"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir,"routing_trace_512.jsonl")  # 修复后输出文件路径

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    buffer = ""
    for line in infile:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        buffer += line
        try:
            obj = json.loads(buffer)  # 尝试是否组成了完整 JSON
            json.dump(obj, outfile)
            outfile.write("\n")
            buffer = ""  # 清空 buffer
        except json.JSONDecodeError:
            continue  # 还没拼好，继续读取下一行
