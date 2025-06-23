import json
import os

num_of_layers=16
num_of_expert_per_layer=64
num_of_gpu_per_node = 2
num_of_node = 2

overall_gpus = num_of_gpu_per_node * num_of_node
experts_per_gpu = num_of_expert_per_layer / overall_gpus

vanilla_placement={}

for layer_id in range(num_of_layers):
    placement_per_layer = [[] for gpu in range(overall_gpus)]

    for expert_id in range(num_of_expert_per_layer):
        gpu_id = int(expert_id // experts_per_gpu)
        placement_per_layer[gpu_id].append(expert_id)

    vanilla_placement[str(layer_id)] = placement_per_layer

placement_dir = f"Occult_test/expert_placement"
os.makedirs(placement_dir, exist_ok=True)

placement_file_path = os.path.join(placement_dir, f"OLMoE_vanilla_placement.json")
with  open(placement_file_path,"w") as f:
    json.dump(vanilla_placement, f, indent=2)
