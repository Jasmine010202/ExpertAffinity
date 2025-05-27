import numpy as np
from solve_affinity_new import placement_plan

# from solve_affinity_topk import placement_plan

###################
model_name = "Switch_Transformer" #Switch_Transformer OLMoE
task_name = "generate" # generate math code
input_name = "sonnet" # sonnet GSM8K mbpp 
phrase_mode = "decode" #encode  decode
# phrase_modes = ["encode","decode"] #encode  decode#
#top_k = 1
###################
prompt_nums = [512]  # 8, 16, 32, 64, 128, 256, 512, 1024

# Switch_Transformer
# for phrase_mode in phrase_modes:
#     for num in prompt_nums:
#         routing_array = np.load(f'different_tasks/trace_by_prompt/expert_trace/{model_name}/{task_name}/{input_name}/{phrase_mode}_routing_trace_{num}.npy') # [num_tokens, num_MOE_layers], expert id for each token at each layer
#         #placement_plan(routing_array, model_name, input_name, phrase_mode, num, top_k)
#         placement_plan(routing_array, model_name, input_name, phrase_mode, num)

for num in prompt_nums:
    routing_array = np.load(f'different_tasks/trace_by_prompt/expert_trace/{model_name}/{task_name}/{input_name}/{phrase_mode}_routing_trace_{num}_front.npy') # [num_tokens, num_MOE_layers], expert id for each token at each layer
    #placement_plan(routing_array, model_name, input_name, phrase_mode, num, top_k)
    placement_plan(routing_array, model_name, input_name, phrase_mode, num)

# routing_array = np.load(f'expert_trace/{model_name}/{input_name}/{topk}/{phrase_mode}_routing_trace_top8.npy') # [num_tokens, num_MOE_layers], expert id for each token at each layer
# # #routing_array = np.load(f'expert_trace/Switch_Transformer/5_input/decode_routing_trace.npy')
# # print(routing_array)
# # print(routing_array.shape)
# placement_plan(routing_array, model_name, input_name, phrase_mode, "top8_test")
