import numpy as np
#from solve_affinity_new import placement_plan

from solve_affinity_topk import placement_plan

###################
model_name = "OLMoE" #Switch_Transformer OLMoE
input_name = "gigaword" # gigaword
phrase_mode = "decode" #encode  decode
top_k = 8
###################
prompt_num = [8, 16, 32, 64, 128, 256, 512, 1024]

# Switch_Transformer
for num in prompt_num:
    routing_array = np.load(f'expert_trace/{model_name}/{input_name}/top{top_k}/{phrase_mode}_routing_trace_{num}.npy') # [num_tokens, num_MOE_layers], expert id for each token at each layer
    placement_plan(routing_array, model_name, input_name, phrase_mode, num, top_k)

# routing_array = np.load(f'expert_trace/{model_name}/{input_name}/{topk}/{phrase_mode}_routing_trace_top8.npy') # [num_tokens, num_MOE_layers], expert id for each token at each layer
# # #routing_array = np.load(f'expert_trace/Switch_Transformer/5_input/decode_routing_trace.npy')
# # print(routing_array)
# # print(routing_array.shape)
# placement_plan(routing_array, model_name, input_name, phrase_mode, "top8_test")
