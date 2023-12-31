# tensor 之间的依赖关系
* [tensor 之间的计算算法](run-without-batch.py)
* [tensor 的 shape 和具体样本](run-without-batch)

```mermaid
graph TD;
input_string-->input_ids;
input_ids[input_ids<br/>7<br/>token序列的长度];
embed_tokens_weight[embed_tokens_weight<br/>32000, 3200<br/>字典大小, 模型 hidden size];
input_ids-->input_embeds;
embed_tokens_weight-->input_embeds;
input_embeds[input_embeds<br/>7, 3200<br/>token序列的长度, 模型 hidden size];
input_embeds-->input_layernormed;
input_layernormed[input_layernormed<br/>7, 3200<br/>token序列的长度, 模型 hidden size];
q_proj_weight[q_proj_weight<br/>3200, 3200<br/>模型 hidden size, 头个数 * 每头 hidden size];
input_layernormed-->query_states;
q_proj_weight-->query_states;
query_states[query_states<br/>32, 7, 100<br/>头的个数, token序列的长度, 每头 hidden size];
k_proj_weight[k_proj_weight<br/>3200, 3200<br/>模型 hidden size, 头个数 * 每头 hidden size];
input_layernormed-->key_states;
k_proj_weight-->key_states;
key_states[key_states<br/>32, 7, 100<br/>头的个数, token序列的长度, 每头 hidden size];
v_proj_weight[v_proj_weight<br/>3200, 3200<br/>模型 hidden size, 头个数 * 每头 hidden size];
input_layernormed-->value_states;
v_proj_weight-->value_states;
value_states[value_states<br/>32, 7, 100<br/>头的个数, token序列的长度, 每头 hidden size];
cos_cached[cos_cached<br/>2048, 100<br/>最大的序列长度, cos 表];
sin_cached[sin_cached<br/>2048, 100<br/>最大的序列长度, sin 表];
position_ids[position_ids!!! shape 描述错误<br/>7<br/>token序列的长度, 位置递增序号];
query_states-->pos_query_states;
cos_cached-->pos_query_states;
sin_cached-->pos_query_states;
position_ids-->pos_query_states;
pos_query_states[pos_query_states<br/>32, 7, 100<br/>头的个数, token序列的长度, 每头 hidden size];
key_states-->pos_key_states;
cos_cached-->pos_key_states;
sin_cached-->pos_key_states;
position_ids-->pos_key_states;
pos_key_states[pos_key_states<br/>32, 7, 100<br/>头的个数, token序列的长度, 每头 hidden size];
pos_query_states-->unmasked_attn_weights;
pos_key_states-->unmasked_attn_weights;
unmasked_attn_weights[unmasked_attn_weights<br/>32, 7, 7<br/>头的个数, token序列的长度, token序列的长度];
causal_mask[causal_mask<br/>1, 7, 7<br/>固定为1,表示对所有的头使用相同的mask, token序列的长度, token序列的长度];
unmasked_attn_weights-->masked_attn_weights;
causal_mask-->masked_attn_weights;
masked_attn_weights[masked_attn_weights<br/>32, 7, 7<br/>头的个数, token序列的长度, token序列的长度];
masked_attn_weights-->attn_weights;
attn_weights[attn_weights<br/>32, 7, 7<br/>头的个数, token序列的长度, token序列的长度];
attn_weights-->attn_tmp_output1;
value_states-->attn_tmp_output1;
attn_tmp_output1[attn_tmp_output1<br/>32, 7, 100<br/>头的个数, token序列的长度, 每头 hidden size];
attn_tmp_output1-->attn_tmp_output2;
attn_tmp_output2[attn_tmp_output2<br/>7, 32, 100<br/>token序列的长度, 头的个数, 每头 hidden size];
attn_tmp_output2-->attn_tmp_output3;
attn_tmp_output3[attn_tmp_output3<br/>7, 3200<br/>token序列的长度, 模型 hidden size];
o_proj_weight[o_proj_weight<br/>3200, 3200<br/>头个数 * 每头 hidden size, 模型 hidden size];
input_layernormed-->attn_output;
attn_tmp_output3-->attn_output;
o_proj_weight-->attn_output;
attn_output[attn_output<br/>7, 3200<br/>token序列的长度, 模型 hidden size];
attn_output-->attn_output_layernormed;
attn_output_layernormed[attn_output_layernormed<br/>7, 3200<br/>token序列的长度, 模型 hidden size];
gate_proj_weight[gate_proj_weight<br/>8640, 3200<br/>intermediate size, 模型 hidden size];
gate_proj_weight-->gate_projected;
attn_output_layernormed-->gate_projected;
gate_projected[gate_projected<br/>7, 8640<br/>token序列的长度, intermediate size];
up_proj_weight[up_proj_weight<br/>8640, 3200<br/>intermediate size, 模型 hidden size];
up_proj_weight-->up_projected;
attn_output_layernormed-->up_projected;
up_projected[up_projected<br/>7, 8640<br/>token序列的长度, intermediate size];
gate_projected-->activated;
up_projected-->activated;
activated[activated<br/>7, 8640<br/>token序列的长度, intermediate size];
down_proj_weight[down_proj_weight<br/>3200, 8640<br/>模型 hidden size, intermediate size];
attn_output-->layer_output;
activated-->layer_output;
down_proj_weight-->layer_output;
layer_output[layer_output<br/>7, 3200<br/>token序列的长度, 模型 hidden size];
layer_output --循环26次--> input_embeds;
layer_output-->output_layernormed;
output_layernormed[output_layernormed<br/>7, 3200<br/>token序列的长度, 模型 hidden size];
lm_head_weight[lm_head_weight<br/>32000, 3200<br/>模型 hidden size, 字典大小];
output_layernormed-->logits;
lm_head_weight-->logits;
logits[logits<br/>7, 32000<br/>token序列的长度, 字典大小];
logits-->last_logit;
last_logit[last_logit<br/>32000<br/>字典大小];
last_logit-->next_tokens;
next_tokens[next_tokens<br/>1<br/>一维数组]
```
