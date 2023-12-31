from dataclasses import dataclass
import torch
from torch import nn
import json
import sentencepiece
import safetensors
import math
import os
import sys

@dataclass
class LayerCache:
    index: int
    input_layernorm_weight: any = None
    post_attention_layernorm_weight: any = None
    q_proj: any = None
    k_proj: any = None
    v_proj: any = None
    o_proj: any = None
    gate_proj: any = None
    down_proj: any = None
    up_proj: any = None
    prefill_key_states: any = None
    prefill_value_states: any = None

@dataclass
class GlobalCache:
    device: any
    layers: list[LayerCache]
    safetensors_index = None
    embed_tokens = None
    config: any = None
    head_dim: any = None
    inv_freq: any = None
    rotary_emb: any = None
    output_layernorm_weight: any = None
    lm_head: any = None
    sp: any = None

def script_name_with_suffix():
    return os.path.basename(sys.argv[0])

def log_dir():
    script_name_without_suffix = os.path.splitext(script_name_with_suffix())[0]
    if not os.path.exists(script_name_without_suffix):
        os.mkdir(script_name_without_suffix)
    return script_name_without_suffix

already_logged = set()
mermaid_lines = []
def log_tensor(all_values, name, description, inputs):
    if name in already_logged:
        return
    already_logged.add(name)
    value = all_values[name]
    for input in inputs:
        mermaid_lines.append(f'{input}-->{name}')
    shape = ', '.join(map(str, value.shape))
    error = ''
    if len(value.shape) != len(description):
        error = '!!! shape 描述错误'
    mermaid_lines.append(f'{name}[{name}{error}<br/>{shape}<br/>{", ".join(description)}]')
    with open(f'{log_dir()}/{name}.txt', 'w') as f:
        f.write(shape)
        f.write('\n')
        f.write(str(value))

def log_mermaid_line(line):
    if line not in mermaid_lines:
        mermaid_lines.append(line)

def main():
    cache = GlobalCache(device=torch.device('cpu'), layers=[LayerCache(index=i) for i in range(26)])
    input_string = 'Once upon a time, '
    input_ids = torch.tensor(tokenizer_encode(cache, input_string), dtype=torch.long)
    log_tensor(locals(), 'input_ids', ['所有输入token的长度'], ['input_string'])
    prefill_input_ids, predict_input_ids = input_ids[:-1], input_ids[-1:]
    log_tensor(locals(), 'prefill_input_ids', ['填充缓存的token序列'], ['input_ids'])
    log_tensor(locals(), 'predict_input_ids', ['预测的token序列'], ['input_ids'])
    mermaid_lines.append('subgraph 用前6个token prefill')
    prefill_kv_cache(cache, prefill_input_ids)
    mermaid_lines.append('end')
    mermaid_lines.append('subgraph 用第7个 token predict 第8个 token')
    output_ids = last_output_ids = decode_one_token(cache, predict_input_ids)
    mermaid_lines.append('end')
    last_output_ids = decode_one_token(cache, last_output_ids)
    output_ids = torch.concat([output_ids, last_output_ids], dim=-1)
    last_output_ids = decode_one_token(cache, last_output_ids)
    output_ids = torch.concat([output_ids, last_output_ids], dim=-1)
    last_output_ids = decode_one_token(cache, last_output_ids)
    output_ids = torch.concat([output_ids, last_output_ids], dim=-1)
    last_output_ids = decode_one_token(cache, last_output_ids)
    output_ids = torch.concat([output_ids, last_output_ids], dim=-1)
    print('output_ids', output_ids)
    print(tokenizer_decode(cache, output_ids.tolist()))
    with open(f'{log_dir()}.md', 'w') as f:
        f.write('# tensor 之间的依赖关系\n')
        f.write(f'* [tensor 之间的计算算法]({script_name_with_suffix()})\n')
        f.write(f'* [tensor 的 shape 和具体样本]({log_dir()})\n\n')
        f.write('```mermaid\n')
        f.write('graph TD;\n')
        f.write(';\n'.join(mermaid_lines))
        f.write('\n```\n')

def prefill_kv_cache(cache: GlobalCache, prefill_input_ids):
    # input_ids is only one sequence
    # embed_tokens want a batch of sequence as input, so need to unsqueeze to add a dimension
    prefill_input_embeds = embed_tokens(cache, prefill_input_ids)
    log_tensor(locals(), 'prefill_input_embeds', ['填充缓存的token序列','模型 hidden size'], ['prefill_input_ids'])
    layer0_output = prefill_layer_kv_cache(cache, cache.layers[0], layer_input=prefill_input_embeds)
    layer1_output = prefill_layer_kv_cache(cache, cache.layers[1], layer_input=layer0_output)
    layer2_output = prefill_layer_kv_cache(cache, cache.layers[2], layer_input=layer1_output)
    layer3_output = prefill_layer_kv_cache(cache, cache.layers[3], layer_input=layer2_output)
    layer4_output = prefill_layer_kv_cache(cache, cache.layers[4], layer_input=layer3_output)
    layer5_output = prefill_layer_kv_cache(cache, cache.layers[5], layer_input=layer4_output)
    layer6_output = prefill_layer_kv_cache(cache, cache.layers[6], layer_input=layer5_output)
    layer7_output = prefill_layer_kv_cache(cache, cache.layers[7], layer_input=layer6_output)
    layer8_output = prefill_layer_kv_cache(cache, cache.layers[8], layer_input=layer7_output)
    layer9_output = prefill_layer_kv_cache(cache, cache.layers[9], layer_input=layer8_output)
    layer10_output = prefill_layer_kv_cache(cache, cache.layers[10], layer_input=layer9_output)
    layer11_output = prefill_layer_kv_cache(cache, cache.layers[11], layer_input=layer10_output)
    layer12_output = prefill_layer_kv_cache(cache, cache.layers[12], layer_input=layer11_output)
    layer13_output = prefill_layer_kv_cache(cache, cache.layers[13], layer_input=layer12_output)
    layer14_output = prefill_layer_kv_cache(cache, cache.layers[14], layer_input=layer13_output)
    layer15_output = prefill_layer_kv_cache(cache, cache.layers[15], layer_input=layer14_output)
    layer16_output = prefill_layer_kv_cache(cache, cache.layers[16], layer_input=layer15_output)
    layer17_output = prefill_layer_kv_cache(cache, cache.layers[17], layer_input=layer16_output)
    layer18_output = prefill_layer_kv_cache(cache, cache.layers[18], layer_input=layer17_output)
    layer19_output = prefill_layer_kv_cache(cache, cache.layers[19], layer_input=layer18_output)
    layer20_output = prefill_layer_kv_cache(cache, cache.layers[20], layer_input=layer19_output)
    layer21_output = prefill_layer_kv_cache(cache, cache.layers[21], layer_input=layer20_output)
    layer22_output = prefill_layer_kv_cache(cache, cache.layers[22], layer_input=layer21_output)
    layer23_output = prefill_layer_kv_cache(cache, cache.layers[23], layer_input=layer22_output)
    layer24_output = prefill_layer_kv_cache(cache, cache.layers[24], layer_input=layer23_output)
    prefill_layer_kv_cache(cache, cache.layers[25], layer_input=layer24_output)
    log_mermaid_line('prefill_layer_output --循环26次--> prefill_input_embeds')

def prefill_layer_kv_cache(cache: GlobalCache, layer: LayerCache, layer_input):
    prefill_input_layernormed = input_layernorm(cache, layer, layer_input)
    log_tensor(locals(), 'prefill_input_layernormed', ['填充缓存的token序列','模型 hidden size'], ['prefill_input_embeds'])
    prefill_attn_output = prefill_self_attn(cache, layer, prefill_input_layernormed) 
    # input_embeds is residual
    prefill_attn_output = layer_input + prefill_attn_output
    log_tensor(locals(), 'prefill_attn_output', 
               ['填充缓存的token序列','模型 hidden size'], 
               ['prefill_input_embeds', 'prefill_attn_tmp_output3'])
    prefill_attn_output_layernormed = post_attention_layernorm(cache, layer, prefill_attn_output)
    log_tensor(locals(), 'prefill_attn_output_layernormed', 
               ['填充缓存的token序列','模型 hidden size'], 
               ['prefill_attn_output'])
    prefill_layer_output = prefill_mlp(cache, layer, prefill_attn_output_layernormed)
    # attn_output is residual
    prefill_layer_output = prefill_attn_output + prefill_layer_output
    log_tensor(locals(), 'prefill_layer_output', 
               ['填充缓存的token序列','模型 hidden size'], 
               ['prefill_attn_output', 'prefill_attn_output_layernormed'])
    return prefill_layer_output

def prefill_self_attn(cache: GlobalCache, layer: LayerCache, input_layernormed):
    config = model_config(cache)
    q_len, _ = input_layernormed.size()
    prefill_query_states = q_proj(cache, layer, input_layernormed).view(q_len, config['num_attention_heads'], head_dim(cache)).transpose(0, 1)
    log_tensor(locals(), 'prefill_query_states', ['头的个数', '填充缓存的token序列', '每头 hidden size'], ['prefill_input_layernormed'])
    prefill_key_states = k_proj(cache, layer, input_layernormed).view(q_len, config['num_attention_heads'], head_dim(cache)).transpose(0, 1)
    log_tensor(locals(), 'prefill_key_states', ['头的个数', '填充缓存的token序列', '每头 hidden size'], ['prefill_input_layernormed'])
    layer.prefill_value_states = prefill_value_states = v_proj(cache, layer, input_layernormed).view(q_len, config['num_attention_heads'], head_dim(cache)).transpose(0, 1)
    log_tensor(locals(), 'prefill_value_states', ['头的个数', '填充缓存的token序列', '每头 hidden size'], ['prefill_input_layernormed'])
    prefill_pos_query_states = apply_rotary_pos_emb(cache, prefill_query_states, 0, q_len)
    log_tensor(locals(), 'prefill_pos_query_states', ['头的个数', '填充缓存的token序列', '每头 hidden size'], ['prefill_query_states'])
    layer.prefill_pos_key_states = prefill_pos_key_states = apply_rotary_pos_emb(cache, prefill_key_states, 0, q_len)
    log_tensor(locals(), 'prefill_pos_key_states', ['头的个数', '填充缓存的token序列', '每头 hidden size'], ['prefill_key_states'])
    prefill_unmasked_attn_weights = torch.matmul(prefill_pos_query_states, prefill_pos_key_states.transpose(1, 2)) / math.sqrt(head_dim(cache))
    log_tensor(locals(), 'prefill_unmasked_attn_weights', 
               ['头的个数', '填充缓存的token序列', '填充缓存的token序列'], 
               ['prefill_pos_query_states', 'prefill_pos_key_states'])
    prefill_causal_mask = causal_mask_of_seq(cache, q_len)
    log_tensor(locals(), 'prefill_causal_mask', 
               ['固定为1,表示对所有的头使用相同的mask', '填充缓存的token序列', '填充缓存的token序列'], [])
    prefill_masked_attn_weights = torch.max(
        prefill_unmasked_attn_weights + prefill_causal_mask, torch.tensor(torch.finfo(prefill_unmasked_attn_weights.dtype).min)
    )
    log_tensor(locals(), 'prefill_masked_attn_weights', 
               ['头的个数', '填充缓存的token序列', '填充缓存的token序列'], 
               ['prefill_unmasked_attn_weights', 'prefill_causal_mask'])
    prefill_attn_weights = nn.functional.softmax(prefill_masked_attn_weights, dim=-1)
    log_tensor(locals(), 'prefill_attn_weights', 
               ['头的个数', '填充缓存的token序列', '填充缓存的token序列'], ['prefill_masked_attn_weights'])
    prefill_attn_tmp_output1 = torch.matmul(prefill_attn_weights, prefill_value_states)
    log_tensor(locals(), 'prefill_attn_tmp_output1', 
               ['头的个数', '填充缓存的token序列',  '每头 hidden size'], 
               ['prefill_attn_weights', 'prefill_value_states'])
    prefill_attn_tmp_output2 = prefill_attn_tmp_output1.transpose(0, 1)
    log_tensor(locals(), 'prefill_attn_tmp_output2', 
               ['填充缓存的token序列', '头的个数', '每头 hidden size'], ['prefill_attn_tmp_output1'])
    prefill_attn_tmp_output3 = prefill_attn_tmp_output2.reshape(q_len, config['hidden_size'])
    log_tensor(locals(), 'prefill_attn_tmp_output3', 
               ['填充缓存的token序列', '模型 hidden size'], ['prefill_attn_tmp_output2'])
    attn_output = o_proj(cache, layer, prefill_attn_tmp_output3)
    return attn_output

def prefill_mlp(cache: GlobalCache, layer: LayerCache, attn_output_layernormed):
    gate_projected = gate_proj(cache, layer, attn_output_layernormed)
    up_projected = up_proj(cache, layer, attn_output_layernormed)
    activated = nn.functional.silu(gate_projected) * up_projected
    return down_proj(cache, layer, activated)

def decode_one_token(cache:GlobalCache, last_output_ids):
    predict_input_embeds = embed_tokens(cache, last_output_ids)
    log_tensor(locals(), 'predict_input_embeds', ['预测的token序列', '模型 hidden size'], ['predict_input_ids'])
    layer0_output = decode_layer(cache, cache.layers[0], layer_input=predict_input_embeds)
    layer1_output = decode_layer(cache, cache.layers[1], layer_input=layer0_output)
    layer2_output = decode_layer(cache, cache.layers[2], layer_input=layer1_output)
    layer3_output = decode_layer(cache, cache.layers[3], layer_input=layer2_output)
    layer4_output = decode_layer(cache, cache.layers[4], layer_input=layer3_output)
    layer5_output = decode_layer(cache, cache.layers[5], layer_input=layer4_output)
    layer6_output = decode_layer(cache, cache.layers[6], layer_input=layer5_output)
    layer7_output = decode_layer(cache, cache.layers[7], layer_input=layer6_output)
    layer8_output = decode_layer(cache, cache.layers[8], layer_input=layer7_output)
    layer9_output = decode_layer(cache, cache.layers[9], layer_input=layer8_output)
    layer10_output = decode_layer(cache, cache.layers[10], layer_input=layer9_output)
    layer11_output = decode_layer(cache, cache.layers[11], layer_input=layer10_output)
    layer12_output = decode_layer(cache, cache.layers[12], layer_input=layer11_output)
    layer13_output = decode_layer(cache, cache.layers[13], layer_input=layer12_output)
    layer14_output = decode_layer(cache, cache.layers[14], layer_input=layer13_output)
    layer15_output = decode_layer(cache, cache.layers[15], layer_input=layer14_output)
    layer16_output = decode_layer(cache, cache.layers[16], layer_input=layer15_output)
    layer17_output = decode_layer(cache, cache.layers[17], layer_input=layer16_output)
    layer18_output = decode_layer(cache, cache.layers[18], layer_input=layer17_output)
    layer19_output = decode_layer(cache, cache.layers[19], layer_input=layer18_output)
    layer20_output = decode_layer(cache, cache.layers[20], layer_input=layer19_output)
    layer21_output = decode_layer(cache, cache.layers[21], layer_input=layer20_output)
    layer22_output = decode_layer(cache, cache.layers[22], layer_input=layer21_output)
    layer23_output = decode_layer(cache, cache.layers[23], layer_input=layer22_output)
    layer24_output = decode_layer(cache, cache.layers[24], layer_input=layer23_output)
    layer25_output = decode_layer(cache, cache.layers[25], layer_input=layer24_output)
    log_mermaid_line('layer_output --循环26次--> predict_input_embeds')

    output_layernormed = output_layernorm(cache, layer25_output)
    log_tensor(locals(), 'output_layernormed', ['预测的token序列', '模型 hidden size'], ['layer_output'])
    logits = lm_head(cache, output_layernormed)
    log_tensor(locals(), 'logits', ['预测的token序列', '字典大小'], ['output_layernormed'])
    next_tokens = torch.argmax(logits, dim=-1)
    log_tensor(locals(), 'next_tokens', ['一维数组'], ['logits'])
    return next_tokens

def decode_layer(cache: GlobalCache, layer: LayerCache, layer_input):
    input_layernormed = input_layernorm(cache, layer, layer_input)
    log_tensor(locals(), 'input_layernormed', ['预测的token序列', '模型 hidden size'], ['predict_input_embeds'])
    attn_output = self_attn(cache, layer, input_layernormed) 
    # input_embeds is residual
    attn_output = layer_input + attn_output
    log_tensor(locals(), 'attn_output', ['预测的token序列', '模型 hidden size'], ['predict_input_embeds', 'attn_tmp_output3'])
    attn_output_layernormed = post_attention_layernorm(cache, layer, attn_output)
    log_tensor(locals(), 'attn_output_layernormed', ['预测的token序列', '模型 hidden size'], ['attn_output'])
    layer_output = mlp(cache, layer, attn_output_layernormed)
    # attn_output is residual
    layer_output = attn_output + layer_output
    log_tensor(locals(), 'layer_output', ['预测的token序列', '模型 hidden size'], ['attn_output', 'activated'])
    return layer_output

def self_attn(cache: GlobalCache, layer: LayerCache, input_layernormed):
    config = model_config(cache)
    q_len, _ = input_layernormed.size()
    query_states = q_proj(cache, layer, input_layernormed).view(q_len, config['num_attention_heads'], head_dim(cache)).transpose(0, 1)
    log_tensor(locals(), 'query_states', ['预测的token序列', '头的个数', '每头 hidden size'], ['input_layernormed'])
    more_key_states = k_proj(cache, layer, input_layernormed).view(q_len, config['num_attention_heads'], head_dim(cache)).transpose(0, 1)
    log_tensor(locals(), 'more_key_states', ['预测的token序列', '头的个数', '每头 hidden size'], ['input_layernormed'])
    more_value_states = v_proj(cache, layer, input_layernormed).view(q_len, config['num_attention_heads'], head_dim(cache)).transpose(0, 1)
    log_tensor(locals(), 'more_value_states', ['预测的token序列', '头的个数', '每头 hidden size'], ['input_layernormed'])
    past_kv_seq_len = layer.prefill_pos_key_states.shape[-2]
    kv_seq_len = past_kv_seq_len + q_len
    pos_query_states = apply_rotary_pos_emb(cache, query_states, past_kv_seq_len, kv_seq_len)
    log_tensor(locals(), 'pos_query_states', ['预测的token序列', '头的个数', '每头 hidden size'], ['prefill_pos_key_states', 'query_states'])
    pos_more_key_states = apply_rotary_pos_emb(cache, more_key_states, past_kv_seq_len, kv_seq_len)
    log_tensor(locals(), 'pos_more_key_states', 
               ['预测的token序列', '头的个数', '每头 hidden size'], 
               ['prefill_pos_key_states', 'more_key_states'])
    
    layer.prefill_pos_key_states = pos_key_states = torch.cat([layer.prefill_pos_key_states, pos_more_key_states], dim=1)
    log_tensor(locals(), 'pos_key_states', 
               ['全token序列', '头的个数', '每头 hidden size'], 
               ['prefill_pos_key_states', 'pos_more_key_states'])
    layer.prefill_value_states = value_states = torch.cat([layer.prefill_value_states, more_value_states], dim=1)
    log_tensor(locals(), 'value_states', 
               ['全token序列', '头的个数', '每头 hidden size'], 
               ['prefill_value_states', 'more_value_states'])

    attn_weights = torch.matmul(pos_query_states, pos_key_states.transpose(1, 2)) / math.sqrt(head_dim(cache))
    # 不需要 causal mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(pos_query_states.dtype)
    log_tensor(locals(), 'attn_weights', ['头的个数', '预测的token序列', '全token序列'], ['pos_query_states', 'pos_key_states'])
    attn_tmp_output1 = torch.matmul(attn_weights, value_states)
    log_tensor(locals(), 'attn_tmp_output1', ['头的个数', '预测的token序列', '每头 hidden size'], ['attn_weights', 'value_states'])
    attn_tmp_output2 = attn_tmp_output1.transpose(0, 1)
    log_tensor(locals(), 'attn_tmp_output2', ['预测的token序列', '头的个数', '每头 hidden size'], ['attn_tmp_output1'])
    attn_tmp_output3 = attn_tmp_output2.reshape(q_len, config['hidden_size'])
    log_tensor(locals(), 'attn_tmp_output3', ['预测的token序列', '模型 hidden size'], ['attn_tmp_output2'])
    attn_output = o_proj(cache, layer, attn_tmp_output3)
    return attn_output

def mlp(cache: GlobalCache, layer: LayerCache, attn_output_layernormed):
    gate_projected = gate_proj(cache, layer, attn_output_layernormed)
    log_tensor(locals(), 'gate_projected', ['预测的token序列', 'intermediate size'], ['attn_output_layernormed'])
    up_projected = up_proj(cache, layer, attn_output_layernormed)
    log_tensor(locals(), 'up_projected', ['预测的token序列', 'intermediate size'], ['attn_output_layernormed'])
    activated = nn.functional.silu(gate_projected) * up_projected
    log_tensor(locals(), 'activated', ['预测的token序列', 'intermediate size'], ['gate_projected', 'up_projected'])
    return down_proj(cache, layer, activated)

def lm_head(cache: GlobalCache, output_layernormed):
    if cache.lm_head is None:
        config = model_config(cache)
        cache.lm_head = torch.nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        cache.lm_head.weight = torch.nn.Parameter(load_safetensors(cache, 'lm_head.weight'))
        print('lm_head.weight.shape', cache.lm_head.weight.shape)
    return cache.lm_head.forward(output_layernormed)

def input_layernorm(cache: GlobalCache, layer: LayerCache, input_embeds):
    return rms_layernorm(cache, input_layernorm_weight(cache, layer), input_embeds)

def post_attention_layernorm(cache: GlobalCache, layer: LayerCache, input_embeds):
    return rms_layernorm(cache, post_attention_layernorm_weight(cache, layer), input_embeds)

def output_layernorm(cache: GlobalCache, output):
    return rms_layernorm(cache, output_layernorm_weight(cache), output)

def rms_layernorm(cache: GlobalCache, weight, input_embeds):
    variance = input_embeds.to(torch.float32).pow(2).mean(-1, keepdim=True)
    input_layernormed = input_embeds * torch.rsqrt(variance + model_config(cache)['rms_norm_eps'])
    input_layernormed = (weight * input_layernormed).to(input_embeds.dtype)
    return input_layernormed

def input_layernorm_weight(cache: GlobalCache, layer: LayerCache):
    if layer.input_layernorm_weight is None:
        weight = load_safetensors(cache, f'model.layers.{layer.index}.input_layernorm.weight')
        layer.input_layernorm_weight = nn.Parameter(weight)
    return layer.input_layernorm_weight

def post_attention_layernorm_weight(cache: GlobalCache, layer: LayerCache):
    if layer.post_attention_layernorm_weight is None:
        weight = load_safetensors(cache, f'model.layers.{layer.index}.post_attention_layernorm.weight')
        layer.post_attention_layernorm_weight = nn.Parameter(weight)
    return layer.post_attention_layernorm_weight

def output_layernorm_weight(cache: GlobalCache):
    if cache.output_layernorm_weight is None:
        weight = load_safetensors(cache, f'model.norm.weight')
        cache.output_layernorm_weight = nn.Parameter(weight)
    return cache.output_layernorm_weight

def model_path():
    return '../weights/open_llama_3b_v2_safetensors'

def model_config(cache: GlobalCache):
    if cache.config:
        return cache.config
    # load config.json
    with open(f'{model_path()}/config.json') as f:
        cache.config = json.loads(f.read())
    print('config', json.dumps(cache.config, indent=2))
    # {'architectures': ['LlamaForCausalLM'], 'bos_token_id': 1, 'eos_token_id': 2, 'hidden_act': 'silu', 'hidden_size': 3200, 'initializer_range': 0.02, 'intermediate_size': 8640, 'max_position_embeddings': 2048, 'model_type': 'llama', 'num_attention_heads': 32, 'num_hidden_layers': 26, 'pad_token_id': 0, 'rms_norm_eps': 1e-06, 'tie_word_embeddings': False, 'torch_dtype': 'float16', 'transformers_version': '4.31.0.dev0', 'use_cache': True, 'vocab_size': 32000}
    return cache.config

def tokenizer_encode(cache: GlobalCache, input: str):
    if cache.sp is None:
        cache.sp = sentencepiece.SentencePieceProcessor()
        cache.sp.load(f'{model_path()}/tokenizer.model')
    # [0, 9038, 2501, 263, 931, 29892, 29871]
    return [model_config(cache)['bos_token_id']] + cache.sp.encode(input)

def tokenizer_decode(cache: GlobalCache, output_ids):
    if cache.sp is None:
        cache.sp = sentencepiece.SentencePieceProcessor()
        cache.sp.load(f'{model_path()}/tokenizer.model')
    return cache.sp.decode(output_ids)

def load_safetensors(cache: GlobalCache, key):
    if cache.safetensors_index is None:
        with open(f'{model_path()}/model.safetensors.index.json') as f:
            cache.safetensors_index = json.loads(f.read())
    safetensor_file = cache.safetensors_index['weight_map'][key]
    with safetensors.safe_open(f'{model_path()}/{safetensor_file}', framework="pt") as f:
        return f.get_tensor(key)
    
def embed_tokens(cache: GlobalCache, input_ids):
    if cache.embed_tokens is None:
        config = model_config(cache)
        cache.embed_tokens = torch.nn.Embedding(
            num_embeddings=config['vocab_size'], # vocab_size=32000
            embedding_dim=config['hidden_size'], # hidden_size=3200
            padding_idx=config['pad_token_id'], # pad_token_id=0
        )
        cache.embed_tokens.weight = torch.nn.Parameter(load_safetensors(cache, 'model.embed_tokens.weight'))
    return cache.embed_tokens.forward(input_ids)

def head_dim(cache: GlobalCache):
    if cache.head_dim is None:
        config = model_config(cache)
        cache.head_dim = config['hidden_size'] // config['num_attention_heads']
        print('head_dim', cache.head_dim)
    return cache.head_dim

def q_proj(cache: GlobalCache, layer: LayerCache, input_layernormed):
    if layer.q_proj is None:
        config = model_config(cache)
        layer.q_proj = nn.Linear(in_features=config['hidden_size'], out_features=config['num_attention_heads'] * head_dim(cache), bias=False)
        layer.q_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.self_attn.q_proj.weight'))
        if layer.index == 0:
            print('q_proj.weight.shape', layer.q_proj.weight.shape) # torch.Size([3200, 32 * 100])
    return layer.q_proj.forward(input_layernormed)

def k_proj(cache: GlobalCache, layer: LayerCache, input_layernormed):
    if layer.k_proj is None:
        config = model_config(cache)
        layer.k_proj = nn.Linear(config['hidden_size'], config['num_attention_heads'] * head_dim(cache), bias=False)
        layer.k_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.self_attn.k_proj.weight'))
        if layer.index == 0:
            print('k_proj.weight.shape', layer.k_proj.weight.shape) # torch.Size([3200, 32 * 100])
    return layer.k_proj.forward(input_layernormed)

def v_proj(cache: GlobalCache, layer: LayerCache, input_layernormed):
    if layer.v_proj is None:
        config = model_config(cache)
        layer.v_proj = nn.Linear(config['hidden_size'], config['num_attention_heads'] * head_dim(cache), bias=False)
        layer.v_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.self_attn.v_proj.weight'))
        if layer.index == 0:
            print('v_proj.weight.shape', layer.v_proj.weight.shape) # torch.Size([3200, 32 * 100])
    return layer.v_proj.forward(input_layernormed)

def o_proj(cache: GlobalCache, layer: LayerCache, attn_tmp_output3):
    if layer.o_proj is None:
        config = model_config(cache)
        layer.o_proj = nn.Linear(config['num_attention_heads'] * head_dim(cache), config['hidden_size'], bias=False)
        layer.o_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.self_attn.o_proj.weight'))
    return layer.o_proj.forward(attn_tmp_output3)

def inv_freq(cache: GlobalCache):
    base=10000
    if cache.inv_freq is None:
        cache.inv_freq = 1.0 / (base ** (torch.arange(0, head_dim(cache), 2).float().to(cache.device) / head_dim(cache)))
    return cache.inv_freq

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(cache: GlobalCache, states, start, end):
    if cache.rotary_emb is None:
        t = torch.arange(model_config(cache)['max_position_embeddings'], device=cache.device, dtype=inv_freq(cache).dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq(cache))
        emb = torch.cat((freqs, freqs), dim=-1)
        cache.rotary_emb = (
            emb.cos(),
            emb.sin()
        )
    cos_cached, sin_cached = cache.rotary_emb
    position_ids = torch.arange(start, end, dtype=torch.long, device=cache.device)
    cos = cos_cached[position_ids]
    sin = sin_cached[position_ids]
    states_embed = (states * cos) + (rotate_half(states) * sin)
    return states_embed

def causal_mask_of_seq(cache: GlobalCache, seq_len: int):
    device = cache.device
    causal_mask = torch.full((seq_len, seq_len), torch.tensor(torch.finfo(torch.float32).min, device=device), device=device)
    mask_cond = torch.arange(causal_mask.size(-1), device=device)
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
    causal_mask = causal_mask.to(torch.float32)
    causal_mask = causal_mask[None, :, :].expand(1, seq_len, seq_len)
    return causal_mask

def gate_proj(cache: GlobalCache, layer: LayerCache, attn_output_layernormed):
    if layer.gate_proj is None:
        config = model_config(cache)
        layer.gate_proj = nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)
        layer.gate_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.mlp.gate_proj.weight'))
    return layer.gate_proj.forward(attn_output_layernormed)

def down_proj(cache: GlobalCache, layer: LayerCache, activation_projected):
    if layer.down_proj is None:
        config = model_config(cache)
        layer.down_proj = nn.Linear(config['intermediate_size'], config['hidden_size'], bias=False)
        layer.down_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.mlp.down_proj.weight'))
    return layer.down_proj.forward(activation_projected)

def up_proj(cache: GlobalCache, layer: LayerCache, attn_output_layernormed):
    if layer.up_proj is None:
        config = model_config(cache)
        layer.up_proj = nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)
        layer.up_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.mlp.up_proj.weight'))
    return layer.up_proj.forward(attn_output_layernormed)

with torch.inference_mode():
    main()