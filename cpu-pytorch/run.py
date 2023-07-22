from dataclasses import dataclass

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

def main():
    import torch
    cache = GlobalCache(device=torch.device('cpu'), layers=[LayerCache(index=i) for i in range(26)])
    input_ids = torch.tensor(tokenizer_encode(cache, 'Once upon a time, '), dtype=torch.long)
    output_ids = decode_one_token(cache, input_ids)
    output_ids = torch.concat([
        output_ids,
        decode_one_token(cache, torch.cat([input_ids, output_ids], dim=-1))
    ], dim=-1)
    output_ids = torch.concat([
        output_ids,
        decode_one_token(cache, torch.cat([input_ids, output_ids], dim=-1))
    ], dim=-1)
    output_ids = torch.concat([
        output_ids,
        decode_one_token(cache, torch.cat([input_ids, output_ids], dim=-1))
    ], dim=-1)
    output_ids = torch.concat([
        output_ids,
        decode_one_token(cache, torch.cat([input_ids, output_ids], dim=-1))
    ], dim=-1)
    print(tokenizer_decode(cache, output_ids.tolist()))

def decode_one_token(cache: GlobalCache, input_ids):
    # input_ids is only one sequence
    # embed_tokens want a batch of sequence as input, so need to unsqueeze to add a dimension
    input_embeds = embed_tokens(cache).forward(input_ids.unsqueeze(0))

    layer0_output = decode_layer(cache, cache.layers[0], layer_input=input_embeds)
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

    output_layernormed = output_layernorm(cache, layer25_output)
    logits = lm_head(cache).forward(output_layernormed)
    next_token_logits = logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    print('next_tokens', next_tokens) # tensor([29532])
    return next_tokens

def decode_layer(cache: GlobalCache, layer: LayerCache, layer_input):
    input_layernormed = input_layernorm(cache, layer, layer_input)
    attn_output = self_attn(cache, layer, input_layernormed)
    # input_embeds is residual
    attn_output = layer_input + attn_output
    attn_output_layernormed = post_attention_layernorm(cache, layer, attn_output)
    layer_output = mlp(cache, layer, attn_output_layernormed)
    # attn_output is residual
    layer_output = attn_output + layer_output
    return layer_output

def lm_head(cache: GlobalCache):
    import torch
    if cache.lm_head is None:
        config = model_config(cache)
        cache.lm_head = torch.nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        cache.lm_head.weight = torch.nn.Parameter(load_safetensors(cache, 'lm_head.weight'))
    return cache.lm_head

def input_layernorm(cache: GlobalCache, layer: LayerCache, input_embeds):
    return rms_layernorm(cache, input_layernorm_weight(cache, layer), input_embeds)

def post_attention_layernorm(cache: GlobalCache, layer: LayerCache, input_embeds):
    return rms_layernorm(cache, post_attention_layernorm_weight(cache, layer), input_embeds)

def output_layernorm(cache: GlobalCache, output):
    return rms_layernorm(cache, output_layernorm_weight(cache), output)

def rms_layernorm(cache: GlobalCache, weight, input_embeds):
    import torch
    variance = input_embeds.to(torch.float32).pow(2).mean(-1, keepdim=True)
    input_layernormed = input_embeds * torch.rsqrt(variance + model_config(cache)['rms_norm_eps'])
    input_layernormed = (weight * input_layernormed).to(input_embeds.dtype)
    return input_layernormed

def input_layernorm_weight(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.input_layernorm_weight is None:
        weight = load_safetensors(cache, f'model.layers.{layer.index}.input_layernorm.weight')
        layer.input_layernorm_weight = nn.Parameter(weight)
    return layer.input_layernorm_weight

def post_attention_layernorm_weight(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.post_attention_layernorm_weight is None:
        weight = load_safetensors(cache, f'model.layers.{layer.index}.post_attention_layernorm.weight')
        layer.post_attention_layernorm_weight = nn.Parameter(weight)
    return layer.post_attention_layernorm_weight

def output_layernorm_weight(cache: GlobalCache):
    from torch import nn
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
    import json
    with open(f'{model_path()}/config.json') as f:
        cache.config = json.loads(f.read())
    print('config:', cache.config)
    # {'architectures': ['LlamaForCausalLM'], 'bos_token_id': 1, 'eos_token_id': 2, 'hidden_act': 'silu', 'hidden_size': 3200, 'initializer_range': 0.02, 'intermediate_size': 8640, 'max_position_embeddings': 2048, 'model_type': 'llama', 'num_attention_heads': 32, 'num_hidden_layers': 26, 'pad_token_id': 0, 'rms_norm_eps': 1e-06, 'tie_word_embeddings': False, 'torch_dtype': 'float16', 'transformers_version': '4.31.0.dev0', 'use_cache': True, 'vocab_size': 32000}
    return cache.config

def tokenizer_encode(cache: GlobalCache, input: str):
    import sentencepiece
    if cache.sp is None:
        cache.sp = sentencepiece.SentencePieceProcessor()
        cache.sp.load(f'{model_path()}/tokenizer.model')
    # [0, 9038, 2501, 263, 931, 29892, 29871]
    return [model_config(cache)['bos_token_id']] + cache.sp.encode(input)

def tokenizer_decode(cache: GlobalCache, output_ids):
    import sentencepiece
    if cache.sp is None:
        cache.sp = sentencepiece.SentencePieceProcessor()
        cache.sp.load(f'{model_path()}/tokenizer.model')
    return cache.sp.decode(output_ids)

def load_safetensors(cache: GlobalCache, key):
    import json
    from safetensors import safe_open
    if cache.safetensors_index is None:
        with open(f'{model_path()}/model.safetensors.index.json') as f:
            cache.safetensors_index = json.loads(f.read())
    safetensor_file = cache.safetensors_index['weight_map'][key]
    with safe_open(f'{model_path()}/{safetensor_file}', framework="pt") as f:
        return f.get_tensor(key)
    
def embed_tokens(cache: GlobalCache):
    import torch
    if cache.embed_tokens is None:
        config = model_config(cache)
        cache.embed_tokens = torch.nn.Embedding(config['vocab_size'], config['hidden_size'], config['pad_token_id'])
        cache.embed_tokens.weight = torch.nn.Parameter(load_safetensors(cache, 'model.embed_tokens.weight'))
    return cache.embed_tokens

def head_dim(cache: GlobalCache):
    if cache.head_dim is None:
        config = model_config(cache)
        cache.head_dim = config['hidden_size'] // config['num_attention_heads']
    return cache.head_dim

def self_attn(cache: GlobalCache, layer: LayerCache, input_layernormed):
    import torch
    from torch import nn
    import math
    bsz, q_len, _ = input_layernormed.size()
    config = model_config(cache)
    query_states = q_proj(cache, layer).forward(input_layernormed).view(bsz, q_len, config['num_attention_heads'], head_dim(cache)).transpose(1, 2)
    key_states = k_proj(cache, layer).forward(input_layernormed).view(bsz, q_len, config['num_attention_heads'], head_dim(cache)).transpose(1, 2)
    value_states = v_proj(cache, layer).forward(input_layernormed).view(bsz, q_len, config['num_attention_heads'], head_dim(cache)).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    cos, sin = rotary_emb(cache, kv_seq_len, value_states.dtype)
    position_ids = position_ids_of_seq(cache, kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim(cache))
    causal_mask = causal_mask_of_seq(cache, kv_seq_len)
    attn_weights = attn_weights + causal_mask
    attn_weights = torch.max(
        attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    )
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, config['hidden_size'])
    attn_output = o_proj(cache, layer).forward(attn_output)
    return attn_output

def q_proj(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.q_proj is None:
        config = model_config(cache)
        layer.q_proj = nn.Linear(config['hidden_size'], config['num_attention_heads'] * head_dim(cache), bias=False)
        layer.q_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.self_attn.q_proj.weight'))
    return layer.q_proj

def k_proj(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.k_proj is None:
        config = model_config(cache)
        layer.k_proj = nn.Linear(config['hidden_size'], config['num_attention_heads'] * head_dim(cache), bias=False)
        layer.k_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.self_attn.k_proj.weight'))
    return layer.k_proj

def v_proj(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.v_proj is None:
        config = model_config(cache)
        layer.v_proj = nn.Linear(config['hidden_size'], config['num_attention_heads'] * head_dim(cache), bias=False)
        layer.v_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.self_attn.v_proj.weight'))
    return layer.v_proj

def o_proj(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.o_proj is None:
        config = model_config(cache)
        layer.o_proj = nn.Linear(config['num_attention_heads'] * head_dim(cache), config['hidden_size'], bias=False)
        layer.o_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.self_attn.o_proj.weight'))
    return layer.o_proj

def inv_freq(cache: GlobalCache):
    import torch
    base=10000
    if cache.inv_freq is None:
        cache.inv_freq = 1.0 / (base ** (torch.arange(0, head_dim(cache), 2).float().to(cache.device) / head_dim(cache)))
    return cache.inv_freq

def rotary_emb(cache: GlobalCache, seq_len, dtype):
    import torch
    if cache.rotary_emb is None:
        t = torch.arange(model_config(cache)['max_position_embeddings'], device=cache.device, dtype=inv_freq(cache).dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq(cache))
        emb = torch.cat((freqs, freqs), dim=-1)
        cache.rotary_emb = (
            emb.cos()[None, None, :, :],
            emb.sin()[None, None, :, :]
        )
    cos_cached, sin_cached = cache.rotary_emb
    if seq_len > model_config(cache)['max_position_embeddings']:
        raise Exception('seq len is longer than max_position_embeddings')
    return (
        cos_cached[:, :, :seq_len, ...].to(dtype=dtype),
        sin_cached[:, :, :seq_len, ...].to(dtype=dtype),
    )

def rotate_half(x):
    import torch
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def position_ids_of_seq(cache:GlobalCache, seq_len):
    import torch
    return torch.arange(0, seq_len, dtype=torch.long, device=cache.device).unsqueeze(0).view(-1, seq_len)

def causal_mask_of_seq(cache: GlobalCache, seq_len: int):
    import torch
    bsz = 1
    device = cache.device
    causal_mask = torch.full((seq_len, seq_len), torch.tensor(torch.finfo(torch.float32).min, device=device), device=device)
    mask_cond = torch.arange(causal_mask.size(-1), device=device)
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
    causal_mask = causal_mask.to(torch.float32)
    causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    return causal_mask

def gate_proj(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.gate_proj is None:
        config = model_config(cache)
        layer.gate_proj = nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)
        layer.gate_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.mlp.gate_proj.weight'))
    return layer.gate_proj

def down_proj(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.down_proj is None:
        config = model_config(cache)
        layer.down_proj = nn.Linear(config['intermediate_size'], config['hidden_size'], bias=False)
        layer.down_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.mlp.down_proj.weight'))
    return layer.down_proj

def up_proj(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.up_proj is None:
        config = model_config(cache)
        layer.up_proj = nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)
        layer.up_proj.weight = nn.Parameter(load_safetensors(cache, f'model.layers.{layer.index}.mlp.up_proj.weight'))
    return layer.up_proj

def mlp(cache: GlobalCache, layer: LayerCache, attn_output_layernormed):
    from torch import nn
    gate_projected = gate_proj(cache, layer).forward(attn_output_layernormed)
    up_projected = up_proj(cache, layer)(attn_output_layernormed)
    return down_proj(cache, layer).forward(nn.functional.silu(gate_projected) * up_projected)


import torch
with torch.inference_mode():
    main()