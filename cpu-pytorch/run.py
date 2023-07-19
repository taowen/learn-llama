from dataclasses import dataclass

@dataclass
class GlobalCache:
    device: any
    safetensors_index = None
    embed_tokens = None
    config: any = None
    head_dim: any = None
    inv_freq: any = None
    rotary_emb: any = None

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

def main():
    import torch
    cache = GlobalCache(device=torch.device('cpu'))
    layer0 = LayerCache(index=0)
    input_ids = torch.tensor(tokenize(cache, 'Once upon a time, '), dtype=torch.long).unsqueeze(0)
    input_embeds = embed_tokens(cache).forward(input_ids)
    layer_input = input_embeds
    layer_input = decode_layer(cache, layer0, layer_input=layer_input)
    layer1 = LayerCache(index=1)
    layer_input = decode_layer(cache, layer1, layer_input=layer_input)
    layer2 = LayerCache(index=2)
    layer_input = decode_layer(cache, layer2, layer_input=layer_input)
    layer3 = LayerCache(index=3)
    layer_input = decode_layer(cache, layer3, layer_input=layer_input)
    layer4 = LayerCache(index=4)
    layer_input = decode_layer(cache, layer4, layer_input=layer_input)
    layer5 = LayerCache(index=5)
    layer_input = decode_layer(cache, layer5, layer_input=layer_input)
    layer6 = LayerCache(index=6)
    layer_input = decode_layer(cache, layer6, layer_input=layer_input)
    layer7 = LayerCache(index=7)
    layer_input = decode_layer(cache, layer7, layer_input=layer_input)
    layer8 = LayerCache(index=8)
    layer_input = decode_layer(cache, layer8, layer_input=layer_input)
    layer9 = LayerCache(index=9)
    layer_input = decode_layer(cache, layer9, layer_input=layer_input)
    layer10 = LayerCache(index=10)
    layer_input = decode_layer(cache, layer10, layer_input=layer_input)
    layer11 = LayerCache(index=11)
    layer_input = decode_layer(cache, layer11, layer_input=layer_input)
    layer12 = LayerCache(index=12)
    layer_input = decode_layer(cache, layer12, layer_input=layer_input)
    layer13 = LayerCache(index=13)
    layer_input = decode_layer(cache, layer13, layer_input=layer_input)
    layer14 = LayerCache(index=14)
    layer_input = decode_layer(cache, layer14, layer_input=layer_input)
    layer15 = LayerCache(index=15)
    layer_input = decode_layer(cache, layer15, layer_input=layer_input)  
    layer16 = LayerCache(index=16)
    layer_input = decode_layer(cache, layer16, layer_input=layer_input)
    layer17 = LayerCache(index=17)
    layer_input = decode_layer(cache, layer17, layer_input=layer_input)
    layer18 = LayerCache(index=18)
    layer_input = decode_layer(cache, layer18, layer_input=layer_input)
    layer19 = LayerCache(index=19)
    layer_input = decode_layer(cache, layer19, layer_input=layer_input)
    layer20 = LayerCache(index=20)
    layer_input = decode_layer(cache, layer20, layer_input=layer_input)
    layer21 = LayerCache(index=21)  
    layer_input = decode_layer(cache, layer21, layer_input=layer_input)  
    layer22 = LayerCache(index=22)
    layer_input = decode_layer(cache, layer22, layer_input=layer_input)
    layer23 = LayerCache(index=23)
    layer_input = decode_layer(cache, layer23, layer_input=layer_input)
    layer24 = LayerCache(index=24)
    layer_input = decode_layer(cache, layer24, layer_input=layer_input)
    layer25 = LayerCache(index=25)
    layer_input = decode_layer(cache, layer25, layer_input=layer_input)


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

def input_layernorm(cache: GlobalCache, layer: LayerCache, input_embeds):
    return rms_layernorm(cache, input_layernorm_weight(cache, layer), input_embeds)

def post_attention_layernorm(cache: GlobalCache, layer: LayerCache, input_embeds):
    return rms_layernorm(cache, post_attention_layernorm_weight(cache, layer), input_embeds)

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

def tokenize(cache: GlobalCache, input: str):
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(f'{model_path()}/tokenizer.model')
    # [0, 9038, 2501, 263, 931, 29892, 29871]
    return [model_config(cache)['bos_token_id']] + sp.encode(input)

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