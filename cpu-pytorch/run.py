from dataclasses import dataclass

def main():
    import torch
    with torch.inference_mode():
        cache = GlobalCache(device=torch.device('cpu'))
        layer1 = LayerCache(index=0)
        decode_layer(cache, layer1, input_embeds=input_embeds(cache))

@dataclass
class GlobalCache:
    device: any
    config: any = None
    input_ids: any = None # REMOVE THIS
    input_embeds: any = None # REMOVE THIS
    head_dim: any = None
    inv_freq: any = None
    rotary_emb: any = None

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

def sp_input_ids(cache: GlobalCache):
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(f'{model_path()}/tokenizer.model')
    # [0, 9038, 2501, 263, 931, 29892, 29871]
    return [model_config(cache)['bos_token_id']] + sp.encode('Once upon a time, ')

def input_ids(cache: GlobalCache):
    # convert input_ids from long[] to 2D tensor (batch_size, sequence_length)
    # we only have 1 batch here, so unsqueeze(0)
    # The 0 in torch.unsqueeze(input, 0) indicates the dimension index at which you want to insert the new axis.
    # By convention:
    # Dimension 0 is the batch dimension
    # Dimension 1 is the sequence length dimension
    # Dimension 2 is the embedding dimension
    # This inserts a new axis (dimension) of size 1 at index 0, which becomes the new batch dimension.
    if cache.input_ids:
        return cache.input_ids
    import torch
    cache.input_ids = torch.tensor(sp_input_ids(cache), dtype=torch.long).unsqueeze(0)
    return cache.input_ids

def safetensors_00001(key):
    from safetensors import safe_open
    with safe_open(f'{model_path()}/model-00001-of-00002.safetensors', framework="pt") as f:
        return f.get_tensor(key)
    
def embed_tokens(cache: GlobalCache):
    import torch
    # the embedding layer created from embed_tokens_tensor and config.json
    return torch.nn.Embedding(
        model_config(cache)['vocab_size'], model_config(cache)['hidden_size'], model_config(cache)['pad_token_id'], 
        _weight=safetensors_00001('model.embed_tokens.weight'))

def input_embeds(cache: GlobalCache):
    if cache.input_embeds is not None:
        return cache.input_embeds
    cache.input_embeds = embed_tokens(cache).forward(input_ids(cache))
    return cache.input_embeds

# def causal_mask():
#     global cache_causal_mask
#     if 'cache_causal_mask' in globals():
#         return cache_causal_mask
#     import torch
#     bsz, seq_len = input_ids().size()
#     device = input_embeds().device
#     causal_mask = torch.full((seq_len, seq_len), torch.tensor(torch.finfo(input_embeds().dtype).min, device=device), device=device)
#     mask_cond = torch.arange(causal_mask.size(-1), device=device)
#     causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
#     causal_mask = causal_mask.to(input_embeds().dtype)
#     causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
#     return causal_mask

# def position_ids():
#     global cache_position_ids
#     if 'cache_position_ids' in globals():
#         return cache_position_ids
#     import torch
#     bsz, seq_len = input_ids().size()
#     cache_position_ids = torch.arange(
#         0, seq_len, dtype=torch.long, device=input_embeds().device
#     )
#     cache_position_ids = cache_position_ids.unsqueeze(0).view(-1, seq_len)
#     return cache_position_ids

@dataclass
class LayerCache:
    index: int
    input_layernorm: any = None
    q_proj: any = None
    k_proj: any = None
    v_proj: any = None

def input_layernorm(layer: LayerCache):
    from torch import nn
    if layer.input_layernorm is None:
        weight = safetensors_00001(f'model.layers.{layer.index}.input_layernorm.weight')
        layer.input_layernorm = nn.Parameter(weight)
    return layer.input_layernorm

def head_dim(cache: GlobalCache):
    if cache.head_dim is None:
        config = model_config(cache)
        cache.head_dim = config['hidden_size'] // config['num_attention_heads']
    return cache.head_dim

def q_proj(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.q_proj is None:
        config = model_config(cache)
        layer.q_proj = nn.Linear(config['hidden_size'], config['num_attention_heads'] * head_dim(cache), bias=False)
        layer.q_proj.weight = nn.Parameter(safetensors_00001(f'model.layers.{layer.index}.self_attn.q_proj.weight'))
    return layer.q_proj

def k_proj(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.k_proj is None:
        config = model_config(cache)
        layer.k_proj = nn.Linear(config['hidden_size'], config['num_attention_heads'] * head_dim(cache), bias=False)
        layer.k_proj.weight = nn.Parameter(safetensors_00001(f'model.layers.{layer.index}.self_attn.k_proj.weight'))
    return layer.k_proj

def v_proj(cache: GlobalCache, layer: LayerCache):
    from torch import nn
    if layer.v_proj is None:
        config = model_config(cache)
        layer.v_proj = nn.Linear(config['hidden_size'], config['num_attention_heads'] * head_dim(cache), bias=False)
        layer.v_proj.weight = nn.Parameter(safetensors_00001(f'model.layers.{layer.index}.self_attn.v_proj.weight'))
    return layer.v_proj

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

def decode_layer(cache: GlobalCache, layer: LayerCache, input_embeds):
    import torch
    variance = input_embeds.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = input_embeds * torch.rsqrt(variance + model_config(cache)['rms_norm_eps'])
    hidden_states = (input_layernorm(layer) * hidden_states).to(input_embeds.dtype)
    bsz, q_len, _ = hidden_states.size()
    config = model_config(cache)
    query_states = q_proj(cache, layer).forward(hidden_states).view(bsz, q_len, config['num_attention_heads'], head_dim(cache)).transpose(1, 2)
    key_states = k_proj(cache, layer).forward(hidden_states).view(bsz, q_len, config['num_attention_heads'], head_dim(cache)).transpose(1, 2)
    value_states = v_proj(cache, layer).forward(hidden_states).view(bsz, q_len, config['num_attention_heads'], head_dim(cache)).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    cos, sin = rotary_emb(cache, kv_seq_len, value_states.dtype)
    # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    return hidden_states

main()