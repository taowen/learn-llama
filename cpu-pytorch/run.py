from dataclasses import dataclass

@dataclass
class GlobalCache:
    config: any = None
    input_ids: any = None
    input_embeds: any = None

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
    if cache.input_embeds:
        return cache.input_embeds
    cache.input_embeds = embed_tokens(cache).forward(input_ids(cache))
    return cache.input_embeds

def causal_mask():
    global cache_causal_mask
    if 'cache_causal_mask' in globals():
        return cache_causal_mask
    import torch
    bsz, seq_len = input_ids().size()
    device = input_embeds().device
    causal_mask = torch.full((seq_len, seq_len), torch.tensor(torch.finfo(input_embeds().dtype).min, device=device), device=device)
    mask_cond = torch.arange(causal_mask.size(-1), device=device)
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
    causal_mask = causal_mask.to(input_embeds().dtype)
    causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    return causal_mask

def position_ids():
    global cache_position_ids
    if 'cache_position_ids' in globals():
        return cache_position_ids
    import torch
    bsz, seq_len = input_ids().size()
    cache_position_ids = torch.arange(
        0, seq_len, dtype=torch.long, device=input_embeds().device
    )
    cache_position_ids = cache_position_ids.unsqueeze(0).view(-1, seq_len)
    return cache_position_ids

@dataclass
class LayerCache:
    index: int
    input: any
    input_layernorm_weight: any = None

def input_layernorm_weight(layer: LayerCache):
    if layer.input_layernorm_weight:
        return layer.input_layernorm_weight
    layer.input_layernorm_weight = safetensors_00001(f'model.layers.{layer.index}.input_layernorm.weight')
    return layer.input_layernorm_weight

def processed_by_layernorm(cache: GlobalCache, layer: LayerCache):
    import torch
    variance = layer.input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = layer.input * torch.rsqrt(variance + model_config(cache)['rms_norm_eps'])
    hidden_states = (input_layernorm_weight(layer) * hidden_states).to(layer.input.dtype)
    return hidden_states

def main():
    import torch
    with torch.inference_mode():
        cache = GlobalCache()
        layer1 = LayerCache(index=1, input=input_embeds(cache))
        print(processed_by_layernorm(cache, layer1))

main()