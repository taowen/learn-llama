def main():
    import sentencepiece
    # tokenize
    sp = sentencepiece.SentencePieceProcessor()
    sp.load('../weights/llama-7b-hf-safetensors/tokenizer.model')
    input_ids = [0] + sp.encode('Once upon a time, ')
    print(input_ids)
    # tensor([[    0,  9038,  2501,   263,   931, 29892, 29871]])
    # [0, 9038, 2501, 263, 931, 29892, 29871]

    # load embed_tokens_tensor
    from safetensors import safe_open

    with safe_open("../weights/llama-7b-hf-safetensors/model-00001-of-00003.safetensors", framework="pt") as f:
        print(f.keys())
        embed_tokens_tensor = f.get_tensor('model.embed_tokens.weight')
    print(embed_tokens_tensor)

    # load config.json
    import json
    with open('../weights/llama-7b-hf-safetensors/config.json') as f:
        config = json.loads(f.read())
    print(config)

    import torch
    # the embedding layer created from embed_tokens_tensor and config.json
    embed_tokens = torch.nn.Embedding(
        config['vocab_size'], config['hidden_size'], config['pad_token_id'], _weight=embed_tokens_tensor)

    device = torch.device('cpu')
    # input ids => input embeddings
    # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    # inputs_embeds = self.embed_tokens(input_ids)
    with torch.inference_mode():
        # the 0 layer: embedding
        # convert input_ids from long[] to 2D tensor (batch_size, sequence_length)
        # we only have 1 batch here, so unsqueeze(0)
        # The 0 in torch.unsqueeze(input, 0) indicates the dimension index at which you want to insert the new axis.
        # By convention:
        # Dimension 0 is the batch dimension
        # Dimension 1 is the sequence length dimension
        # Dimension 2 is the embedding dimension
        # This inserts a new axis (dimension) of size 1 at index 0, which becomes the new batch dimension.
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        inputs_embeds = embed_tokens(input_ids)
        # Make causal mask
        bsz, tgt_len = input_ids.size()
        mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(inputs_embeds.dtype).min, device=device), device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(inputs_embeds.dtype)
        mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)
        print(mask)
        # Position ids
        position_ids = torch.arange(
                0, tgt_len, dtype=torch.long, device=device
            )
        position_ids = position_ids.unsqueeze(0).view(-1, tgt_len)
        print(position_ids)
        decode_layer(config, index=0, hidden_states=inputs_embeds)

def decode_layer(config, index, hidden_states):
    from safetensors import safe_open
    import torch
    # layer norm
    with safe_open("../weights/llama-7b-hf-safetensors/model-00001-of-00003.safetensors", framework="pt") as f:
        input_layernorm_tensor = f.get_tensor(f'model.layers.{index}.input_layernorm.weight')
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + config['rms_norm_eps'])
    hidden_states = (input_layernorm_tensor * hidden_states).to(input_dtype)
    # self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    # hidden_states = self.input_layernorm(hidden_states)

main()