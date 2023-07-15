import sentencepiece

# tokenize
sp = sentencepiece.SentencePieceProcessor()
sp.load('../weights/llama-7b-hf-safetensors/tokenizer.model')
input_ids = sp.encode('Once upon a time, ')
print(input_ids)
# tensor([[    0,  9038,  2501,   263,   931, 29892, 29871]])
# [9038, 2501, 263, 931, 29892, 29871]

# load embed_tokens_tensor
from safetensors import safe_open

with safe_open("../weights/llama-7b-hf-safetensors/model-00001-of-00003.safetensors", framework="pt") as f:
    embed_tokens_tensor = f.get_tensor('model.embed_tokens.weight')
print(embed_tokens_tensor)

# load config.json
import json
with open('../weights/llama-7b-hf-safetensors/config.json') as f:
    config = json.loads(f.read())
print(config)

# the embedding layer created from embed_tokens_tensor and config.json
embed_tokens = torch.nn.Embedding(
    config['vocab_size'], config['hidden_size'], config['pad_token_id'], _weight=embed_tokens_tensor)

# input ids => input embeddings
# self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
# inputs_embeds = self.embed_tokens(input_ids)
import torch
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
    print(input_ids)
    inputs_embeds = embed_tokens(input_ids)
    print(inputs_embeds)