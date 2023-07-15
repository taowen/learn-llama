from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.generation.utils import GenerationConfig
import torch
import json

assert(torch.cuda.is_available())
assert(torch.version.hip)
assert(not torch.version.cuda)

tokenizer = LlamaTokenizer.from_pretrained('../weights/llama-7b-hf')
input_ids = tokenizer.encode("Once upon a time, ", return_tensors="pt")
print(input_ids)

model = LlamaForCausalLM.from_pretrained('../weights/llama-7b-hf', device_map='auto', torch_dtype=torch.float32)
print(json.dumps(model.hf_device_map, indent=4))
# {
#     "model.embed_tokens": 0,
#     "model.layers.0": 0,
#     "model.layers.1": 0,
#     "model.layers.2": 0,
#     "model.layers.3": 0,
#     "model.layers.4": 0,
#     "model.layers.5": 0,
#     "model.layers.6": 0,
#     "model.layers.7": 0,
#     "model.layers.8": 0,
#     "model.layers.9": 0,
#     "model.layers.10": 0,
#     "model.layers.11": 0,
#     "model.layers.12": 0,
#     "model.layers.13": 0,
#     "model.layers.14": 0,
#     "model.layers.15": 0,
#     "model.layers.16": 0,
#     "model.layers.17": 0,
#     "model.layers.18": 0,
#     "model.layers.19": 0,
#     "model.layers.20": 0,
#     "model.layers.21": 0,
#     "model.layers.22": 0,
#     "model.layers.23": 0,
#     "model.layers.24": 0,
#     "model.layers.25": 0,
#     "model.layers.26": 0,
#     "model.layers.27": 0,
#     "model.layers.28": 0,
#     "model.layers.29": "cpu",
#     "model.layers.30": "cpu",
#     "model.layers.31": "cpu",
#     "model.norm": "cpu",
#     "lm_head": "cpu"
# }

generated = model.generate(input_ids.to('cuda:0'), max_length=50, generation_config=GenerationConfig())
generated_tokens = generated[:, input_ids.shape[-1]:] # slice to get generated tokens
print(generated_tokens)

# The generate() method on the LLaMAForCausalLM model returns a 2D tensor of shape (batch_size, sequence_length).
decoded = tokenizer.decode(generated_tokens[0])
print(decoded)

# 100 years ago, a young man named Jack was born. He was a very brave boy, and he loved to play with his friends. One day, Jack and his friends were playing in the woods.