from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.generation.utils import GenerationConfig
import torch
import json

assert(torch.cuda.is_available())
assert(not torch.version.hip)
assert(torch.version.cuda)

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
#     "model.layers.7": "cpu",
#     "model.layers.8": "cpu",
#     "model.layers.9": "cpu",
#     "model.layers.10": "cpu",
#     "model.layers.11": "cpu",
#     "model.layers.12": "cpu",
#     "model.layers.13": "cpu",
#     "model.layers.14": "cpu",
#     "model.layers.15": "cpu",
#     "model.layers.16": "cpu",
#     "model.layers.17": "cpu",
#     "model.layers.18": "cpu",
#     "model.layers.19": "cpu",
#     "model.layers.20": "cpu",
#     "model.layers.21": "cpu",
#     "model.layers.22": "cpu",
#     "model.layers.23": "cpu",
#     "model.layers.24": "cpu",
#     "model.layers.25": "cpu",
#     "model.layers.26": "cpu",
#     "model.layers.27": "cpu",
#     "model.layers.28": "cpu",
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