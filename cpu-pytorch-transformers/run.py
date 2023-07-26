from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.generation.utils import GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained('../weights/open_llama_3b_v2')
input_ids = tokenizer.encode("Once upon a time, ", return_tensors="pt")
print(input_ids)

model = LlamaForCausalLM.from_pretrained('../weights/open_llama_3b_v2')
generated = model.generate(input_ids, max_length=50, generation_config=GenerationConfig())
generated_tokens = generated[:, input_ids.shape[-1]:] # slice to get generated tokens
print(generated_tokens)

decoded = tokenizer.decode(generated_tokens[0])
print(decoded)

# 10 years ago, I was a young, single, and very naive woman. I was living in a small town in the middle of nowhere, and I was working at a local grocery store. I was