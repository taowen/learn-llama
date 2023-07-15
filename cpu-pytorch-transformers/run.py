from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.generation.utils import GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained('../weights/llama-7b-hf')
input_ids = tokenizer.encode("Once upon a time, ", return_tensors="pt")
print(input_ids)

model = LlamaForCausalLM.from_pretrained('../weights/llama-7b-hf')
generated = model.generate(input_ids, max_length=50, generation_config=GenerationConfig())
generated_tokens = generated[:, input_ids.shape[-1]:] # slice to get generated tokens
print(generated_tokens)

# The generate() method on the LLaMAForCausalLM model returns a 2D tensor of shape (batch_size, sequence_length).
decoded = tokenizer.decode(generated_tokens[0])
print(decoded)

# 100 years ago, a young man named Jack was born. He was a very brave boy, and he loved to play with his friends. One day, Jack and his friends were playing in the woods.
