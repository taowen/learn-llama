from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained('../weights/llama-7b-hf')
tokenizer.save_pretrained('../weights/llama-7b-hf-safetensors')

model = LlamaForCausalLM.from_pretrained('../weights/llama-7b-hf')
model.save_pretrained('../weights/llama-7b-hf-safetensors')