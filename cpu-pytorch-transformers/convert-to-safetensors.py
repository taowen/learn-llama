from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained('../weights/open_llama_3b_v2')
tokenizer.save_pretrained('../weights/open_llama_3b_v2_safetensors')

model = LlamaForCausalLM.from_pretrained('../weights/open_llama_3b_v2')
model.save_pretrained('../weights/open_llama_3b_v2_safetensors', safe_serialization=True)