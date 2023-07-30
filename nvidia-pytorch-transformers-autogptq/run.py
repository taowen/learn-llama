from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "../weights/open_llama_3b_v2"
quantized_model_dir = "../weights/open_llama_3b_v2_gptq"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=False, legacy=False)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

# load quantized model to the first GPU
model = AutoGPTQForCausalLM.from_quantized(
    quantized_model_dir,  use_cuda_fp16=False, 
    use_triton=True, device='cuda:0')

# download quantized model from Hugging Face Hub and load to the first GPU
# model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0", use_safetensors=True, use_triton=False)

# inference with model.generate
input_ids = tokenizer("Once upon a time, ", return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**input_ids, max_new_tokens=24)[0]))

# or you can also use pipeline
# pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# print(pipeline("once upon a time")[0]["generated_text"])