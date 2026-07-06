#https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct





import transformers
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "meta-llama/Llama-3.1-8B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True,bnb_8bit_compute_dtype=torch.bfloat16,)

model= AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    quantization_config= quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Can the sun become a black hole?"
start= time.time()
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=128)
end = time.time()
print((tokenizer.decode(outputs[0])))
length=end-start
print(f"It took {length} seconds to run")