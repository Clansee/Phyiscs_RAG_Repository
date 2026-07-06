import transformers
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True,bnb_8bit_compute_dtype=torch.bfloat16,)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
                  "quantization_config": quantization_config},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "teacher "},
    {"role": "user", "content": "Can the sun become a black hole?"},
]
start=time.time()

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
end = time.time()
length= end-start
print(outputs[0]["generated_text"][-1])
print(f"The time taken is {length} seconds")
