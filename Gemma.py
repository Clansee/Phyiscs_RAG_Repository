# @article{gemma_2024,
#     title={Gemma},
#     url={https://www.kaggle.com/m/3301},
#     DOI={10.34740/KAGGLE/M/3301},
#     publisher={Kaggle},
#     author={Gemma Team},
#     year={2024}
# }

import transformers
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "google/gemma-2-9b-it"
quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         bnb_8bit_compute_dtype=torch.bfloat16)

model= AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype= "auto",
    device_map="auto",
    quantization_config= quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Can the sun become a black hole?"
start = time.time()

input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=128)
end = time.time()
print((tokenizer.decode(outputs[0])))
length= end - start
print(f"It took {length} seconds to run")