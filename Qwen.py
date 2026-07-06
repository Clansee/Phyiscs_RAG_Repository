# @misc{qwen2.5,
#     title = {Qwen2.5: A Party of Foundation Models},
#     url = {https://qwenlm.github.io/blog/qwen2.5/},
#     author = {Qwen Team},
#     month = {September},
#     year = {2024}
# }

# @article{qwen2,
#       title={Qwen2 Technical Report}, 
#       author={An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
#       journal={arXiv preprint arXiv:2407.10671},
#       year={2024}
# 
import transformers
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True,bnb_8bit_compute_dtype=torch.bfloat16)

model= AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    quantization_config= quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Can the sun become a black hole?"
start=time.time()
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**input_ids, max_new_tokens=128)
end=time.time()
print((tokenizer.decode(outputs[0])))
length= end- start
print(f"It took {length} seconds to run")

