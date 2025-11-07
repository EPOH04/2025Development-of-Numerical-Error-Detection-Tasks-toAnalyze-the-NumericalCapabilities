from typing import List, Dict, Tuple
from zhipuai import ZhipuAI
from tqdm import tqdm
import json
import os

client = ZhipuAI(api_key="60a107de2b381145b3ab9955187f2e50.Z4hUWiul0ggb2pFg")  # 请填写您自己的APIKey


def GLM_analysis(content):
    client = ZhipuAI(api_key="60a107de2b381145b3ab9955187f2e50.Z4hUWiul0ggb2pFg")
    system_prompt = '''### ROLE. You are an expert fact-checker specializing in numerical accuracy across biology, physics, history, mathematics, and everyday scenarios. ### TASK. Determine if the given number contains a factual error within the provided context. ### ANALYSIS PROCESS. Follow this reasoning sequence: 1. CONTEXT ANALYSIS: Identify the domain and type of measurement being described 2. GENERATED KNOWLEDGE: Recall established facts, typical ranges, and known standards for this specific domain and measurement type 3. PLAUSIBILITY CHECK: Compare the number against expected ranges, physical laws, biological constraints, historical accuracy, and mathematical consistency 4. ERROR DETECTION: Check for biological impossibilities, physical violations, historical inaccuracies, mathematical contradictions, or scale/magnitude errors. ### OUTPUT. Answer must only be "Yes" or "No",do not provide explanations. Yes = Contains factual error, No = Factually accurate."
Answer:'''
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        temperature=0.3,
        top_p=0.5
    )
    markdown_content = response.choices[0].message.content
    markdown_content = markdown_content.replace("```python", "").replace("```json", "").replace("```", "")
    return markdown_content



"""Performance under different parameters:
1. t=0.3;t-p=0.5:
    Overall Metrics:
    FP: 321 (0.134)
    Generation Error: 2 (0.001)
    FN: 432 (0.180)
    TP: 768 (0.320)
    TN: 879 (0.366)
    Accuracy: 0.686

2. t=0.3;t-p=0.7:
    Overall Metrics:
    TN: 827 (0.345)
    FN: 393 (0.164)
    TP: 807 (0.336)
    FP: 373 (0.155)
    Accuracy: 0.681

3. t=0.3;t-p=0.3:
    Overall Metrics:
    TN: 853 (0.355)
    TP: 777 (0.324)
    FN: 423 (0.176)
    FP: 347 (0.145)
    Accuracy: 0.679
4. t=0.3;t-p=0.6:
    Overall Metrics:
    TN: 847 (0.353)
    TP: 776 (0.323)
    FN: 424 (0.177)
    FP: 353 (0.147)
    Generation Error: 2 (0.001)
    Accuracy: 0.676
5. t=0.3;t-p=0.4：

MV:
Overall Metrics:
FP: 1144 (0.477)
TP: 1176 (0.490)
TN: 56 (0.023)
FN: 24 (0.010)
Accuracy: 0.513
"""


def create_zero_shot_prompt(passage: str, number: str) -> str:
    return f"""Answer with only 'Yes' or 'No'. Do not provide explanations. Is "{number}" in the following passage an error? "{passage}"
Answer:"""

def create_few_shot_prompt(passage: str, number: str) -> str:
    examples = [
        {"passage": "Spiders have 9 limbs.", "number": "9", "answer": "Yes"},
        {"passage": "Spiders have 8 limbs.", "number": "8", "answer": "No"},
        {"passage": "Mike's height is -3.6 meters.", "number": "-3.6", "answer": "Yes"},
        {"passage": "Mike's height is 1.8 meters.", "number": "1.8", "answer": "No"}
    ]
    prompt = "Answer with only 'Yes' or 'No'. Do not provide explanations.\n"
    for ex in examples:
        prompt += f"""Question: Is "{ex['number']}" in the following passage an error? "{ex['passage']}"
Answer: {ex['answer']}\n"""
    prompt += f"""Question: Is "{number}" in the following passage an error? "{passage}"
Answer:"""
    return prompt

def load_benedect_dataset(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件 {file_path} 不存在，请确认路径！")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 文件解析错误：{e}")
    
    dataset = list(dataset_dict.values())
    save_list = []
    
    for i, data in enumerate(tqdm(dataset, desc="Processing dataset")):
        required_fields = ['correct_number', 'correct_passage', 'error_number', 'error_passage', 'dataset', 'operation']
        for field in required_fields:
            if field not in data:
                print(f"样本 {data.get('id', '未知')} 缺少字段 {field}")
                continue
        
        prompt_fn = create_few_shot_prompt if i % 48 == 0 else create_zero_shot_prompt
        correct_item = {
            "prompt": prompt_fn(data['correct_passage'], data['correct_number']),
            "expected_answer": "No",
            "dataset": data['dataset'],
            "operation": data['operation'],
            "error_annotation": data.get('error_annotation', {}),
            "passage": data['correct_passage'],
            "number": data['correct_number'],
            "prompt_type": "few_shot" if i % 48 == 0 else "zero_shot"
        }
        error_item = {
            "prompt": prompt_fn(data['error_passage'], data['error_number']),
            "expected_answer": "Yes",
            "dataset": data['dataset'],
            "operation": data['operation'],
            "error_annotation": data.get('error_annotation', {}),
            "passage": data['error_passage'],
            "number": data['error_number'],
            "prompt_type": "few_shot" if i % 48 == 0 else "zero_shot"
        }
        save_list.append(correct_item)
        save_list.append(error_item)
    
    return save_list


# #halved dataset for training
# import random
# def load_benedect_dataset(file_path: str) -> List[Dict]:
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"数据集文件 {file_path} 不存在，请确认路径！")
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             dataset_dict = json.load(f)
#     except json.JSONDecodeError as e:
#         raise ValueError(f"JSON 文件解析错误：{e}")
    
#     dataset = list(dataset_dict.values())
#     # Randomly select half of the dataset (2400 items)
#     selected_dataset = random.sample(dataset, 1200)#change again to 1200 items
#     save_list = []
    
#     for i, data in enumerate(tqdm(selected_dataset, desc="Processing dataset")):
#         required_fields = ['correct_number', 'correct_passage', 'error_number', 'error_passage', 'dataset', 'operation']
#         for field in required_fields:
#             if field not in data:
#                 print(f"样本 {data.get('id', '未知')} 缺少字段 {field}")
#                 continue
        
#         prompt_fn = create_few_shot_prompt if i % 48 == 0 else create_zero_shot_prompt
#         correct_item = {
#             "prompt": prompt_fn(data['correct_passage'], data['correct_number']),
#             "expected_answer": "No",
#             "dataset": data['dataset'],
#             "operation": data['operation'],
#             "error_annotation": data.get('error_annotation', {}),
#             "passage": data['correct_passage'],
#             "number": data['correct_number'],
#             "prompt_type": "few_shot" if i % 48 == 0 else "zero_shot"
#         }
#         error_item = {
#             "prompt": prompt_fn(data['error_passage'], data['error_number']),
#             "expected_answer": "Yes",
#             "dataset": data['dataset'],
#             "operation": data['operation'],
#             "error_annotation": data.get('error_annotation', {}),
#             "passage": data['error_passage'],
#             "number": data['error_number'],
#             "prompt_type": "few_shot" if i % 48 == 0 else "zero_shot"
#         }
#         save_list.append(correct_item)
#         save_list.append(error_item)
    
#     return save_list
# #---------------------------------

file_path = "model_eval_first/BeNEDect_all.json"
dataset = load_benedect_dataset(file_path)

result_list = []
for datas in tqdm(dataset):
    try:
        raw_prediction = GLM_analysis(datas['prompt'])
    except Exception as e:
        raw_prediction = "generation_error"
        print(f"parsel error! {e}")
    expected = datas['expected_answer'].lower()
    datas.update({"raw_prediction": raw_prediction})
    result_list.append(datas)

        
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(result_list, f, ensure_ascii=False, indent=4)

with open('results.json', 'r', encoding='utf-8') as f_json:
    data = json.load(f_json)  # 加载为列表

with open('predictions.jsonl', 'w', encoding='utf-8') as f_jsonl:
    for item in data:
        json_line = json.dumps(item, ensure_ascii=False)
        f_jsonl.write(json_line + '\n')

print("转换完成：JSON ➜ JSONL")
