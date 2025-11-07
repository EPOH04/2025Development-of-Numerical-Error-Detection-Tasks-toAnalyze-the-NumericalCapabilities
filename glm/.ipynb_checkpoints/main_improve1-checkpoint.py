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
        messages=messages
    )
    markdown_content = response.choices[0].message.content
    markdown_content = markdown_content.replace("```python", "").replace("```json", "").replace("```", "")
    return markdown_content

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

"""
Changes:
changing the prompt

Overall Metrics:
TN: 3043 (0.317)
FN: 1781 (0.186)
FP: 1757 (0.183)
TP: 3019 (0.314)
Generation Error: 2 (0.000)
Accuracy: 0.631

Metrics by Domain:
Numeracy_600K_article_title: {'TN': 703, 'FN': 356, 'FP': 297, 'TP': 644}
aclsent: {'FP': 237, 'FN': 462, 'TN': 711, 'TP': 486}
DROP: {'FP': 435, 'TP': 708, 'TN': 559, 'FN': 286}
qa-text-source-comparison: {'TN': 577, 'FN': 319, 'TP': 605, 'FP': 347, 'Generation Error': 2}
FinNum: {'TN': 493, 'FN': 358, 'TP': 576, 'FP': 441}

Metrics by Error Type:
Error in Number Relationships: {'TN': 121, 'FN': 75, 'TP': 121, 'FP': 75}
Undetectable Error: {'FP': 183, 'TP': 210, 'FN': 254, 'TN': 281}
Type Error: {'TN': 328, 'FN': 143, 'FP': 190, 'TP': 375, 'Generation Error': 2}
Anomaly: {'TN': 153, 'TP': 168, 'FN': 62, 'FP': 77}
Improper Data: {'FP': 10, 'TP': 10, 'TN': 19, 'FN': 19}
Factual Error: {'FP': 18, 'TP': 43, 'TN': 39, 'FN': 14}

Metrics by Operation:
*2: {'TN': 101, 'FN': 74, 'FP': 58, 'TP': 85}
-10: {'FP': 69, 'TP': 111, 'TN': 114, 'FN': 72}
+1: {'FP': 78, 'FN': 101, 'TP': 87, 'TN': 110}
*0.9: {'TN': 115, 'FN': 74, 'TP': 112, 'FP': 71}
*1.1: {'TN': 139, 'TP': 113, 'FN': 86, 'FP': 60}
-0.5: {'FP': 56, 'FN': 57, 'TP': 99, 'TN': 100}
+1000: {'TN': 111, 'TP': 124, 'FP': 48, 'FN': 35}
*1.5: {'FP': 69, 'TP': 94, 'TN': 95, 'FN': 70}
*0.1: {'TN': 104, 'FN': 69, 'TP': 92, 'FP': 57}
*0: {'TN': 102, 'TP': 129, 'FP': 76, 'FN': 49}
-0.1: {'TN': 114, 'FN': 75, 'FP': 74, 'TP': 113, 'Generation Error': 2}
swap: {'FP': 128, 'TP': 177, 'TN': 220, 'FN': 171}
*0.01: {'TN': 112, 'TP': 115, 'FP': 69, 'FN': 66}
+10: {'TN': 106, 'FN': 93, 'TP': 81, 'FP': 68}
+0.1: {'TN': 93, 'TP': 93, 'FP': 70, 'FN': 70}
*(-1): {'FP': 63, 'TP': 134, 'TN': 97, 'FN': 26}
-1: {'TN': 133, 'FN': 118, 'FP': 67, 'TP': 82}
*0.5: {'TN': 128, 'TP': 105, 'FP': 81, 'FN': 104}
*100: {'TN': 110, 'TP': 115, 'FP': 51, 'FN': 46}
+0.5: {'TN': 113, 'TP': 120, 'FP': 56, 'FN': 49}
-1000: {'TN': 104, 'FN': 19, 'TP': 145, 'FP': 60}
-100: {'TN': 108, 'TP': 125, 'FP': 47, 'FN': 30}
*0.001: {'TN': 97, 'TP': 107, 'FP': 60, 'FN': 50}
*0.7: {'TN': 76, 'FN': 53, 'TP': 76, 'FP': 53}
*1000: {'FP': 56, 'FN': 18, 'TN': 115, 'TP': 153}
+100: {'TN': 113, 'TP': 117, 'FP': 52, 'FN': 48}
*10: {'TN': 113, 'FN': 58, 'FP': 60, 'TP': 115}

Metrics by Prompt Type:
few_shot: {'TN': 87, 'FN': 48, 'TP': 52, 'FP': 13}
zero_shot: {'FP': 1744, 'TP': 2967, 'FN': 1733, 'TN': 2956, 'Generation Error': 2}"""