from typing import List, Dict, Tuple
from zhipuai import ZhipuAI
from tqdm import tqdm
import json
import os

client = ZhipuAI(api_key="60a107de2b381145b3ab9955187f2e50.Z4hUWiul0ggb2pFg")  # 请填写您自己的APIKey


def GLM_analysis(content):
    client = ZhipuAI(api_key="60a107de2b381145b3ab9955187f2e50.Z4hUWiul0ggb2pFg")
    system_prompt = '''Roleplay: 你现在是一个专业的分析师，那你需要分析用户提供的问题，判断问题的对错。
    Task: 根据用户提供的问题完成下面的任务:
    1. 根据用户输入的信息，判断问题是否有问题，有问题回答yes，没问题回答no。
    2. 输出内容必须是yes或no。
    '''
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

file_path = "BeNEDect_all.json"
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
