import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

def parse_prediction(raw_prediction: str) -> str:
    """解析模型原始预测，返回 'yes', 'no' 或 'generation_error'"""
    raw_prediction = raw_prediction.lower()
    if 'yes' in raw_prediction:
        return 'yes'
    elif 'no' in raw_prediction:
        return 'no'
    else:
        print(f"无法解析响应: {raw_prediction}")
        return 'generation_error'

def evaluate_model(data_list: List[Dict], unparsed_output_file: str = 'unparsed_predictions.json') -> Tuple[Dict, Dict]:
    """评测模型性能，计算总体和分维度指标，并保存无法解析的数据到 JSON"""
    metrics = Counter()
    detailed_metrics = {
        'by_domain': defaultdict(Counter),
        'by_error_type': defaultdict(Counter),
        'by_operation': defaultdict(Counter),
        'by_prompt_type': defaultdict(Counter)
    }
    unparsed_data = {}  # 存储无法解析的样本，格式为 {id: {...}}
    
    for idx, item in enumerate(data_list):
        expected = item['expected_answer'].lower()  # Yes/No 转为小写
        pred = parse_prediction(item['raw_prediction'])
        item['parsel_prediction'] = pred  # 保存解析结果
        
        # 如果无法解析，添加到 unparsed_data
        if pred == 'generation_error':
            # 只保存 expected_answer == "Yes" 的样本（错误样本）
            if expected == 'yes':
                sample_id = f"unparsed_{idx}"
                unparsed_data[sample_id] = {
                    "error_number": item['number'],
                    "error_passage": item['passage'],
                    "dataset": item['dataset'],
                    "operation": item['operation'],
                    "error_annotation": item['error_annotation'],
                    # 以下字段需补充（若有正确数据）
                    "correct_number": "",  # 需手动补充或从原始数据推导
                    "correct_passage": ""  # 需手动补充或从原始数据推导
                }
        
        domain = item['dataset']
        operation = item['operation']
        prompt_type = item['prompt_type']
        error_types = [k for k, v in item['error_annotation'].items() if v > 0]
        
        # 计算总体指标
        if pred == expected:
            if expected == 'yes':
                metrics['TP'] += 1
                for et in error_types:
                    detailed_metrics['by_error_type'][et]['TP'] += 1
                detailed_metrics['by_domain'][domain]['TP'] += 1
                detailed_metrics['by_operation'][operation]['TP'] += 1
                detailed_metrics['by_prompt_type'][prompt_type]['TP'] += 1
            else:  # expected == 'no'
                metrics['TN'] += 1
                for et in error_types:
                    detailed_metrics['by_error_type'][et]['TN'] += 1
                detailed_metrics['by_domain'][domain]['TN'] += 1
                detailed_metrics['by_operation'][operation]['TN'] += 1
                detailed_metrics['by_prompt_type'][prompt_type]['TN'] += 1
        else:
            if expected == 'yes':
                metrics['FN'] += 1
                for et in error_types:
                    detailed_metrics['by_error_type'][et]['FN'] += 1
                detailed_metrics['by_domain'][domain]['FN'] += 1
                detailed_metrics['by_operation'][operation]['FN'] += 1
                detailed_metrics['by_prompt_type'][prompt_type]['FN'] += 1
            else:  # expected == 'no'
                metrics['FP'] += 1
                for et in error_types:
                    detailed_metrics['by_error_type'][et]['FP'] += 1
                detailed_metrics['by_domain'][domain]['FP'] += 1
                detailed_metrics['by_operation'][operation]['FP'] += 1
                detailed_metrics['by_prompt_type'][prompt_type]['FP'] += 1
        
        if pred == 'generation_error':
            metrics['Generation Error'] += 1
            for et in error_types:
                detailed_metrics['by_error_type'][et]['Generation Error'] += 1
            detailed_metrics['by_domain'][domain]['Generation Error'] += 1
            detailed_metrics['by_operation'][operation]['Generation Error'] += 1
            detailed_metrics['by_prompt_type'][prompt_type]['Generation Error'] += 1
    
    # 保存无法解析的数据到 JSON
    if unparsed_data:
        with open(unparsed_output_file, 'w', encoding='utf-8') as f:
            json.dump(unparsed_data, f, indent=2, ensure_ascii=False)
        print(f"无法解析的 {len(unparsed_data)} 条数据已保存到 {unparsed_output_file}")
        print("注意：JSON 文件仅包含 expected_answer='Yes' 的样本，correct_number 和 correct_passage 需手动补充")
    else:
        print("没有无法解析的数据")
    
    total = len(data_list)
    metrics['Accuracy'] = (metrics['TP'] + metrics['TN']) / total if total > 0 else 0
    
    return metrics, detailed_metrics

def main():
    # 读取 predictions.jsonl
    data_list = []
    input_file = 'predictions.jsonl'  # 确认路径
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    
    if not data_list:
        print("数据为空，退出！")
        return
    
    # 评测模型并保存无法解析的数据
    unparsed_output_file = 'unparsed_predictions.json'
    metrics, detailed_metrics = evaluate_model(data_list, unparsed_output_file)
    
    # 打印总体指标
    print("\nOverall Metrics:")
    total = len(data_list)
    for key, value in metrics.items():
        if key == 'Accuracy':
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value} ({value / total:.3f})")
    
    # 打印分维度指标
    print("\nMetrics by Domain:")
    for domain, counts in detailed_metrics['by_domain'].items():
        print(f"{domain}: {dict(counts)}")
    
    print("\nMetrics by Error Type:")
    for error_type, counts in detailed_metrics['by_error_type'].items():
        print(f"{error_type}: {dict(counts)}")
    
    print("\nMetrics by Operation:")
    for operation, counts in detailed_metrics['by_operation'].items():
        print(f"{operation}: {dict(counts)}")
    
    print("\nMetrics by Prompt Type:")
    for prompt_type, counts in detailed_metrics['by_prompt_type'].items():
        print(f"{prompt_type}: {dict(counts)}")

if __name__ == "__main__":
    main()

"""
完成一个中文的分析报告
这是qwen3-7B的评测结果:
注意：JSON 文件仅包含 expected_answer='Yes' 的样本，correct_number 和 correct_passage 需手动补充

Overall Metrics:
FP: 4644 (0.484)
TP: 4443 (0.463)
TN: 156 (0.016)
FN: 357 (0.037)
Generation Error: 577 (0.060)
Accuracy: 0.479

Metrics by Domain:
Numeracy_600K_article_title: {'FP': 966, 'TP': 984, 'TN': 34, 'FN': 16}
aclsent: {'FP': 896, 'TP': 931, 'TN': 52, 'FN': 17}
DROP: {'FP': 973, 'TP': 700, 'TN': 21, 'Generation Error': 569, 'FN': 294}
qa-text-source-comparison: {'FP': 901, 'TP': 905, 'TN': 23, 'FN': 19, 'Generation Error': 8}
FinNum: {'FP': 908, 'TP': 923, 'TN': 26, 'FN': 11}

Metrics by Error Type:
Error in Number Relationships: {'FP': 193, 'TP': 192, 'TN': 3, 'FN': 4, 'Generation Error': 6}
Undetectable Error: {'FP': 453, 'TP': 445, 'TN': 11, 'FN': 19, 'Generation Error': 18}
Type Error: {'FP': 499, 'TP': 508, 'TN': 19, 'FN': 10, 'Generation Error': 16}
Anomaly: {'FP': 223, 'TP': 219, 'FN': 11, 'TN': 7, 'Generation Error': 14}
Improper Data: {'FP': 28, 'TP': 29, 'TN': 1}
Factual Error: {'FP': 55, 'TP': 55, 'TN': 2, 'FN': 2, 'Generation Error': 2}

Metrics by Operation:
*2: {'FP': 153, 'TP': 146, 'TN': 6, 'FN': 13, 'Generation Error': 22}
-10: {'FP': 175, 'TP': 167, 'TN': 8, 'FN': 16, 'Generation Error': 22}
+1: {'FP': 182, 'TP': 174, 'TN': 6, 'FN': 14, 'Generation Error': 18}
*0.9: {'FP': 182, 'TP': 171, 'TN': 4, 'Generation Error': 28, 'FN': 15}
*1.1: {'FP': 190, 'TP': 180, 'TN': 9, 'FN': 19, 'Generation Error': 34}
-0.5: {'FP': 149, 'TP': 148, 'TN': 7, 'Generation Error': 11, 'FN': 8}
+1000: {'FP': 157, 'TP': 146, 'Generation Error': 24, 'FN': 13, 'TN': 2}
*1.5: {'FP': 161, 'TP': 156, 'TN': 3, 'Generation Error': 14, 'FN': 8}
*0.1: {'FP': 155, 'TP': 152, 'TN': 6, 'FN': 9, 'Generation Error': 12}
*0: {'FP': 174, 'TP': 161, 'TN': 4, 'FN': 17, 'Generation Error': 24}
-0.1: {'FP': 186, 'TP': 173, 'TN': 2, 'Generation Error': 30, 'FN': 15}
swap: {'FP': 338, 'TP': 318, 'TN': 10, 'FN': 30, 'Generation Error': 42}
*0.01: {'FP': 174, 'TP': 172, 'TN': 7, 'Generation Error': 14, 'FN': 9}
+10: {'FP': 169, 'TP': 162, 'TN': 5, 'FN': 12, 'Generation Error': 20}
+0.1: {'FP': 155, 'TP': 149, 'TN': 8, 'FN': 14, 'Generation Error': 17}
*(-1): {'FP': 154, 'TP': 147, 'TN': 6, 'FN': 13, 'Generation Error': 24}
-1: {'FP': 195, 'TP': 189, 'TN': 5, 'FN': 11, 'Generation Error': 18}
*0.5: {'FP': 202, 'TP': 193, 'TN': 7, 'FN': 16, 'Generation Error': 20}
*100: {'FP': 156, 'TP': 144, 'FN': 17, 'TN': 5, 'Generation Error': 29}
+0.5: {'FP': 161, 'TP': 154, 'TN': 8, 'Generation Error': 27, 'FN': 15}
-1000: {'FP': 160, 'TP': 149, 'TN': 4, 'Generation Error': 30, 'FN': 15}
-100: {'FP': 148, 'TP': 144, 'FN': 11, 'TN': 7, 'Generation Error': 16}
*0.001: {'FP': 151, 'TP': 147, 'TN': 6, 'Generation Error': 20, 'FN': 10}
*0.7: {'FP': 125, 'TP': 120, 'TN': 4, 'FN': 9, 'Generation Error': 13}
*1000: {'FP': 167, 'TP': 162, 'TN': 4, 'Generation Error': 18, 'FN': 9}
+100: {'FP': 158, 'TP': 155, 'TN': 7, 'FN': 10, 'Generation Error': 14}
*10: {'FP': 167, 'TP': 164, 'TN': 6, 'FN': 9, 'Generation Error': 16}

Metrics by Prompt Type:
few_shot: {'FP': 34, 'TP': 55, 'TN': 66, 'FN': 45, 'Generation Error': 18}
zero_shot: {'FP': 4610, 'TP': 4388, 'FN': 312, 'TN': 90, 'Generation Error': 559}

这是xiaomi-7B的模型评测：
注意：JSON 文件仅包含 expected_answer='Yes' 的样本，correct_number 和 correct_passage 需手动补充

Overall Metrics:
FP: 4619 (0.481)
TP: 4347 (0.453)
TN: 181 (0.019)
FN: 453 (0.047)
Generation Error: 634 (0.066)
Accuracy: 0.472

Metrics by Domain:
Numeracy_600K_article_title: {'FP': 994, 'TP': 997, 'TN': 6, 'FN': 3, 'Generation Error': 1}
aclsent: {'FP': 904, 'TP': 921, 'TN': 44, 'FN': 27, 'Generation Error': 8}
DROP: {'FP': 921, 'TP': 669, 'TN': 73, 'FN': 325, 'Generation Error': 550}
qa-text-source-comparison: {'FP': 872, 'TP': 867, 'FN': 57, 'TN': 52, 'Generation Error': 35}
FinNum: {'FP': 928, 'TP': 893, 'FN': 41, 'Generation Error': 40, 'TN': 6}

Metrics by Error Type:
Error in Number Relationships: {'FP': 190, 'TP': 187, 'TN': 6, 'FN': 9, 'Generation Error': 7}
Undetectable Error: {'FP': 457, 'TP': 446, 'TN': 7, 'FN': 18, 'Generation Error': 24}
Type Error: {'FP': 502, 'TP': 495, 'TN': 16, 'FN': 23, 'Generation Error': 23}
Anomaly: {'FP': 216, 'TP': 214, 'TN': 14, 'FN': 16, 'Generation Error': 15}
Improper Data: {'FP': 29, 'TP': 28, 'FN': 1}
Factual Error: {'FP': 49, 'TP': 52, 'TN': 8, 'FN': 5, 'Generation Error': 3}

Metrics by Operation:
*2: {'FP': 155, 'TP': 143, 'TN': 4, 'FN': 16, 'Generation Error': 24}
-10: {'FP': 178, 'TP': 162, 'FN': 21, 'TN': 5, 'Generation Error': 25}
+1: {'FP': 177, 'TP': 166, 'TN': 11, 'FN': 22, 'Generation Error': 26}
*0.9: {'FP': 179, 'TP': 167, 'Generation Error': 29, 'FN': 19, 'TN': 7}
*1.1: {'FP': 190, 'TP': 180, 'FN': 19, 'TN': 9, 'Generation Error': 35}
-0.5: {'FP': 148, 'TP': 148, 'TN': 8, 'Generation Error': 11, 'FN': 8}
+1000: {'FP': 150, 'TP': 139, 'FN': 20, 'TN': 9, 'Generation Error': 27}
*1.5: {'FP': 157, 'TP': 153, 'TN': 7, 'FN': 11, 'Generation Error': 15}
*0.1: {'FP': 155, 'TP': 148, 'FN': 13, 'TN': 6, 'Generation Error': 14}
*0: {'FP': 173, 'TP': 163, 'TN': 5, 'FN': 15, 'Generation Error': 20}
-0.1: {'FP': 184, 'TP': 170, 'TN': 4, 'FN': 18, 'Generation Error': 30}
swap: {'FP': 335, 'TP': 311, 'FN': 37, 'TN': 13, 'Generation Error': 45}
*0.01: {'FP': 177, 'TP': 170, 'FN': 11, 'Generation Error': 16, 'TN': 4}
+10: {'FP': 166, 'TP': 160, 'Generation Error': 21, 'TN': 8, 'FN': 14}
+0.1: {'FP': 160, 'TP': 150, 'TN': 3, 'FN': 13, 'Generation Error': 24}
*(-1): {'FP': 157, 'TP': 143, 'TN': 3, 'Generation Error': 29, 'FN': 17}
-1: {'FP': 192, 'TP': 185, 'FN': 15, 'TN': 8, 'Generation Error': 15}
*0.5: {'FP': 200, 'TP': 187, 'TN': 9, 'FN': 22, 'Generation Error': 24}
*100: {'FP': 153, 'TP': 141, 'TN': 8, 'FN': 20, 'Generation Error': 29}
+0.5: {'FP': 162, 'TP': 153, 'FN': 16, 'Generation Error': 29, 'TN': 7}
-1000: {'FP': 161, 'TP': 144, 'TN': 3, 'FN': 20, 'Generation Error': 32}
-100: {'FP': 148, 'TP': 136, 'TN': 7, 'FN': 19, 'Generation Error': 21}
*0.001: {'FP': 151, 'TP': 142, 'TN': 6, 'FN': 15, 'Generation Error': 24}
*0.7: {'FP': 124, 'TP': 122, 'TN': 5, 'FN': 7, 'Generation Error': 11}
*1000: {'FP': 162, 'TP': 154, 'TN': 9, 'Generation Error': 25, 'FN': 17}
+100: {'FP': 157, 'TP': 153, 'TN': 8, 'FN': 12, 'Generation Error': 12}
*10: {'FP': 168, 'TP': 157, 'TN': 5, 'FN': 16, 'Generation Error': 21}

Metrics by Prompt Type:
few_shot: {'FP': 79, 'TP': 70, 'TN': 21, 'FN': 30, 'Generation Error': 28}
zero_shot: {'FP': 4540, 'TP': 4277, 'TN': 160, 'FN': 423, 'Generation Error': 606}

这是glm_4v_flash的模型评测结果
注意：JSON 文件仅包含 expected_answer='Yes' 的样本，correct_number 和 correct_passage 需手动补充

Overall Metrics:
FP: 2450 (0.255)
TP: 3353 (0.349)
TN: 2350 (0.245)
FN: 1447 (0.151)
Generation Error: 2 (0.000)
Accuracy: 0.594

Metrics by Domain:
Numeracy_600K_article_title: {'FP': 454, 'TP': 684, 'TN': 546, 'FN': 316}
aclsent: {'FP': 475, 'TP': 658, 'TN': 473, 'FN': 290}
DROP: {'TN': 491, 'TP': 744, 'FN': 250, 'FP': 503}
qa-text-source-comparison: {'FP': 448, 'FN': 307, 'TN': 476, 'TP': 617, 'Generation Error': 2}
FinNum: {'TN': 364, 'FN': 284, 'TP': 650, 'FP': 570}

Metrics by Error Type:
Error in Number Relationships: {'FP': 117, 'TP': 149, 'TN': 79, 'FN': 47}
Undetectable Error: {'TN': 217, 'TP': 278, 'FP': 247, 'FN': 186}
Type Error: {'TN': 252, 'FN': 129, 'FP': 266, 'TP': 389, 'Generation Error': 2}
Anomaly: {'FP': 123, 'FN': 57, 'TN': 107, 'TP': 173}
Improper Data: {'FP': 18, 'FN': 11, 'TN': 11, 'TP': 18}
Factual Error: {'FP': 27, 'TP': 43, 'TN': 30, 'FN': 14}

Metrics by Operation:
*2: {'FP': 74, 'TP': 99, 'TN': 85, 'FN': 60}
-10: {'TN': 85, 'TP': 113, 'FP': 98, 'FN': 70}
+1: {'FP': 97, 'TP': 109, 'TN': 91, 'FN': 79}
*0.9: {'TN': 105, 'FN': 59, 'TP': 127, 'FP': 81}
*1.1: {'FP': 92, 'FN': 59, 'TN': 107, 'TP': 140}
-0.5: {'TN': 81, 'FN': 51, 'TP': 105, 'FP': 75}
+1000: {'TN': 79, 'TP': 117, 'FP': 80, 'FN': 42}
*1.5: {'TN': 74, 'FN': 57, 'TP': 107, 'FP': 90}
*0.1: {'FP': 88, 'FN': 50, 'TN': 73, 'TP': 111}
*0: {'TN': 83, 'FN': 51, 'TP': 127, 'FP': 95}
-0.1: {'TN': 81, 'TP': 147, 'FP': 107, 'FN': 41, 'Generation Error': 2}
swap: {'TN': 163, 'TP': 233, 'FP': 185, 'FN': 115}
*0.01: {'FP': 88, 'TP': 146, 'TN': 93, 'FN': 35}
+10: {'FP': 91, 'TP': 109, 'TN': 83, 'FN': 65}
+0.1: {'FP': 84, 'TP': 112, 'TN': 79, 'FN': 51}
*(-1): {'FP': 76, 'TP': 125, 'TN': 84, 'FN': 35}
-1: {'TN': 103, 'TP': 102, 'FP': 97, 'FN': 98}
*0.5: {'TN': 93, 'TP': 128, 'FP': 116, 'FN': 81}
*100: {'FP': 75, 'TP': 116, 'TN': 86, 'FN': 45}
+0.5: {'TN': 75, 'FN': 36, 'FP': 94, 'TP': 133}
-1000: {'TN': 75, 'TP': 140, 'FP': 89, 'FN': 24}
-100: {'TN': 87, 'TP': 116, 'FP': 68, 'FN': 39}
*0.001: {'TN': 77, 'TP': 120, 'FP': 80, 'FN': 37}
*0.7: {'TN': 64, 'FN': 34, 'TP': 95, 'FP': 65}
*1000: {'FP': 89, 'FN': 33, 'TP': 138, 'TN': 82}
+100: {'TN': 85, 'TP': 114, 'FP': 80, 'FN': 51}
*10: {'TN': 77, 'FN': 49, 'TP': 124, 'FP': 96}

Metrics by Prompt Type:
few_shot: {'FP': 26, 'TP': 52, 'FN': 48, 'TN': 74}
zero_shot: {'TN': 2276, 'TP': 3301, 'FP': 2424, 'FN': 1399, 'Generation Error': 2}

这是deepseek_8b的评测结果
注意：JSON 文件仅包含 expected_answer='Yes' 的样本，correct_number 和 correct_passage 需手动补充

Overall Metrics:
FP: 4207 (0.438)
TP: 3849 (0.401)
TN: 593 (0.062)
FN: 951 (0.099)
Generation Error: 509 (0.053)
Accuracy: 0.463

Metrics by Domain:
Numeracy_600K_article_title: {'FP': 860, 'TP': 822, 'TN': 140, 'FN': 178}
aclsent: {'FP': 905, 'TP': 905, 'FN': 43, 'TN': 43}
DROP: {'TN': 180, 'FN': 474, 'FP': 814, 'TP': 520, 'Generation Error': 499}
qa-text-source-comparison: {'TN': 67, 'FN': 92, 'FP': 857, 'TP': 832, 'Generation Error': 8}
FinNum: {'FP': 771, 'TP': 770, 'TN': 163, 'FN': 164, 'Generation Error': 2}

Metrics by Error Type:
Error in Number Relationships: {'FP': 172, 'TP': 161, 'TN': 24, 'FN': 35, 'Generation Error': 4}
Undetectable Error: {'FP': 395, 'TP': 387, 'TN': 69, 'FN': 77, 'Generation Error': 14}
Type Error: {'FP': 436, 'TP': 415, 'TN': 82, 'FN': 103, 'Generation Error': 10}
Anomaly: {'FP': 203, 'TP': 186, 'TN': 27, 'FN': 44, 'Generation Error': 10}
Improper Data: {'FP': 26, 'TP': 27, 'TN': 3, 'FN': 2}
Factual Error: {'FP': 52, 'TP': 45, 'FN': 12, 'TN': 5, 'Generation Error': 2}

Metrics by Operation:
*2: {'FP': 136, 'TP': 125, 'TN': 23, 'FN': 34, 'Generation Error': 21}
-10: {'FP': 161, 'TP': 148, 'FN': 35, 'TN': 22, 'Generation Error': 18}
+1: {'TN': 27, 'FN': 41, 'FP': 161, 'TP': 147, 'Generation Error': 18}
*0.9: {'FP': 162, 'TP': 147, 'FN': 39, 'TN': 24, 'Generation Error': 26}
*1.1: {'FP': 174, 'TP': 150, 'TN': 25, 'FN': 49, 'Generation Error': 32}
-0.5: {'TN': 16, 'TP': 142, 'FP': 140, 'FN': 14, 'Generation Error': 10}
+1000: {'FP': 142, 'TP': 133, 'TN': 17, 'FN': 26, 'Generation Error': 22}
*1.5: {'FP': 140, 'TP': 139, 'TN': 24, 'FN': 25, 'Generation Error': 10}
*0.1: {'FP': 142, 'TP': 127, 'TN': 19, 'FN': 34, 'Generation Error': 10}
*0: {'FP': 158, 'TP': 135, 'FN': 43, 'TN': 20, 'Generation Error': 16}
-0.1: {'FP': 162, 'TP': 154, 'TN': 26, 'FN': 34, 'Generation Error': 26}
swap: {'FP': 305, 'FN': 79, 'TP': 269, 'TN': 43, 'Generation Error': 42}
*0.01: {'FP': 152, 'TP': 135, 'TN': 29, 'FN': 46, 'Generation Error': 14}
+10: {'FP': 154, 'TP': 148, 'TN': 20, 'FN': 26, 'Generation Error': 16}
+0.1: {'FP': 142, 'TP': 127, 'FN': 36, 'TN': 21, 'Generation Error': 16}
*(-1): {'FP': 146, 'TP': 131, 'TN': 14, 'FN': 29, 'Generation Error': 20}
-1: {'FP': 177, 'TP': 168, 'TN': 23, 'FN': 32, 'Generation Error': 10}
*0.5: {'FP': 184, 'TP': 171, 'TN': 25, 'FN': 38, 'Generation Error': 22}
*100: {'FP': 141, 'TP': 127, 'TN': 20, 'FN': 34, 'Generation Error': 24}
+0.5: {'TN': 23, 'TP': 131, 'FP': 146, 'FN': 38, 'Generation Error': 26}
-1000: {'FP': 142, 'TP': 129, 'TN': 22, 'FN': 35, 'Generation Error': 20}
-100: {'FP': 138, 'TP': 119, 'TN': 17, 'FN': 36, 'Generation Error': 16}
*0.001: {'FP': 144, 'TP': 127, 'FN': 30, 'TN': 13, 'Generation Error': 18}
*0.7: {'FP': 112, 'TP': 106, 'FN': 23, 'TN': 17, 'Generation Error': 10}
*1000: {'FP': 148, 'TP': 138, 'TN': 23, 'FN': 33, 'Generation Error': 18}
+100: {'FP': 145, 'TP': 134, 'FN': 31, 'TN': 20, 'Generation Error': 12}
*10: {'FP': 153, 'TP': 142, 'FN': 31, 'TN': 20, 'Generation Error': 16}

Metrics by Prompt Type:
few_shot: {'FP': 77, 'TP': 88, 'TN': 23, 'FN': 12, 'Generation Error': 12}
zero_shot: {'FP': 4130, 'TP': 3761, 'TN': 570, 'FN': 939, 'Generation Error': 497}
"""