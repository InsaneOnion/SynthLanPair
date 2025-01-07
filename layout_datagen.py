import json
import os
import glob
import numpy as np


def validate_data(data, file_path):
    """验证json数据的完整性"""
    required_fields = ["line_texts", "lineBB", "charBB", "text_index"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in {file_path}")


def process_json_files(base_dir):
    """处理json文件并生成配对数据"""
    # 获取所有json文件路径
    en_json_files = glob.glob(os.path.join(base_dir, "label_en", "*.json"))
    if not en_json_files:
        print(
            f"Warning: No English json files found in {os.path.join(base_dir, 'label_en')}"
        )
        return {}

    # 创建一个字典来存储配对数据
    paired_data = {}
    error_files = []
    id = 0

    # 处理英文文件
    for en_file in en_json_files:
        base_name = os.path.basename(en_file)

        # 读取英文数据
        with open(en_file, "r", encoding="utf-8") as f:
            en_data = json.load(f)
        validate_data(en_data, en_file)

        # 查找对应的中文文件
        cn_file = os.path.join(base_dir, "label_cn", base_name)
        if not os.path.exists(cn_file):
            print(f"Warning: No matching Chinese file for {base_name}")
            continue

        # 读取中文数据
        with open(cn_file, "r", encoding="utf-8") as f:
            cn_data = json.load(f)
        validate_data(cn_data, cn_file)

        # 获取所有不同的text_index
        unique_indices = sorted(set(en_data["text_index"]))

        # 按text_index分组处理数据
        for idx in unique_indices:
            # 获取当前text_index的所有位置
            en_positions = [i for i, x in enumerate(en_data["text_index"]) if x == idx]
            cn_positions = [i for i, x in enumerate(cn_data["text_index"]) if x == idx]

            # 提取对应的文本和边界框
            en_texts = [en_data["line_texts"][i] for i in en_positions]
            cn_texts = [cn_data["line_texts"][i] for i in cn_positions]

            # 提取对应的边界框
            if isinstance(en_data["lineBB"], np.ndarray):
                en_boxes = en_data["lineBB"][:, :, en_positions].tolist()
                cn_boxes = cn_data["lineBB"][:, :, cn_positions].tolist()
            else:
                en_boxes = [
                    np.array(en_data["lineBB"])[:, :, i].tolist() for i in en_positions
                ]
                cn_boxes = [
                    np.array(cn_data["lineBB"])[:, :, i].tolist() for i in cn_positions
                ]

            # 保存分组后的数据
            paired_data[str(id)] = {
                "en": {
                    "line_texts": en_texts,
                    "lineBB": en_boxes,
                },
                "cn": {
                    "line_texts": cn_texts,
                    "lineBB": cn_boxes,
                },
            }
            id += 1

    if error_files:
        print("\nFiles with errors:")
        for file in error_files:
            print(f"- {file}")

    # 保存配对数据
    if paired_data:
        output_file = os.path.join(base_dir, "paired_data.json")
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(paired_data, f, ensure_ascii=False, indent=2)
            print(f"Paired data saved to {output_file}")
        except Exception as e:
            print(f"\nError saving paired data: {str(e)}")
    else:
        print("\nNo valid paired data to save")

    return paired_data


def analyze_paired_data(paired_data):
    """分析配对数据的统计信息"""
    if not paired_data:
        print("No data to analyze")
        return

    total_pairs = len(paired_data)
    print("Data Analysis:")
    print(f"Total paired files: {total_pairs}")


if __name__ == "__main__":
    base_dir = "data/results/img"
    paired_data = process_json_files(base_dir)
    analyze_paired_data(paired_data)
