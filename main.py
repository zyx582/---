import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict

# 加载模型和tokenizer
model_path = "E:/roberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5, ignore_mismatched_sizes=True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# 改进的目标群体分类函数（基于训练数据增强）
def classify_target_group(text, is_hate, train_data_stats=None):
    if not is_hate or is_hate == "non-hate":
        return "non-hate"

    # 基础关键词匹配
    hate_keywords = {
        "Region": ["外地人", "某省人", "某地人", "地域黑", "东北人", "河南人", "上海人", "北京人", "广东人"],
        "Racism": ["黑人", "白人", "种族", "民族", "回族", "维吾尔", "藏人", "飞周", "老黑", "黑鬼","黑马"],
        "Sexism": ["女人", "男人", "性别", "女权", "女拳", "男权", "娘们", "汉子婊", "打拳", "小仙女"],
        "LGBTQ": ["同性恋", "LGBT", "同志", "彩虹", "基佬", "拉拉", "txl", "通讯录", "艾滋病"],
        "Others": ["残疾", "智障", "傻子", "弱智", "肥宅", "死胖子", "丑八怪", "loser","飞舞"]
    }

    # 使用训练数据统计增强分类
    if train_data_stats:
        for group in train_data_stats["group_keywords"]:
            for kw in train_data_stats["group_keywords"][group]:
                if kw in text:
                    return group

    # 回退到基础关键词匹配
    for group, keywords in hate_keywords.items():
        for kw in keywords:
            if kw in text:
                return group
    return "Others"


# 从训练数据中提取统计信息
def extract_train_stats(train_data):
    stats = {
        "group_keywords": defaultdict(list),
        "hate_phrases": [],
        "non_hate_phrases": []
    }

    for item in train_data:
        content = item["content"]
        output = item.get("output", "")

        # 提取目标群体关键词
        if "|" in output:
            parts = output.split("|")
            if len(parts) >= 3:
                group = parts[2].strip()
                argument = parts[1].strip()
                if group != "non-hate" and argument != "NULL":
                    stats["group_keywords"][group].append(argument.split()[0])  # 取论点的第一个词作为关键词

        # 收集仇恨和非仇恨短语
        if "hate [END]" in output:
            stats["hate_phrases"].extend(re.findall(r'\w{2,}', content))
        else:
            stats["non_hate_phrases"].extend(re.findall(r'\w{2,}', content))

    # 去重并保留常见关键词
    for group in stats["group_keywords"]:
        counter = defaultdict(int)
        for kw in stats["group_keywords"][group]:
            counter[kw] += 1
        stats["group_keywords"][group] = [kw for kw, cnt in
                                          sorted(counter.items(), key=lambda x: -x[1])[:20]]  # 取前20个最常见关键词

    return stats


# 加载训练数据
def load_train_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    return train_data


train_data = load_train_data("train.json")
train_stats = extract_train_stats(train_data)

# 将训练数据中的标签映射到数字
label_map = {
    "Region": 0,
    "Racism": 1,
    "Sexism": 2,
    "LGBTQ": 3,
    "Others": 4
}

# 改进的训练数据处理，使用训练统计信息
for item in train_data:
    text = item["content"]
    # 优先使用训练数据中的标注
    if "output" in item and "|" in item["output"]:
        parts = item["output"].split("|")
        if len(parts) >= 4:
            label = parts[3].strip().split()[0]  # 提取hate/non-hate
            group = parts[2].strip() if label == "hate" else "non-hate"
    else:
        label = "hate" if any(phrase in text for phrase in train_stats["hate_phrases"]) else "non-hate"
        group = classify_target_group(text, label, train_stats)

    item["label"] = label_map.get(group, 4)  # Others作为默认


# 定义数据集类（保持不变）
class HateSpeechDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["content"]
        label = item["label"]

        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# 创建数据集
train_dataset = HateSpeechDataset(train_data, tokenizer)

# 训练参数（保持不变）
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 开始训练
trainer.train()

# 保存微调后的模型
model.save_pretrained("E:/roberta-finetuned")

# 加载测试数据（保持不变）
try:
    with open("test1.json", "r", encoding="utf-8") as f:
        try:
            test_data = json.load(f)
            if not isinstance(test_data, list):
                test_data = [test_data]
        except json.JSONDecodeError:
            f.seek(0)
            test_data = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        test_data.append(item)
                    except json.JSONDecodeError:
                        test_data.append({"content": line})
except FileNotFoundError:
    print("错误：test1.json 文件未找到")
    exit(1)
except Exception as e:
    print(f"读取文件时出错: {str(e)}")
    exit(1)


# 改进的仇恨言论判断（使用训练数据统计）
def is_hate_speech(text, train_stats):
    # 检查是否包含仇恨短语
    hate_words = set(train_stats["hate_phrases"])
    text_words = set(re.findall(r'\w{2,}', text))
    if hate_words & text_words:
        return True

    # 检查是否匹配仇恨模式
    hate_patterns = [
        r'[你您他她它].+?[死傻笨蠢丑贱]',
        r'[狗猪废物垃圾白痴]',
        r'[不配该死去死]'
    ]
    for pattern in hate_patterns:
        if re.search(pattern, text):
            return True
    return False


# 改进的预测函数
def predict_hate_speech_enhanced(text, train_stats):
    if not text or not isinstance(text, str):
        return "NULL | NULL | non-hate | non-hate [END]"

    # 使用训练数据增强的仇恨判断
    is_hate = is_hate_speech(text, train_stats)

    # 情感分析作为辅助
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        sentiment = torch.argmax(outputs.logits, dim=1).item()
    except:
        sentiment = 1

    # 结合两种判断
    final_hate = "hate" if is_hate or (sentiment == 0 and is_hate is not False) else "non-hate"
    targeted_group = classify_target_group(text, final_hate, train_stats)

    # 改进的目标和论点提取
    target = "NULL"
    argument = text[:50].replace("\n", " ").strip()

    # 从训练数据中学习的目标提取模式
    target_patterns = [
        r'[你您他她它这那].+?[，。！？]',
        r'[^，。！？]+?(狗|猪|废物|垃圾|白痴)[^，。！？]*',
        r'[^，。！？]+?(人|群体|分子)[^，。！？]*'
    ]

    for pattern in target_patterns:
        match = re.search(pattern, text)
        if match:
            target = match.group().strip('，。！？')
            argument_start = match.end()
            argument = text[argument_start:argument_start + 50].strip()
            break

    return f"{target} | {argument} | {targeted_group} | {final_hate} [END]"


# 处理测试集并保存结果
results = []
for item in tqdm(test_data, desc="Processing"):
    try:
        content = item.get("content", "")
        result = predict_hate_speech_enhanced(content, train_stats)
        results.append(result)
    except Exception as e:
        print(f"处理条目时出错: {str(e)}")
        results.append("NULL | NULL | non-hate | non-hate [END]")

# 确保有2000行结果
if len(results) < 2000:
    results.extend(["NULL | NULL | non-hate | non-hate [END]"] * (2000 - len(results)))
elif len(results) > 2000:
    results = results[:2000]

# 保存结果
with open("demo.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

print("处理完成，结果已保存到demo.txt")