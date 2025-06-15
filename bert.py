import json
import torch
import re
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import torch.optim as optim


# 自定义数据集类
class HateSpeechDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['content']

        # 对文本进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 解析输出标签
        output = item.get('output', 'NULL | NULL | Others | non-hate [END]')
        parts = [p.strip() for p in output.split('|')]
        if len(parts) >= 4:
            target_group = parts[2].strip()
            hateful = parts[3].strip().split()[0]
        else:
            target_group = 'Others'
            hateful = 'non-hate'

        # 将标签转换为数值
        group_map = {'Region': 0, 'Racism': 1, 'Gender': 2, 'LGBTQ': 3, 'Others': 4}
        hate_map = {'hate': 1, 'non-hate': 0}

        group_label = group_map.get(target_group, 4)
        hate_label = hate_map.get(hateful, 0)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'group_labels': torch.tensor(group_label, dtype=torch.long),
            'hate_labels': torch.tensor(hate_label, dtype=torch.long),
            'text': text,
            'target': parts[0] if len(parts) >= 4 else 'NULL',
            'argument': parts[1] if len(parts) >= 4 else 'NULL'
        }


# 改进的模型定义
class HateSpeechClassifier(torch.nn.Module):
    def __init__(self, model_path='E:\\bert\\bert-base-chinese-V1\\bert-base-chinese-V1'):
        super(HateSpeechClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = torch.nn.Dropout(0.2)
        hidden_size = self.bert.config.hidden_size

        # 群体分类器 (5类)
        self.group_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, 5))

        # 仇恨分类器 (二分类)
        self.hate_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, 1))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 使用[CLS] token的表示作为整个序列的表示
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        group_logits = self.group_classifier(pooled_output)
        hate_logits = self.hate_classifier(pooled_output)

        return group_logits, torch.sigmoid(hate_logits).squeeze()


# 训练函数
def train(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    group_correct = 0
    hate_correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="训练中"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        group_labels = batch['group_labels'].to(device)
        hate_labels = batch['hate_labels'].float().to(device)

        optimizer.zero_grad()

        group_logits, hate_probs = model(input_ids, attention_mask)

        # 计算群体分类损失
        group_loss = torch.nn.CrossEntropyLoss()(group_logits, group_labels)
        # 计算仇恨分类损失
        hate_loss = torch.nn.BCELoss()(hate_probs, hate_labels)
        # 总损失
        loss = group_loss + hate_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

        # 计算准确率
        _, group_preds = torch.max(group_logits, 1)
        group_correct += torch.sum(group_preds == group_labels).item()

        hate_preds = (hate_probs > 0.5).long()
        hate_correct += torch.sum(hate_preds == hate_labels.long()).item()

        total += len(group_labels)

    avg_loss = total_loss / len(dataloader)
    group_acc = group_correct / total
    hate_acc = hate_correct / total

    return avg_loss, group_acc, hate_acc


# 改进的评估函数
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    group_correct = 0
    hate_correct = 0
    total = 0

    all_group_preds = []
    all_group_labels = []
    all_hate_preds = []
    all_hate_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            group_labels = batch['group_labels'].to(device)
            hate_labels = batch['hate_labels'].float().to(device)

            group_logits, hate_probs = model(input_ids, attention_mask)

            # 计算损失
            group_loss = torch.nn.CrossEntropyLoss()(group_logits, group_labels)
            hate_loss = torch.nn.BCELoss()(hate_probs, hate_labels)
            loss = group_loss + hate_loss
            total_loss += loss.item()

            # 获取预测结果
            _, group_preds = torch.max(group_logits, 1)
            hate_preds = (hate_probs > 0.5).long()

            group_correct += torch.sum(group_preds == group_labels).item()
            hate_correct += torch.sum(hate_preds == hate_labels.long()).item()
            total += len(group_labels)

            all_group_preds.extend(group_preds.cpu().numpy())
            all_group_labels.extend(group_labels.cpu().numpy())
            all_hate_preds.extend(hate_preds.cpu().numpy())
            all_hate_labels.extend(hate_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    group_acc = group_correct / total
    hate_acc = hate_correct / total

    # 动态获取实际存在的类别
    unique_groups = sorted(set(all_group_labels))
    group_names = ['Region', 'Racism', 'Gender', 'LGBTQ', 'Others']
    present_group_names = [group_names[i] for i in unique_groups]

    print("\n群体分类报告:")
    print(classification_report(
        all_group_labels,
        all_group_preds,
        target_names=present_group_names,
        labels=unique_groups,
        zero_division=0
    ))

    print("\n仇恨分类报告:")
    print(classification_report(
        all_hate_labels,
        all_hate_preds,
        target_names=['non-hate', 'hate'],
        zero_division=0
    ))

    return avg_loss, group_acc, hate_acc


# 改进的预测函数，包含Target和Argument提取逻辑
def predict(model, dataloader, device):
    model.eval()
    predictions = []

    # 定义一些常见仇恨词汇模式
    hate_patterns = {
        'Racism': ['黑', '白皮', '黄皮', '种族', '民族','飞周'],
        'Region': ['地域', '地方人', '某省人', '某地人'],
        'Gender': ['女权', '男权', '女人', '男人', '性别','打拳'],
        'LGBTQ': ['同性恋', 'LGBT', '同志', '变性','gay']
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['text']

            group_logits, hate_probs = model(input_ids, attention_mask)

            # 获取预测结果
            group_preds = torch.argmax(group_logits, dim=1)
            hate_preds = (hate_probs > 0.5).long()

            for i in range(len(texts)):
                text = texts[i]
                group_idx = group_preds[i].item()
                hate_label = hate_preds[i].item()

                group_map = {0: 'Region', 1: 'Racism', 2: 'Gender', 3: 'LGBTQ', 4: 'Others'}
                hate_map = {0: 'non-hate', 1: 'hate'}

                group = group_map[group_idx]
                hate = hate_map[hate_label]

                # 改进的Target和Argument提取逻辑
                target = 'NULL'
                argument = 'NULL'

                # 如果预测为仇恨言论，尝试提取更精确的Target和Argument
                if hate == 'hate':
                    # 根据预测的群体类型查找相关关键词
                    patterns = hate_patterns.get(group, [])
                    for pattern in patterns:
                        if pattern in text:
                            # 简单的提取逻辑 - 实际应用中可以使用更复杂的NLP技术
                            start = max(0, text.index(pattern) - 10)
                            end = min(len(text), text.index(pattern) + len(pattern) + 10)
                            argument = text[start:end].strip()
                            target = pattern
                            break

                    # 如果没有找到特定模式，使用更通用的提取方法
                    if target == 'NULL':
                        # 提取名词短语作为潜在Target
                        words = re.findall(r'[\w]+', text)
                        if len(words) > 0:
                            target = words[0]
                            argument = ' '.join(words[1:3]) if len(words) > 2 else words[1] if len(
                                words) > 1 else 'NULL'
                else:
                    # 对于非仇恨言论，使用简单提取
                    words = text.split()[:2]
                    if len(words) > 0:
                        target = words[0]
                        argument = words[1] if len(words) > 1 else 'NULL'

                # 确保Target和Argument不为空
                target = target if target else 'NULL'
                argument = argument if argument else 'NULL'

                # 特殊处理：如果群体是Others但预测为hate，保持hate标签
                if group == 'Others' and hate == 'hate':
                    hate = 'hate'
                # 如果预测为non-hate，确保hate标签为non-hate
                elif hate == 'non-hate':
                    hate = 'non-hate'

                predictions.append(f"{target} | {argument} | {group} | {hate} [END]")

    return predictions


def main():
    # 参数设置
    BATCH_SIZE = 16
    EPOCHS = 1
    MAX_LEN = 128
    LEARNING_RATE = 3e-5
    MODEL_PATH = 'E:\\bert\\bert-base-chinese-V1\\bert-base-chinese-V1'

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    with open('train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open('test1.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = HateSpeechClassifier(MODEL_PATH).to(device)

    # 创建数据集和数据加载器
    train_dataset = HateSpeechDataset(train_data, tokenizer, MAX_LEN)
    test_dataset = HateSpeechDataset(test_data, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 训练循环
    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss, train_group_acc, train_hate_acc = train(model, train_loader, optimizer, device, scheduler)
        print(f"训练损失: {train_loss:.4f} | 群体准确率: {train_group_acc:.4f} | 仇恨准确率: {train_hate_acc:.4f}")

        val_loss, val_group_acc, val_hate_acc = evaluate(model, test_loader, device)
        print(f"验证损失: {val_loss:.4f} | 群体准确率: {val_group_acc:.4f} | 仇恨准确率: {val_hate_acc:.4f}")

        # 保存最佳模型
        if val_hate_acc > best_acc:
            best_acc = val_hate_acc
            torch.save(model.state_dict(), 'best_model1.bin')
            print("保存新的最佳模型")

    # 加载最佳模型进行预测
    print("\n加载最佳模型进行预测...")
    model.load_state_dict(torch.load('best_model1.bin'))
    predictions = predict(model, test_loader, device)

    # 保存预测结果
    with open('demo1.txt', 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(pred + '\n')

    print(f"\n预测结果已保存到demo1.txt，共{len(predictions)}条预测")


if __name__ == '__main__':
    main()