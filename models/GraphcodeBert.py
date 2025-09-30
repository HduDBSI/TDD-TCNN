# coding: UTF-8
"""
使用 GraphCodeBERT 进行方法级别技术债务 (TD) 检测
"""

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
import os


# -----------------------------
# 1. 数据集类
# -----------------------------
class TDDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data.iloc[idx]['method_code']
        label = self.data.iloc[idx]['label']
        encoded = self.tokenizer(
            code,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }


# -----------------------------
# 2. 主函数
# -----------------------------
def main(args):
    # 读取 CSV
    df = pd.read_csv(args.input_file)  # CSV 包含: project,file_name,method_code,label

    # 划分训练/验证
    train_df, val_df = train_test_split(
        df, test_size=args.val_ratio, random_state=42, stratify=df['label']
    )

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/graphcodebert-base",
        num_labels=1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # DataLoader
    train_dataset = TDDataset(train_df, tokenizer, max_length=args.max_length)
    val_dataset = TDDataset(val_df, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # -----------------------------
    # 训练循环
    # -----------------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(train_loader):.4f}")

    # -----------------------------
    # 验证集评估
    # -----------------------------
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1: {f1:.4f}")

    # 保存模型
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"模型已保存到 {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphCodeBERT for TD Detection")
    parser.add_argument("--input_file", type=str, default="../dataset/td-dataset.csv", help="输入CSV文件路径")
    parser.add_argument("--save_dir", type=str, default="saved_models/graphcodebert", help="保存模型目录")
    parser.add_argument("--batch_size", type=int, default=4, help="训练 batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="验证 batch size")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--val_ratio", type=float, default=0.4, help="验证集比例")
    args = parser.parse_args()

    main(args)
