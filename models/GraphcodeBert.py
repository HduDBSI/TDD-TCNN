import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score

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
        encoded = self.tokenizer(code,
                                 padding='max_length',
                                 truncation=True,
                                 max_length=self.max_length,
                                 return_tensors='pt')
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# -----------------------------
# 2. 读取 CSV 数据
# -----------------------------
df = pd.read_csv(r"F:\paper_experiment\TDD-TCNN-Prediction-github\TDD-TCNN\dataset\td-dataset.csv")  # CSV 包含 columns: project,file_name,method_code,label

# 划分训练集和验证集
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['label'])

# -----------------------------
# 3. 初始化 tokenizer 和模型
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/graphcodebert-base",
    num_labels=1  # 二分类任务
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# 4. DataLoader
# -----------------------------
train_dataset = TDDataset(train_df, tokenizer)
val_dataset = TDDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# -----------------------------
# 5. 训练循环
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)
        logits = outputs.logits
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# 6. 验证集评估
# -----------------------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].cpu().numpy()

        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask).logits
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
