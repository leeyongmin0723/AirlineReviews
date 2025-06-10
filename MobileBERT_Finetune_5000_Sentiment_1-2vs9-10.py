import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, AdamW
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 데이터 불러오기
df = pd.read_csv("data/AirlineReviews_EngClean_1-2vs9-10_5000.csv")
texts = df["Review"].astype(str).tolist()
labels = df["label"].tolist()

# 토큰화
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
tokens = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

# 데이터 분할
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    tokens['input_ids'], labels, test_size=0.2, random_state=42)
train_masks, val_masks = train_test_split(tokens['attention_mask'], test_size=0.2, random_state=42)

# 커스텀 Dataset 클래스
class ReviewDataset(Dataset):
    def __init__(self, inputs, masks, labels):
        self.inputs = inputs
        self.masks = masks
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx],
            "attention_mask": self.masks[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = ReviewDataset(train_inputs, train_masks, train_labels)
val_dataset = ReviewDataset(val_inputs, val_masks, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 모델 로드
model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)
model.to(device)

# 옵티마이저 & 스케줄러
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=len(train_loader) * 10)

# 학습 루프
for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} - Train Loss: {total_loss:.4f} - Train Acc: {correct / total:.4f}")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    print(f"Epoch {epoch+1} - Validation Acc: {val_correct / val_total:.4f}")

# 모델 저장
model.save_pretrained("./mobilebert_1-2vs9-10_5000_e10")
tokenizer.save_pretrained("./mobilebert_1-2vs9-10_5000_e10")
