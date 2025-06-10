import pandas as pd
import torch
import numpy as np
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import classification_report

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 데이터 로드
df = pd.read_csv("AirlineReviews_EngClean_1-2vs9-10_remaining.csv")
df = df[df['Review'].notna()]  # 결측치 제거

texts = df['Review'].astype(str).tolist()
true_labels = df['label'].tolist()

# 모델과 토크나이저 로드
model_path = "./mobilebert_1-2vs9-10_5000_e10"
tokenizer = MobileBertTokenizer.from_pretrained(model_path)
model = MobileBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# 토크나이징
inputs = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 배치 처리
batch_size = 32
test_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, torch.tensor(true_labels))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Inference
all_preds = []
all_true = []

for batch in tqdm(test_loader, desc="Running Inference"):
    b_input_ids, b_mask, b_labels = [b.to(device) for b in batch]

    with torch.no_grad():
        outputs = model(input_ids=b_input_ids, attention_mask=b_mask)
        preds = torch.argmax(outputs.logits, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_true.extend(b_labels.cpu().numpy())

# 정확도 및 리포트 출력
accuracy = np.mean(np.array(all_preds) == np.array(all_true))
print(f"\n✅ 감정 분류 정확도: {accuracy:.4f}\n")

print("📊 분류 리포트:")
print(classification_report(all_true, all_preds, target_names=["부정", "긍정"]))
