# Hướng dẫn Tinh chỉnh (Fine-tuning) Mô hình ViSoBERT cho Bài toán Phân tích Cảm xúc trên tập dữ liệu SA-VLSP2016

Tài liệu này cung cấp hướng dẫn chi tiết quy trình tinh chỉnh mô hình ngôn ngữ lớn ViSoBERT nhằm thực hiện tác vụ phân loại cảm xúc văn bản tiếng Việt. Mô hình được huấn luyện để phân loại văn bản đầu vào thành ba nhãn cảm xúc: **Tiêu cực (Negative)**, **Trung lập (Neutral)** và **Tích cực (Positive)**.

---

## Bước 1: Khởi tạo Môi trường và Khai báo Thư viện

Bước đầu tiên yêu cầu thiết lập môi trường lập trình và tải các thư viện máy học cần thiết phục vụ cho quá trình xử lý dữ liệu, xây dựng kiến trúc mô hình và đánh giá kết quả thực nghiệm.

```python
# Cài đặt các thư viện phụ thuộc (Bỏ chú thích nếu thực thi trên môi trường Google Colab/Kaggle)
# !pip install pandas torch transformers scikit-learn matplotlib seaborn tqdm

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
```

**Đặc tả các thư viện:**
* `torch`: Framework PyTorch hỗ trợ xây dựng, tính toán tensor và huấn luyện mạng nơ-ron nhân tạo.
* `transformers`: Thư viện cung cấp kiến trúc và bộ trọng số của các mô hình ngôn ngữ tiền huấn luyện (Pre-trained Language Models) từ HuggingFace.
* `pandas`: Thư viện hỗ trợ thao tác, tiền xử lý và phân tích dữ liệu dạng bảng.
* `matplotlib` & `seaborn`: Các công cụ hỗ trợ trực quan hóa dữ liệu và xuất biểu đồ đánh giá.
* `scikit-learn`: Thư viện cung cấp các hàm đo lường và đánh giá hiệu năng mô hình học máy (metrics).

---

## Bước 2: Tải và Tiền xử lý Dữ liệu (Data Loading & Preprocessing)

Quy trình nạp dữ liệu bao gồm việc đọc các tập tin chứa dữ liệu thực nghiệm, loại bỏ các bản ghi khuyết thiếu và chuyển đổi nhãn phân loại văn bản sang định dạng số học (mã hóa nhãn).

```python
# Ánh xạ nhãn phân loại từ định dạng chuỗi văn bản sang giá trị số nguyên
label_to_idx = {
    "NEG": 0,  # Negative - Tiêu cực
    "NEU": 1,  # Neutral - Trung lập
    "POS": 2   # Positive - Tích cực
}

def load_sentiment_data(csv_path):
    df = pd.read_csv(csv_path)                 # Tải dữ liệu từ định dạng CSV
    df = df.dropna(subset=["texts", "labels"]) # Loại bỏ các bản ghi chứa giá trị rỗng (NaN)
    texts = df["texts"].astype(str).tolist()   # Trích xuất đặc trưng văn bản đầu vào
    labels = [label_to_idx[label] for label in df["labels"]] # Chuyển đổi nhãn rời rạc thành số
    return texts, labels

# Khai báo cấu trúc đường dẫn thư mục chứa tập dữ liệu
DATA_PATH = "/kaggle/input/datasets/easterharry/vlsp-2016" 

# Phân tách bộ dữ liệu thành ba tập con: Huấn luyện (Train), Kiểm định (Dev/Val) và Kiểm thử (Test)
train_texts, train_labels = load_sentiment_data(os.path.join(DATA_PATH, "train.csv"))
val_texts, val_labels = load_sentiment_data(os.path.join(DATA_PATH, "dev.csv"))
test_texts, test_labels = load_sentiment_data(os.path.join(DATA_PATH, "test.csv"))
```

---

## Bước 3: Chuẩn bị Cấu trúc Dữ liệu (Dataset & Dataloader)

Nhằm đáp ứng yêu cầu đầu vào của mô hình Transformer, các chuỗi ký tự thô cần được phân giải thành chuỗi token thông qua bộ Tokenizer tương ứng. Dữ liệu sau đó được đóng gói thành các lô (batch) thông qua `DataLoader` để tối ưu hóa quá trình tính toán trên GPU.

```python
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len # Giới hạn chiều dài chuỗi token (áp dụng kỹ thuật padding/truncation)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Biến đổi văn bản thành biểu diễn ma trận (input_ids, attention_mask)
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Khởi tạo Dataloader với kích thước lô (batch_size) là 40
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True) # Áp dụng cơ chế xáo trộn cho tập huấn luyện

val_dataset = TextDataset(val_texts, val_labels, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False)

test_dataset = TextDataset(test_texts, test_labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)
```

---

## Bước 4: Khởi tạo Kiến trúc Mô hình (Model Initialization)

Thực nghiệm sử dụng kiến trúc ViSoBERT, một mô hình được tiền huấn luyện trên quy mô lớn ngữ liệu tiếng Việt. Mô hình được tải cấu hình phục vụ bài toán Sequence Classification với không gian nhãn đầu ra là 3.

```python
# Xác định thiết bị tính toán phần cứng (Khuyến nghị sử dụng GPU để gia tốc quá trình huấn luyện)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Môi trường thực thi hiện tại: {device}")

model_name = "uitnlp/visobert" 
num_labels = 3 # Số lượng lớp phân loại (Negative, Neutral, Positive)

# Nạp Tokenizer và toàn bộ trọng số mô hình từ nguồn HuggingFace
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device) # Cấp phát bộ nhớ mô hình lên thiết bị tính toán
```

---

## Bước 5: Cấu hình Thuật toán Tối ưu hóa (Optimizer & Scheduler)

Thiết lập thuật toán AdamW nhằm cực tiểu hóa hàm mục tiêu. Kết hợp sử dụng bộ lập lịch suy giảm tuyến tính (Linear Scheduler) để điều phối và kiểm soát động tốc độ học (Learning Rate).

```python
epochs = 10 # Số lượng chu kỳ huấn luyện toàn bộ tập dữ liệu (Epochs)
optimizer = AdamW(model.parameters(), lr=2e-5) # Tham số hóa tốc độ học khởi tạo
total_steps = len(train_loader) * epochs
warm = int(0.1 * total_steps) # Tỷ lệ số bước làm nóng (Warmup steps) ở pha đầu
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warm, num_training_steps=total_steps)
```

---

## Bước 6: Quy trình Huấn luyện và Kiểm định (Training & Validation Loop)

Quy trình lặp qua các epochs, chia làm hai pha độc lập. Pha huấn luyện sử dụng cơ chế lan truyền ngược (Backpropagation) để cập nhật trọng số. Pha kiểm định thực hiện đo lường hàm mất mát trên tập dữ liệu Validation nhằm ngăn chặn hiện tượng quá khớp (Overfitting).

```python
best_val_loss = float('inf')

for epoch in range(epochs):
    print(f'\n=== Kỷ nguyên huấn luyện (Epoch) {epoch + 1}/{epochs} ===')
    
    # --- 1. Pha Huấn luyện (Training Phase) ---
    model.train() # Kích hoạt chế độ cập nhật gradient (bật Dropout/BatchNorm)
    total_train_loss = 0
    
    for batch in tqdm(train_loader, desc="Tiến trình Huấn luyện"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad() # Khởi tạo lại gradient về 0
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss # Đo lường hàm mất mát (Cross-Entropy Loss)
        
        loss.backward() # Lan truyền ngược để tính toán ma trận Jacobian (Gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Chuẩn hóa gradient để tránh Exploding Gradients
        optimizer.step() # Cập nhật trọng số theo thuật toán tối ưu
        scheduler.step() # Cập nhật tham số tốc độ học
        
        total_train_loss += loss.item()

    # --- 2. Pha Kiểm định (Validation Phase) ---
    model.eval() # Vô hiệu hóa quá trình tính toán đạo hàm nhằm tối ưu không gian bộ nhớ
    total_val_loss = 0
    
    with torch.no_grad(): 
        for batch in tqdm(val_loader, desc="Tiến trình Kiểm định"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Chiến lược lưu trữ trọng số mô hình có hàm mất mát kiểm định (Validation Loss) tối ưu nhất
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth') 
        print(">>> Lưu thành công trọng số cấu hình tối ưu (Best Model Checkpoint).")
```

---

## Bước 7: Đánh giá Hiệu năng Tổng thể (Model Evaluation)

Quá trình suy luận (Inference) được thực hiện trên tập Test ẩn (Unseen data) bằng trọng số tối ưu nhất để phản ánh chính xác khả năng khái quát hóa của mô hình.

```python
model.load_state_dict(torch.load('best_model.pth')) # Nạp checkpoint mô hình tối ưu
model.eval()

test_preds, test_true = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Đánh giá Kiểm thử"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1) # Xác suất lớp cực đại thông qua Argmax

        test_preds.extend(preds.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

# Báo cáo các chỉ số định lượng bao gồm Precision, Recall, F1-Score và Accuracy
target_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
print(classification_report(test_true, test_preds, target_names=target_names, digits=4))
```

---

## Bước 8: Trực quan hóa Phân tích Lỗi (Error Analysis & Visualization)

Xây dựng Ma trận nhầm lẫn (Confusion Matrix) để định tính và định lượng khả năng phân tách giữa các biên quyết định của mô hình, đặc biệt là quan sát độ lệch phân phối giữa các lớp nhãn.

```python
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Triển khai biểu đồ dạng Heatmap biểu diễn mật độ dự đoán
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)')
    plt.xlabel('Nhãn Dự Đoán (Predicted Label)')
    plt.ylabel('Nhãn Thực Tế (Ground Truth)')
    plt.show()

# Thực thi hàm trực quan hóa
plot_confusion_matrix(test_true, test_preds, target_names)
```