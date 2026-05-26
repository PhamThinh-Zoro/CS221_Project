# Hướng dẫn Thực nghiệm: Tinh chỉnh (Fine-tuning) Mô hình ViSoBERT cho Bài toán Phân loại Cảm xúc trên Tập dữ liệu UIT-VSFC

Tài liệu này trình bày quy trình thực nghiệm chi tiết từng bước nhằm hướng dẫn sinh viên triển khai huấn luyện mô hình ngôn ngữ lớn ViSoBERT cho tác vụ phân loại cảm xúc văn bản tiếng Việt. Tập dữ liệu được sử dụng trong thực nghiệm này là UIT-VSFC, bao gồm các câu phản hồi của sinh viên được gán ba nhãn cảm xúc: **Negative (0)**, **Neutral (1)** và **Positive (2)**.

---

## Bước 1: Khởi tạo Môi trường và Khai báo Thư viện

Để tiến hành thực nghiệm, bước đầu tiên là thiết lập không gian làm việc và tải các thư viện máy học cốt lõi. Khuyến nghị thực thi mã nguồn trên môi trường có hỗ trợ bộ xử lý đồ họa (GPU) như Google Colab hoặc Kaggle để tối ưu hóa thời gian tính toán.

```python
# Cài đặt các gói phụ thuộc (Bỏ chú thích nếu chạy trên môi trường đám mây)
# !pip install torch transformers scikit-learn matplotlib seaborn tqdm

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
```

**Mô tả các thư viện:**
* `torch`: Nền tảng PyTorch hỗ trợ xây dựng và huấn luyện mạng nơ-ron.
* `transformers`: Cung cấp giao diện truy xuất các mô hình ngôn ngữ tiền huấn luyện (Pre-trained Models) từ nền tảng HuggingFace.
* `scikit-learn`: Cung cấp công cụ đo lường và đánh giá hiệu năng mô hình.
* `matplotlib` & `seaborn`: Hỗ trợ trực quan hóa kết quả thực nghiệm.

---

## Bước 2: Khai thác và Tiền xử lý Dữ liệu (Data Loading)

Đặc thù của tập dữ liệu UIT-VSFC được tổ chức dưới dạng các tệp văn bản thô (`.txt`). Chúng ta cần xây dựng một hàm đọc dữ liệu nhằm trích xuất đồng thời nội dung văn bản (`sents.txt`) và nhãn tương ứng (`sentiments.txt`).

```python
# Hàm nạp dữ liệu từ cấu trúc thư mục UIT-VSFC
def load_vsfc_data(folder_path):
    # Trích xuất đặc trưng văn bản
    with open(os.path.join(folder_path, 'sents.txt'), 'r', encoding='utf-8') as f:
        sents = [line.strip() for line in f.readlines()]
        
    # Trích xuất nhãn phân loại
    with open(os.path.join(folder_path, 'sentiments.txt'), 'r', encoding='utf-8') as f:
        labels = [int(line.strip()) for line in f.readlines()]
        
    return sents, labels

# Khai báo đường dẫn và phân tách thành ba tập: Huấn luyện, Kiểm định và Kiểm thử
train_texts, train_labels = load_vsfc_data('/kaggle/input/datasets/phamthinhzoro/uit-vsfc/new_data/train')
val_texts, val_labels = load_vsfc_data('/kaggle/input/datasets/phamthinhzoro/uit-vsfc/new_data/dev')
test_texts, test_labels = load_vsfc_data('/kaggle/input/datasets/phamthinhzoro/uit-vsfc/new_data/test')
```

---

## Bước 3: Đóng gói và Phân lô Dữ liệu (Dataset & Dataloader)

Nhằm đáp ứng không gian đầu vào của kiến trúc Transformer, các chuỗi văn bản cần được lượng hóa thành chuỗi token thông qua `Tokenizer`. Tiếp theo, dữ liệu được đóng gói vào đối tượng `DataLoader` để hỗ trợ cơ chế nạp theo lô (batch-processing), giúp tiết kiệm bộ nhớ và gia tốc phần cứng.

```python
class UIT_VSFC_Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len # Áp dụng kỹ thuật Padding/Truncation để đồng nhất kích thước

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Ánh xạ văn bản thành ma trận định danh (input_ids) và mặt nạ chú ý (attention_mask)
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

# Khởi tạo Dataloader với kích thước lô (batch_size) là 40, xáo trộn dữ liệu ở tập huấn luyện
train_dataset = UIT_VSFC_Dataset(train_texts, train_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)

val_dataset = UIT_VSFC_Dataset(val_texts, val_labels, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False)

test_dataset = UIT_VSFC_Dataset(test_texts, test_labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)
```

---

## Bước 4: Khởi tạo Kiến trúc Mô hình (Model Initialization)

Thực nghiệm sử dụng kiến trúc `uitnlp/visobert`, một mô hình tối ưu cho ngôn ngữ tự nhiên tiếng Việt trên mạng xã hội. Cấu hình mô hình được thiết lập với 3 lớp (neurons) tại tầng xuất (Output Layer) tương ứng với 3 nhãn cảm xúc.

```python
# Cấp phát tài nguyên tính toán (Kích hoạt GPU nếu khả dụng)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Môi trường phần cứng đang sử dụng: {device}")

model_name = "uitnlp/visobert" 
num_labels = 3 # Định nghĩa không gian nhãn đầu ra

# Tải bộ từ điển (Tokenizer) và trọng số tiền huấn luyện (Pre-trained Weights)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device) # Chuyển mô hình vào không gian bộ nhớ của thiết bị tính toán
```

---

## Bước 5: Cấu hình Hàm mục tiêu và Bộ tối ưu hóa (Optimizer & Scheduler)

Chúng ta triển khai thuật toán `AdamW` để cực tiểu hóa hàm mất mát (Cross-Entropy). Bộ lập lịch (Scheduler) được tích hợp để điều chỉnh tuyến tính tốc độ học (Learning Rate), kết hợp cơ chế làm nóng (Warmup) ở 10% số bước khởi đầu nhằm ổn định độ dốc (Gradients).

```python
epochs = 10 # Số chu kỳ huấn luyện (Epochs)
optimizer = AdamW(model.parameters(), lr=2e-5) # Khởi tạo tốc độ học ở ngưỡng 2e-5
total_steps = len(train_loader) * epochs
warm = int(0.1 * total_steps) # Tỷ lệ làm nóng (Warmup steps)

scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=warm, num_training_steps=total_steps
)
```

---

## Bước 6: Kỷ nguyên Huấn luyện và Đánh giá (Training & Validation Loop)

Quy trình học sâu được lặp lại qua các chu kỳ. Mỗi chu kỳ bao gồm hai pha:
1. **Pha Huấn luyện:** Tính toán giá trị gradient qua lan truyền ngược (Backpropagation) và cập nhật trọng số.
2. **Pha Kiểm định:** Đánh giá độ hội tụ trên tập Validation mà không thực hiện đạo hàm, nhằm giám sát hiện tượng quá khớp (Overfitting).

```python
best_val_loss = float('inf')
bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'

for epoch in range(epochs):
    print(f'\n{"="*20} Chu kỳ (Epoch) {epoch + 1}/{epochs} {"="*20}')
    
    # --- 1. Giai đoạn Huấn luyện (Training Phase) ---
    model.train() # Kích hoạt chế độ huấn luyện
    total_train_loss = 0
    
    train_loop = tqdm(train_loader, desc="Huấn luyện", bar_format=bar_format)
    for batch in train_loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad() # Thiết lập lại ma trận gradient
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward() # Lan truyền ngược
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Tránh nổ gradient (Gradient Clipping)
        optimizer.step()
        scheduler.step()
        
        total_train_loss += loss.item()
        train_loop.set_postfix(loss=f"{loss.item():.4f}")

    # --- 2. Giai đoạn Kiểm định (Validation Phase) ---
    model.eval() # Kích hoạt chế độ suy luận
    total_val_loss = 0
    
    with torch.no_grad(): # Tắt cơ chế đồ thị tính toán đạo hàm
        for batch in tqdm(val_loader, desc="Kiểm định", bar_format=bar_format):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Cơ chế lưu trữ mô hình (Model Checkpointing) dựa trên hàm mất mát tốt nhất
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth') 
        print(">>> Lưu thành công trọng số mô hình tối ưu (Best Model Checkpoint).")
```

---

## Bước 7: Suy luận và Báo cáo Hiệu năng (Model Inference & Evaluation)

Mô hình đạt trạng thái hội tụ tốt nhất (`best_model.pth`) sẽ được khôi phục để tiến hành quá trình suy luận độc lập trên tập kiểm thử ẩn (Test set). Kết quả đánh giá được thể hiện thông qua báo cáo phân loại tổng hợp.

```python
# Phục hồi trọng số và chuyển sang chế độ đánh giá
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_preds, test_true = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Kiểm thử (Testing)"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1) # Xác định lớp có độ phân phối xác suất cao nhất

        test_preds.extend(preds.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

# Khởi tạo ma trận báo cáo độ đo (Precision, Recall, F1-Score)
print("\n--- Báo cáo kết quả phân loại (Test Classification Report) ---")
target_names = ['0 - Negative', '1 - Neutral', '2 - Positive']
print(classification_report(test_true, test_preds, target_names=target_names, digits=4))
```

---

## Bước 8: Phân tích Lỗi Thông qua Trực quan hóa (Error Analysis & Visualization)

Cuối cùng, ma trận nhầm lẫn (Confusion Matrix) được khởi tạo nhằm định lượng sai số chéo giữa các lớp phân loại. Thao tác này hỗ trợ nhà nghiên cứu chẩn đoán hành vi thiên lệch (bias) của mô hình đối với các nhóm nhãn chiếm đa số.

```python
def plot_confusion_matrix(y_true, y_pred, classes):
    # Triển khai ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Sử dụng bản đồ nhiệt (Heatmap) để biểu diễn mật độ
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)')
    plt.xlabel('Nhãn Dự Đoán (Predicted Label)')
    plt.ylabel('Nhãn Thực Tế (True Label)')
    plt.tight_layout()
    plt.show()

# Thực thi trực quan hóa
plot_confusion_matrix(test_true, test_preds, target_names)
```