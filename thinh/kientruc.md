# 1 Kiến trúc mô hình

## Vấn đề
- Các PLMs tiếng Việt hoạt động chưa tốt trên dữ liệu mạng xã hội
- Dữ liệu social media chứa:
  - teencode
  - emojis
  - ngôn ngữ không chuẩn

---

## ViSoBERT

Mô hình Transformer theo kiến trúc XLM-R:

- 768 hidden units
- 12 attention layers
- 12 attention heads

### Training Objective
- Masked Language Modeling (MLM)

---

# 2 Vietnamese Social Media Tokenizer

## Ý tưởng
Xây dựng tokenizer riêng cho dữ liệu mạng xã hội tiếng Việt.

---

## SentencePiece

Sử dụng SentencePiece vì:

- Xử lý raw text tốt
- Giảm mất thông tin
- Phù hợp social media

---

## Ưu điểm của tokenizer

- Xử lý tốt:
  - teencode
  - emojis
  - từ viết tắt

- Tạo token ngắn và hiệu quả hơn
- Bao phủ dữ liệu tốt hơn các tokenizer cũ

---

## Kết quả

So với PhoBERT:

- Tokenization tốt hơn
- Hiểu văn bản informal tốt hơn
- Phù hợp social media hơn

![ViSoBERT Architecture](../architecture.png)

### Ông chỉ cần import tấm ảnh này là được khỏi phải ghi lại nội dung bên trên, mình sẽ nhìn hình để nói

### Phần fine tuning thì soạn slides đi

# 3 Fine-tuning

## Hyperparameters

- Batch size: 40
- Max token length: 128
- Learning rate: 2e-5
- Optimizer: AdamW
- Epsilon: 1e-8
- Số epoch: 10


- Không áp dụng preprocessing trên dataset

### Mục tiêu
Đánh giá khả năng xử lý raw text của PLM trên dữ liệu mạng xã hội

## script kiến trúc
Ở slide này là pipeline của ViSoBERT cho bài toán classification thông thường, tức là mỗi câu chỉ thuộc một nhãn duy nhất.

ViSoBERT được pre-train bằng Masked Language Modeling (MLM), tức là mô hình sẽ che ngẫu nhiên một số từ trong câu và học cách dự đoán lại chúng dựa trên ngữ cảnh xung quanh.
Cách học này giúp mô hình hiểu tốt ngữ nghĩa và ngữ cảnh tiếng Việt trước khi fine-tune cho downstream tasks.

Đầu tiên, đầu vào của hệ thống là một câu tiếng Việt, ví dụ ở đây là ‘hôm nay vui quá’.

Câu này sẽ được đưa qua bước Tokenizer và Encoding bằng SentencePiece.

Tokenizer sẽ tách câu thành các token và chuyển thành dạng số gọi là input_ids.

Ngoài ra, hệ thống còn tạo attention_mask để đánh dấu đâu là token thật và đâu là padding, giúp mô hình chỉ attention vào nội dung thực sự của câu.

Sau đó, dữ liệu được đưa vào ViSoBERT Encoder.

Mô hình gồm 12 attention layers, 12 attention heads và hidden size là 768 dimensions.

Các Transformer layers sẽ học ngữ cảnh và quan hệ giữa các từ trong câu để tạo semantic representation.

Sau khi encoder xử lý xong, ta lấy vector biểu diễn của token [CLS].

Đây là vector đại diện cho toàn bộ ý nghĩa của câu và có kích thước 768 chiều.

Tiếp theo, vector này được đưa qua Fully Connected Layer.

Layer này thực hiện phép biến đổi tuyến tính từ 768 chiều xuống còn 3 outputs, tương ứng với 3 lớp phân loại.

Kết quả đầu ra lúc này được gọi là logits.

Các logits sau đó đi qua hàm Softmax để chuyển thành xác suất cho từng lớp.

Ví dụ:

negative có xác suất p1
neutral có xác suất p2
positive có xác suất p3

Tổng các xác suất này sẽ bằng 1.

Cuối cùng, hệ thống sử dụng argmax để chọn lớp có xác suất cao nhất làm kết quả dự đoán cuối cùng.

Trong ví dụ này, lớp có xác suất cao nhất là positive, nên mô hình dự đoán câu mang cảm xúc tích cực.

Điểm khác biệt quan trọng ở pipeline này là sử dụng Softmax cho single-label classification, nghĩa là mỗi câu chỉ được chọn đúng một nhãn duy nhất

## script siêu tham số
Tiếp theo là phần thiết lập fine-tuning.

Về hyperparameters, nhóm sử dụng batch size là 40 và giới hạn chiều dài câu ở mức 128 tokens để cân bằng giữa tốc độ huấn luyện và khả năng giữ ngữ cảnh.

Learning rate được đặt là 2 nhân 10 mũ âm 5, đây là mức learning rate phổ biến khi fine-tune các pre-trained language models.

Optimizer được sử dụng là AdamW với epsilon bằng 1e-8 nhằm giúp quá trình cập nhật trọng số ổn định hơn.

Mô hình được train trong 10 epochs.

Một điểm quan trọng trong thí nghiệm này là nhóm không áp dụng preprocessing trên dataset.

Tức là dữ liệu được đưa trực tiếp dưới dạng raw text thay vì chuẩn hóa hay làm sạch trước.

Mục tiêu của cách tiếp cận này là đánh giá khả năng xử lý dữ liệu thực tế của pre-trained language model trên dữ liệu mạng xã hội tiếng Việt, nơi thường tồn tại nhiều nhiễu như viết tắt, typo hoặc ngôn ngữ không chuẩn.

Qua đó, nhóm muốn kiểm tra mức độ robust của ViSoBERT khi làm việc với dữ liệu ngoài thực tế.