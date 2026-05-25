# Giới thiệu method ViSoBERT

## 1. ViSoBERT là gì?

**ViSoBERT** là một mô hình ngôn ngữ được huấn luyện trước cho **văn bản mạng xã hội tiếng Việt**.

Mục tiêu của ViSoBERT là xử lý tốt các hiện tượng thường gặp trong bình luận mạng xã hội, ví dụ:

- emoji;
- teencode;
- từ viết tắt;
- từ không dấu;
- từ sai chính tả;
- cách viết kéo dài ký tự, ví dụ: `hayyyyy`, `cườiiii`;
- ngôn ngữ không trang trọng trong bình luận Facebook, TikTok, YouTube.

Khác với các mô hình tiếng Việt tổng quát như PhoBERT, ViSoBERT tập trung vào **social media text**. Đây là loại văn bản ngắn, nhiễu, không chuẩn và chứa nhiều tín hiệu cảm xúc.

---

## 2. Vấn đề mà ViSoBERT muốn giải quyết

Nhiều mô hình tiếng Việt trước đó được huấn luyện trên dữ liệu tương đối chuẩn, ví dụ Wikipedia, báo chí hoặc văn bản chính quy. Tuy nhiên, dữ liệu mạng xã hội có nhiều đặc điểm khác:

| Đặc điểm | Ví dụ | Khó khăn |
|---|---|---|
| Emoji | `😂`, `😭`, `😡` | Emoji có thể mang ý nghĩa cảm xúc mạnh |
| Teencode | `ko`, `k`, `iu`, `d4y` | Mô hình khó hiểu nếu chưa từng thấy dạng viết này |
| Không dấu | `toi buon qua` | Dễ gây nhập nhằng nghĩa |
| Viết sai chính tả | `cườiiii`, `đẹpppp` | Tokenizer có thể chia từ thành nhiều mảnh nhỏ |
| Từ lóng | `xỉu`, `hề`, `toang` | Nghĩa phụ thuộc vào ngữ cảnh mạng xã hội |

Vì vậy, ViSoBERT được xây dựng để học trực tiếp từ dữ liệu mạng xã hội tiếng Việt.

---

## 3. Pipeline tổng quát của method

```text
Vietnamese social media data
        ↓
Data cleaning
        ↓
Vietnamese social media tokenizer
        ↓
Pre-training with Masked Language Modeling
        ↓
Fine-tuning on downstream tasks
        ↓
Prediction / Evaluation
```

Nói đơn giản, method của ViSoBERT gồm hai giai đoạn chính:

1. **Pre-training**: cho mô hình học cách biểu diễn ngôn ngữ từ dữ liệu mạng xã hội lớn.
2. **Fine-tuning**: điều chỉnh mô hình cho từng bài toán cụ thể, ví dụ nhận diện cảm xúc hoặc phát hiện hate speech.

---

## 4. Dữ liệu pre-training

ViSoBERT được huấn luyện trên dữ liệu mạng xã hội tiếng Việt. Dữ liệu được thu thập từ các nền tảng phổ biến:

- Facebook;
- TikTok;
- YouTube.

Sau bước làm sạch, tập dữ liệu pre-training có khoảng **1GB văn bản thô**.

Trong quá trình tiền xử lý, nhóm tác giả loại bỏ các thành phần quá nhiễu, ví dụ:

- bình luận chỉ chứa link;
- bình luận spam lặp lại nhiều lần;
- bình luận vô nghĩa;
- bình luận chỉ chứa tài khoản người dùng.

Tuy nhiên, **emoji được giữ lại** vì emoji là tín hiệu quan trọng trong văn bản mạng xã hội.

---

## 5. Tokenizer riêng cho mạng xã hội tiếng Việt

Một đóng góp quan trọng của ViSoBERT là xây dựng **custom tokenizer** cho văn bản mạng xã hội tiếng Việt.

### 5.1. Tokenizer là gì?

Tokenizer là bộ chia văn bản thành các đơn vị nhỏ hơn để mô hình có thể xử lý.

Ví dụ:

```text
Tôi vui quá 😂
```

có thể được chia thành:

```text
["Tôi", "vui", "quá", "😂"]
```

### 5.2. Vì sao cần tokenizer riêng?

Văn bản mạng xã hội có nhiều từ không chuẩn. Nếu dùng tokenizer của mô hình huấn luyện trên văn bản chính quy, nhiều từ có thể bị chia thành các mảnh nhỏ khó hiểu.

Ví dụ:

```text
d4y l4 vj du cko mot cau teencode
```

Một tokenizer không phù hợp có thể chia câu này thành nhiều mảnh rất nhỏ. Khi đó, mô hình khó học được ý nghĩa thật của câu.

ViSoBERT dùng **SentencePiece** để học tokenizer trực tiếp từ dữ liệu mạng xã hội tiếng Việt. Nhờ đó, mô hình xử lý tốt hơn:

- emoji;
- teencode;
- từ viết sai;
- từ không dấu;
- từ lóng.

---

## 6. Kiến trúc mô hình

ViSoBERT dùng kiến trúc theo kiểu **XLM-R**.

Các thông tin chính:

| Thành phần | Mô tả |
|---|---|
| Kiến trúc nền | XLM-R style Transformer |
| Hidden size | 768 |
| Số self-attention layers | 12 |
| Số attention heads | 12 |
| Objective | Masked Language Modeling |

Điểm cần chú ý: ViSoBERT không phải là một kiến trúc hoàn toàn mới. Điểm mạnh chính của nó nằm ở **dữ liệu huấn luyện**, **tokenizer**, và **khả năng thích nghi với văn bản mạng xã hội tiếng Việt**.

---

## 7. Pre-training bằng Masked Language Modeling

ViSoBERT được huấn luyện bằng **Masked Language Modeling**.

### 7.1. Masked Language Modeling là gì?

Masked Language Modeling là phương pháp che một số token trong câu, sau đó yêu cầu mô hình đoán token bị che.

Ví dụ:

```text
Tôi rất [MASK] khi xem video này 😂
```

Mô hình cần đoán từ phù hợp, ví dụ:

```text
vui
```

Cách học này giúp mô hình hiểu ngữ cảnh xung quanh một từ.

### 7.2. Masking rate

Trong ViSoBERT, nhóm tác giả thử nghiệm nhiều mức masking rate khác nhau. Kết quả cho thấy mức **30%** phù hợp với nhiều bài toán mạng xã hội tiếng Việt hơn so với mức truyền thống thấp hơn.

---

## 8. Fine-tuning trên các bài toán cụ thể

Sau pre-training, ViSoBERT được fine-tune trên nhiều bài toán xử lý ngôn ngữ tự nhiên cho mạng xã hội tiếng Việt.

Các bài toán gồm:

| Task | Nghĩa tiếng Việt |
|---|---|
| Emotion Recognition | Nhận diện cảm xúc |
| Hate Speech Detection | Phát hiện ngôn từ thù ghét |
| Sentiment Analysis | Phân tích cảm xúc tích cực / tiêu cực |
| Spam Reviews Detection | Phát hiện đánh giá spam |
| Hate Speech Spans Detection | Xác định đoạn chứa ngôn từ thù ghét |

Ở giai đoạn fine-tuning, ViSoBERT được dùng như một encoder. Sau đó, một lớp phân loại được thêm vào để dự đoán nhãn của từng task.

---

## 9. Vai trò của ViSoBERT trong ViGoEmotions

Trong bài toán **ViGoEmotions**, ViSoBERT được dùng để phân loại cảm xúc chi tiết trên bình luận tiếng Việt.

Pipeline có thể hiểu như sau:

```text
Vietnamese comment
        ↓
ViSoBERT tokenizer
        ↓
ViSoBERT encoder
        ↓
Dropout layer
        ↓
Fully Connected layer
        ↓
Emotion labels
```

Vì ViGoEmotions là bài toán **multi-label classification**, một bình luận có thể có nhiều cảm xúc cùng lúc.

Ví dụ:

```text
"tội quá, tức thật"
```

có thể có nhiều nhãn:

```text
sadness = 1
anger = 1
```

ViSoBERT phù hợp với ViGoEmotions vì dữ liệu của ViGoEmotions cũng đến từ mạng xã hội và chứa nhiều emoji, từ lóng, teencode, cũng như cách viết không chuẩn.

---

## 10. Tóm tắt ngắn gọn

ViSoBERT là một method dựa trên ý tưởng **domain-specific pre-training**.

Nghĩa là mô hình không chỉ học tiếng Việt nói chung, mà học trực tiếp từ **văn bản mạng xã hội tiếng Việt**.

Các điểm chính của method:

1. Thu thập dữ liệu mạng xã hội tiếng Việt.
2. Làm sạch dữ liệu nhưng giữ emoji.
3. Xây tokenizer riêng cho social media text.
4. Pre-train mô hình bằng Masked Language Modeling.
5. Fine-tune mô hình trên các task mạng xã hội.
6. Đánh giá mô hình trên nhiều benchmark tiếng Việt.

Kết luận: ViSoBERT mạnh không phải vì thay đổi lớn về kiến trúc, mà vì nó được thiết kế đúng với miền dữ liệu mạng xã hội tiếng Việt.


## 11. Nguồn tham khảo

- Nguyen et al. (2023). *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing*. EMNLP 2023.
- Tran et al. (2026). *ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts*. EACL 2026.