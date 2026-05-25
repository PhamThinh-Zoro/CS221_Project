# Nội dung đóng góp chính của method ViSoBERT

## 1. Tổng quan đóng góp

ViSoBERT đóng góp chính trong hướng nghiên cứu **Vietnamese social media NLP**.

Nói đơn giản, method này giải quyết một vấn đề quan trọng: nhiều mô hình tiếng Việt trước đó xử lý tốt văn bản chuẩn, nhưng chưa xử lý đủ tốt văn bản mạng xã hội. Văn bản mạng xã hội có nhiều emoji, teencode, từ không dấu, từ viết sai và từ lóng.

ViSoBERT đóng góp bằng cách xây dựng một mô hình ngôn ngữ được huấn luyện trực tiếp trên dữ liệu mạng xã hội tiếng Việt.

---

## 2. Đóng góp 1: Mô hình PLM cho mạng xã hội tiếng Việt

Đóng góp đầu tiên là giới thiệu **ViSoBERT**, một mô hình ngôn ngữ được huấn luyện trước cho **Vietnamese social media text processing**.

### Ý nghĩa

Trước ViSoBERT, nhiều mô hình tiếng Việt phổ biến như PhoBERT chủ yếu được huấn luyện trên văn bản chuẩn hơn, ví dụ Wikipedia hoặc tin tức. Những nguồn này không phản ánh đầy đủ cách người dùng viết trên mạng xã hội.

ViSoBERT tập trung vào dữ liệu từ:

- Facebook;
- TikTok;
- YouTube.

Nhờ đó, mô hình học tốt hơn các hiện tượng như:

- emoji;
- teencode;
- từ viết tắt;
- không dấu;
- từ sai chính tả;
- câu ngắn và không đầy đủ ngữ pháp.

### Giá trị nghiên cứu

ViSoBERT tạo ra một baseline mạnh cho các bài toán NLP tiếng Việt trên mạng xã hội.

---

## 3. Đóng góp 2: Tokenizer riêng cho social media text

Một đóng góp quan trọng là xây dựng **Vietnamese social media tokenizer**.

### Vì sao tokenizer quan trọng?

Tokenizer quyết định cách văn bản được chia thành token. Nếu tokenizer không phù hợp, mô hình sẽ khó hiểu dữ liệu.

Ví dụ, với teencode:

```text
d4y l4 vj du cko mot cau teencode
```

Một tokenizer không phù hợp có thể chia câu thành nhiều phần nhỏ không có nghĩa. Điều này làm mô hình khó học được biểu diễn tốt.

### Cách ViSoBERT giải quyết

ViSoBERT dùng **SentencePiece** để học tokenizer từ dữ liệu mạng xã hội tiếng Việt. Vì tokenizer được học từ đúng miền dữ liệu, nó xử lý tốt hơn:

- emoji;
- teencode;
- từ viết sai;
- từ không dấu;
- cách viết không chuẩn.

### Giá trị nghiên cứu

Đây không chỉ là cải thiện kỹ thuật nhỏ. Với social media NLP, tokenizer có ảnh hưởng lớn vì dữ liệu rất nhiễu và không chuẩn.

---

## 4. Đóng góp 3: Domain-specific pre-training

ViSoBERT áp dụng hướng **domain-specific pre-training**.

### Domain-specific pre-training là gì?

Đây là cách huấn luyện mô hình trên dữ liệu thuộc một miền cụ thể.

Trong ViSoBERT:

```text
Domain = Vietnamese social media text
```

Thay vì học chủ yếu từ văn bản chuẩn, mô hình học trực tiếp từ bình luận và bài đăng mạng xã hội.

### Ý nghĩa

Cách này giúp mô hình hiểu tốt hơn:

- cách diễn đạt cảm xúc trên mạng;
- ngôn ngữ không trang trọng;
- emoji như một phần của ngữ nghĩa;
- từ lóng và cách viết phổ biến của người dùng Việt Nam.

### Kết luận

Đóng góp này cho thấy dữ liệu pre-training phù hợp với miền ứng dụng có thể quan trọng không kém kiến trúc mô hình.

---

## 5. Đóng góp 4: Đạt SOTA trên nhiều task mạng xã hội tiếng Việt

ViSoBERT đạt kết quả SOTA trên nhiều bài toán Vietnamese social media NLP.

Các task được đánh giá gồm:

| Task | Mục tiêu |
|---|---|
| Emotion Recognition | Nhận diện cảm xúc |
| Hate Speech Detection | Phát hiện ngôn từ thù ghét |
| Sentiment Analysis | Phân tích cảm xúc |
| Spam Reviews Detection | Phát hiện review spam |
| Hate Speech Spans Detection | Xác định đoạn chứa hate speech |

### Ý nghĩa

Kết quả này chứng minh rằng ViSoBERT không chỉ tốt trên một bài toán riêng lẻ. Nó có khả năng tổng quát trên nhiều loại nhiệm vụ liên quan đến mạng xã hội tiếng Việt.

---

## 6. Đóng góp 5: Phân tích ảnh hưởng của emoji, teencode và dấu tiếng Việt

Bài báo không chỉ đưa ra mô hình, mà còn phân tích các yếu tố đặc trưng của mạng xã hội tiếng Việt.

Các yếu tố được phân tích gồm:

- emoji;
- teencode;
- dấu tiếng Việt;
- masking rate;
- khả năng trích xuất đặc trưng cho mô hình task-specific.

### Ý nghĩa

Phân tích này giúp hiểu rõ hơn vì sao ViSoBERT hoạt động tốt.

Ví dụ:

- Emoji không chỉ là ký tự phụ. Emoji có thể mang thông tin cảm xúc.
- Teencode không chỉ là lỗi viết. Nó là một dạng biến thể ngôn ngữ phổ biến trên mạng xã hội.
- Dấu tiếng Việt có ảnh hưởng lớn đến nghĩa của từ.

---

## 7. Đóng góp 6: Hữu ích cho ViGoEmotions

Trong paper ViGoEmotions, ViSoBERT là mô hình đạt kết quả tốt nhất khi giữ emoji gốc.

Kết quả tốt nhất trên tập test của ViGoEmotions:

| Model | Cách xử lý | Macro F1 | Weighted F1 |
|---|---|---:|---:|
| ViSoBERT | Giữ emoji gốc | 61.50% | 63.26% |

### Ý nghĩa

Điều này cho thấy ViSoBERT phù hợp với bài toán nhận diện cảm xúc chi tiết trên bình luận tiếng Việt.

Lý do là ViGoEmotions cũng có đặc điểm giống dữ liệu pre-training của ViSoBERT:

- nhiều bình luận mạng xã hội;
- nhiều emoji;
- nhiều cách viết không chuẩn;
- cảm xúc phụ thuộc vào ngữ cảnh ngắn.

---

## 8. Điểm khác biệt so với PhoBERT

| Tiêu chí | PhoBERT | ViSoBERT |
|---|---|---|
| Miền dữ liệu chính | Văn bản tiếng Việt tổng quát | Văn bản mạng xã hội tiếng Việt |
| Dữ liệu phù hợp | Tin tức, Wikipedia, văn bản chuẩn | Bình luận, bài đăng, ngôn ngữ không chuẩn |
| Emoji | Không phải trọng tâm chính | Được giữ và xử lý như tín hiệu quan trọng |
| Teencode | Không phải trọng tâm chính | Là một hiện tượng được quan tâm |
| Mục tiêu | NLP tiếng Việt tổng quát | NLP tiếng Việt trên mạng xã hội |

Kết luận: PhoBERT mạnh với văn bản chuẩn, còn ViSoBERT phù hợp hơn với dữ liệu mạng xã hội.

---

## 9. Đóng góp thực tiễn

ViSoBERT có thể được dùng cho nhiều ứng dụng thực tế:

1. Phân tích cảm xúc bình luận người dùng.
2. Phát hiện bình luận độc hại.
3. Phát hiện spam review.
4. Theo dõi phản ứng của cộng đồng mạng.
5. Hỗ trợ nghiên cứu về hành vi và cảm xúc trên mạng xã hội tiếng Việt.

---

## 10. Giới hạn của method

ViSoBERT vẫn có một số giới hạn:

1. Mô hình phụ thuộc vào chất lượng và độ mới của dữ liệu mạng xã hội.
2. Ngôn ngữ mạng xã hội thay đổi rất nhanh, nên mô hình cần được cập nhật theo thời gian.
3. ViSoBERT là mô hình base-size, không phải mô hình large-size.
4. Cần thêm phân tích để hiểu rõ chính xác yếu tố nào tạo ra thành công của mô hình.
5. Mô hình có thể gặp khó với các cảm xúc mơ hồ hoặc các câu có nhiều nghĩa.

---

## 11. Kết luận ngắn gọn

Đóng góp chính của ViSoBERT không nằm ở việc tạo ra một kiến trúc hoàn toàn mới. Đóng góp chính nằm ở việc đưa mô hình ngôn ngữ tiếng Việt vào đúng miền dữ liệu: **mạng xã hội tiếng Việt**.

Các điểm đóng góp quan trọng nhất:

1. Đề xuất PLM cho Vietnamese social media text.
2. Xây dựng tokenizer riêng cho mạng xã hội tiếng Việt.
3. Huấn luyện trên dữ liệu Facebook, TikTok và YouTube.
4. Đạt SOTA trên nhiều task mạng xã hội tiếng Việt.
5. Phân tích tác động của emoji, teencode và dấu tiếng Việt.
6. Cho thấy domain-specific pre-training rất quan trọng trong NLP tiếng Việt.

---

## 12. Nguồn tham khảo

- Nguyen et al. (2023). *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing*. EMNLP 2023.
- Tran et al. (2026). *ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts*. EACL 2026.