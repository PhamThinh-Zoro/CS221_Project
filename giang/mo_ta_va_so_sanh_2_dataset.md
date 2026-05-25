# Trình bày dataset: ViGoEmotions, UIT-VSFC và so sánh hai dataset

## 1. Mục tiêu của tài liệu

Tài liệu này trình bày hai dataset tiếng Việt liên quan đến bài toán phân loại cảm xúc và cảm xúc thái độ trong văn bản.

- **ViGoEmotions**: dataset ban đầu được dùng trong paper về nhận diện cảm xúc chi tiết trên bình luận mạng xã hội tiếng Việt.
- **UIT-VSFC**: dataset mới dự kiến áp dụng method **ViSoBERT** để phân loại phản hồi sinh viên.
- **So sánh hai dataset**: làm rõ điểm giống nhau, khác nhau và lý do ViSoBERT có thể được áp dụng cho UIT-VSFC.

---

## 2. Dataset ban đầu: ViGoEmotions

### 2.1. Giới thiệu chung

**ViGoEmotions** là một benchmark dataset cho bài toán **fine-grained emotion detection** trên văn bản tiếng Việt.

Nói đơn giản, dataset này dùng để huấn luyện và đánh giá mô hình nhận diện nhiều loại cảm xúc khác nhau trong một câu hoặc một bình luận.

ViGoEmotions tập trung vào **bình luận mạng xã hội**, vì vậy dữ liệu thường có:

- emoji,
- teencode,
- từ viết tắt,
- cách viết không chuẩn,
- câu ngắn,
- cảm xúc ẩn trong ngữ cảnh.

Ví dụ:

```text
cần tìm gấp bạn trai đáng yêu như anh này trời đất ơi =))
```

Câu này có thể mang các cảm xúc như:

```text
amusement, desire
```

---

### 2.2. Nguồn dữ liệu

ViGoEmotions được xây dựng từ hai nguồn chính:

| Nguồn dữ liệu | Mô tả |
|---|---|
| UIT-VSMEC | Dataset cảm xúc tiếng Việt có sẵn, sau đó được gán nhãn lại theo hệ nhãn mới |
| Dữ liệu mạng xã hội mới | Bình luận từ Facebook, YouTube, Reddit, TikTok, Threads và X |

Tổng số dữ liệu cuối cùng là **20,664 bình luận tiếng Việt**.

---

### 2.3. Hệ nhãn của ViGoEmotions

ViGoEmotions sử dụng **27 nhãn cảm xúc chi tiết** và thêm một nhãn **neutral**.

Vì vậy, tổng số nhãn đầu ra là:

```text
27 emotion labels + 1 neutral label = 28 labels
```

Một số nhãn cảm xúc gồm:

| Nhóm | Ví dụ nhãn |
|---|---|
| Positive | joy, love, admiration, gratitude, optimism |
| Negative | sadness, anger, fear, disgust, disappointment |
| Ambiguous | curiosity, confusion, surprise, realization |
| Neutral | neutral |

---

### 2.4. Kiểu bài toán

ViGoEmotions là bài toán **multi-label classification**.

Điều này có nghĩa là một câu có thể có **nhiều nhãn cảm xúc cùng lúc**.

Ví dụ:

```text
Input:
tội cả con quá một lũ vô nhân tính

Output:
sadness = 1
grief = 1
anger = 1
```

Mô hình không chỉ chọn một cảm xúc duy nhất. Nó cần dự đoán tất cả cảm xúc phù hợp.

---

### 2.5. Quy trình gán nhãn

ViGoEmotions sử dụng quy trình **LLM + Human Annotation**.

Pipeline tổng quát:

```text
Raw social media comments
        ↓
Cleaning and filtering
        ↓
LLM annotation
        ↓
Human verification
        ↓
Final emotion labels
```

Các LLM được dùng để hỗ trợ gán nhãn gồm:

- Gemini 2.0 Flash,
- Meta-Llama-3-70B,
- Gemma 3.

Tuy nhiên, nhãn từ LLM không được dùng trực tiếp. Con người kiểm tra lại để đảm bảo chất lượng nhãn.

---

### 2.6. Method được áp dụng trên ViGoEmotions

Trong paper ViGoEmotions, nhiều mô hình Transformer được thử nghiệm, ví dụ:

- mBERT,
- XLM-R,
- PhoBERT,
- ViBERT,
- BARTpho,
- ViT5,
- CafeBERT,
- ViSoBERT.

Mô hình đạt kết quả tốt nhất là **ViSoBERT** khi giữ emoji ở dạng gốc.

Pipeline mô hình có thể hiểu như sau:

```text
Vietnamese social media comment
        ↓
ViSoBERT tokenizer
        ↓
ViSoBERT encoder
        ↓
Dropout layer
        ↓
Fully Connected layer
        ↓
28 emotion outputs
```

Vì đây là bài toán multi-label, loss function phù hợp là:

```text
Binary Cross-Entropy with Logits Loss
```

---

## 3. Dataset mới sẽ áp dụng method: UIT-VSFC

### 3.1. Giới thiệu chung

**UIT-VSFC** là viết tắt của **Vietnamese Students’ Feedback Corpus**.

Đây là dataset tiếng Việt gồm các câu phản hồi của sinh viên về hoạt động học tập và giảng dạy ở trường đại học.

Dataset này được xây dựng cho hai bài toán chính:

1. **Sentiment-based classification**  
   Phân loại sắc thái cảm xúc của câu phản hồi.

2. **Topic-based classification**  
   Phân loại chủ đề mà câu phản hồi đang nói đến.

---

### 3.2. Nguồn dữ liệu

Dữ liệu UIT-VSFC được thu thập từ phản hồi của sinh viên tại một trường đại học ở Việt Nam.

Các phản hồi được thu thập thông qua khảo sát cuối học kỳ.

Tổng số dữ liệu sau khi xử lý là **hơn 16,000 câu phản hồi tiếng Việt**.

---

### 3.3. Đặc điểm ngôn ngữ của UIT-VSFC

Phản hồi sinh viên thường là câu ngắn và có thể chứa:

- từ viết tắt,
- lỗi chính tả,
- biểu tượng cảm xúc,
- câu không hoàn chỉnh,
- cách viết tự nhiên của sinh viên.

Ví dụ từ paper:

```text
Giảng viên hướng dẫn tận tình và chu đáo.
```

Câu này có thể được gán:

```text
Sentiment: positive
Topic: Lecturer
```

---

### 3.4. Task 1: Sentiment-based classification

Trong task này, mỗi câu được phân loại vào một trong ba nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| Positive | Sinh viên hài lòng, khen hoặc đánh giá tốt |
| Negative | Sinh viên không hài lòng, phàn nàn hoặc yêu cầu cải thiện |
| Neutral | Câu không rõ cảm xúc hoặc không thể hiện ý kiến rõ ràng |

Ví dụ:

| Câu | Sentiment |
|---|---|
| Giảng viên hướng dẫn tận tình và chu đáo. | Positive |
| Nội dung môn học chưa đủ và chưa đúng với đề cương. | Negative |
| Em không có bất cứ một lời phê bình nào. | Neutral |

---

### 3.5. Task 2: Topic-based classification

Trong task này, mỗi câu được phân loại theo chủ đề.

UIT-VSFC có bốn nhãn topic chính:

| Nhãn | Ý nghĩa |
|---|---|
| Lecturer | Câu nói về giảng viên, phương pháp dạy, thái độ, kiến thức |
| Curriculum | Câu nói về môn học, bài tập, đề cương, điểm số, thời lượng học |
| Facility | Câu nói về cơ sở vật chất như máy tính, máy chiếu, phòng học |
| Others | Câu không thuộc ba nhóm trên hoặc không rõ chủ đề |

Ví dụ:

| Câu | Topic |
|---|---|
| Giảng viên hướng dẫn tận tình và chu đáo. | Lecturer |
| Nội dung môn học chưa đủ và chưa đúng với đề cương. | Curriculum |
| Nhà trường cần cải thiện hệ thống điện và máy chiếu. | Facility |
| Em không có bất cứ một lời phê bình nào. | Others |

---

### 3.6. Cách áp dụng ViSoBERT vào UIT-VSFC

Với UIT-VSFC, có thể áp dụng ViSoBERT theo hai hướng.

#### Hướng 1: Huấn luyện hai mô hình riêng

Một mô hình cho sentiment classification:

```text
Student feedback sentence
        ↓
ViSoBERT
        ↓
Classifier head
        ↓
Positive / Negative / Neutral
```

Một mô hình cho topic classification:

```text
Student feedback sentence
        ↓
ViSoBERT
        ↓
Classifier head
        ↓
Lecturer / Curriculum / Facility / Others
```

Cách này đơn giản, dễ cài đặt và dễ đánh giá.

---

#### Hướng 2: Multi-task learning

Một mô hình ViSoBERT dùng chung encoder, nhưng có hai classifier head.

```text
Student feedback sentence
        ↓
ViSoBERT shared encoder
        ↓
 ┌───────────────────────┬───────────────────────┐
 ↓                       ↓
Sentiment head           Topic head
 ↓                       ↓
Positive/Negative/Neutral Lecturer/Curriculum/Facility/Others
```

Cách này có thể giúp mô hình học mối quan hệ giữa **sentiment** và **topic**.

Ví dụ:

```text
Máy chiếu phòng học quá mờ.
```

Có thể có:

```text
Sentiment: Negative
Topic: Facility
```

---

## 4. So sánh ViGoEmotions và UIT-VSFC

### 4.1. Bảng so sánh tổng quan

| Tiêu chí | ViGoEmotions | UIT-VSFC |
|---|---|---|
| Ngôn ngữ | Tiếng Việt | Tiếng Việt |
| Domain | Mạng xã hội | Giáo dục |
| Loại văn bản | Bình luận mạng xã hội | Phản hồi sinh viên |
| Số lượng mẫu | 20,664 bình luận | Hơn 16,000 câu |
| Task chính | Fine-grained emotion detection | Sentiment classification và topic classification |
| Số nhãn | 27 cảm xúc + neutral | 3 sentiment labels + 4 topic labels |
| Kiểu nhãn | Multi-label | Single-label cho từng task |
| Mức độ cảm xúc | Chi tiết | Tổng quát hơn |
| Ví dụ cảm xúc | joy, anger, sadness, gratitude | positive, negative, neutral |
| Nhiễu ngôn ngữ | Cao | Trung bình |
| Có emoji / teencode | Có nhiều | Có thể có, nhưng ít hơn mạng xã hội |
| Method phù hợp | ViSoBERT multi-label classifier | ViSoBERT single-label hoặc multi-task classifier |

---

### 4.2. Điểm giống nhau

ViGoEmotions và UIT-VSFC có một số điểm giống nhau.

Thứ nhất, cả hai đều là dataset tiếng Việt. Điều này giúp ViSoBERT có cơ sở ngôn ngữ phù hợp để xử lý.

Thứ hai, cả hai đều liên quan đến cảm xúc hoặc thái độ của người viết. ViGoEmotions tập trung vào cảm xúc chi tiết, còn UIT-VSFC tập trung vào thái độ positive, negative và neutral.

Thứ ba, cả hai đều có văn bản ngắn. Điều này phù hợp với mô hình Transformer như ViSoBERT, vì mô hình có thể biểu diễn câu ngắn thành vector ngữ nghĩa hiệu quả.

Thứ tư, cả hai đều có thể chứa ngôn ngữ không hoàn toàn chuẩn. Ví dụ, người viết có thể dùng từ viết tắt, biểu tượng cảm xúc hoặc câu không đầy đủ.

---

### 4.3. Điểm khác nhau

Điểm khác biệt lớn nhất là **mức độ chi tiết của nhãn**.

ViGoEmotions có 27 loại cảm xúc chi tiết. Một câu có thể có nhiều cảm xúc cùng lúc. Vì vậy, đây là bài toán **multi-label classification**.

UIT-VSFC có nhãn sentiment tổng quát hơn, gồm positive, negative và neutral. Ngoài ra, dataset còn có nhãn topic gồm Lecturer, Curriculum, Facility và Others. Mỗi task thường được xem là **single-label classification**.

Khác biệt thứ hai là **domain**.

ViGoEmotions thuộc domain mạng xã hội. Dữ liệu có nhiều từ lóng, emoji và teencode. UIT-VSFC thuộc domain giáo dục. Dữ liệu thường nghiêm túc hơn và tập trung vào chất lượng giảng dạy, môn học hoặc cơ sở vật chất.

Khác biệt thứ ba là **mục tiêu ứng dụng**.

ViGoEmotions phù hợp cho các hệ thống hiểu cảm xúc xã hội, phân tích cộng đồng hoặc phát hiện nội dung có hại. UIT-VSFC phù hợp cho hệ thống phân tích phản hồi sinh viên, hỗ trợ nhà trường cải thiện giảng dạy và quản lý đào tạo.

---

## 5. Vì sao có thể áp dụng ViSoBERT từ ViGoEmotions sang UIT-VSFC?

ViSoBERT phù hợp với UIT-VSFC vì ba lý do chính.

### 5.1. Cả hai đều là văn bản tiếng Việt ngắn

ViSoBERT đã học cách biểu diễn văn bản tiếng Việt ngắn từ dữ liệu mạng xã hội. Phản hồi sinh viên trong UIT-VSFC cũng thường ngắn, nên mô hình có thể xử lý tốt kiểu dữ liệu này.

### 5.2. UIT-VSFC có hiện tượng ngôn ngữ không chuẩn

Paper UIT-VSFC cho biết phản hồi sinh viên có thể chứa từ viết tắt, lỗi chính tả, biểu tượng cảm xúc và các ký tự đặc biệt. Đây cũng là nhóm hiện tượng mà ViSoBERT được thiết kế để xử lý tốt.

### 5.3. Sentiment của UIT-VSFC gần với emotion của ViGoEmotions

ViGoEmotions học các cảm xúc chi tiết như joy, sadness, anger, disappointment. Các cảm xúc này có liên hệ với nhãn sentiment trong UIT-VSFC.

Ví dụ:

| Emotion trong ViGoEmotions | Có thể liên hệ với sentiment trong UIT-VSFC |
|---|---|
| joy, gratitude, admiration | Positive |
| anger, disappointment, sadness | Negative |
| neutral | Neutral |

Vì vậy, ViSoBERT có thể tận dụng biểu diễn cảm xúc đã học để hỗ trợ phân loại sentiment trong UIT-VSFC.

---

## 6. Vấn đề cần chú ý khi áp dụng method

Khi chuyển method từ ViGoEmotions sang UIT-VSFC, cần chú ý một số điểm.

### 6.1. Khác biệt domain

ViGoEmotions là mạng xã hội, còn UIT-VSFC là giáo dục. Vì vậy, mô hình cần được fine-tune lại trên UIT-VSFC. Không nên dùng trực tiếp mô hình từ ViGoEmotions mà không huấn luyện lại.

### 6.2. Khác biệt kiểu nhãn

ViGoEmotions là multi-label. UIT-VSFC thường là single-label cho từng task.

Do đó, cần thay đổi output layer và loss function.

| Dataset | Output layer | Loss function phù hợp |
|---|---|---|
| ViGoEmotions | 28 sigmoid outputs | Binary Cross-Entropy with Logits Loss |
| UIT-VSFC Sentiment | 3 softmax outputs | Cross-Entropy Loss |
| UIT-VSFC Topic | 4 softmax outputs | Cross-Entropy Loss |

### 6.3. Mất cân bằng dữ liệu

UIT-VSFC có thể mất cân bằng nhãn, đặc biệt là nhãn neutral và topic Others. Vì vậy, nên kiểm tra phân bố nhãn trước khi train.

Một số cách xử lý:

- dùng class weight,
- dùng macro F1 để đánh giá,
- dùng stratified split,
- phân tích confusion matrix.

---

## 7. Cách trình bày hướng nghiên cứu

Có thể trình bày hướng áp dụng method như sau:

> Nghiên cứu này áp dụng ViSoBERT, một pre-trained language model cho văn bản mạng xã hội tiếng Việt, vào bài toán phân loại phản hồi sinh viên trên UIT-VSFC. Trong khi ViGoEmotions kiểm tra khả năng nhận diện cảm xúc chi tiết trên bình luận mạng xã hội, UIT-VSFC cho phép đánh giá khả năng chuyển giao của ViSoBERT sang domain giáo dục. Mục tiêu là kiểm tra liệu khả năng xử lý văn bản ngắn, không chuẩn và giàu sắc thái cảm xúc của ViSoBERT có giúp cải thiện sentiment classification và topic classification trên phản hồi sinh viên hay không.

---

## 8. Kết luận

ViGoEmotions và UIT-VSFC đều là dataset tiếng Việt liên quan đến cảm xúc hoặc thái độ trong văn bản.

ViGoEmotions có nhãn cảm xúc chi tiết hơn và thuộc domain mạng xã hội. Dataset này phù hợp để đánh giá khả năng nhận diện nhiều cảm xúc cùng lúc.

UIT-VSFC có nhãn đơn giản hơn nhưng có hai hướng phân loại: sentiment và topic. Dataset này phù hợp để đánh giá khả năng ứng dụng mô hình vào domain giáo dục.

ViSoBERT có thể được áp dụng vào UIT-VSFC vì mô hình được thiết kế cho văn bản tiếng Việt ngắn, không chuẩn và có nhiều tín hiệu cảm xúc. Tuy nhiên, cần fine-tune lại mô hình và điều chỉnh output layer theo task của UIT-VSFC.

---

## 9. Từ vựng quan trọng

| Từ | Giải thích đơn giản | Nghĩa tiếng Việt |
|---|---|---|
| Dataset | A collection of data used for training or testing models | Bộ dữ liệu |
| Benchmark | A standard dataset used to compare models | Bộ chuẩn đánh giá |
| Domain | The field or context of data | Miền dữ liệu / lĩnh vực |
| Fine-grained | Very detailed | Chi tiết |
| Emotion detection | Finding emotions in text | Nhận diện cảm xúc |
| Sentiment classification | Classifying positive, negative, or neutral opinion | Phân loại sắc thái cảm xúc |
| Topic classification | Classifying what the text is about | Phân loại chủ đề |
| Multi-label | One sample can have many labels | Nhiều nhãn |
| Single-label | One sample has one label | Một nhãn |
| Fine-tune | Train a pre-trained model on a specific dataset | Huấn luyện tinh chỉnh |
| Output layer | The final layer that gives predictions | Lớp đầu ra |
| Loss function | A function that measures model error | Hàm mất mát |

---

## 10. Tài liệu nguồn

- Tran et al. (2026). **ViGoEmotions: A Benchmark Dataset For Fine-grained Emotion Detection on Vietnamese Texts**.
- Nguyen et al. (2018). **UIT-VSFC: Vietnamese Students’ Feedback Corpus for Sentiment Analysis**.
- Nguyen et al. (2023). **ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing**.