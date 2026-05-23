## 5.1 Chuẩn bị Dữ liệu (Corpus Preparation)

* **Phân chia tập dữ liệu:** Dữ liệu được chia thành 3 phần bao gồm tập huấn luyện (training set), tập phát triển (development set) và tập kiểm tra (test set) theo tỷ lệ 8:1:1.
* **Tiền xử lý cơ bản:** Trước khi đưa vào mô hình, dữ liệu phải trải qua quá trình tiền xử lý kỹ lưỡng để đảm bảo chất lượng và tính đồng nhất.
    * **Rút gọn dấu câu:** Các mẫu dấu câu kéo dài được tinh gọn (ví dụ: ":))))" chuyển thành ":))").
    * **Chuẩn hóa ký tự và emoji:** Các ký tự và emoji lặp lại liên tiếp được chuẩn hóa về dạng đơn (ví dụ: "cườiiiiii" chuyển thành "cười").
* **Tiền xử lý nâng cao (Tùy chọn):** Tùy thuộc vào từng kịch bản thử nghiệm, các biểu tượng cảm xúc (emoji), từ ngữ không trang trọng và teencode (ngôn ngữ mạng) sẽ được chuẩn hóa theo các cách khác nhau.

---

## 5.2 Thiết lập Thử nghiệm (Experimental Settings)

Mục đích của các thử nghiệm là đánh giá xem các chiến lược tiền xử lý văn bản và cấu hình dữ liệu khác nhau ảnh hưởng như thế nào đến hiệu suất của mô hình. Các thử nghiệm được tiến hành dựa trên 3 kịch bản tiền xử lý:

* **Kịch bản 1 (Scenario 1):** Giữ nguyên các emoji ở dạng gốc và áp dụng phương pháp chuẩn hóa dựa trên luật (rule-based normalization) để sửa teencode và các từ ngữ không trang trọng thông qua một từ điển được biên soạn thủ công.
* **Kịch bản 2 (Scenario 2):** Chuyển đổi emoji thành văn bản tiếng Việt mô tả (ví dụ: biểu tượng cảm xúc khóc được chuyển thành chữ "khóc"). Tương tự như kịch bản 1, phương pháp chuẩn hóa dựa trên luật thủ công cũng được áp dụng cho teencode và lỗi chính tả.
* **Kịch bản 3 (Scenario 3):** Thay thế bước chuẩn hóa thủ công bằng **ViSoLex** - một hệ thống chuẩn hóa từ vựng dựa trên mô hình học máy. ViSoLex sử dụng kiến trúc ViSOBERT để dự đoán dạng chuẩn xác của các từ viết sai trên mạng xã hội (ví dụ: "dung v" thành "Đúng vậy."). Ở kịch bản này, emoji được giữ nguyên, mô hình chỉ tự động chuẩn hóa các nhiễu văn bản.

**Cấu hình Huấn luyện và Mô hình:**

* **Đánh giá mô hình:** 8 mô hình được lựa chọn sẽ được huấn luyện qua cả 3 kịch bản trên nhằm tìm ra chiến lược tiền xử lý tối ưu nhất cho tập dữ liệu.
* **Kiến trúc bổ sung để tối ưu:**
    * **Lớp Dropout:** Bổ sung một lớp Dropout với tỷ lệ 0.2 để tắt ngẫu nhiên 20% các node trong mạng khi huấn luyện, giúp mô hình tăng độ tráng kiện và tránh tình trạng học vẹt (overfitting).
    * **Lớp Fully Connected (FC):** Thêm một lớp kết nối đầy đủ ở đầu ra với số lượng node tương ứng với 28 nhãn phân loại mục tiêu.
* **Thông số huấn luyện:**
    * Mô hình được huấn luyện trong vòng 12 epochs (vòng lặp) bằng trình tối ưu hóa AdamW.
    * Tỷ lệ học (Learning rate) ban đầu được đặt ở mức 0.00005.
    * Lịch trình tỷ lệ học: Tăng dần trong pha khởi động (ở epoch đầu tiên), sau đó giảm tuyến tính dần về 0 cho đến khi kết thúc quá trình huấn luyện.
* **Hàm mất mát (Loss function):** Sử dụng hàm *Binary Cross-Entropy with Logits* để xử lý bài toán phân loại đa nhãn (multi-label), nơi mỗi nhãn cảm xúc được mô hình dự đoán một cách độc lập.
* **Xử lý mất cân bằng dữ liệu:** Để giải quyết vấn đề có nhãn xuất hiện nhiều, nhãn xuất hiện ít, trọng số lớp dương (pos_weight) được tính cho từng nhãn để giúp mô hình chú trọng hơn vào việc học các nhóm cảm xúc thiểu số.
* **Chỉ số đánh giá:** Nghiên cứu sử dụng hai chỉ số *Macro F1-score* và *Weighted F1-score* để đem lại cái nhìn phân tích toàn diện về hiệu suất của mô hình trên cả các lớp cảm xúc lớn và nhỏ.

# Kiến trúc Mô hình Phân loại Cảm xúc Đa nhãn (Multilabel Sentiment Classifier)

## 1. Tổng quan Mô hình (Model Overview)
- **Kiến trúc lõi:** Dựa trên các mô hình ngôn ngữ lớn họ BERT (Pre-trained Language Models).
- **Mô hình pre-trained cụ thể:** Hỗ trợ linh hoạt chuyển đổi giữa `ViSoBERT` (mặc định đang cấu hình), `PhoBERT-base-v2`, `CafeBERT`, và `viBERT`.
- **Nhiệm vụ chính:** Phân loại văn bản đa nhãn (Multilabel Text Classification) để nhận diện 28 loại cảm xúc khác nhau trên cùng một câu.

## 2. Các thành phần của Mạng Neural (Network Architecture)
Kiến trúc luồng truyền xuôi (Forward Pass) đi qua các lớp sau:

* **Lớp đầu vào (Input Layer):** - Tiếp nhận văn bản đã được làm sạch và mã hóa bởi `XLMRobertaTokenizer`.
  - Các tensor đầu vào bao gồm `input_ids` (mã thông báo) và `attention_mask` (mặt nạ chú ý).

* **Khối Mã hóa (Pre-trained BERT Encoder):**
  - Chứa 12 lớp Transformer (XLMRobertaLayer), mỗi lớp sử dụng cơ chế Self-Attention (`XLMRobertaSdpaSelfAttention`) để nắm bắt ngữ cảnh từ vựng. 
  - Kích thước vector đặc trưng ẩn (hidden size) là 768.
  - Có áp dụng xác suất dropout nội bộ là 0.1 cho hidden state và attention probs để kiểm soát nhiễu.

* **Lớp chống học vẹt (Dropout Layer):**
  - Đặc trưng đầu ra (output) từ lớp BERT được đi qua một lớp `nn.Dropout(p=0.2)`.
  - Tác dụng: Ngắt ngẫu nhiên 20% kết nối của các nơ-ron nhằm giảm thiểu tình trạng Overfitting trong quá trình fine-tuning.

* **Lớp phân loại tuyến tính (Fully Connected / Linear Layer):**
  - `nn.Linear(hidden_size, n_classes)`: Chuyển đổi vector biểu diễn kích thước 768 chiều xuống một không gian vector 28 chiều (tương ứng với 28 nhãn cảm xúc).
  - Trả về `logits` (giá trị thô chưa qua hàm kích hoạt).

## 3. Quá trình Tối ưu hóa và Đánh giá (Optimization & Evaluation)
- **Hàm mất mát (Loss Function):** Sử dụng `BCEWithLogitsLoss` (Binary Cross Entropy with Logits) thay vì CrossEntropy. Đây là hàm chuẩn chuẩn chỉnh cho bài toán **đa nhãn** (multilabel classification), nơi một câu có thể thuộc nhiều nhãn.
- **Xử lý mất cân bằng dữ liệu:** Trọng số `pos_weight` được tích hợp vào hàm Loss để cân bằng tỷ lệ mẫu (tăng phạt khi dự đoán sai ở các nhãn thiểu số).
- **Suy luận (Inference):** Tại bước dự đoán, `logits` được đi qua hàm kích hoạt `Sigmoid` và ngưỡng (threshold) `0.5` được thiết lập để quyết định trạng thái xuất hiện của từng cảm xúc một cách độc lập.

