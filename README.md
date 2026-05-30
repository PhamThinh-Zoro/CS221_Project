# CS231 Project Nhóm 3

## Cấu trúc dataset và code thực nghiệm

```text
├── Experiment/
│   ├── thongke-UIT-VSFC.ipynb
│   │      # Thống kê số nhãn trong dataset UIT-VSFC
│   │
│   ├── UIT-VSFC.ipynb
│   │      # Code thực nghiệm fine-tuning trên dataset UIT-VSFC
│   │
│   ├── VLSP2016.ipynb
│   │      # Code thực nghiệm fine-tuning trên dataset VLSP2016
│   │
│   ├── readme1.md
│   │      # Mô tả thực nghiệm fine-tune trên dataset gốc của paper (SA-VLSP2016)
│   │
│   └── readme2.md
│          # Mô tả thực nghiệm fine-tune trên dataset mới (UIT-VSFC)
│
├── vlsp2016/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
│
└── vsfc/
    ├── README.txt
    │      # Mô tả cấu trúc dataset UIT-VSFC
    │      # Bao gồm ý nghĩa các file:
    │      # sentiments.txt, sents.txt, topics.txt
    │
    ├── train/
    │   ├── sentiments.txt
    │   ├── sents.txt
    │   └── topics.txt
    │
    ├── dev/
    │   ├── sentiments.txt
    │   ├── sents.txt
    │   └── topics.txt
    │
    └── test/
        ├── sentiments.txt
        ├── sents.txt
        └── topics.txt
```

### Ghi chú

- `readme1.md` (thuộc folder `Experiment`) là file mô tả thực nghiệm fine-tune trên dataset gốc của paper (`SA-VLSP2016`).

- `readme2.md` (thuộc folder `Experiment`) là file mô tả thực nghiệm fine-tune trên dataset mới (`UIT-VSFC`).

- `README.txt` trong folder `vsfc` mô tả cấu trúc và ý nghĩa của dataset UIT-VSFC:
  - `sents.txt`: chứa nội dung văn bản.
  - `sentiments.txt`: chứa nhãn cảm xúc tương ứng.
  - `topics.txt`: chứa chủ đề của từng câu dữ liệu.

### Hướng dẫn sử dụng

- Sử dụng kaggle để chạy thực nghiệm.


## Mô hình ViSoBERT cho Bài toán Phân tích Cảm xúc trên tập dữ liệu SA-VLSP2016

- Import file `VLSP2016.ipynb` lên notebook của bạn trên kaggle.
- Import data [vlsp2016](https://www.kaggle.com/datasets/easterharry/vlsp-2016) vào notebook.
- Ấn run all để chạy thực nghiệm và in ra kết quả.

## Mô hình ViSoBERT cho Bài toán Phân loại Cảm xúc trên Tập dữ liệu UIT-VSFC

- Import file `UIT-VSFC.ipynb` lên notebook của bạn trên kaggle.
- Import data [vsfc](https://drive.google.com/drive/folders/1xclbjHHK58zk2X6iqbvMPS2rcy9y9E0X) vào notebook.
- Chỉnh lại path /kaggle/input/datasets/phamthinhzoro/uit-vsfc/new_data/train sao cho phù hợp với đường dẫn data input sao cho hợp lý.
- Ấn run all để chạy thực nghiệm và in ra kết quả.