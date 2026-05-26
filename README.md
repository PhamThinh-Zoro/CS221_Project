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