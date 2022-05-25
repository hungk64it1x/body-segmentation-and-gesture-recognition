## BKAI NAVER CHALLENGE 2022
### BODY SEGMENTATION AND GESTURE RECOGNITION
------------------------------------------
### 1. CÀI ĐẶT PHẦN CỨNG VÀ MÔI TRƯỜNG
• PHẦN CỨNG: 1 x RTX 3090 GPU, 16 vCPUs, 24GB VRAM, 64GB RAM

• Ubuntu 18.04.5 LTS

• CUDA 11.2

• Python 3.9.7

*Sau đây là cách cài đặt môi trường và các packages cần thiết
```
$ conda create -n envs python=3.9.7
$ conda activate envs
$ conda install -c conda-forge gdcm
$ cd cls
$ pip install -r requirements.txt
$ cd segmentation
$ pip install -r requirements.txt 
```
-----------------------------------------------------------------------
### 2. CHUẨN BỊ DỮ LIỆU
#### 2.1 Extract các frames từ video gốc
- Tải 3 file .zip dataset của BTC bao gồm gesture_training_data.zip, private_test_data.zip, train_segment.zip và cho vào folder download

- Chuẩn bị toàn bộ data:
```
bash prepare.sh
```

- Tiếp theo đó
```
cd cls
bash extract_data.sh
```

2.2 Chia dữ liệu thành 5 folds cho phần body segmentation
- Cấu trúc thư mục sẽ như sau:
```
dataset
├── train
│   ├── images
│   ├── masks
└── test
    
```
Trước khi chia thành 5 folds chúng ta sẽ loại bỏ đi 1 ảnh bị lỗi do thiếu mask (BTC cung cấp thiếu)
```
cd segmentation/dataset
rm -r ./train/images/scratch_Subject1_Tuan1_cam-1-index_1.avi_1.jpg
```
Tiếp theo gõ lệnh sau để chia dataset thành 5 folds
```
cd ../utilizes
python split_5folds.py
```
BTC có thể mở file split_5folds trong thư mục segmentation/utilizes để tùy chỉnh:
+ seed: 42 (để có thể lặp lại thí nghiệm)
+ train_dir: chứa images và masks huấn luyện của BTC
+ folds_dir: Chứa dữ liệu sau khi chia

```
├── fold0
│   ├── images
│   ├── masks
└── fold1
    ├── images
    ├── masks
...
```
---------------------------------------
### 3. HUẤN LUYỆN MÔ HÌNH
#### 3.1 Task body segmentation

```
cd segmentation
bash train.sh
```
+ Chạy câu lệnh trên để huấn luyện 5 folds của mô hình và checkpoint sẽ được lưu trong thư mục bd-segmentation/checkpoint
+ Một số tham số để set file config như:
    + train_data_path: Mỗi một gạch đầu dòng là một thư mục chứa images và masks.
    + val_data_path" tương tự train nhưng là fold để valid
    + phần model["pretrained"] sẽ được để thành imagenet nó sẽ tự động tải pretrained weight về nếu BTC không đồng ý có thể xóa comment phần pretrained ở dưới cùng để đưa path của pretrained vào.
    + Ở mục test["checkpoint_dir"] thì là path sẽ lưu checkpoint.

+ Sau khi huấn luyện xong tiếp theo chúng ta sẽ sinh masks cho tập train của task gesture recoginition để tiền xử lý đầu vào (loại bỏ nền)
```
bash gen_masks_smp.sh
```
+ Nếu lúc chạy sinh nhãn có lỗi thì kiểm tra lại 2 file config gen-ges-test-mask.yaml và gen-ges-train-mask.yaml trong thư mục segmentation/configs và sửa lại đường dẫn theo hướng dẫn sau:
    + phase: train hoặc test (phải sinh cả 2 để làm đầu vào cho task còn lại)
    + train_data_path: dữ liệu train của task còn lại
    + test_data_path: dữ liệu test của task còn lại
    + save_train_data: thư mục lưu dữ liệu mới tạo ra (mặc định là cls/dataset/segment_dataset)
    + save_test_data: tương tự train...
+ Sinh file rle để nộp lên hệ thống:
```
bash gen_rle.sh
```
Sau khi chạy câu lệnh này thì trong mục segmentation/csv sẽ xuất hiện file sub1.csv, copy file này cho vào thư mục output
#### 3.2 Task gesture recognition
```
cd cls
```
Lần lượt thay tên các backbone trong file config.py bao gồm 2 backbone: dm_nfnet_f0, eca_nfnet_l1 và ứng với mỗi lần chạy:
```
bash train.sh
```
Mỗi lần chạy tương ứng với chạy 5 folds của mô hình, checkpoint sẽ được lưu vào thư mục checkpoint trong cls.

Sau khi có được checkpoint chúng ta sẽ infer:
```
bash infer.sh
```
file sub2.csv sẽ xuất hiện trong thư mục csv nằm trong mục cls copy sub2.csv vào thử mục output bên ngoài.

-----------------------------------
### 4. NỘP KẾT QUẢ LÊN HỆ THỐNG
#### Bây giờ trong thư mục output đã có 2 file sub1.csv và sub2.csv việc còn lại là chạy câu lệnh sau để có được kết quả cuối cùng:
```
bash submit.sh
```
Kết quả cuối cùng sẽ ở dạng csv (results.csv), nén nó thành định dạng results.zip và nộp lên hệ thống.



