# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603069 - Lã Thái Hoà

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Phân loại dựa trên cơ cấu truyền động: có bàn đạp và xích đạp $\rightarrow$ `bicycle`; kiểu dáng scooter, sàn để chân phẳng, không có bàn đạp $\rightarrow$ `motorbike`. | `img_004.jpg` | `CAL-02` |
| 2 | Khối xe máy đỗ dày đặc ở cự ly gần và xa: từng xe hay `ignore_region`? | Xe nhìn thấy rõ $\ge 20\%$ diện tích và có thể phân biệt đầu/đuôi/yên $\rightarrow$ gán từng box `motorbike` riêng. Cụm xe nền xa, chồng lấn không tách rời hoặc chiều cao $< 20\text{ px}$ $\rightarrow$ gom bằng 1 khung `ignore_region`. | `img_018.jpg` | `CAL-03` |
| 3 | Hàng hóa nhô ra khỏi xe: box xe có gồm hàng không? Mask thì sao? | Giỏ hàng, giá đèo hàng, thùng xe gắn cố định thuộc về xe. Hàng hóa cồng kềnh (tre, rổ, thùng carton) nhô ra quá $20\%$ chiều dài xe $\rightarrow$ không tính vào box xe. Với mask: chỉ bao trọn khung xe kim loại và bánh xe, loại trừ hàng hóa nhô ra. | `img_017.jpg` | `CAL-01` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_013.jpg` | Xe tuk-tuk chở khách có mui sắt hở nền | YOLO26 gọi là `car`; YOLOE sinh 2 box chồng đè `three_wheeler` (IoU $\ge 0.6$) | Ưu tiên gán `three_wheeler`, xóa 1 box trùng; tách người lái thành box `person` riêng. | Mục 2 & Mục 5: Xe ba bánh tuk-tuk/xích lô luôn là `three_wheeler`, người lái luôn tách riêng. |
| `img_004.jpg` | Xe đạp điện dựng cạnh người bán hàng | YOLO26 gọi là `bicycle`, conf thấp | Kiểm tra có bàn đạp xích $\rightarrow$ giữ nguyên `bicycle`, để trống `conf`. | Mục 2 & Mục 6: Tiêu chuẩn phân loại xe đạp điện có bàn đạp. |
| `img_018.jpg` | Hàng xe máy đỗ san sát nhau trên phố chợ | YOLO26 vẽ hàng loạt box nhỏ li ti $< 20\text{ px}$ | Xóa các box rời rạc $< 20\text{ px}$, vẽ 1 khung `ignore_region` bao quanh cụm xe xa. | Mục 4.1.6: Quy tắc gom cụm phương tiện nhỏ ở nền xa. |
| `img_023.jpg` | Khung bao chữ nhật bị nới rộng hở nền (bẫy loose lọt) | Máy sinh box `loose` nuốt cả vỉa hè | Nhận diện lỗi thiên kiến tự động hóa; siết lại quy tắc kéo khít 4 cạnh viền chạm mép ngoài cùng vật ($\le 2\text{ px}$). | Mục 4.1.1: Quy tắc kiểm soát viền bám khít hình học. |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên viết sẵn | Khởi tạo hợp đồng nhãn ban đầu |
| v1.1 | Chốt phân loại xe đạp điện (`CAL-02`); chuẩn hóa quy tắc cụm xe máy dày đặc dùng `ignore_region` (`CAL-03`); chốt ưu tiên `three_wheeler` và loại bỏ box trùng lặp (`CAL-01`); siết viền ôm sát $\le 2\text{ px}$ sau khi phát hiện bẫy lọt `img_023.jpg` (`CAL-04`). | Khắc phục điểm mù của YOLO26/YOLOE, giải quyết câu hỏi mở v1.0 và chặn đứng automation bias (`CAL-01` đến `CAL-04`). |
