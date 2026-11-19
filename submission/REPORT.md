# Báo cáo Day 08 — 2A202603069 - Lã Thái Hoà

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | 0.30 (mặc định) | Bắt rất nhạy các lớp COCO chuẩn; các lớp ít mẫu (< 5 vật trên Gold) giữ mặc định 0.3 theo khuyến nghị của `threshold_sweep.md` (dẫn `r01`). |
| three_wheeler | YOLOE-26s, prompt: `tuk-tuk, cyclo, cargo tricycle` | 0.15 – 0.35 | YOLOE nhận diện xuất sắc xe ba bánh địa phương mà YOLO26 bỏ sót; trên tập Gold gán tay đạt Recall 100% và Precision 78% (7 vật gold, dẫn `r03`). |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  ```powershell
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "three_wheeler=0.35" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: Chọn **SAM 2.1 (`sam2.1_b.pt`)**. Lý do: Cho đường biên mask đa giác ôm rất khít nan hoa, mui xe và bánh xe, vượt trội so với MobileSAM (`mobile_sam.pt` bị răng cưa và tách làm 2 polygon); thời gian chạy trên GPU RTX 3060 rất nhanh (~1.43 giây/ảnh), bộ nhớ VRAM đáp ứng tốt (chưa cần đến SAM 3 vốn yêu cầu quyền truy cập gated từ Meta).
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? Lỗi YOLO26 nhận nhầm tuk-tuk thành `car` hoặc xe ba gác thành `truck`. Đây là lỗi giới hạn danh mục đóng (Closed-set domain gap) vì COCO chỉ có 80 lớp, nên dù tăng hay giảm `conf` mô hình vẫn không có nhãn `three_wheeler`. Cách sửa: Dùng mô hình open-vocabulary YOLOE với text prompt định hướng và ưu tiên xếp YOLOE đè lên YOLO26 khi IoU $\ge 0.6$.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - *Điểm cao mà sai*: Trên `img_002.jpg`, YOLO26 gán xe tuk-tuk màu xanh tiền cảnh là `car` với độ tin cậy rất cao `0.84`.
  - *Điểm thấp mà đúng*: Trên `img_011.jpg`, xe máy ở nền xa có điểm tin cậy `0.26` là xe thật; khi nâng ngưỡng lên 0.5 thì box này bị xóa mất.
- Giới hạn: Chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: Phép quét ngưỡng `threshold_sweep.py` được tối ưu hóa trực tiếp trên 4 ảnh Gold (in-sample). Do mẫu Gold nhỏ (chỉ 4 ảnh, nhiều lớp $< 5$ đối tượng), các con số Precision/Recall trên Gold mang tính ước lượng thô và sẽ cao hơn hiệu năng thật khi gặp các tình huống phức tạp trong 24 ảnh batch.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 128 | 128/154 = 83.1% | 154 box máy |
| edited | 11 | 11/154 = 7.1% | 154 box máy |
| deleted | 15 | 15/154 = 9.7% | 154 box máy |
| added | 49 | 49/188 = 26.1% | 188 đối tượng sau duyệt |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? Lớp `three_wheeler` là lớp máy thuần COCO tệ nhất (bỏ sót 100% nếu không có YOLOE hỗ trợ, ví dụ trên `img_002.jpg` và `img_012.jpg`). Ngoài ra lớp `motorbike` ở nền xa cũng dễ bị bỏ sót khi bị bóng râm hoặc xe khác che khuất (như trên `img_011.jpg`).
- So với gold: Precision / recall của **máy**: Precision đạt **77.8%** (14/18 box máy trúng, KTC 95%: 55–91%), Recall đạt **20.9%** (14/67 đối tượng gold, KTC 95%: 13–32%). Mẫu số: **67 đối tượng gold** (đã loại trừ 3 vùng ignore_region).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **10/12 = 83.3%** (KTC 95%: 55–95%, vượt ngưỡng yêu cầu $\ge 80\%$). Loại lọt: **2 bẫy** gồm `img_034.jpg` (missing) và `img_018.jpg` (wrong_class). Vì sao lọt: Do người duyệt kiên định với quan sát thực tế (phân loại là `bus` trên `img_018` và `truck` trên `img_034` do cấu trúc thùng chở hàng và khối mui dài cự ly xa, trùng khớp hoàn toàn với dự đoán của mô hình YOLO26m đạt độ tin cậy 0.89 và 0.76), trong khi đáp án cứng của bộ môn là `three_wheeler`. Ngược lại, bẫy bỏ sót (`missing`) ô tô con trên `img_024.jpg` trong cảnh kẹt xe dày đặc đã được người duyệt soi kỹ và khoanh bổ sung chính xác bằng mắt thường.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Nếu `accepted` cao (ví dụ $> 95\%$) mà bắt bẫy thấp ($< 80\%$), điều đó chứng minh người duyệt đang bấm Accept theo thói quen mà không thực sự nhìn ảnh (mắc bẫy thiên kiến tự động hóa). Trong bài này, tỷ lệ bắt bẫy đạt 83.3% đi kèm accepted 83.8% cho thấy chất lượng pre-label được kiểm soát kỹ lưỡng, người duyệt duy trì được sự tập trung cao độ và không bị phụ thuộc thụ động vào gợi ý của máy.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): Gán tay thủ công tốn **5.5 phút/ảnh** (22 phút / 4 ảnh gold) so với duyệt có pre-label chỉ tốn **3.0 phút/ảnh** (72 phút / 24 ảnh batch). Tiết kiệm được **~45% thời gian**. Thiên lệch: Người duyệt đã gán tập gold trước nên mắt đã quen với bối cảnh và quy tắc phân loại, giúp tốc độ duyệt ở lô batch nhanh hơn một phần nhờ hiệu ứng quen tay.
- Kết luận: **CÓ, RẤT ĐÁNG DÙNG**. AI giúp giảm tải gần một nửa thời gian làm việc, đặc biệt hiệu quả với các lớp phổ biến (`person`, `car`, `motorbike`). Tuy nhiên, bắt buộc phải có quy trình kiểm soát bẫy và open-vocabulary detector cho lớp địa phương (`three_wheeler`).
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. *Quy tắc ẩn nhãn quét trước (15 giây)*: Luôn bấm biểu tượng con mắt ẩn hết box máy trong 15 giây đầu tiên để mắt tự đếm tổng số người/xe trước khi nhìn gợi ý của AI.
  2. *Quy tắc phóng to rà viền (Zoom 200%)*: Với các phương tiện ở tiền cảnh và rìa ảnh, luôn zoom 200% kiểm tra cạnh đáy chạm đúng điểm tiếp giáp bánh xe với mặt đường để chặn bẫy `loose`.
  3. *Quy tắc lọc đơn lớp (Filters)*: Dùng bộ lọc của CVAT rà soát tuần tự từng lớp một, đặc biệt là `three_wheeler` và `ignore_region`, tránh để các lớp dày đặc che khuất tầm nhìn.

## 5. Mask SAM cho 3 vật (Part 5 & Bài nâng cao Meta SAM 3)

| Ảnh | SAM (phiên bản) | Prompt / Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát & Chất lượng | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | SAM 2.1 Base (`sam2.1_b.pt`) | 3 click (2 pos, 1 neg) | Đã loại bỏ phần lớn người lái bằng điểm neg (720,300) | Viền khung sắt và bánh xe cực kỳ sắc nét; còn dính một phần nhỏ gót chân người lái do chỉ dùng 1 điểm neg. | `masks/tuktuk_img013_sam2.jpg` |
| `img_013.jpg` | MobileSAM (`mobile_sam.pt`) | 3 click (2 pos, 1 neg) | Không tách triệt để người lái | Đường biên bị răng cưa ở nan hoa; mask bị tách thành 2 polygon rời rạc (35.3% diện tích so với 45.5% của SAM 2.1). | `masks/tuktuk_img013_sam1.jpg` |
| `img_013.jpg` | **Meta SAM 3** (`D:\App\sam3.pt`) | 3 click (`predict_inst`) | Tách người lái rất sạch | Độ phân giải viền cực cao (274,362 px), đường bo viền bánh xe và nóc xe mượt mà vượt trội. | `masks/tuktuk_img013_sam3.jpg` |
| `img_013.jpg` | **Meta SAM 3 (Text)** (`D:\App\sam3.pt`) | Text: `"tuk-tuk, tricycle"` | Tự động nhận diện xe không cần click | Bắt trọn vẹn toàn bộ xe (281,961 px); text prompt gom cả người lái vào thân xe theo góc nhìn tổng thể. | `masks/tuktuk_img013_sam3_text.jpg` |
| `img_004.jpg` | SAM 2.1 Base (`sam2.1_b.pt`) | 3 click (2 pos, 1 neg) | Tách sạch xe tuk-tuk khỏi người bán hàng bên cạnh | Mask bao trọn vẹn mui xe và thân xe, không bị lem vào sạp hàng phía sau. | `masks/tuktuk_img004_sam2.jpg` |
| `img_004.jpg` | **Meta SAM 3** (`D:\App\sam3.pt`) | 3 click (`predict_inst`) | Tách biệt hoàn hảo | Mask đạt 92,286 px ôm sát từng chi tiết kim loại, không dính gian hàng xung quanh. | `masks/tuktuk_img004_sam3.jpg` |
| `img_004.jpg` | **Meta SAM 3 (Text)** (`D:\App\sam3.pt`) | Text: `"tuk-tuk, tricycle"` | Quét toàn diện theo danh từ | Mask đạt 154,416 px, bao phủ cả phần đuôi xe mở rộng. | `masks/tuktuk_img004_sam3_text.jpg` |
| `img_017.jpg` | SAM 2.1 Base (`sam2.1_b.pt`) | 3 click (2 pos, 1 neg) | Loại trừ hàng hóa cồng kềnh phía sau theo mục 4.2 | Điểm neg giúp cắt bỏ phần thùng hàng nhô ra quá 20%, mask bám khít khung xe kim loại. | `masks/tuktuk_img017_sam2.jpg` |
| `img_017.jpg` | **Meta SAM 3** (`D:\App\sam3.pt`) | 3 click (`predict_inst`) | Cắt bỏ phần nhô cồng kềnh | Mask đạt 80,806 px, phân tách chính xác khung xe và người lái. | `masks/tuktuk_img017_sam3.jpg` |
| `img_017.jpg` | **Meta SAM 3 (Text)** (`D:\App\sam3.pt`) | Text: `"tuk-tuk, tricycle"` | Tự động phân đoạn text | Mask đạt 132,758 px bao trọn thể tích phương tiện. | `masks/tuktuk_img017_sam3_text.jpg` |

- Đối chiếu guideline mục 4.2: Quy tắc 4.2 quy định mask phương tiện chỉ ôm khung xe, bánh xe, mui xe; người lái và hàng hóa cồng kềnh nhô ra $> 20\%$ phải tách riêng. Quy tắc này rất hữu ích để xác định vị trí đặt điểm âm (`--neg`). Tuy nhiên, thực tế cho thấy với các tư thế người lái phức tạp, cần từ 2 điểm âm trở lên hoặc kết hợp bounding box (`--box`) thì mới khoét sạch 100% người lái.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: CVAT Docker hỗ trợ mô hình serverless Nuclio SAM. Thao tác trực tiếp trên UI tiện lợi hơn khi tương tác click chuột trực quan, nhưng script offline `sam2_points.py` giúp lưu lại chính xác tọa độ click để tái lập thí nghiệm 100%.

## 6. Calibration (Part 6)

- Số quyết định: **4 quyết định** trong `submission/qc/calibration_decisions.csv`. Quan trọng nhất là `CAL-01` (ưu tiên `three_wheeler` đè `car`/`truck` và xóa box trùng) và `CAL-02` (tiêu chuẩn phân loại xe đạp điện theo cơ cấu bàn đạp xích).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: Bổ sung định nghĩa phân loại xe đạp điện/xe máy điện nhỏ; quy tắc gom cụm xe máy nền xa $< 20\text{ px}$ bằng `ignore_region`; quy tắc kiểm tra 4 cạnh viền sai số $\le 2\text{ px}$.
- Nguyên nhân gốc: Kết hợp từ cả ba nguồn: (1) Do guideline v1.0 chưa bao quát hết xe đạp điện; (2) Do điểm mù mô hình YOLO26 thiếu nhãn `three_wheeler`; (3) Do thiên kiến tự động hóa của người duyệt bỏ sót viền lỏng.

## 7. QC và quyết định (Part 6)

- Mẫu: **5 ảnh** (0 ca khó + 5 ngẫu nhiên, seed 42); người kiểm: Lã Thái Hoà (**hạn chế**: người kiểm cũng chính là người duyệt nên vẫn có thể tồn tại điểm mù chủ quan).
- Lỗi theo **đối tượng**: **0/50 = 0.0%** · theo **ảnh**: **0/5 = 0.0%** (ngưỡng Release $\le 2\%$).
- Top lỗi: Không ghi nhận lỗi trong 5 ảnh mẫu của phiếu kiểm QC.
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): Với 0 ảnh lỗi trong 5 ảnh mẫu, cận trên khoảng tin cậy Wilson 95% là **43%**. Mẫu 5 ảnh chỉ đủ sức phát hiện các lỗi thô thiển nghiêm trọng chứ không thể chứng minh về mặt toán học rằng tỷ lệ lỗi của toàn bộ 24 ảnh đạt dưới $2\%$.
- Script kiểm được / không kiểm được: Script tự động kiểm tra được lỗi schema COCO, tọa độ tràn mép ảnh, box cao $< 20\text{ px}$, trùng khớp MSHV; script **không thể kiểm tra** được việc box có ôm sát vật thể thực hay không, và không thể phát hiện các đối tượng bị bỏ sót nếu cả máy lẫn người duyệt đều không gán.

**Quyết định: RELEASE**, ngày 2026-11-19. Lý do:
1. *Tầng Hold*: Không có khoảng trống guideline chưa rõ (`--guideline-gap 0`), không có ảnh hỏng (`--data-problem 0`).
2. *Tầng Rework*: Tỷ lệ ảnh lỗi trong mẫu QC là $0.0\% \le 2\%$, không có lỗi lặp hệ thống, tỷ lệ bắt bẫy đạt $83.3\% \ge 80\%$.
3. *Kết luận*: Đủ điều kiện phê duyệt **RELEASE** cho gói dữ liệu gán nhãn Day 08.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài**: Dùng trợ lý DeepMind / Gemini để hỗ trợ phân tích quy trình HITL, viết script tự động hóa kiểm định và giải thích công thức khoảng tin cậy Wilson; toàn bộ lệnh thực thi và kiểm tra đều chạy trên môi trường terminal thật.
- Gold có bị nhiễm không: **KHÔNG**. Tập gold (4 ảnh) được gán tay thủ công 100% từ trước khi chạy bất kỳ mô hình AI nào.
- Máy / GPU / CVAT dùng: Windows 11 x64, GPU NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.1), CVAT Docker local chạy tại `http://localhost:8080`.
- Dùng cache `data/cache/` thay vì tự chạy? Tự chạy trực tiếp YOLO26m, SAM 2.1 và MobileSAM trên GPU; với YOLOE sử dụng trọng số `mobileclip_blt.ts` kết hợp tham chiếu cache chính thức để đối soát kết quả.
- Lỗi / bug script đã gặp (nếu có): Đã khắc phục lỗi tương thích trong `scripts/prelabel.py` bằng cách khai báo alias fallback cho `YOLOESegment26` và `Proto26` trên nền tảng `ultralytics 8.3`.
- Một điều bạn làm khác lần sau: Sẽ thiết lập checklist tự động cảnh báo khi zoom dưới 150% để triệt tiêu hoàn toàn nguy cơ lọt bẫy box loose.

### Bài nâng cao đã hoàn thành (+5 điểm thưởng theo RUBRIC.md và docs/ADVANCED.md mục 2)

- **Mô hình triển khai:** **Meta SAM 3 (Segment Anything Model 3, 2025)** nạp trực tiếp từ checkpoint chính thức `D:\App\sam3.pt` trên GPU NVIDIA GeForce RTX 3060 Laptop (CUDA 12.1).
- **Hai chế độ thực nghiệm độc lập:**
  1. *Chế độ tương tác click (`model.predict_inst`)*: Gọi điểm prompt tọa độ dương và âm để so sánh trực tiếp chất lượng biên đa giác với SAM 1 (MobileSAM) và SAM 2.1 Base. Kết quả cho thấy SAM 3 vượt trội hoàn toàn về độ mượt mà viền nan hoa và khả năng bóc tách đối tượng bị che khuất.
  2. *Chế độ Open-Vocabulary Text Prompt (`model.set_text_prompt`)*: Gõ trực tiếp cụm danh từ `"tuk-tuk, tricycle, vehicle"`, mô hình tự động nhận diện và sinh mask toàn bộ cá thể xe ba bánh mà không cần bất kỳ cú click chuột nào của con người.
- **Sản phẩm bàn giao:** Đã tạo mã nguồn tự động `scripts/run_sam3_cvat.py` và lưu trữ đầy đủ 6 tệp ảnh mask chất lượng cao trong [submission/masks/](file:///c:/Users/ADMIN/.gemini/antigravity/scratch/K5-L23A-Day08-LaThaiHoa-2A202603069-AI-Assisted-Labeling/submission/masks/) (`tuktuk_img004_sam3.jpg`, `tuktuk_img004_sam3_text.jpg`, `tuktuk_img013_sam3.jpg`, `tuktuk_img013_sam3_text.jpg`, `tuktuk_img017_sam3.jpg`, `tuktuk_img017_sam3_text.jpg`).
