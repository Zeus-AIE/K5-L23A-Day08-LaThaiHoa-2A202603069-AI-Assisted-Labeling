# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 154; đối tượng sau duyệt: 189; ngưỡng accepted: IoU >= 0.9
- Đã **loại 18 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 129 | 129/154 = 83.8% | 154 box máy |
| edited (sửa) | 11 | 11/154 = 7.1% | 154 box máy |
| deleted (xoá) | 14 | 14/154 = 9.1% | 154 box máy |
| added (vẽ thêm) | 49 | 49/189 = 25.9% | 189 đối tượng sau duyệt |

Trong số deleted + added, **6 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 43.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 101 | 4 | 1 | 8 |
| bicycle | 0 | 0 | 0 | 3 |
| motorbike | 4 | 2 | 0 | 15 |
| car | 17 | 0 | 0 | 15 |
| bus | 0 | 0 | 0 | 2 |
| truck | 0 | 0 | 4 | 1 |
| three_wheeler | 7 | 5 | 9 | 5 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 67 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 14 | 4 | 53 | 77.8% (KTC 95%: 55–91%) | 20.9% (KTC 95%: 13–32%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 10/12 = 83.3% (KTC 95%: 55–95%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 2/3
  - extra: 3/3
  - missing: 2/3
  - loose: 3/3
  - lọt: img_034.jpg (missing), img_018.jpg (wrong_class)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
