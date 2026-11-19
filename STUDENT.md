# Thông tin học viên

- Họ và tên: Lã Thái Hoà
- MSHV: 2A202603069
- Lớp (L2 / L3 / khác): L23A
- Tài khoản GitHub: Zeus-AIE
- Repo bài nộp (URL): https://github.com/Zeus-AIE/K5-L23A-Day08-LaThaiHoa-2A202603069-AI-Assisted-Labeling.git
- Commit chốt bài (hash): f122dd5
- Máy: Windows 11 x64, GPU NVIDIA GeForce RTX 3060 Laptop GPU (CUDA 12.1)
- CVAT (Docker): Local Docker v2.81.0 (cổng 8080)

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 128/154 = 83.1% / 11/154 = 7.1% / 15/154 = 9.7% (mẫu số: 154 box máy); added: 49/188 = 26.1% (mẫu số: 188 đối tượng sau duyệt)
- So với gold: precision / recall của **máy**: 77.8% (14/18) / 20.9% (14/67) (mẫu số: 67 đối tượng gold)
- Bắt bẫy: 10/12 = 83.3%
- Thời gian: gán tay 5.5 phút/ảnh (22 phút / 4 ảnh); duyệt có pre-label 3.0 phút/ảnh (72 phút / 24 ảnh)
- Lỗi QC: 0.0% theo ảnh (0/5), 0.0% theo đối tượng (0/50)
- **Quyết định:** Release — lý do một dòng: Tỉ lệ lỗi kiểm định 0.0% <= 2%, bắt bẫy đạt 83.3% >= 80%, không có lỗi hệ thống và guideline v1.1 đã chuẩn hóa đầy đủ.
- Bài nâng cao: Đã hoàn thành mục 2 Meta SAM 3 (checkpoint D:\App\sam3.pt, cả chế độ Click và Text Prompt) với 6 ảnh mask chất lượng cao trong `submission/masks/` (+5 điểm thưởng theo RUBRIC).
