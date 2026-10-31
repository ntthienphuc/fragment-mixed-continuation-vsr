# Bản trao đổi với thầy: fragment-mixed continuation

**Tên bài:** Fragment-Mixed Continuation for Cross-Corpus Visual Speech Recognition.
**Hướng nộp đầu tiên:** Pattern Recognition. Bản thảo ẩn danh; tác giả và cơ quan bổ sung sau.

## Có thể trình bày trong khoảng một phút

“Em nghiên cứu cách học tiếp một mô hình đọc khẩu hình đã có: thay vì luôn đưa cả
câu, em trộn câu đầy đủ với đoạn ngắn có nhãn lấy từ transcript nguồn. Kiến trúc
và cách nhận dạng nguyên câu lúc sử dụng được giữ nguyên. Ở cặp dữ liệu tiếng Việt,
trộn đoạn ngắn cải thiện kết quả ngoài corpus qua hai seed, nhưng làm giảm một
phần kết quả trên nguồn so với chỉ học câu đầy đủ. Crop trọn từ cũng có lợi, nên
em không nhận cách cắt qua biên từ là yếu tố mới hoặc cần thiết. Đối chứng chỉ
dùng đoạn ngắn làm kết quả kém đi ở cả nguồn và đích, chủ yếu do xóa nhiều đơn vị
trong transcript. Kết quả tiếng Anh không có lợi ích ngoài corpus rõ ràng. Đóng
góp của bài là làm rõ lợi ích, đánh đổi và giới hạn của cách thay đổi độ dài
supervision trong một thiết kế đối chứng có kiểm soát.”

## Bảng matched chính

Cùng initializer, thứ tự 48.000 lượt trình bày, seed 17, 6.000 cập nhật và decoder
chung; mỗi trạng thái được đánh giá trên đủ 11.307 clip VLR và 1.167 clip ViCocktail.
Các trạng thái này khác checkpoint của thí nghiệm FC/FM hai seed ban đầu.

| Dữ liệu trình bày khi học tiếp | VLR WER (%) | ViCocktail WER (%) |
| --- | ---: | ---: |
| Full-only | 38,029 | 67,369 |
| Fixed-time mixed | 38,580 | 63,846 |
| Whole-word mixed | 38,606 | 63,388 |
| Whole-word short-only | 44,844 | 70,715 |

Short-only kém whole-word mixed **6,238 điểm phần trăm trên nguồn** và **7,328 điểm
trên đích**. Theo chiều mixed trừ short-only đã khóa, khoảng tin cậy 95% lần lượt
là **[−6,592; −5,887]** và **[−8,864; −5,488]**. Có thêm 33.541 lỗi xóa trên nguồn
và 5.735 lỗi xóa trên đích; số substitution và insertion giảm nhưng không bù được.

“Short-only” vẫn giữ 598 lượt full fallback khi không tìm được đoạn trọn từ hợp
lệ; 47.402 lượt còn lại là đoạn ngắn. Không có clip nào bị bỏ khỏi thứ tự nguồn.
Ngân sách bằng nhau về số cập nhật, không bằng nhau về tổng frame hoặc token.

## Những đối chứng trả lời được gì

- **Hai seed tiếng Việt ban đầu:** common-decoder gain 3,410 và 3,170 điểm WER;
  không gọi toàn bộ bảng matched và short-only là kết quả hai seed.
- **Crop trọn từ:** cũng tạo lợi ích chuyển corpus; không cần khẳng định nhãn
  chồng lên biên fixed-time crop là nguyên nhân của cải thiện.
- **Loss chuẩn hóa theo token:** fixed-time gain còn 3,460 điểm; loại được giả
  thuyết cụ thể rằng lợi ích sẽ biến mất dưới phép đổi reduction này.
- **Context và self-teaching:** không có lợi thế được chứng minh so với FM trong
  các cấu hình đã thử; không khái quát thành mọi hình thức distillation đều vô ích.
- **Tiếng Anh:** common WER của FC 66,817%, FM 66,884%; không có destination gain
  rõ, trong khi có chi phí trên nguồn. Đây là bằng chứng giới hạn phạm vi kết luận.

## Điểm mới nên trình bày thế nào

Temporal crop, chuỗi từ liên tiếp và mixed-length training đã có trong nghiên
cứu trước. Bài không đề xuất kiến trúc nhận dạng mới. Giá trị cần bảo vệ là câu
hỏi **source-only continuation có chuyển được sang corpus khác không, phải đánh
đổi gì trên nguồn, và vai trò của việc giữ lượt học nguyên câu là gì**; cùng chuỗi
đối chứng matched crop, loss reduction, context và short-only.

Chủ đề phù hợp với Pattern Recognition, nhưng mức insight có đủ vượt một cách
augmentation đã biết hay không vẫn là điểm biên tập viên có thể chất vấn. Nên thử
bản đã hoàn thiện ở mục tiêu này theo chủ đích của tác giả, không xem bộ đối chứng
hiện tại là bảo đảm nhận bài.

## Các giới hạn cần trả lời thẳng

Một họ recognizer, ít seed, annotation thời gian tự động và các test set đã được
quan sát trước đó. Khoảng tin cậy bootstrap là biến thiên theo các nhóm nguồn
đã lưu, có điều kiện trên model đã fit; không phải độ bất định qua mọi lần train.
Nhóm recording không đồng nghĩa speaker đã xác minh. WER tiếng Việt dùng đơn vị
tách bằng khoảng trắng, không cộng chung mẫu số với tiếng Anh. Chưa kiểm chứng
độc lập toàn bộ overlap trong lịch sử pretraining.

Giữ full presentations có lợi trong phép thử đã thực hiện; chưa chứng minh tỉ
lệ trộn tối ưu, mọi mục tiêu short-only đều thất bại, hay một cơ chế nội tại duy
nhất giải thích được kết quả. Nếu cần bước xác nhận mạnh hơn sau vòng góp ý,
ưu tiên corpus độc lập hoặc họ recognizer thứ hai thay vì tiếp tục chọn cấu hình
trên cùng test set.

## Code và khả năng tái lập

Gói số liệu tái lập đủ 60 dòng kết quả và 49 khoảng tin cậy bằng CPU, không cần
download video hay checkpoint. Gói research code đầy đủ có 1.719 file nguồn/phụ
thuộc, kiểm đúng identity train gốc và sáu identity control; 27/27 CPU test đã qua.
Bản ZIP được giải nén mới và chạy lại thật. Các kiểm tra này không tương đương
với train lại toàn bộ mạng 6.000 cập nhật. Video và checkpoint lớn có danh mục
identity và yêu cầu quyền truy cập riêng của nhà cung cấp.

Phép thử greedy CTC bổ sung đã hoàn tất và được tính lại độc lập trên cùng
ba model, cùng pixels và đủ cả hai quần thể. WER (%) là:

| Corpus | Full-only | Whole-word mixed | Short-only |
| --- | ---: | ---: | ---: |
| VLR | 42,477 | 43,151 | 47,060 |
| ViCocktail | 69,187 | 64,385 | 72,424 |

Short-only vẫn kém cả hai trạng thái đối chiếu và có nhiều lỗi xóa hơn.
Cả bốn khoảng tin cậy paired theo nhóm nguồn loại 0 theo chiều bất lợi
cho short-only. Đây là kiểm tra readout mang tính khám phá trên dữ liệu đã
dùng, không phải xác nhận độc lập hoặc bằng chứng về một cơ chế duy nhất.
Các số này không thay thế bảng decoder chung trong paper.
