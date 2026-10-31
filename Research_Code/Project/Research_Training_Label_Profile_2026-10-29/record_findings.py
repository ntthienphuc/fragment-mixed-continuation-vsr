"""Record verified training-only findings and the follow-up review without changing experiments."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    output = HERE / "Results_01"
    summary = read(output / "Summary.json")
    proof = read(output / "Verification.json")
    assert digest((output / "Summary.json").read_bytes()) == proof["summary_sha256"]
    assert proof["all_original_train_text_rows"] == 92632
    assert len(proof["individual_word_csv_checks"]) == 24
    assert len(proof["sampled_full_text_vs_word_csv_discrepancies"]) == 1
    assert not proof["sampled_full_text_vs_word_csv_discrepancies"][0]["scheduled"]
    assert all(row["audio_streams"] == 0 for row in proof["motivating_video_streams"])
    assert not summary["assessment_read"] and not summary["train_rows_changed"]
    now = datetime.now(timezone.utc)
    heartbeat = read(ROOT / "Research_Source_Retention_2026-10-29/Analysis_01/Watcher_01/Heartbeat.json")
    text = "\n".join([
        "", "=" * 72,
        "PHẢN HỒI REVIEW VÀ HỒ SƠ NHÃN TRAIN — " + now.astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "=" * 72,
        "Lượt trước: PROGRESS — dữ liệu server và provider đã chạy kiểm tra thật; phân tích English đã backup.",
        "Lượt này: PROGRESS — đọc phản hồi reviewer, kiểm nhãn nguồn train và supervision exposure thực.",
        "Reviewer đồng ý giữ bốn câu hỏi: VLR retention, whole-word, normalized loss, fragment-only.",
        "Không gọi source/destination WER profile là transfer efficiency khi chưa đo chi phí tính toán.",
        "Không chọn ngưỡng retention chấp nhận được sau khi nhìn kết quả; nonsignificant không là equivalent.",
        "External_Review_Response.md đã thêm phản hồi, sửa trạng thái staging/504 validation windows.",
        "Phát hiện khi đọc resource extrema: một số nhãn VLR có câu tiếng Anh/cụm lặp trong CSV gốc.",
        "Đã rà đủ 92.632 transcript train, nối 48.000 lượt seed17 và rebuild original FM targets/tokenizer.",
        "Tổng token/frame khớp hoàn toàn exposure đã có; không thay lịch mẫu hay preprocessing.",
        "ASCII/English-cue: 1.218/48.000 mẫu (2,5375%); token share FC 5,0738%, FM 4,6844%.",
        "Repeated fourgram: 159 mẫu (0,3313%); majority zero-duration word rows: 242 mẫu (0,5042%).",
        "Các nhóm chồng lấp, là heuristic/timing flags, không phải số nhãn sai đã xác nhận.",
        "Verifier độc lập từ CSV gốc khớp tất cả text-flag IDs; rà thêm 24 word CSV, khớp prepared words.",
        "Một discrepancy U+0E3A word CSV/full transcript được giữ lại; clip không thuộc schedule seed17.",
        "Ba video motivating không có audio; chưa xác nhận dịch nhầm hay sai ngôn ngữ.",
        "Không suy token share là gradient/loss contribution hoặc nguyên nhân tăng WER/giảm WER.",
        "Profile và verifier exit0 với discrepancy được ghi rõ; Findings.md nêu đầy đủ phạm vi kiểm chứng.",
        "Giữ nguyên thí nghiệm đang chạy. Không tạo nhánh data-cleaning hoặc sửa nhãn theo dấu hiệu này.",
        f"English tại {heartbeat['utc']}: {len(heartbeat['verified_tasks'])}/14 shard đã verify/backup.",
        "Driver production chưa nối provider mới; đây vẫn là bước tích hợp kế tiếp sau kiểm nhãn này.",
        "GPU probe vẫn trong hàng đợi. Không có train dài mới, không thuê thêm cloud, goal ACTIVE.", "",
    ])
    (HERE / "Status_Update.txt").write_text(text, encoding="utf-8")
    with (ROOT / "result.txt").open("a", encoding="utf-8") as stream:
        stream.write(text)
    review = ROOT / "PatternRecognition_Vietnamese_Transfer_Draft_2026-10-29/Review_02/External_Review_Response.md"
    files = [path for path in HERE.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    files.append(review)
    folder = Path("C:/Users/thien/Lip_Research_Backups/Training_Label_Profile_20261029")
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / ("Training_Only_Profile_" + now.strftime("%Y%m%d_%H%M%S") + ".zip")
    contents = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in files}
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as pack:
        for name, content in contents.items():
            pack.writestr(name, content)
    with zipfile.ZipFile(archive) as pack:
        assert set(pack.namelist()) == set(contents)
        for name, content in contents.items():
            assert digest(pack.read(name)) == digest(content)
    receipt = {"utc": now.isoformat(), "archive": str(archive), "sha256": digest(archive.read_bytes()),
               "verified_files": len(contents), "assessment_changed": False, "manuscript_changed": False}
    with (HERE / "Backup.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
