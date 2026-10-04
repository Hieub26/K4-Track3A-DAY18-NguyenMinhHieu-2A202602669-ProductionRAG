# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Minh Hiếu (MSSV 2A202602669)  
**Khóa:** K4 - Track 3A  

Nguồn số liệu: `reports/ragas_report.json`, `reports/naive_baseline_report.json`, `reports/latency_report.json`
(20 câu hỏi trong `test_set.json`; LLM trả lời và LLM judge đều là `gemini-3.5-flash-lite`, embedding judge là `gemini-embedding-001`).

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.6417 | 0.9583 | +0.3167 |
| Answer Relevancy | 0.7742 | 0.9069 | +0.1327 |
| Context Precision | 0.8750 | 0.9250 | +0.0500 |
| Context Recall | 0.8250 | 0.9250 | +0.1000 |

- **Naive baseline:** chunk theo đoạn văn (57 chunks) + dense search (`bge-m3`) top-3, không rerank, không enrichment.
- **Production:** hierarchical chunking (112 child chunks 256 ký tự) → enrichment 1 call/chunk → hybrid BM25 + dense + RRF (top-20) → rerank `bge-reranker-v2-m3` → trả về parent chunk (tối đa 3) cho LLM.
- Cả 4 metric đều tăng và đều ≥ 0.90. Metric tăng mạnh nhất là **faithfulness (+0.32)**; metric thấp nhất sau cải tiến là **answer relevancy (0.9069)**.

### Hai lần chạy production — tác động của prompt

| Metric | Lần 1 (prompt "ngắn gọn") | Lần 2 (prompt "câu hoàn chỉnh + nêu căn cứ") |
|--------|---------------------------|----------------------------------------------|
| Faithfulness | 0.8458 | **0.9583** |
| Answer Relevancy | 0.9062 | 0.9069 |
| Context Precision | 0.9500 | 0.9250 |
| Context Recall | 0.9250 | 0.9250 |

Ở lần 1, hai câu trả lời **đúng** nhưng chỉ là một cụm từ — "Tổng Giám đốc (CEO)" và "1.000.000 VNĐ/tháng" — bị chấm faithfulness = 0.0 dù context precision = recall = 1.0. Chỉ sửa system prompt (retrieval giữ nguyên) đã đưa faithfulness từ 0.8458 lên 0.9583 và cả hai câu đó ra khỏi bottom-5. Report đang nộp là của lần 2. Context precision lệch 0.95 → 0.925 dù retrieval không đổi: đây là dao động của LLM judge giữa hai lần chấm, không phải thay đổi của pipeline.

### Latency breakdown (production, 20 queries, CPU)

| Bước | Thời gian |
|------|-----------|
| 1. Chunking (M1) | 1.37 s |
| 2. Enrichment (M5) | 0.02 s khi đã có cache — lần đầu chưa cache: 557 s cho 112 chunks (~5 s/chunk, do tự giới hạn 12 request/phút) |
| 3. Indexing BM25 + Dense (M2) | 99.7 s |
| 4. Load reranker (M3) | 7.7 s |
| 5. Hybrid search / query (M2) | 0.21 s |
| 6. Rerank / query (M3) | **9.09 s** |
| 7. LLM generate / query | 1.57 s (gồm thời gian chờ rate limit) |
| 8. RAGAS evaluation (M4) | 686 s (80 phép chấm) |

**Nút thắt lúc query là rerank:** 9.09 s/query (lần chạy đầu đo được 13.7 s), gấp ~40 lần hybrid search, vì cross-encoder 568M tham số chạy trên CPU cho 20 candidate/query.

---

## Bottom-5 Failures

Xếp theo điểm trung bình 4 metric tăng dần (lấy từ `failures` trong `reports/ragas_report.json`). Nguồn của context ở từng câu đã được kiểm tra lại bằng cách chạy riêng bước retrieval.

### #1 — avg 0.59
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Vế phép năm đúng (15 + 3 = **18 ngày**, có nêu bản 2023 đã bị thay thế). Vế lương trả lời **sai ý**: "Trong 18 ngày phép năm này, nhân viên được hưởng lương đầy đủ (100% lương)…" — LLM hiểu "lương" thành "nghỉ có lương" thay vì khung lương Senior.
- **Worst metric:** context_precision = 0.00 (faithfulness 1.00, answer_relevancy 0.85, context_recall 0.50)
- **Error Tree:** Output sai (1 trong 2 ý) → Context đúng? **Không** — 3 context là `nghi_phep_nam_v2024.md`, `nghi_phep_nam_v2023.md`, `nghi_phep_khong_luong.md`; không có `bang_luong_2024.md` → Query OK? **Không** — câu hỏi multi-hop gồm 2 chủ đề nhưng được search bằng 1 query duy nhất, nên cả 3 slot context đều bị chủ đề "nghỉ phép" chiếm.
- **Root cause:** Thiếu bước query decomposition cho câu hỏi multi-hop. Vì prompt yêu cầu "trả lời đủ từng ý" mà context không có bảng lương, LLM đã diễn giải lại câu hỏi để trả lời bằng thứ có trong context — faithfulness vẫn 1.0 nên **metric này không bắt được lỗi**; chỉ context recall/precision lộ ra.
- **Suggested fix:** Tách câu hỏi thành sub-query ("phép năm theo thâm niên", "khung lương Senior"), search + rerank riêng từng sub-query rồi gộp context. Thêm vào prompt: ý nào context không có thì nói rõ "không tìm thấy" cho riêng ý đó.

### #2 — avg 0.82
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** "Theo Chính sách mật khẩu (Phiên bản 2.0, hiệu lực từ 01/07/2024), tất cả nhân viên bắt buộc kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ."
- **Worst metric:** context_recall = 0.50 (faithfulness 1.00, answer_relevancy 0.78, context_precision 1.00)
- **Error Tree:** Output sai? Đúng ý chính nhưng **thiếu vế so sánh phiên bản** → Context đúng? **Thiếu một nửa** — context là `mat_khau_v2.md`, `mua_sam.md`, `vpn_truy_cap.md`; không có `mat_khau_v1.md` → Query OK? Có → lỗi ở bước **retrieval**.
- **Root cause:** Chính sách v1.0 *không nhắc* tới MFA, nên không chunk nào của v1 khớp với query "MFA" — cả BM25 lẫn dense đều không thể tìm ra "sự vắng mặt" của một quy định. Hai slot còn lại bị tài liệu khác (mua sắm, VPN) chiếm.
- **Suggested fix:** Gắn metadata `policy` + `version` + `status: superseded` cho từng tài liệu ở bước enrichment; khi top-1 thuộc một chính sách có nhiều phiên bản thì luôn kéo thêm phiên bản còn lại vào context.

### #3 — avg 0.84
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi 120 ngày. Chính sách cũ yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** "Theo Chính sách mật khẩu phiên bản 2.0 (hiện hành…), mật khẩu phải được thay đổi mỗi 120 ngày. (Lưu ý: Phiên bản cũ v1.0 quy định … mỗi 90 ngày nhưng đã bị thay thế)." — **câu trả lời đúng và đủ**.
- **Worst metric:** context_precision = 0.50 (faithfulness 1.00, answer_relevancy 0.86, context_recall 1.00)
- **Error Tree:** Output sai? **Không** → Context đúng? Đủ (recall 1.0) nhưng **xếp hạng sai**: `mat_khau_v1.md` (bản đã bị thay thế) đứng trên `mat_khau_v2.md` → Query OK? Có → lỗi ở bước **rerank**.
- **Root cause:** Cross-encoder chỉ chấm độ khớp ngữ nghĩa giữa câu hỏi và đoạn văn; đoạn "thay đổi mỗi 90 ngày" của v1 khớp câu hỏi ngang với đoạn "mỗi 120 ngày" của v2. Reranker không biết phiên bản nào còn hiệu lực. Lần này LLM vẫn trả lời đúng nhờ prompt xử lý xung đột phiên bản, nhưng thứ tự context sai là rủi ro tiềm ẩn.
- **Suggested fix:** Sau rerank, đẩy tài liệu `status: current` lên trên `superseded` (cùng metadata như #2), hoặc cộng điểm theo ngày hiệu lực.

### #4 — avg 0.85
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Nêu đúng quy định, xác định quá hạn 5 ngày, tính được "15.000.000 VNĐ x 2% = **300.000 VNĐ/tháng**" — nhưng **không tính pro-rata** cho 5 ngày.
- **Worst metric:** context_recall = 0.50 (faithfulness 1.00, answer_relevancy 0.88, context_precision 1.00)
- **Error Tree:** Output sai? Gần đúng, thiếu bước tính cuối → Context đúng? **Có** — `tam_ung.md` đứng đầu, chứa quy định 15 ngày và 2%/tháng → Query OK? Có → lỗi ở bước **generation** (suy luận số học nhiều bước).
- **Root cause:** Context recall 0.5 vì phần "300.000 VNĐ / khoảng 50.000 VNĐ" trong ground truth là kết quả *tính ra*, không xuất hiện nguyên văn trong context nên judge không quy được về context. Về phía LLM: prompt mới đã khiến nó tính 300.000 VNĐ/tháng (lần 1 chỉ chép lại quy định), nhưng nó dừng ở mức phí theo tháng.
- **Suggested fix:** Với câu hỏi numeric, yêu cầu trình bày từng bước tính tới con số cuối cùng cho đúng khoảng thời gian trong câu hỏi. Chấp nhận rằng context recall của RAGAS đánh giá thấp các ground truth chứa số được suy ra.

### #5 — avg 0.90
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** "Căn cứ theo mục "Cam kết hoàn chi" trong Chính sách hoàn chi đào tạo (Phiên bản: 1.1)… phải hoàn trả 100% chi phí… số tiền là 25.000.000 VNĐ." — **đáp án đúng**.
- **Worst metric:** faithfulness = 0.67 (answer_relevancy 0.93, context_precision 1.00, context_recall 1.00)
- **Error Tree:** Output sai? **Không** → Context đúng? **Có** — `hoan_chi_dao_tao.md` đứng đầu → Query OK? Có → điểm bị trừ ở bước **generation/đánh giá**.
- **Root cause:** Faithfulness tách câu trả lời thành các mệnh đề; 1/3 mệnh đề không được judge xác nhận. Mệnh đề dễ bị trượt nhất là kết luận "phải hoàn trả 25.000.000 VNĐ": con số này là suy ra (100% × 25 triệu trong câu hỏi), không có nguyên văn trong context. Đây là suy đoán từ điểm số — report không lưu phán quyết cho từng mệnh đề.
- **Suggested fix:** Lưu thêm danh sách mệnh đề + phán quyết của judge vào report để biết chính xác mệnh đề nào bị trượt, thay vì đoán.

### Nhận xét chung

| Nhóm lỗi | Câu | Tầng cần sửa |
|----------|-----|--------------|
| Multi-hop, 1 query không phủ đủ 2 chủ đề | #1 | Query (decomposition) |
| Xung đột phiên bản: thiếu bản cũ / bản cũ xếp trên bản mới | #2, #3 | Retrieval + rerank (metadata `version`/`status`) |
| Số được suy ra, không có nguyên văn trong context | #4, #5 | Generation prompt + giới hạn của phép đo |

Chỉ #1 là câu trả lời sai thật sự. Ở #3 và #5 câu trả lời đúng, điểm thấp đến từ thứ tự context và từ cách RAGAS chấm số được suy ra. Hai trong năm failure liên quan tới **phiên bản chính sách** — đặc điểm riêng của corpus này (v2023/v2024, v1.0/v2.0) mà cả hybrid search lẫn cross-encoder đều không xử lý được nếu thiếu metadata.

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?" (failure #1, dạng multi-hop)

**Error Tree walkthrough:**
1. Output đúng? → **Đúng một nửa.** Vế phép năm đúng (18 ngày). Vế lương bị hiểu thành "nghỉ có hưởng lương" thay vì khung lương 20–35 triệu.
2. Context đúng? → **Không.** Cả 3 context đều là tài liệu nghỉ phép; thiếu `bang_luong_2024.md`. Context recall = 0.5, context precision = 0.0 — thấp nhất trong 20 câu.
3. Query rewrite OK? → **Không có bước rewrite.** Câu hỏi được đưa nguyên văn vào hybrid search. Cụm "9 năm thâm niên … ngày phép năm" áp đảo cả BM25 (nhiều token trùng) lẫn dense, nên top-20 candidate và top-3 sau rerank đều thuộc chủ đề nghỉ phép.
4. Fix ở bước: **Query** — thêm query decomposition trước M2, rerank riêng từng sub-query rồi gộp context.

Điểm đáng chú ý: faithfulness của câu này là **1.0**. Câu trả lời bám sát context nhưng trả lời sai câu hỏi, nên phải đọc cả context recall/precision mới thấy lỗi — một metric riêng lẻ không đủ.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Query decomposition cho câu hỏi multi-hop (#1).
- Metadata `version` / `status` ở bước enrichment, dùng để kéo đủ các phiên bản và ưu tiên bản hiện hành sau rerank (#2, #3).
- Giảm latency rerank 9 s/query: giảm candidate từ 20 xuống 10, hoặc đổi sang reranker nhỏ hơn (FlashRank) / chạy GPU.
- Lưu phán quyết từng mệnh đề của RAGAS vào report để chẩn đoán faithfulness chính xác hơn (#5).
- OCR 2 file PDF scan (`BCTC.pdf`, Nghị định 13/2023) đang bị bỏ qua vì không có text layer — hiện không câu hỏi nào trong test set hỏi về 2 file này.
