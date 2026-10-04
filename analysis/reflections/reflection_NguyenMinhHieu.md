# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Minh Hiếu (MSSV 2A202602669)  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Trên toàn corpus, threshold 0.85 tạo **208 chunks** (trung bình 99 ký tự, nhỏ nhất 6 ký tự) so với basic **51 chunks** (trung bình 410 ký tự). Ngưỡng 0.85 quá cao với `all-MiniLM-L6-v2` trên tiếng Việt: hai câu liền nhau hiếm khi đạt similarity ≥ 0.85 nên gần như mỗi câu thành một chunk. Semantic chunking chỉ có ích khi ngưỡng được chỉnh theo model embedding và ngôn ngữ. |
| Hierarchical chunking (parent-child) | M1 | `chunk_hierarchical()` | 103–112 child (≤ 256 ký tự) để search chính xác; pipeline map child về parent (~800 ký tự) trước khi đưa cho LLM. Child ngắn giúp rerank đúng đoạn, parent đủ dài để LLM thấy cả điều khoản. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | Với câu "Mật khẩu phải đổi sau bao lâu?", BM25 xếp `mat_khau_v1.md` (bản cũ) lên đầu vì trùng nhiều token, còn dense xếp `mat_khau_v2.md`. RRF đưa đúng đoạn "mỗi 120 ngày" của v2 lên top-1 vì đoạn này đứng cao ở **cả hai** danh sách. RRF chỉ dùng thứ hạng nên không phải chuẩn hóa điểm BM25 (10–12) với cosine (0.7–0.8). |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | `underthesea` nối từ ghép bằng `_` ("nghỉ_phép"); phải đổi lại thành khoảng trắng để query và corpus tokenize giống nhau. Thêm lowercase + bỏ token dấu câu thì BM25 mới khớp "Nghỉ" với "nghỉ". |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Latency **9–14 s/query** trên CPU cho 20 candidate (đo 13.7 s và 9.1 s ở hai lần chạy; hybrid search chỉ ~0.2 s). Đổi lại context precision tăng từ 0.875 lên **0.925**. Reranker không biết phiên bản nào còn hiệu lực: với câu "Bao lâu phải đổi mật khẩu", nó xếp `mat_khau_v1.md` (bản cũ) trên v2. Cross-encoder là bước đắt nhất lúc query, nên chỉ rerank top-20 chứ không rerank cả corpus. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Production: faithfulness 0.9583, answer relevancy 0.9069, context precision 0.925, context recall 0.925 (baseline: 0.6417 / 0.7742 / 0.875 / 0.825). Ở lần chạy đầu **faithfulness thấp nhất (0.8458)** vì 2 câu trả lời đúng nhưng quá cụt ("Tổng Giám đốc (CEO)") bị chấm 0; chỉ sửa prompt thành "trả lời bằng câu hoàn chỉnh, nêu căn cứ" đã đưa lên 0.9583. Metric đo cả *cách diễn đạt*, không chỉ đúng/sai. |
| Diagnostic / Error Tree | M4 | `failure_analysis()` | Trong bottom-5 chỉ 1 câu trả lời sai thật (multi-hop, faithfulness vẫn = 1.0 nhưng context precision = 0). Nhìn metric thấp nhất của từng câu chỉ ra đúng tầng cần sửa — query, rerank hay prompt — thay vì chỉnh mò cả pipeline. |
| Contextual embeddings | M5 | `_enrich_single_call()` / `contextual_prepend()` | 108/112 chunk được Gemini thêm 1 câu ngữ cảnh ("Đoạn văn nằm trong tài liệu … về chính sách nghỉ phép năm 2024…") trước khi embed. Chunk 256 ký tự tự nó thường thiếu chủ thể (không biết là chính sách nào, phiên bản nào); câu ngữ cảnh bù lại phần đó. Combined mode dùng 1 call/chunk thay vì 4. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### Lỗi 1 — Không cài được dependencies
- **Exact error message:**
  ```
  ..\meson.build:1:0: ERROR: Unknown compiler(s): [['icl'], ['cl'], ['cc'], ['gcc'], ['clang'], ['clang-cl'], ['pgcc']]
  error: metadata-generation-failed
  × Encountered error while generating package metadata.
  ```
- **Nguyên nhân gốc rễ & cách debug:** Log cho thấy pip đang *build* `numpy 1.26.4` từ source. `.venv` được tạo từ Anaconda base là Python 3.13, trong khi `langchain 0.2` ép `numpy<2` và `numpy 1.26.4` không có wheel cho 3.13. File `.python-version` của repo ghi 3.11. Sửa bằng cách tạo conda env Python 3.11 rồi dựng lại `.venv` từ interpreter đó → pip lấy được wheel `cp311`.

### Lỗi 2 — Gemini không tương thích hoàn toàn với tham số của OpenAI
- **Exact error messages:**
  ```
  BadRequestError: Error code: 400 - 'Multiple candidates is not enabled for this model'
  ```
  và câu trả lời bị cắt: `finish_reason='length'`, `completion_tokens=3, total_tokens=90` khi đặt `max_tokens=80`.
- **Nguyên nhân gốc rễ & cách debug:** Dùng Gemini qua endpoint tương thích OpenAI nên viết một script gọi thử từng tham số trước khi code M4/M5. Phát hiện: (1) `n=3` không được hỗ trợ, mà RAGAS `answer_relevancy` mặc định sinh 3 câu hỏi ngược → đặt `strictness = 1`; (2) thinking token bị tính vào `max_tokens` → bỏ `max_tokens` nhỏ, thêm `reasoning_effort="low"`; (3) embedding phải gửi text thô → `check_embedding_ctx_length=False`.

### Lỗi 3 — Hết quota free tier
- **Exact error messages:**
  ```
  RateLimitError: Error code: 429 - Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, model: gemini-3.8-flash
  ...limit: 500, model: gemini-3.5-flash-lite  Please retry in 14h27m56s.
  ```
- **Nguyên nhân gốc rễ & cách debug:** `gemini-3.8-flash` chỉ cho 5 request/phút → đổi sang `gemini-3.5-flash-lite`, viết `src/llm.py` tự giãn 12 request/phút và retry theo "Please retry in Xs", cache kết quả enrichment ra file. Một lượt `main.py` tốn khoảng 400 request (112 enrichment + 40 câu trả lời + 2 lượt RAGAS), nên lần chạy lại để thử prompt mới chạm trần **500 request/ngày** giữa lúc RAGAS đang chấm và bị treo; phải đổi sang API key khác mới chạy lại được. Bài học: phải ước lượng tổng số call *trước khi* chạy, và mỗi thí nghiệm RAGAS là một khoản chi quota.

### Lỗi 4 — Tải model 2.3 GB bị đứt
- **Hiện tượng:** `SentenceTransformer("BAAI/bge-m3")` đứng im; cache chỉ có file config, file `.incomplete` hiển thị 0 MB.
- **Nguyên nhân gốc rễ & cách debug:** Mạng ~300 KB/s và trình tải của Hugging Face không ghi ra đĩa cho tới khi xong, nên trông như bị treo; tiến trình tải lại chết khi phiên làm việc đóng. Chuyển sang `curl -C -` để tải nối tiếp từ phần đã có, rồi so SHA-256 với tên blob trước khi đưa vào cache.

### Kiến thức còn thiếu & cách khắc phục
- Chưa nắm cách RAGAS tính từng metric (faithfulness tách mệnh đề, answer relevancy sinh câu hỏi ngược) → cần đọc prompt nội bộ của RAGAS để giải thích chắc chắn vì sao câu trả lời cụt bị 0 điểm (hiện mới là suy luận từ điểm số).
- Chưa có thói quen kiểm tra giới hạn của API bên thứ ba → từ giờ viết script thăm dò (rate limit, tham số được hỗ trợ) trước khi tích hợp.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Lọc & matching JD–CV cho HR

Hệ thống nhận một JD (job description) và một kho CV (PDF), trả về danh sách ứng viên phù hợp nhất kèm lý do, để HR sàng lọc vòng đầu nhanh hơn. Về bản chất đây là bài toán retrieval + rerank + sinh giải thích, với "query" là JD và "tài liệu" là CV.

#### 1. Hiện trạng
- **Pipeline hiện tại:** CV dạng PDF được trích text (có OCR cho CV scan) → embed cả CV thành vector → so cosine similarity với embedding của JD → LLM đọc top CV và viết nhận xét.
- **Vấn đề / Bottlenecks đang gặp:**
  - Embed nguyên CV làm loãng tín hiệu: một CV dài 2 trang có 1 dòng đúng kỹ năng cần tìm vẫn bị xếp thấp.
  - Dense-only bỏ sót từ khóa cứng mà HR lọc theo (tên công nghệ, chứng chỉ, "IELTS 7.0", "5 năm kinh nghiệm") — đúng hiện tượng đã thấy ở lab: dense và BM25 xếp hạng khác nhau.
  - LLM nhận xét đôi khi gán cho ứng viên kỹ năng không có trong CV (hallucination) — với HR đây là lỗi nặng nhất.
  - Chưa có bộ đo: không biết một thay đổi làm kết quả tốt lên hay xấu đi.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** **Structure-aware + hierarchical.** CV có cấu trúc mục rõ (Kinh nghiệm, Kỹ năng, Học vấn, Dự án) nên cắt theo mục giống `chunk_structure_aware()`, mỗi mục/mỗi vị trí công việc là 1 child, cả CV là parent. Search trên child để bắt đúng dòng kỹ năng, trả về parent (cả CV) cho bước đánh giá. Không dùng semantic chunking: lab cho thấy ngưỡng 0.85 cắt vụn thành 208 chunk, CV vốn toàn câu ngắn sẽ còn vụn hơn.
2. **Search retrieval:** **Hybrid BM25 + dense + RRF.** BM25 bắt chính xác tên công nghệ/chứng chỉ/số năm; dense bắt diễn đạt khác từ ("xây dựng API" ↔ "backend development"). Tách JD thành từng yêu cầu (must-have / nice-to-have) và search riêng từng yêu cầu rồi gộp — chính là query decomposition mà failure #1 của lab chỉ ra là còn thiếu.
3. **Reranking:** **Có.** Cross-encoder `bge-reranker-v2-m3` chấm cặp (yêu cầu trong JD, đoạn CV) cho top-20. Lab đo 9–14 s/query trên CPU nên phải chạy theo batch offline khi HR tạo JD mới (không cần real-time), hoặc chạy GPU; nếu cần phản hồi tức thì thì dùng FlashRank.
4. **Evaluation:** Hai lớp. (a) **Retrieval:** bộ ~30 cặp JD–CV do HR gán nhãn phù hợp/không phù hợp, đo Precision@5 và Recall@10 — phù hợp hơn RAGAS vì đầu ra là danh sách xếp hạng. (b) **Phần nhận xét của LLM:** RAGAS **faithfulness** (nhận xét có bám CV không) và **context precision**. Bài học từ lab: ước lượng số request trước khi chạy (20 câu × 4 metric ≈ 200 call) và yêu cầu LLM trả lời thành câu đầy đủ có dẫn chứng để faithfulness không bị 0 oan.
5. **Enrichment:** **Auto metadata + contextual prepend, 1 call/chunk, có cache.** Trích metadata có cấu trúc cho từng CV (`skills`, `years_experience`, `education_level`, `languages`, `last_title`) để lọc cứng trước khi search (ví dụ loại CV dưới số năm kinh nghiệm yêu cầu). Prepend 1 câu ngữ cảnh vào mỗi child ("Đoạn kinh nghiệm tại công ty X, vị trí Y, 2021–2023 của ứng viên Z") vì một dòng kỹ năng đứng riêng không cho biết thuộc ai, giai đoạn nào. HyQA ít giá trị ở đây vì query là JD chứ không phải câu hỏi.

#### 3. Timeline triển khai
- **Tuần 1:** Dựng bộ đánh giá trước: 30 cặp JD–CV có nhãn của HR, đo baseline hiện tại (Precision@5, Recall@10, faithfulness). Cắt CV theo mục + parent/child.
- **Tuần 2:** Thêm BM25 + RRF, tách JD thành từng yêu cầu. Đo lại, so với baseline.
- **Tuần 3:** Enrichment (metadata + contextual prepend, có cache) và bộ lọc cứng theo metadata. Thêm rerank chạy batch; đo latency từng bước như bảng trong lab.
- **Tuần 4:** Sửa prompt nhận xét (bắt buộc trích dẫn dòng CV làm căn cứ), failure analysis theo Error Tree trên bottom-5, chốt cấu hình và demo cho HR.
