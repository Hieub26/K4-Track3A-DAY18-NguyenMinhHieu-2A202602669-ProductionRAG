from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (TEST_SET_PATH, GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL,
                    GEMINI_EMBEDDING_MODEL, GEMINI_RPM)

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {**{m: 0.0 for m in METRICS}, "per_question": []}
    # RAGAS cần LLM judge → không có key thì trả 0 luôn, không gọi API.
    if not GEMINI_API_KEY or not questions:
        return zeros
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.run_config import RunConfig
        from datasets import Dataset
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        # RAGAS mặc định gọi OpenAI → truyền llm + embeddings trỏ sang Gemini (OpenAI-compatible).
        llm = ChatOpenAI(
            model=GEMINI_MODEL, api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL, max_retries=6,
            model_kwargs={"reasoning_effort": "low"},
            rate_limiter=InMemoryRateLimiter(requests_per_second=GEMINI_RPM / 60, max_bucket_size=1),
        )
        # check_embedding_ctx_length=False: gửi text thô thay vì token ids của tiktoken (Gemini không nhận).
        embeddings = OpenAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, api_key=GEMINI_API_KEY,
                                      base_url=GEMINI_BASE_URL, check_embedding_ctx_length=False)
        # Gemini không hỗ trợ n > 1 ("Multiple candidates is not enabled") → sinh 1 câu hỏi ngược/answer.
        answer_relevancy.strictness = 1

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm, embeddings=embeddings,
            # Free tier giới hạn request/phút → chạy tuần tự, timeout dài để chờ rate limiter.
            run_config=RunConfig(max_workers=2, timeout=600, max_retries=6, max_wait=90),
            raise_exceptions=False,
        )
        df = result.to_pandas()

        def _score(row, metric: str) -> float:
            value = row.get(metric)
            # RAGAS trả NaN khi judge lỗi/parse hỏng → tính là 0 để không làm hỏng trung bình & JSON.
            return 0.0 if value is None or value != value else float(value)

        per_question = [EvalResult(
            question=row["question"], answer=row["answer"],
            contexts=list(row["contexts"]), ground_truth=row["ground_truth"],
            **{m: _score(row, m) for m in METRICS},
        ) for _, row in df.iterrows()]
        aggregate = {m: sum(getattr(r, m) for r in per_question) / len(per_question) for m in METRICS}
        return {**aggregate, "per_question": per_question}
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


# Diagnostic Tree: metric thấp nhất → (chẩn đoán, hướng sửa)
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
    "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    analyzed = []
    for r in eval_results:
        scores = {m: getattr(r, m) for m in METRICS}
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[worst_metric]
        analyzed.append({
            "question": r.question,
            "answer": r.answer,
            "ground_truth": r.ground_truth,
            "avg_score": sum(scores.values()) / len(scores),
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "scores": scores,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    analyzed.sort(key=lambda f: f["avg_score"])
    return analyzed[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
