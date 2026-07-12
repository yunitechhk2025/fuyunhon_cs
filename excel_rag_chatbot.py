import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class FaqItem:
    sheet: str
    row_index: int
    serial: str
    question: str
    answer: str

    def kb_text(self) -> str:
        prefix = f"序号: {self.serial} | " if self.serial else ""
        return f"{prefix}问题: {self.question} | 回答: {self.answer}"


class ExcelFaqRagBot:
    def __init__(self, excel_path: str, top_k: int = 5, min_score: float = 0.1) -> None:
        self.excel_path = excel_path
        self.top_k = top_k
        self.min_score = min_score
        self.items: List[FaqItem] = []

        # 中文 FAQ 场景，字符 n-gram 检索更稳（无需分词）
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self.doc_vectors = None

    def _find_header_row(
        self, df: pd.DataFrame, question_key: str = "咨询问题", answer_key: str = "解答"
    ) -> Optional[int]:
        for ridx, row in df.iterrows():
            values = [str(v).strip() for v in row.tolist() if str(v).strip() and str(v) != "nan"]
            joined = " | ".join(values)
            if question_key in joined and answer_key in joined:
                return int(ridx)
        return None

    def _sheet_to_faq_items(self, sheet_name: str, df_raw: pd.DataFrame) -> List[FaqItem]:
        if df_raw.empty:
            return []

        header_row_idx = self._find_header_row(df_raw)
        if header_row_idx is None:
            return []

        header_values = df_raw.iloc[header_row_idx].tolist()
        columns = [
            str(v).strip() if str(v).strip() and str(v) != "nan" else f"col_{i}"
            for i, v in enumerate(header_values)
        ]

        df = df_raw.iloc[header_row_idx + 1 :].copy()
        df.columns = columns
        df = df.fillna("")

        question_col = next((c for c in df.columns if "咨询问题" in c or c == "问题"), None)
        answer_col = next((c for c in df.columns if "解答" in c or "答案" in c), None)
        serial_col = next((c for c in df.columns if c in {"序号", "编号"}), None)

        if question_col is None or answer_col is None:
            return []

        items: List[FaqItem] = []
        for idx, row in df.iterrows():
            question = str(row.get(question_col, "")).strip()
            answer = str(row.get(answer_col, "")).strip()
            serial = str(row.get(serial_col, "")).strip() if serial_col else ""

            if not question or question.lower() == "nan":
                continue
            if not answer or answer.lower() == "nan":
                continue

            items.append(
                FaqItem(
                    sheet=sheet_name,
                    row_index=int(idx),
                    serial=serial,
                    question=question,
                    answer=answer,
                )
            )
        return items

    def build_index(self) -> None:
        sheets = pd.read_excel(self.excel_path, sheet_name=None, header=None)
        items: List[FaqItem] = []
        for sheet_name, df in sheets.items():
            items.extend(self._sheet_to_faq_items(sheet_name, df))

        if not items:
            raise ValueError("未识别到 FAQ 问答列（需要包含“咨询问题/解答”字段）。")

        self.items = items
        corpus = [f"问题: {it.question}\n回答: {it.answer}" for it in self.items]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, user_query: str) -> List[Tuple[float, FaqItem]]:
        if self.doc_vectors is None:
            raise RuntimeError("索引尚未构建，请先执行 build_index()。")

        q_vec = self.vectorizer.transform([user_query])
        scores = cosine_similarity(q_vec, self.doc_vectors)[0]
        ranked = sorted(
            [(float(score), self.items[i]) for i, score in enumerate(scores)],
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked[: self.top_k]

    def _render_source_block(self, ranked: Sequence[Tuple[float, FaqItem]]) -> str:
        lines = []
        for i, (score, item) in enumerate(ranked, start=1):
            lines.append(
                (
                    f"[{i}] score={score:.3f} | sheet={item.sheet} | row={item.row_index}\n"
                    f"问题：{item.question}\n"
                    f"回答：{item.answer}"
                )
            )
        return "\n\n".join(lines)

    def _grounded_fallback_answer(self, ranked: Sequence[Tuple[float, FaqItem]]) -> str:
        best_score, best_item = ranked[0]
        if best_score < self.min_score:
            return (
                "抱歉，题库里没有足够匹配的信息。\n"
                "你可以补充关键词（比如：孕妇、敏感肌、使用顺序、搭配禁忌、见效周期）我再帮你查。"
            )

        return (
            "根据题库内容，建议这样回复客户：\n"
            f"{best_item.answer}\n\n"
            f"（命中问题：{best_item.question}，匹配分数：{best_score:.3f}）"
        )

    def _call_ai(
        self,
        user_query: str,
        ranked: Sequence[Tuple[float, FaqItem]],
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("未安装 openai 依赖，请执行: pip install openai") from exc

        sources = self._render_source_block(ranked)
        system_prompt = (
            "你是电商客服助手。你只能依据给定题库片段回答，禁止使用外部知识。\n"
            "如果题库无法支持结论，必须明确说明“题库未覆盖”，并引导用户补充问题。\n"
            "输出风格：口语化、礼貌、简洁，可直接发给客户。\n"
            "不要暴露“分数、sheet、row”等技术信息。"
        )
        user_prompt = (
            f"客户问题：{user_query}\n\n"
            f"题库检索结果：\n{sources}\n\n"
            "请给出最终客服回复。"
        )

        client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def answer(
        self,
        user_query: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        ranked = self.retrieve(user_query)
        if not ranked:
            return "抱歉，题库为空或尚未建立索引。"

        try:
            ai_reply = self._call_ai(
                user_query=user_query,
                ranked=ranked,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
            if ai_reply:
                return ai_reply
        except Exception as exc:  # noqa: BLE001
            return f"{self._grounded_fallback_answer(ranked)}\n\n（AI生成失败，已切换检索直出：{exc}）"

        return self._grounded_fallback_answer(ranked)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Excel FAQ RAG 客服机器人（仅 AI 模式）")
    parser.add_argument("--excel", required=True, help="Excel 文件路径（支持 .xls/.xlsx）")
    parser.add_argument("--top-k", type=int, default=5, help="每次检索候选条数，默认 5")
    parser.add_argument("--min-score", type=float, default=0.1, help="最低命中分数阈值，默认 0.1")
    parser.add_argument("--model", default="gpt-4o-mini", help="AI 模型名")
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容接口地址（可选）")
    parser.add_argument("--api-key", default=None, help="API Key（可选，默认读取 OPENAI_API_KEY）")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="执行一次索引和样例问答后退出，便于快速验证",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    bot = ExcelFaqRagBot(excel_path=args.excel, top_k=args.top_k, min_score=args.min_score)

    try:
        bot.build_index()
    except FileNotFoundError:
        print(f"找不到 Excel 文件: {args.excel}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"加载 Excel 失败: {exc}")
        sys.exit(1)

    print(f"索引完成，共 {len(bot.items)} 条 FAQ。")

    if args.self_test:
        sample_q = "孕妇能用吗"
        print(f"\n[自测问题] {sample_q}")
        print(bot.answer(sample_q, model=args.model, base_url=args.base_url, api_key=args.api_key))
        return

    print("客服机器人已启动。输入 exit 退出。")
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("机器人: 已退出。")
            break

        reply = bot.answer(
            user_input,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        print(f"\n机器人:\n{reply}")


if __name__ == "__main__":
    main()
