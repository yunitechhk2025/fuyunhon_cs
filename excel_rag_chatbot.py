import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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


@dataclass
class AnswerResult:
    text: str
    matched: bool
    score: float
    matched_question: Optional[str] = None
    matched_answer: Optional[str] = None


class ExcelFaqRagBot:
    def __init__(self, excel_path: str, top_k: int = 5, min_score: float = 0.05) -> None:
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
        # 问题权重更高，回答作为补充，提升短问命中率
        corpus = [f"{it.question} {it.question} {it.answer}" for it in self.items]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, user_query: str) -> List[Tuple[float, FaqItem]]:
        if self.doc_vectors is None:
            raise RuntimeError("索引尚未构建，请先执行 build_index()。")

        q_vec = self.vectorizer.transform([user_query])
        scores = cosine_similarity(q_vec, self.doc_vectors)[0]

        # 额外叠加关键词重合分，避免纯 TF-IDF 对短中文问句过严
        query_chars = set(user_query.replace(" ", ""))
        for i, item in enumerate(self.items):
            q_chars = set(item.question.replace(" ", ""))
            if not query_chars or not q_chars:
                continue
            overlap = len(query_chars & q_chars) / max(len(query_chars), 1)
            scores[i] = float(scores[i]) + 0.25 * overlap

        ranked = sorted(
            [(float(score), self.items[i]) for i, score in enumerate(scores)],
            key=lambda x: x[0],
            reverse=True,
        )
        return ranked[: self.top_k]

    def _not_found_text(self) -> str:
        return (
            "亲，这个问题题库里暂时没有完全对应的说明。"
            "你可以再补充一下具体场景，比如：孕妇/备孕、敏感肌、使用顺序、搭配禁忌、见效周期，我再帮你查。"
        )

    def _call_ai_match(
        self,
        user_query: str,
        ranked: Sequence[Tuple[float, FaqItem]],
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Tuple[int, str]:
        """让 AI 在关键词检索出的候选题库问题中做语义匹配（允许措辞、语序、同义表达不同，
        只要客户意图与某条候选实质相同即可），并生成引出官方说明的开场白。
        绝不让 AI 生成或复述专业内容本身，专业内容始终由调用方原文拼接，确保逐字一致。

        返回 (候选序号, 开场白)；序号为 0 表示 AI 判断所有候选都不匹配。
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("未安装 openai 依赖，请执行: pip install openai") from exc

        candidates = "\n".join(f"{i}. {item.question}" for i, (_, item) in enumerate(ranked, start=1))
        system_prompt = (
            "你是电商客服助手的题库匹配模块，只能依据候选题库问题做判断，禁止使用外部知识回答专业内容。\n"
            "任务：\n"
            "1. 判断客户问题在意图上最匹配下面哪一条候选题库问题——即使措辞、语序、同义表达不同，"
            "只要客户实际想问的事情和某条候选实质相同，就算匹配。\n"
            "2. 如果确实匹配到某一条，生成一句简短亲切、口语化的开场白/过渡语（不超过20个字），"
            "用于引出接下来将原文展示的官方说明；开场白中严禁提及、复述或猜测任何成分、功效、用法用量、"
            "禁忌等专业内容，那部分会在开场白之后原文附加，不需要你生成。\n"
            "3. 如果没有任何一条候选真正匹配客户的意图，index 填 0，greeting 填空字符串。\n"
            "4. 只输出如下格式的 JSON，不要输出任何多余文字、解释或代码块标记：\n"
            '{"index": 数字, "greeting": "字符串"}'
        )
        user_prompt = f"客户问题：{user_query}\n\n候选题库问题：\n{candidates}"

        client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return 0, ""
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            return 0, ""

        try:
            idx = int(payload.get("index", 0) or 0)
        except (TypeError, ValueError):
            idx = 0
        greeting = str(payload.get("greeting", "") or "").strip()
        return idx, greeting

    def answer(
        self,
        user_query: str,
        model: str = "qwen3.6-flash",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AnswerResult:
        ranked = self.retrieve(user_query)
        if not ranked:
            return AnswerResult(text="抱歉，题库为空或尚未建立索引。", matched=False, score=0.0)

        selection: Optional[Tuple[int, str]] = None
        try:
            # 始终把候选交给 AI 做语义判断，不因关键词分数偏低而提前拒判，
            # 这样措辞不同但意思相同的相似问题也能命中题库答案。
            selection = self._call_ai_match(
                user_query=user_query,
                ranked=ranked,
                model=model,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] AI 语义匹配失败，已回退到关键词检索结果: {exc}", file=sys.stderr)

        if selection is not None:
            idx, greeting = selection
            if idx and 1 <= idx <= len(ranked):
                score, item = ranked[idx - 1]
                if not greeting:
                    greeting = "亲，为您查询到以下官方说明："
                # 专业内容始终是题库原文的直接拼接，不经过 AI 改写，确保逐字一致
                text = f"{greeting}\n{item.answer}"
                return AnswerResult(
                    text=text,
                    matched=True,
                    score=score,
                    matched_question=item.question,
                    matched_answer=item.answer,
                )
            return AnswerResult(text=self._not_found_text(), matched=False, score=ranked[0][0])

        # AI 调用异常时的兜底：仅依赖关键词检索分数
        best_score, best_item = ranked[0]
        if best_score < self.min_score:
            return AnswerResult(text=self._not_found_text(), matched=False, score=best_score)
        text = f"亲，为您查询到以下官方说明：\n{best_item.answer}"
        return AnswerResult(
            text=text,
            matched=True,
            score=best_score,
            matched_question=best_item.question,
            matched_answer=best_item.answer,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Excel FAQ RAG 客服机器人（仅 AI 模式）")
    parser.add_argument("--excel", required=True, help="Excel 文件路径（支持 .xls/.xlsx）")
    parser.add_argument("--top-k", type=int, default=5, help="每次检索候选条数，默认 5")
    parser.add_argument("--min-score", type=float, default=0.05, help="最低命中分数阈值，默认 0.05")
    parser.add_argument("--model", default="qwen3.6-flash", help="AI 模型名（DashScope 如 qwen3.6-flash）")
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
        result = bot.answer(sample_q, model=args.model, base_url=args.base_url, api_key=args.api_key)
        print(result.text)
        print(f"\n[检索标注] matched={result.matched} score={result.score:.3f} matched_question={result.matched_question!r}")
        return

    print("客服机器人已启动。输入 exit 退出。")
    while True:
        user_input = input("\n你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("机器人: 已退出。")
            break

        result = bot.answer(
            user_input,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
        print(f"\n机器人:\n{result.text}")
        print(f"[检索标注] matched={result.matched} score={result.score:.3f} matched_question={result.matched_question!r}")


if __name__ == "__main__":
    main()
