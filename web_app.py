import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from excel_rag_chatbot import ExcelFaqRagBot


DEFAULT_EXCEL = "2026.01.26_肤润康-常见咨询问题_v2(1).xls"
DEFAULT_MODEL = "qwen3.6-flash"


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str


app = FastAPI(title="Excel FAQ AI 客服机器人")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

bot = ExcelFaqRagBot(
    excel_path=os.getenv("FAQ_EXCEL_PATH", str(BASE_DIR / DEFAULT_EXCEL)),
    top_k=int(os.getenv("FAQ_TOP_K", "5")),
    min_score=float(os.getenv("FAQ_MIN_SCORE", "0.1")),
)


@app.on_event("startup")
def startup() -> None:
    bot.build_index()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "faq_count": len(bot.items)}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    answer = bot.answer(
        question,
        model=req.model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return ChatResponse(answer=answer)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
