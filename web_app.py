import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
from auth import create_token, decode_token, get_current_agent, require_admin
from excel_rag_chatbot import AnswerResult, ExcelFaqRagBot
from ws_manager import manager

DEFAULT_EXCEL = "2026.01.26_肤润康-常见咨询问题_v2(1).xls"
DEFAULT_MODEL = "qwen3.6-flash"
VALID_MODES = {"auto", "manual", "collab"}


class AskRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None


class AskResponse(BaseModel):
    conversation_id: int
    status: str  # 'answered' | 'pending'
    answer: Optional[str] = None
    mode: str


class LoginRequest(BaseModel):
    username: str
    password: str


class AnswerRequest(BaseModel):
    answer: str


class ModeRequest(BaseModel):
    mode: str


class CreateAgentRequest(BaseModel):
    username: str
    password: str
    display_name: str
    role: Optional[str] = "agent"


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


app = FastAPI(title="Excel FAQ AI 客服机器人")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

bot = ExcelFaqRagBot(
    excel_path=os.getenv("FAQ_EXCEL_PATH", str(BASE_DIR / DEFAULT_EXCEL)),
    top_k=int(os.getenv("FAQ_TOP_K", "8")),
    min_score=float(os.getenv("FAQ_MIN_SCORE", "0.1")),
)


@app.on_event("startup")
def startup() -> None:
    database.init_db()
    bot.build_index()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/agent")
def agent_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "agent.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "faq_count": len(bot.items)}


def _ai_model(model: Optional[str]) -> str:
    return model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _generate_ai_reply(question: str, model: Optional[str]) -> AnswerResult:
    return bot.answer(
        question,
        model=_ai_model(model),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _log_retrieval_only(conversation_id: int, question: str) -> None:
    """全人工模式下不调用 AI，但仍在后台记录题库检索结果，便于复核。"""
    try:
        ranked = bot.retrieve(question)
        if ranked and ranked[0][0] >= bot.min_score:
            score, item = ranked[0]
            database.set_retrieval_info(conversation_id, True, item.question, item.answer, score)
        else:
            database.set_retrieval_info(conversation_id, False, None, None, ranked[0][0] if ranked else 0.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 检索标注失败: {exc}", file=sys.stderr)


# ---------------- 模式设置 ----------------

@app.get("/api/mode")
def get_mode() -> dict:
    mode = database.get_setting("global_mode", "auto")
    return {"mode": mode}


@app.post("/api/mode")
def set_mode(req: ModeRequest, agent: dict = Depends(require_admin)) -> dict:
    if req.mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail="模式必须是 auto / manual / collab 之一")
    database.set_setting("global_mode", req.mode)
    return {"mode": req.mode}


# ---------------- 客户端提问 ----------------

@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = req.session_id or str(uuid.uuid4())
    mode = database.get_setting("global_mode", "auto")
    conversation_id = database.create_conversation(session_id, question, mode)

    if mode == "auto":
        try:
            result = _generate_ai_reply(question, req.model)
            reply = result.text
            database.set_retrieval_info(
                conversation_id, result.matched, result.matched_question, result.matched_answer, result.score
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 全AI模式生成回复失败: {exc}", file=sys.stderr)
            reply = "抱歉，暂时无法生成回复，请稍后再试或联系人工客服。"
        database.mark_answered(conversation_id, reply)
        return AskResponse(conversation_id=conversation_id, status="answered", answer=reply, mode=mode)

    # manual / collab：不直接回复客户，进入人工队列
    if mode == "collab":
        try:
            result = _generate_ai_reply(question, req.model)
            database.set_ai_suggestion(conversation_id, result.text, result.score)
            database.set_retrieval_info(
                conversation_id, result.matched, result.matched_question, result.matched_answer, result.score
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 协同模式生成AI建议失败: {exc}", file=sys.stderr)
    elif mode == "manual":
        _log_retrieval_only(conversation_id, question)

    conversation = database.get_conversation(conversation_id)
    await manager.broadcast({"type": "new_question", "conversation": dict(conversation)})

    return AskResponse(conversation_id=conversation_id, status="pending", answer=None, mode=mode)


@app.get("/api/conversations/{conversation_id}")
def get_conversation_status(conversation_id: int) -> dict:
    conversation = database.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {
        "id": conversation["id"],
        "status": conversation["status"],
        "answer": conversation["final_answer"],
    }


# ---------------- 客服端 ----------------

@app.post("/api/agent/login")
def agent_login(req: LoginRequest) -> dict:
    agent_row = database.get_agent_by_username(req.username)
    if agent_row is None or not database.verify_password(req.password, agent_row["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_token(agent_row)
    return {
        "token": token,
        "agent": {
            "id": agent_row["id"],
            "username": agent_row["username"],
            "displayName": agent_row["display_name"],
            "role": agent_row["role"],
        },
    }


@app.get("/api/agent/me")
def agent_me(agent: dict = Depends(get_current_agent)) -> dict:
    return {"agent": agent}


@app.post("/api/agent/change-password")
def agent_change_password(
    req: ChangePasswordRequest, agent: dict = Depends(get_current_agent)
) -> dict:
    agent_row = database.get_agent_by_id(agent["id"])
    if not database.verify_password(req.old_password, agent_row["password_hash"]):
        raise HTTPException(status_code=401, detail="旧密码不正确")
    database.update_agent_password(agent["id"], req.new_password)
    return {"success": True}


@app.get("/api/agent/queue")
def agent_queue(agent: dict = Depends(get_current_agent)) -> dict:
    return {"items": database.list_queue()}


@app.get("/api/agent/history")
def agent_history(agent: dict = Depends(get_current_agent)) -> dict:
    return {"items": database.list_recent(50)}


@app.post("/api/agent/claim/{conversation_id}")
async def agent_claim(conversation_id: int, agent: dict = Depends(get_current_agent)) -> dict:
    ok = database.claim_conversation(conversation_id, agent["id"], agent["display_name"])
    if not ok:
        raise HTTPException(status_code=409, detail="该问题已被其他客服认领或已完成")
    await manager.broadcast({"type": "claimed", "id": conversation_id, "agent": agent["display_name"]})
    return {"success": True}


@app.post("/api/agent/release/{conversation_id}")
async def agent_release(conversation_id: int, agent: dict = Depends(get_current_agent)) -> dict:
    ok = database.release_conversation(conversation_id, agent["id"])
    if not ok:
        raise HTTPException(status_code=409, detail="无法释放：该问题不是由你认领")
    await manager.broadcast({"type": "released", "id": conversation_id})
    return {"success": True}


@app.post("/api/agent/answer/{conversation_id}")
async def agent_answer(
    conversation_id: int, req: AnswerRequest, agent: dict = Depends(get_current_agent)
) -> dict:
    answer_text = req.answer.strip()
    if not answer_text:
        raise HTTPException(status_code=400, detail="回复内容不能为空")

    conversation = database.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="对话不存在")

    database.mark_answered(conversation_id, answer_text, agent["id"], agent["display_name"])
    await manager.broadcast({"type": "answered", "id": conversation_id})
    return {"success": True}


@app.get("/api/agent/agents")
def list_agent_accounts(agent: dict = Depends(require_admin)) -> dict:
    return {"agents": database.list_agents()}


@app.post("/api/agent/agents")
def create_agent_account(req: CreateAgentRequest, agent: dict = Depends(require_admin)) -> dict:
    existing = database.get_agent_by_username(req.username)
    if existing is not None:
        raise HTTPException(status_code=409, detail="用户名已存在")
    role = req.role if req.role in {"agent", "admin"} else "agent"
    agent_id = database.create_agent(req.username, req.password, req.display_name, role)
    return {"id": agent_id, "username": req.username, "displayName": req.display_name, "role": role}


@app.websocket("/ws/agent")
async def agent_ws(websocket: WebSocket, token: str = "") -> None:
    try:
        decode_token(token)
    except HTTPException:
        await websocket.close(code=4401)
        return

    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
