import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
from auth import create_token, decode_token, get_current_agent, require_admin
from doc_rag_chatbot import DocRagBot
from excel_rag_chatbot import AnswerResult, ExcelFaqRagBot
from ws_manager import manager

DEFAULT_EXCEL = "2026.01.26_肤润康-常见咨询问题_v2(1).xls"
UREA_DOC = "urea_hand_cream_info.md"
DEFAULT_MODEL = "qwen3.6-flash"
VALID_MODES = {"auto", "manual", "collab"}
DEFAULT_COLLAB_AUTO_SEND_SECONDS = 5
AUTO_SEND_AGENT_NAME = "AI自动发送"

# 两款产品用了两种不同的知识来源：
# - 杜鹃花酸乳霜：现成的"问题-答案"题库（Excel），专业内容原文照搬，逐字不改写。
# - 10%尿素护手霜：暂无题库，只有一份产品说明文档（doc），没有固定问答对，
#   AI 需要现场组织语言回答，但内容必须严格限定在文档范围内——文档没提到的内容
#   （如孕妇能否使用等）一律视为未命中，与题库场景未命中时的转人工规则完全一致。
PRODUCTS: dict = {
    "azelaic_cream": {"label": "澳洲肤润康 杜鹃花酸乳霜", "excel": DEFAULT_EXCEL},
    "urea_hand_cream": {"label": "澳洲肤润康 10%尿素护手霜", "doc": UREA_DOC},
}
DEFAULT_PRODUCT = "azelaic_cream"
NO_KB_TEXT = "亲，这款产品的常见问题库还在整理中，已为您转接人工客服，请稍候~"


class AskRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    product: Optional[str] = None


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


class CollabTimeoutRequest(BaseModel):
    seconds: float


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

# 每个有题库的产品各自一个 bot 实例；没有题库的产品不在此字典中出现。
bots: dict = {}


@app.on_event("startup")
def startup() -> None:
    database.init_db()

    top_k = int(os.getenv("FAQ_TOP_K", "8"))
    min_score = float(os.getenv("FAQ_MIN_SCORE", "0.1"))

    for product_id, meta in PRODUCTS.items():
        excel_name = meta.get("excel")
        doc_name = meta.get("doc")

        if excel_name:
            default_path = str(BASE_DIR / excel_name)
            # 兼容旧的 FAQ_EXCEL_PATH 环境变量，仅对默认产品（杜鹃花酸乳霜）生效
            excel_path = os.getenv("FAQ_EXCEL_PATH", default_path) if product_id == DEFAULT_PRODUCT else default_path
            if not Path(excel_path).exists():
                print(f"[warn] 产品「{meta['label']}」配置的题库文件不存在，跳过: {excel_path}", file=sys.stderr)
                continue
            excel_bot = ExcelFaqRagBot(excel_path=excel_path, top_k=top_k, min_score=min_score)
            excel_bot.build_index()
            bots[product_id] = excel_bot
        elif doc_name:
            doc_path = str(BASE_DIR / doc_name)
            if not Path(doc_path).exists():
                print(f"[warn] 产品「{meta['label']}」配置的说明文档不存在，跳过: {doc_path}", file=sys.stderr)
                continue
            doc_bot = DocRagBot(doc_path=doc_path, top_k=4, min_score=float(os.getenv("DOC_MIN_SCORE", "0.15")))
            doc_bot.build_index()
            bots[product_id] = doc_bot

    # 品牌类问题（如"是澳洲品牌？"）会被标记为 shared=True，属于跨产品共用问题：
    # 无论客户当前选的是哪款产品，都应该能命中——即便该产品自己还没有专属题库。
    shared_items = [it for b in bots.values() for it in b.items if it.shared]
    if shared_items:
        for product_id in PRODUCTS:
            existing_bot = bots.get(product_id)
            if existing_bot is not None:
                existing_bot.load_extra_items(shared_items)
            else:
                bots[product_id] = ExcelFaqRagBot.from_items(shared_items, top_k=top_k, min_score=min_score)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/agent")
def agent_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "agent.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "products": {pid: len(b.items) for pid, b in bots.items()},
    }


@app.get("/api/products")
def get_products() -> dict:
    return {
        "items": [
            {"id": pid, "label": meta["label"], "has_kb": pid in bots}
            for pid, meta in PRODUCTS.items()
        ],
        "default": DEFAULT_PRODUCT,
    }


def _normalize_product(product: Optional[str]) -> str:
    return product if product in PRODUCTS else DEFAULT_PRODUCT


def _ai_model(model: Optional[str]) -> str:
    return model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _generate_ai_reply(product: str, question: str, model: Optional[str]) -> AnswerResult:
    product_bot = bots.get(product)
    if product_bot is None:
        # 该产品还没有题库，直接判定未命中，不消耗 AI 调用
        return AnswerResult(text=NO_KB_TEXT, matched=False, score=0.0)
    return product_bot.answer(
        question,
        model=_ai_model(model),
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _client_ip(request: Request) -> Optional[str]:
    """获取访客真实 IP：优先取反向代理头（若未来接入 nginx/CDN），否则取连接的源地址。
    仅用于客服工作台展示参考，不作为区分用户的依据（同一 IP 下可能有多个真实客户）。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else None


async def _auto_send_after_timeout(conversation_id: int, ai_answer: str, timeout: float) -> None:
    """人机协同模式下，若客服在超时时间内还没有认领处理，则自动把 AI 建议发送给客户，
    避免客户长时间等待；一旦客服已认领（状态不再是 pending），则尊重人工处理，不再自动发送。"""
    try:
        await asyncio.sleep(timeout)
        conversation = database.get_conversation(conversation_id)
        if conversation is None or conversation["status"] != "pending":
            return
        database.mark_answered(conversation_id, ai_answer, answered_by=None, answered_by_name=AUTO_SEND_AGENT_NAME)
        await manager.broadcast({"type": "answered", "id": conversation_id})
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 协同模式自动发送失败: {exc}", file=sys.stderr)


def _log_retrieval_only(conversation_id: int, product: str, question: str) -> None:
    """全人工模式下不调用 AI 生成回复，但仍在后台记录题库检索结果，便于客服复核。"""
    product_bot = bots.get(product)
    if product_bot is None:
        database.set_retrieval_info(conversation_id, False, None, None, 0.0)
        return
    try:
        ranked = product_bot.retrieve(question)
        if ranked and ranked[0][0] >= product_bot.min_score:
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


@app.get("/api/collab-timeout")
def get_collab_timeout() -> dict:
    seconds = float(database.get_setting("collab_auto_send_seconds", str(DEFAULT_COLLAB_AUTO_SEND_SECONDS)))
    return {"seconds": seconds}


@app.post("/api/collab-timeout")
def set_collab_timeout(req: CollabTimeoutRequest, agent: dict = Depends(require_admin)) -> dict:
    if req.seconds < 1 or req.seconds > 300:
        raise HTTPException(status_code=400, detail="超时时间需在 1-300 秒之间")
    database.set_setting("collab_auto_send_seconds", str(req.seconds))
    return {"seconds": req.seconds}


# ---------------- 客户端提问 ----------------

@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request) -> AskResponse:
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = req.session_id or str(uuid.uuid4())
    product = _normalize_product(req.product)
    mode = database.get_setting("global_mode", "auto")
    conversation_id = database.create_conversation(session_id, question, mode, _client_ip(request), product)

    # 未命中题库（包括该产品尚未建立题库的情况）时，任何模式都不允许 AI 直接回复或编造答案，
    # 统一转人工处理；只有确认命中题库时，才允许由 AI 直接回复（全AI模式）或生成建议（协同模式）。
    if mode == "auto":
        result: Optional[AnswerResult] = None
        try:
            result = _generate_ai_reply(product, question, req.model)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 全AI模式生成回复失败: {exc}", file=sys.stderr)

        if result is not None and result.matched:
            database.set_retrieval_info(
                conversation_id, True, result.matched_question, result.matched_answer, result.score
            )
            database.mark_answered(conversation_id, result.text)
            return AskResponse(conversation_id=conversation_id, status="answered", answer=result.text, mode=mode)

        database.set_retrieval_info(
            conversation_id, False, None, None, result.score if result is not None else 0.0
        )

    elif mode == "collab":
        try:
            result = _generate_ai_reply(product, question, req.model)
            database.set_retrieval_info(
                conversation_id, result.matched, result.matched_question, result.matched_answer, result.score
            )
            if result.matched:
                database.set_ai_suggestion(conversation_id, result.text, result.score)
                timeout = float(
                    database.get_setting("collab_auto_send_seconds", str(DEFAULT_COLLAB_AUTO_SEND_SECONDS))
                )
                auto_send_at = (datetime.utcnow() + timedelta(seconds=timeout)).strftime("%Y-%m-%dT%H:%M:%SZ")
                database.set_auto_send_at(conversation_id, auto_send_at)
                asyncio.create_task(_auto_send_after_timeout(conversation_id, result.text, timeout))
            # 未命中：不生成 AI 建议、不安排自动发送，完全交给客服人工处理
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 协同模式生成AI建议失败: {exc}", file=sys.stderr)
            database.set_retrieval_info(conversation_id, False, None, None, 0.0)
    elif mode == "manual":
        _log_retrieval_only(conversation_id, product, question)

    # 走到这里说明本次提问需要人工处理（全人工模式 / 协同或全AI模式下未命中题库）
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


@app.get("/api/agent/sessions")
def agent_sessions(agent: dict = Depends(get_current_agent)) -> dict:
    """按用户（session_id）分组的对话列表：一个用户对应一个对话框，而不是每条问题单独一个。"""
    return {"items": database.list_sessions()}


@app.get("/api/agent/sessions/{session_id}")
def agent_session_messages(session_id: str, agent: dict = Depends(get_current_agent)) -> dict:
    """某个用户会话下的全部提问/回复，按时间顺序返回，用于渲染连续对话。"""
    return {"items": database.list_session_messages(session_id)}


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
    if conversation["status"] == "answered":
        raise HTTPException(status_code=409, detail="该问题已被回复，请勿重复发送")

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
