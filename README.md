# Excel FAQ RAG 客服机器人

这是一个可直接落地的版本：把 Excel 常见问题库作为唯一知识源，先做检索，再用 AI 生成更自然的客服话术。  
机器人只提供“题库 + AI”模式，且被限制为“只依据检索结果回答”。

## 1) 准备 Excel 题库

把你的 Excel 文件放到当前目录，例如：`客服知识库.xlsx`。

支持 `.xls` / `.xlsx`，支持多 sheet。

当前脚本优先识别包含以下字段的 FAQ 表头：

- `咨询问题`
- `解答`
- （可选）`序号` / `编号`

当前 `docker-compose.yml` 默认挂载这个文件：

```text
2026.01.26_肤润康-常见咨询问题_v2(1).xls
```

如果你更换文件名，需要同步修改 `docker-compose.yml` 里的 `volumes`。

## 2) 配置 .env

打开 `.env`，把第一行替换成真实 Key：

```env
OPENAI_API_KEY=你的真实key
```

如果你是兼容接口（例如本地网关），再加：

```env
OPENAI_BASE_URL=https://你的兼容接口/v1
```

常用配置：

```env
OPENAI_MODEL=gpt-4o-mini
FAQ_TOP_K=5
FAQ_MIN_SCORE=0.1
APP_PORT=8000
```

## 3) Docker 启动

```bash
docker compose up --build
```

然后打开：

```text
http://127.0.0.1:8000
```

后台运行：

```bash
docker compose up -d --build
```

停止：

```bash
docker compose down
```

查看日志：

```bash
docker compose logs -f
```

## 4) 本地开发（可选）

安装依赖：

```bash
pip install -r requirements.txt
```

启动网页客服：

```bash
uvicorn web_app:app --reload --host 127.0.0.1 --port 8000
```

命令行版本：

```bash
python excel_rag_chatbot.py --excel "2026.01.26_肤润康-常见咨询问题_v2(1).xls" --model gpt-4o-mini
```

## 5) 参数说明

- `--top-k 3`：每次返回的知识条数，默认 3
- `--min-score 0.1`：最低命中阈值，太低就会拒答并引导补充问题
- `--self-test`：跑一次样例问答后退出，便于快速验收

网页版本也支持这些环境变量：

- `FAQ_EXCEL_PATH`：Excel 路径，默认读取当前目录的肤润康题库
- `FAQ_TOP_K`：检索候选条数，默认 5
- `FAQ_MIN_SCORE`：最低命中阈值，默认 0.1
- `OPENAI_MODEL`：模型名，默认 `gpt-4o-mini`

---

如果你要继续升级，我可以下一步直接给你加：

- FastAPI 接口（可接企微/小程序）
- 会话上下文记忆（多轮客服）
- 敏感词和合规兜底规则
