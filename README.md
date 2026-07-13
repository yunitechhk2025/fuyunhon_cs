# Excel FAQ RAG 客服机器人

这是一个可直接落地的版本：把 Excel 常见问题库作为唯一知识源，先做检索，再用 AI 生成更自然的客服话术。

现已支持 **三种工作模式**，可在客服工作台中随时切换：

| 模式 | 说明 |
| --- | --- |
| 全AI (`auto`) | AI 自动检索题库并直接回复客户，无需人工介入（原有行为，默认模式） |
| 全人工 (`manual`) | 所有问题进入人工队列，由客服手动查看并回复，AI 不参与 |
| 人机协同 (`collab`) | AI 先生成建议回复（仅客服可见），客服审核/编辑后再发送给客户 |

客户网页 (`/`) 始终只显示一个聊天框；如命中全人工或协同模式，客户会看到“正在转接人工客服”的提示并自动轮询等待回复，无需手动刷新。

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
FAQ_TOP_K=8
FAQ_MIN_SCORE=0.1
APP_PORT=8000
```

客服工作台相关配置（新增）：

```env
# JWT 签名密钥，务必修改为随机字符串
JWT_SECRET=change_this_to_a_random_secret_string

# 首次启动自动创建的管理员账号，登录后请立即修改密码
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change_this_password

# 系统首次启动时的默认模式：auto / manual / collab
DEFAULT_MODE=auto
```

> 首次启动会自动在数据库里创建管理员账号（用户名/密码取自上面的环境变量），日志里也会打印一次。数据库文件保存在 `data/qa.db`，`docker-compose.yml` 已挂载该目录为宿主机卷，重启/重新构建容器不会丢失客服账号、模式设置与历史对话。

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

## 4) 客服工作台

访问：

```text
http://127.0.0.1:8000/agent
```

用 `.env` 中配置的管理员账号登录（默认 `admin` / 见 `ADMIN_PASSWORD`）。

- **工作模式卡片**（仅管理员可切换）：点击即可在全AI / 全人工 / 人机协同之间切换，立即生效，无需重启服务。
- **待处理队列**：全人工/协同模式下的新问题会实时推送到所有在线客服（WebSocket + 5 秒兜底轮询），支持多人同时在线。
  - 协同模式下，队列条目会显示 AI 生成的建议回复（含匹配度），客服可直接采纳或编辑后发送。
  - 点击「认领此问题」后，该问题会标记为你在处理，其他客服会看到「客服 X 正在处理中」，避免重复回复。
  - 「释放认领」可把问题放回待处理队列，交给其他客服处理。
- **历史记录**：展示最近 50 条问题（覆盖全AI、全人工、人机协同三种模式），每条都会标注「题库检索标注」——命中了题库中的哪个问题及对应答案，或提示未命中，方便复核 AI 是否按题库准确回复。
- **客服账号管理**（仅管理员）：可在工作台内直接创建新的客服/管理员账号。

> 建议登录后立即修改默认管理员密码：可调用 `POST /api/agent/change-password`（`{"old_password":"...","new_password":"..."}`，需携带登录返回的 `token`）。

## 5) 本地开发（可选）

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

## 6) 参数说明

**回答机制**：每次提问分别用关键词检索（TF-IDF）和语义向量检索（embedding 余弦相似度）
各取 `top-k` 条候选，取并集后交给 AI 在候选中做语义判断——即使客户的措辞、语序、同义/口语化
表达（例如"涨豆豆"对应"长痘"）与题库原文不同，只要意图相同即可命中；语义检索弥补了纯关键词
检索对这类改写问句召回不足的问题。命中后，AI 只生成一句开场白/过渡语，专业内容（成分、功效、
用法用量、禁忌等）始终是题库答案原文的直接拼接，不经过 AI 改写，确保逐字一致。如果 AI 判断
所有候选都不匹配，或调用失败兜底走关键词分数判断，会返回引导客户补充关键词的话术。每次提问的
检索结果（命中的题库问题/答案，或未命中）都会记录在后台，客服工作台的问题卡片上会显示
「题库检索标注」供人工核对。首次启动时会调用一次 embedding 接口为所有题库问题建立语义索引，
如果该接口调用失败（无网络/无 API Key），会自动降级为纯关键词检索，不影响其余功能。

- `--top-k 3`：每次返回的知识条数，默认 3
- `--min-score 0.1`：最低命中阈值，太低就会拒答并引导补充问题
- `--self-test`：跑一次样例问答后退出，便于快速验收

网页版本也支持这些环境变量：

- `FAQ_EXCEL_PATH`：Excel 路径，默认读取当前目录的肤润康题库
- `FAQ_TOP_K`：关键词检索和语义检索各自的候选条数（取并集后交给 AI 做语义匹配），默认 8
- `FAQ_MIN_SCORE`：AI 语义匹配调用失败时，兜底关键词检索所用的最低命中阈值，默认 0.1
- `FAQ_EMBED_MODEL`：语义检索所用的 embedding 模型，默认 `text-embedding-v3`
- `OPENAI_MODEL`：模型名，默认 `qwen3.6-flash`
- `OPENAI_BASE_URL`：默认 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`

## 7) GitHub 自动部署到阿里云 ECS

仓库已包含 `.github/workflows/deploy.yml`。

在 GitHub → Settings → Secrets and variables → Actions 添加：

- `ECS_HOST`：服务器公网 IP
- `ECS_USER`：SSH 用户名（如 `root`）
- `ECS_PASSWORD`：SSH 密码

服务器需先手动部署一次，并保留 `~/fuyunhon_cs/.env`（不要提交到 Git）。

之后每次 `push` 到 `main`，GitHub Actions 会自动：

1. SSH 登录 ECS
2. `git pull origin main`
3. `docker compose up -d --build`

> **升级到三种模式后的一次性操作**：由于 `.env` 不会被 `git pull` 更新，首次部署本次更新前，请手动 SSH 登录服务器，在 `~/fuyunhon_cs/.env` 中补充 `JWT_SECRET`、`ADMIN_USERNAME`、`ADMIN_PASSWORD`、`DEFAULT_MODE`（参考 `.env.example`），保存后再让 GitHub Actions 触发部署，或手动执行一次 `docker compose up -d --build`。同时确保服务器上存在 `~/fuyunhon_cs/data` 目录（`docker-compose.yml` 已配置挂载，用于持久化客服账号与对话数据）。
