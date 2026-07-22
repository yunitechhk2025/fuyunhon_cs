"""一次性维护脚本：清空全部对话相关数据（测试数据清理用）。

会删除：
  - conversations（所有提问/回复记录）
  - chat_messages（客户端气泡明细，刷新重放用的那份）

不会动：
  - agents（客服/管理员账号）
  - settings（工作模式、SMTP、日报、无关闲聊过滤等后台配置）

用法（务必先不带 --yes 预览一次）：

    # 在 Docker 容器里（生产环境推荐）：
    docker compose exec faq-chatbot python scripts/clear_conversations.py
    docker compose exec faq-chatbot python scripts/clear_conversations.py --yes

    # 本机直接跑（连的是本机 data/qa.db，不是服务器上的）：
    python scripts/clear_conversations.py
    python scripts/clear_conversations.py --yes

执行前建议先备份一次：
    docker compose exec faq-chatbot cp /app/data/qa.db /app/data/qa.db.bak
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import database  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="真正执行删除；不加此参数只预览")
    args = parser.parse_args()

    database.init_db()

    with database.get_conn() as conn:
        conv_count = conn.execute("SELECT COUNT(*) AS c FROM conversations").fetchone()["c"]
        msg_count = conn.execute("SELECT COUNT(*) AS c FROM chat_messages").fetchone()["c"]
        session_count = conn.execute(
            "SELECT COUNT(DISTINCT session_id) AS c FROM conversations"
        ).fetchone()["c"]

        print(f"数据库路径：{database.DB_PATH}")
        print(f"将清空：{session_count} 位访客、{conv_count} 条对话、{msg_count} 条气泡明细")
        print("保留：agents（账号）、settings（后台配置）")

        if not args.yes:
            print("\n（当前是预览模式，未做任何修改。确认后加上 --yes 重新执行即可真正清空。）")
            return

        conn.execute("DELETE FROM chat_messages")
        conn.execute("DELETE FROM conversations")
        conn.execute("VACUUM")
        print("\n已清空全部对话数据。")


if __name__ == "__main__":
    main()
