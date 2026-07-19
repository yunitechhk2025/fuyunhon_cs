"""邮件提醒工具：客户问题需要转人工处理时，通过邮件提醒客服/管理员。

所有 SMTP 配置均从环境变量读取，未配置或发送失败时只打印警告、不抛出异常，
避免邮件故障影响客户提问、客服回复等主业务流程。
"""

import os
import smtplib
import sys
from email.mime.text import MIMEText
from email.utils import formatdate
from typing import Optional

# 测试阶段默认转发到该邮箱；正式上线后可通过环境变量 NOTIFY_EMAIL_TO 覆盖（支持用逗号分隔多个收件人）。
DEFAULT_NOTIFY_EMAIL_TO = "yunitechhk@gmail.com"


def _smtp_configured() -> bool:
    return bool(os.getenv("SMTP_HOST"))


def send_email(subject: str, body: str, to: Optional[str] = None) -> bool:
    """发送一封纯文本提醒邮件。成功返回 True，任何失败（未配置/连接失败/认证失败等）返回 False。"""
    host = os.getenv("SMTP_HOST")
    if not host:
        print("[warn] 未配置 SMTP_HOST，跳过邮件发送（请在 .env 中配置 SMTP_HOST 等变量）", file=sys.stderr)
        return False

    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or username or "no-reply@fuyunhon.local"
    use_tls = os.getenv("SMTP_USE_TLS", "true").strip().lower() != "false"
    use_ssl = os.getenv("SMTP_USE_SSL", "false").strip().lower() == "true"
    recipients = [
        addr.strip()
        for addr in (to or os.getenv("NOTIFY_EMAIL_TO", DEFAULT_NOTIFY_EMAIL_TO)).split(",")
        if addr.strip()
    ]
    if not recipients:
        print("[warn] 未配置收件邮箱，跳过邮件发送", file=sys.stderr)
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)

    try:
        smtp_cls = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
        with smtp_cls(host, port, timeout=15) as server:
            if use_tls and not use_ssl:
                server.starttls()
            if username and password:
                server.login(username, password)
            server.sendmail(sender, recipients, msg.as_string())
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 邮件发送失败: {exc}", file=sys.stderr)
        return False
