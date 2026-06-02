from flask import Flask, request, jsonify
import redis
import requests
import json
import uuid
import os

app = Flask(__name__)

# some flask example

#@app.route("/")
#def hello():
#    return "<h1>Hello, Flask!</h1>"
#
#@app.route("/user/<name>")
#def user(name):
#    return f"Hello {name}"
#
#@app.route("/hi/<name>")
#def hi(name):
#    return render_template("index.html", name=name)
#
#@app.route("/login", methods=["GET", "POST"])
#def login():
#    if request.method == "POST":
#        return "Logged in!"
#    return render_template("login.html")


# -------------------
# Redis 初始化
# -------------------
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)
# 历史最大条数
MAX_HISTORY = 64

# 默认角色提示词
default_prompt = """
你是一名中文语音助手。
你的回答将被中文文本转语音系统朗读，因此必须适合语音播放。
请严格遵守以下规则：
1. 使用自然的中文口语回答。
2. 句子尽量简短、清晰。
3. 不要使用 Markdown 格式，例如 **加粗**、`代码`、列表或表格。
4. 不要使用表情符号或特殊符号。
5. 尽量避免 URL 或网址。
6. 如果出现英文缩写，请在字母之间加空格，例如：
   AI → A I
   GPU → G P U
7. 尽量使用中文表达技术概念，例如“人工智能”而不是 AI。
8. 如果必须出现数字，请使用适合朗读的形式，例如：
   2024 → 二零二四

最后，请确保你的回答读出来像自然的人类语音。
"""

# 创建新 session
# 尝试兼容OpenAI标准
def create_new_session() -> str:
    session_id = str(uuid.uuid4())
    system_msg = {
        "role": "system",
        "content": default_prompt
    }
    redis_client.rpush(
        f"chat:{session_id}:messages",
        json.dumps(system_msg)
    )
    redis_client.expire(f"chat:{session_id}:messages", 3600)
    print(f"[INFO] New session created: {session_id}")
    return session_id

# 读取所有对话历史
def get_messages(session_id: str):
    raw = redis_client.lrange(f"chat:{session_id}:messages", 0, -1)
    return [json.loads(x) for x in raw]

# 写入新的对话记录
def append_message(session_id, role, content):
    msg = {
        "role": role,
        "content": content
    }
    redis_client.rpush(
        f"chat:{session_id}:messages",
        json.dumps(msg)
    )
    redis_client.ltrim(
        f"chat:{session_id}:messages",
        -MAX_HISTORY,
        -1
    )
    redis_client.expire(f"chat:{session_id}:messages", 3600)

def call_llm_openai(server, model, messages, token_limit, api_key=None):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": token_limit
    }
    headers = {
        "Content-Type": "application/json"
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        server,
        headers=headers,
        json=payload,
        timeout=60
    )

    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]

def llm_chat_handler(provider):
    data = request.get_json()

    if data.get("version") != "v1":
        return jsonify({"error": "unrecognized json schema"}), 400

    session_id = data.get("session")
    user_msg = data.get("message")
    token_limit = data.get("token_limit", 128)

    if not session_id or not redis_client.exists(f"chat:{session_id}:messages"):
        print("[INFO] Creating new session")
        session_id = create_new_session()

    # 写入用户消息
    append_message(session_id, "user", user_msg)
    # 读取历史
    messages = get_messages(session_id)
    # provider 配置
    cfg = LLM_PROVIDERS[provider]

    try:
        assistant_reply = call_llm_openai(
            cfg["server"],
            cfg["model"],
            messages,
            token_limit,
            cfg["api_key"]
        )
    except Exception as e:
        return jsonify({"error": f"llm request failed: {str(e)}"}), 500

    append_message(session_id, "assistant", assistant_reply)

    return jsonify({
        "session": session_id,
        "reply": assistant_reply
    })

LLM_PROVIDERS = {
    "local": {
        "server": "http://192.168.31.109:8080/v1/chat/completions",
        "model": "local-model",
        "api_key": None
    },
    "deepseek": {
        "server": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY")
    }
}

# API

@app.route("/llm/local/chat", methods=["POST"])
def local_chat():
    return llm_chat_handler("local")

@app.route("/llm/deepseek/chat", methods=["POST"])
def deepseek_chat():
    return llm_chat_handler("deepseek")
