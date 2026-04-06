import json, os, uuid
from datetime import datetime

CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)

def create_chat():
    return str(uuid.uuid4())

def save_chat(chat_id, title, messages):
    data = {
        "chat_id": chat_id,
        "title": title,
        "messages": messages,
        "updated_at": datetime.now().isoformat()
    }

    with open(f"{CHAT_DIR}/{chat_id}.json", "w") as f:
        json.dump(data, f, indent=4)

def load_chats():
    chats = []
    for file in os.listdir(CHAT_DIR):
        if file.endswith(".json"):
            with open(f"{CHAT_DIR}/{file}") as f:
                chats.append(json.load(f))

    chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return chats

def load_chat(chat_id):
    path = f"{CHAT_DIR}/{chat_id}.json"
    if not os.path.exists(path):
        return {"chat_id": chat_id, "title": "New Chat", "messages": []}

    with open(path) as f:
        return json.load(f)