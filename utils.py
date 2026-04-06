import re

def generate_chat_title(query):
    query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
    words = query.split()
    title = " ".join(words[:6]).capitalize()
    return title if title else "New Chat"