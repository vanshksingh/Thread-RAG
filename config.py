# config.py
import os


# Using qwen2.5:0.5b for extraction/logic and tinyllama for fast summarization
EMBEDDER_MODEL = "nomic-embed-text"
SUMMARY_MODEL = "qwen2.5:0.5b"
DB_PATH = "./local_rag_db"
DB_NAME = "chunk_summaries.json"
CACHE_PATH = "./doc_cache"
CACHE_NAME = "catalog.json"


MAIN_MODEL = "gpt-oss:20b"
TEMPERATURE = 1   #set to 1 to avoid getting stuck in tool call loops

os.makedirs(CACHE_PATH, exist_ok=True)


if __name__ == "__main__":
    # Local import prevents circular dependency error
    from runner import start_chat

    print(f"🚀 Launching RAG Session [Model: {MAIN_MODEL}]")
    start_chat()