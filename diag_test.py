from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.client import User
from diagrams.onprem.compute import Server
from diagrams.onprem.database import Mongodb
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.filesystem import S3  # Filesystem usually has S3 or similar
from diagrams.onprem.mlops import Polyaxon
from diagrams.programming.language import Python

graph_attr = {
    "fontsize": "20",
    "bgcolor": "white"
}

with Diagram("Local RAG Assistant Architecture", show=False, direction="LR", filename="rag_arch",
             graph_attr=graph_attr):
    user = User("User Input")

    with Cluster("Agent Execution Layer (LangGraph)"):
        agent = Python("Agent Executor")
        memory = Redis("In-Memory Saver")
        prompt = Server("System Prompt\n(Sequential Strategy)")

        user >> agent
        agent >> Edge(color="blue", style="dashed") >> memory
        agent >> prompt

    with Cluster("Tool Logic Layer"):
        with Cluster("Tools"):
            list_docs = Python("List Docs")
            index_doc = Python("Index New Doc")
            search = Python("RAG Search")
            fetch = Python("Fetch Chunks")

        context_manager = Python("Context Assembler\n(Prev/Next Summaries)")

        agent >> [list_docs, index_doc, search, fetch]
        search >> context_manager
        fetch >> context_manager

    with Cluster("Storage & Models (Ollama/Local)"):
        llm_main = Polyaxon("Main LLM\n(gpt-oss)")
        llm_sum = Polyaxon("Summarizer\n(qwen2.5:0.5b)")
        vector_db = Mongodb("ChromaDB\n(Vector Store)")
        cache = S3("JSON Caches\n(Catalog/Summaries)")

        index_doc >> vector_db
        index_doc >> llm_sum >> cache
        context_manager >> vector_db
        context_manager >> cache
        agent >> llm_main

    # Operational Process
    preheat = Python("Pre-heat Loop")
    preheat >> vector_db
    preheat >> llm_sum >> cache