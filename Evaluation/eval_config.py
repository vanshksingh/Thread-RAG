"""
eval_config.py - Evaluation Framework Configuration
Central configuration for all evaluation parameters
"""

import os

# ============= DATASET CONFIGURATION =============
EVAL_DATASET_FILE = "./research_context_stability_dataset.json"

# ============= MODEL CONFIGURATIONS =============
# Mix of different model sizes for comprehensive evaluation
OLLAMA_MODELS = [
    "qwen2.5:0.5b",    # Small model (fast, lightweight)
    "qwen2.5:3b",      # Medium-small model
    "qwen2.5:7b",      # Medium model
    "qwen2.5:14b",     # Large model (comprehensive)
    "gpt-oss:20b",     # Very large model (deep reasoning)
]

# Embedding models for different quality levels
EMBEDDING_MODELS = [
    "nomic-embed-text",      # Standard quality
    "mxbai-embed-large",     # High quality
    "bge-large-en-v1.5",     # Specialized for retrieval
]

# ============= RAG MODES =============
RAG_MODES = {
    "normal_rag": {
        "description": "Standard RAG with raw chunks",
        "use_context_window": False,
        "sequential": False,
    },
    "thread_rag": {
        "description": "Thread-RAG with context windows",
        "use_context_window": True,
        "sequential": True,
    }
}

# ============= RETRIEVAL STRATEGIES =============
RETRIEVAL_STRATEGIES = {
    "semantic_search": {
        "description": "Pure semantic/vector similarity",
        "k": 3,
    },
    "bm25": {
        "description": "BM25 keyword-based retrieval",
        "k": 3,
    },
    "rrf": {
        "description": "Reciprocal Rank Fusion (hybrid)",
        "k": 3,
        "rrf_k": 60,
    },
    "mmr": {
        "description": "Maximal Marginal Relevance (diversity-aware)",
        "k": 3,
        "fetch_k": 6,
        "lambda_mult": 0.5,
    }
}

# ============= EVALUATION METRICS =============
METRICS = {
    "ragas": {
        "enabled": True,
        "metrics": [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall"
        ]
    },
    "custom": {
        "enabled": True,
        "metrics": [
            "perplexity",
            "mrr",
            "ndcg",
            "f1_score",
            "context_stability_score"
        ]
    }
}

# ============= STORAGE CONFIGURATION =============
EVAL_DIR = "./eval_results"
EVAL_RESULTS_FILE = os.path.join(EVAL_DIR, "context_stability_evaluation.xlsx")
EVAL_CHECKPOINT_FILE = os.path.join(EVAL_DIR, "eval_checkpoint.json")

# Ensure directories exist
os.makedirs(EVAL_DIR, exist_ok=True)

# ============= EXCEL SHEET CONFIGURATION =============
EXCEL_SHEETS = {
    "summary": {
        "columns": [
            "Timestamp",
            "Test_Run_ID",
            "Total_Questions",
            "Completed",
            "Failed",
            "Avg_Faithfulness",
            "Avg_Answer_Relevancy",
            "Avg_Context_Precision",
            "Avg_Context_Recall",
            "Avg_F1_Score",
            "Avg_Context_Stability",
            "Avg_MRR",
            "Avg_NDCG",
            "Avg_Token_Savings_Pct"
        ]
    },
    "detailed": {
        "columns": [
            "Row_ID",
            "Timestamp",
            "Question",
            "Reference_Answer",
            "Document_Type",
            "Context_Dependency",
            "Chunk_Span",
            "RAG_Mode",
            "Retrieval_Strategy",
            "Main_Model",
            "Embedding_Model",
            "Retrieved_Chunks",
            "Generated_Answer",
            "Faithfulness",
            "Answer_Relevancy",
            "Context_Precision",
            "Context_Recall",
            "F1_Score",
            "Context_Stability_Score",
            "MRR",
            "NDCG",
            "Perplexity",
            "Retrieval_Time_ms",
            "Generation_Time_ms",
            "Total_Time_ms",
            "Token_Count",
            "Estimated_Token_Savings",
            "Status",
            "Error_Message"
        ]
    },
    "comparison": {
        "columns": [
            "Metric",
            "Normal_RAG_Avg",
            "Thread_RAG_Avg",
            "Difference",
            "Percent_Change",
            "Statistical_Significance"
        ]
    },
    "strategy_comparison": {
        "columns": [
            "Metric",
            "Semantic_Search_Avg",
            "BM25_Avg",
            "RRF_Avg",
            "MMR_Avg",
            "Best_Strategy"
        ]
    },
    "model_comparison": {
        "columns": [
            "Model",
            "Avg_F1_Score",
            "Avg_Context_Stability",
            "Avg_Token_Savings_Pct",
            "Avg_Processing_Time_ms",
            "Performance_Rank"
        ]
    },
    "context_stability_analysis": {
        "columns": [
            "Document_Type",
            "Context_Dependency",
            "Normal_RAG_Stability",
            "Thread_RAG_Stability",
            "Stability_Improvement_Pct",
            "Chunk_Span_Type",
            "Sample_Size"
        ]
    }
}

# ============= CHECKPOINT CONFIGURATION =============
CHECKPOINT_CONFIG = {
    "save_interval": 5,
    "auto_resume": True,
    "max_retries": 3,
}

# ============= EVALUATION PARAMETERS =============
EVAL_PARAMS = {
    "batch_size": 1,
    "timeout_seconds": 120,  # Increased for larger models
    "num_workers": 1,
    "context_stability_weight": 0.3,  # Weight for custom stability metric
}

# ============= TOKEN SAVINGS CALCULATIONS =============
TOKEN_SAVINGS_CONFIG = {
    "baseline_tokens_per_chunk": 1000,  # Average tokens in full chunk
    "context_window_tokens": 300,       # Average tokens in context window
    "summary_tokens": 50,              # Average tokens per summary
    "thread_rag_multiplier": 0.75,     # Token efficiency multiplier
    "api_cost_per_1k_tokens": 0.002,   # Example API cost (like Gemini)
}

# ============= CONTEXT STABILITY EVALUATION =============
CONTEXT_STABILITY_CONFIG = {
    "semantic_coherence_weight": 0.4,
    "information_flow_weight": 0.3,
    "chunk_boundary_preservation_weight": 0.3,
    "min_stability_threshold": 0.0,
    "max_stability_threshold": 1.0,
}
