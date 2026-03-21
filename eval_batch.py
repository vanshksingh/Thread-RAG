"""
eval_batch.py - Batch evaluation runner with predefined configurations
Useful for common evaluation scenarios
"""

import logging
from eval_framework import RAGEvaluator
from eval_config import OLLAMA_MODELS, EMBEDDING_MODELS, RAG_MODES, RETRIEVAL_STRATEGIES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchEvaluationRunner:
    """Predefined batch evaluation scenarios"""
    
    @staticmethod
    def run_mode_comparison():
        """Compare Normal RAG vs Thread-RAG with best configuration"""
        logger.info("Running Mode Comparison Evaluation...")
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            models=["gpt-oss:20b"],
            embedding_models=["nomic-embed-text"],
            modes=list(RAG_MODES.keys()),
            strategies=["rrf"]
        )
        evaluator.generate_comparison_sheets()

    @staticmethod
    def run_strategy_comparison():
        """Compare all retrieval strategies"""
        logger.info("Running Strategy Comparison Evaluation...")
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            models=["gpt-oss:20b"],
            embedding_models=["nomic-embed-text"],
            modes=["thread_rag"],
            strategies=list(RETRIEVAL_STRATEGIES.keys())
        )
        evaluator.generate_comparison_sheets()

    @staticmethod
    def run_model_benchmark():
        """Benchmark all available models"""
        logger.info("Running Model Benchmark Evaluation...")
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            models=OLLAMA_MODELS,
            embedding_models=["nomic-embed-text"],
            modes=["thread_rag"],
            strategies=["rrf"]
        )
        evaluator.generate_comparison_sheets()

    @staticmethod
    def run_embedding_benchmark():
        """Benchmark all embedding models"""
        logger.info("Running Embedding Model Benchmark Evaluation...")
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            models=["gpt-oss:20b"],
            embedding_models=EMBEDDING_MODELS,
            modes=["thread_rag"],
            strategies=["rrf"]
        )
        evaluator.generate_comparison_sheets()

    @staticmethod
    def run_quick_test():
        """Quick test with minimal configuration (good for testing setup)"""
        logger.info("Running Quick Test Evaluation...")
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            models=["gpt-oss:20b"],
            embedding_models=["nomic-embed-text"],
            modes=["thread_rag"],
            strategies=["semantic_search"]
        )
        evaluator.generate_comparison_sheets()

    @staticmethod
    def run_comprehensive():
        """Comprehensive evaluation (long-running)"""
        logger.info("Running Comprehensive Evaluation...")
        logger.warning("⚠️  This will take several hours!")
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            models=OLLAMA_MODELS,
            embedding_models=EMBEDDING_MODELS,
            modes=list(RAG_MODES.keys()),
            strategies=list(RETRIEVAL_STRATEGIES.keys())
        )
        evaluator.generate_comparison_sheets()

    @staticmethod
    def run_hybrid_strategies():
        """Focus on hybrid retrieval strategies"""
        logger.info("Running Hybrid Strategies Evaluation...")
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            models=["gpt-oss:20b", "qwen2.5:14b"],
            embedding_models=["nomic-embed-text"],
            modes=["thread_rag"],
            strategies=["rrf", "mmr", "bm25"]
        )
        evaluator.generate_comparison_sheets()

    @staticmethod
    def run_context_window_evaluation():
        """Evaluate impact of context window assembly"""
        logger.info("Running Context Window Impact Evaluation...")
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            models=["gpt-oss:20b"],
            embedding_models=["nomic-embed-text"],
            modes=list(RAG_MODES.keys()),
            strategies=["semantic_search", "rrf"]
        )
        evaluator.generate_comparison_sheets()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Evaluation Runner")
    parser.add_argument(
        '--scenario',
        choices=[
            'mode_comparison',
            'strategy_comparison',
            'model_benchmark',
            'embedding_benchmark',
            'quick_test',
            'hybrid_strategies',
            'context_window',
            'comprehensive'
        ],
        default='quick_test',
        help='Evaluation scenario to run'
    )
    
    args = parser.parse_args()
    
    scenarios = {
        'mode_comparison': BatchEvaluationRunner.run_mode_comparison,
        'strategy_comparison': BatchEvaluationRunner.run_strategy_comparison,
        'model_benchmark': BatchEvaluationRunner.run_model_benchmark,
        'embedding_benchmark': BatchEvaluationRunner.run_embedding_benchmark,
        'quick_test': BatchEvaluationRunner.run_quick_test,
        'hybrid_strategies': BatchEvaluationRunner.run_hybrid_strategies,
        'context_window': BatchEvaluationRunner.run_context_window_evaluation,
        'comprehensive': BatchEvaluationRunner.run_comprehensive,
    }
    
    try:
        scenarios[args.scenario]()
        logger.info("✅ Evaluation completed!")
    except KeyboardInterrupt:
        logger.info("\n⏸️  Evaluation interrupted. Checkpoint saved.")
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")


