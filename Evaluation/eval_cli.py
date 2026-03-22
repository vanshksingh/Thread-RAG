"""
eval_cli.py - Command-line interface for RAG evaluation framework
Handles running evaluations, reading results, validating data, and resuming checkpoints
"""

import argparse
import logging
import sys

from eval_framework import RAGEvaluator
from eval_reader import EvalReader
from eval_validator import EvalValidator
from eval_config import OLLAMA_MODELS, EMBEDDING_MODELS, RAG_MODES, RETRIEVAL_STRATEGIES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_command(args):
    """Validate evaluation results"""
    logger.info("Starting validation...")
    validator = EvalValidator()
    is_valid = validator.validate_all()
    
    if is_valid:
        logger.info("✅ Validation passed!")
        return 0
    else:
        logger.error("❌ Validation failed! Check warnings and issues above.")
        return 1


def read_command(args):
    """Read and analyze evaluation results"""
    logger.info("Reading evaluation results...")
    reader = EvalReader()
    
    # Print report
    reader.print_comparison_report()
    
    # Export comparison
    if args.export:
        reader.export_comparison_to_text(args.export)
    
    # Get best configurations
    if args.best:
        best = reader.get_best_configuration(args.metric, args.best)
        logger.info(f"\nTop {args.best} Configurations by {args.metric}:")
        for i, config in enumerate(best, 1):
            logger.info(
                f"{i}. {config.get('Main_Model')} + "
                f"{config.get('Embedding_Model')} + "
                f"{config.get('RAG_Mode')} + "
                f"{config.get('Retrieval_Strategy')}: "
                f"{config.get(args.metric, 0):.4f}"
            )
    
    # Run specific query
    if args.query:
        logger.info(f"\nRunning query: {args.query}")
        # Parse query format: "Main_Model=gpt-oss:20b,RAG_Mode=thread_rag"
        query_dict = {}
        for pair in args.query.split(','):
            key, value = pair.split('=')
            query_dict[key.strip()] = value.strip()
        
        results = reader.run_specific_query(query_dict)
        logger.info(f"Found {len(results)} results")
        
        if args.limit:
            results = results[:args.limit]
            logger.info(f"Showing first {args.limit} results")
        
        for i, result in enumerate(results, 1):
            logger.info(f"\n{i}. {result.get('Question')[:50]}...")
            logger.info(f"   F1: {result.get('F1_Score', 0):.4f}, "
                       f"MRR: {result.get('MRR', 0):.4f}, "
                       f"Time: {result.get('Total_Time_ms', 0):.0f}ms")
    
    return 0


def run_command(args):
    """Run evaluation"""
    logger.info("Starting RAG evaluation...")
    
    # Parse model lists
    models = args.models if args.models else OLLAMA_MODELS
    embedding_models = args.embedding_models if args.embedding_models else EMBEDDING_MODELS
    modes = args.modes if args.modes else list(RAG_MODES.keys())
    strategies = args.strategies if args.strategies else list(RETRIEVAL_STRATEGIES.keys())
    
    # Parse specific rows if provided
    specific_rows = None
    if args.rows:
        try:
            specific_rows = [int(x) - 1 for x in args.rows.split(',')]
        except ValueError:
            logger.error("Invalid row format. Use: --rows 1,5,10")
            return 1
    
    logger.info(f"Configuration:")
    logger.info(f"  Models: {models}")
    logger.info(f"  Embedding Models: {embedding_models}")
    logger.info(f"  RAG Modes: {modes}")
    logger.info(f"  Strategies: {strategies}")
    if specific_rows:
        logger.info(f"  Specific Rows: {[x+1 for x in specific_rows]}")
    
    try:
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            models=models,
            embedding_models=embedding_models,
            modes=modes,
            strategies=strategies,
            specific_rows=specific_rows,
            resume_from_checkpoint=args.resume
        )
        evaluator.generate_comparison_sheets()
        logger.info("✅ Evaluation complete!")
        return 0
    
    except KeyboardInterrupt:
        logger.info("\n⏸️  Evaluation interrupted. Checkpoint saved for resume.")
        return 130
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        logger.info("💾 Checkpoint saved. Run with --resume to continue.")
        return 1


def list_command(args):
    """List available models and configurations"""
    logger.info("\n" + "="*60)
    logger.info("AVAILABLE CONFIGURATIONS")
    logger.info("="*60)
    
    logger.info("\n🤖 OLLAMA MODELS:")
    for i, model in enumerate(OLLAMA_MODELS, 1):
        logger.info(f"  {i}. {model}")
    
    logger.info("\n🔤 EMBEDDING MODELS:")
    for i, model in enumerate(EMBEDDING_MODELS, 1):
        logger.info(f"  {i}. {model}")
    
    logger.info("\n🎯 RAG MODES:")
    for mode, config in RAG_MODES.items():
        logger.info(f"  - {mode}: {config['description']}")
    
    logger.info("\n🔍 RETRIEVAL STRATEGIES:")
    for strategy, config in RETRIEVAL_STRATEGIES.items():
        logger.info(f"  - {strategy}: {config['description']}")
    
    logger.info("\n" + "="*60 + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation
  python eval_cli.py run

  # Run specific models and modes
  python eval_cli.py run --models gpt-oss:20b qwen2.5:14b --modes thread_rag

  # Resume from checkpoint
  python eval_cli.py run --resume

  # Re-run specific questions
  python eval_cli.py run --rows 1,5,10

  # Read and analyze results
  python eval_cli.py read --best 10 --metric f1_score

  # Query specific configuration
  python eval_cli.py read --query "Main_Model=gpt-oss:20b,RAG_Mode=thread_rag"

  # Validate data integrity
  python eval_cli.py validate

  # List available options
  python eval_cli.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run evaluation')
    run_parser.add_argument(
        '--models',
        nargs='+',
        help='Ollama models to test (default: all)'
    )
    run_parser.add_argument(
        '--embedding-models',
        nargs='+',
        help='Embedding models to test (default: all)'
    )
    run_parser.add_argument(
        '--modes',
        nargs='+',
        help='RAG modes to test (default: all)'
    )
    run_parser.add_argument(
        '--strategies',
        nargs='+',
        help='Retrieval strategies to test (default: all)'
    )
    run_parser.add_argument(
        '--rows',
        help='Specific rows to evaluate (comma-separated, 1-indexed)'
    )
    run_parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from checkpoint if available'
    )
    run_parser.set_defaults(func=run_command)
    
    # Read command
    read_parser = subparsers.add_parser('read', help='Read and analyze results')
    read_parser.add_argument(
        '--export',
        help='Export comparison to text file'
    )
    read_parser.add_argument(
        '--best',
        type=int,
        help='Show top-K best configurations'
    )
    read_parser.add_argument(
        '--metric',
        default='f1_score',
        help='Metric to sort by (default: f1_score)'
    )
    read_parser.add_argument(
        '--query',
        help='Query results (format: "Key1=Value1,Key2=Value2")'
    )
    read_parser.add_argument(
        '--limit',
        type=int,
        help='Limit query results'
    )
    read_parser.set_defaults(func=read_command)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate results')
    validate_parser.set_defaults(func=validate_command)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available configurations')
    list_parser.set_defaults(func=list_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


