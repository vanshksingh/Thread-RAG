"""
analysis_production.py - Professional RAG evaluation analysis
Generates peer-review ready findings backed by statistical analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGAnalysis:
    """Statistical analysis of Thread-RAG evaluation results"""
    
    def __init__(self, data_file: str = "./eval_results/thread_rag_analysis.xlsx"):
        self.data_file = data_file
        self.df = pd.read_excel(data_file, sheet_name='detailed_results')
        logger.info(f"Loaded {len(self.df)} evaluation records")
    
    def generate_findings_document(self, output_file: str = "./FINDINGS.md"):
        """Generate professional findings document"""
        
        findings = []
        findings.append("# Thread-RAG Evaluation: Empirical Findings\n")
        findings.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        findings.append(f"Sample Size: {len(self.df)} evaluations\n")
        findings.append(f"Models: {', '.join(self.df['model'].unique())}\n")
        findings.append(f"Strategies: {', '.join(self.df['retrieval_strategy'].unique())}\n\n")
        
        # 1. RAG Mode Comparison
        findings.append("## RAG Mode Performance\n")
        mode_comparison = self._analyze_modes()
        findings.append(mode_comparison)
        
        # 2. Model Performance
        findings.append("\n## Model Performance Analysis\n")
        model_analysis = self._analyze_models()
        findings.append(model_analysis)
        
        # 3. Strategy Impact
        findings.append("\n## Retrieval Strategy Effectiveness\n")
        strategy_analysis = self._analyze_strategies()
        findings.append(strategy_analysis)
        
        # 4. Embedding Model Impact
        findings.append("\n## Embedding Model Evaluation\n")
        embedding_analysis = self._analyze_embeddings()
        findings.append(embedding_analysis)
        
        # 5. Interaction Effects
        findings.append("\n## Cross-Factor Analysis\n")
        interaction_analysis = self._analyze_interactions()
        findings.append(interaction_analysis)
        
        # 6. Statistical Summary
        findings.append("\n## Statistical Summary\n")
        summary = self._generate_statistical_summary()
        findings.append(summary)
        
        output = "\n".join(findings)
        
        with open(output_file, 'w') as f:
            f.write(output)
        
        logger.info(f"Findings saved to {output_file}")
        return output
    
    def _analyze_modes(self) -> str:
        """Analyze Thread-RAG vs Normal RAG performance"""
        text = []
        
        for metric in ['f1_score', 'context_precision', 'faithfulness', 'answer_relevancy', 'mrr']:
            normal = self.df[self.df['rag_mode'] == 'normal_rag'][metric]
            thread = self.df[self.df['rag_mode'] == 'thread_rag'][metric]
            
            improvement = ((thread.mean() - normal.mean()) / normal.mean() * 100)
            
            text.append(f"- {metric.replace('_', ' ').title()}: {normal.mean():.4f} → {thread.mean():.4f} ({improvement:+.1f}%)")
        
        # Token savings
        thread_tokens = self.df[self.df['rag_mode'] == 'thread_rag']['token_savings_percent'].mean()
        text.append(f"- Token Savings: {thread_tokens:.2f}% (Thread-RAG only)")
        
        return "\n".join(text)
    
    def _analyze_models(self) -> str:
        """Analyze model-specific performance"""
        text = []
        
        model_perf = self.df.groupby('model').agg({
            'f1_score': ['mean', 'std'],
            'faithfulness': ['mean', 'std'],
            'answer_relevancy': ['mean', 'std'],
            'retrieval_time_ms': 'mean',
            'generation_time_ms': 'mean'
        }).round(4)
        
        for model in sorted(self.df['model'].unique()):
            model_data = self.df[self.df['model'] == model]
            f1 = model_data['f1_score'].mean()
            faith = model_data['faithfulness'].mean()
            rel = model_data['answer_relevancy'].mean()
            total_time = model_data['total_time_ms'].mean()
            
            text.append(f"- {model}: F1={f1:.4f}, Faithfulness={faith:.4f}, Relevancy={rel:.4f}, Time={total_time:.0f}ms")
        
        return "\n".join(text)
    
    def _analyze_strategies(self) -> str:
        """Analyze retrieval strategy effectiveness"""
        text = []
        
        for strategy in sorted(self.df['retrieval_strategy'].unique()):
            strat_data = self.df[self.df['retrieval_strategy'] == strategy]
            f1 = strat_data['f1_score'].mean()
            mrr = strat_data['mrr'].mean()
            ndcg = strat_data['ndcg'].mean()
            retrieval_time = strat_data['retrieval_time_ms'].mean()
            
            text.append(f"- {strategy.upper()}: F1={f1:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}, Retrieval={retrieval_time:.1f}ms")
        
        return "\n".join(text)
    
    def _analyze_embeddings(self) -> str:
        """Analyze embedding model impact"""
        text = []
        
        for embedding in sorted(self.df['embedding_model'].unique()):
            embed_data = self.df[self.df['embedding_model'] == embedding]
            f1 = embed_data['f1_score'].mean()
            context_precision = embed_data['context_precision'].mean()
            mrr = embed_data['mrr'].mean()
            
            text.append(f"- {embedding}: F1={f1:.4f}, Context Precision={context_precision:.4f}, MRR={mrr:.4f}")
        
        return "\n".join(text)
    
    def _analyze_interactions(self) -> str:
        """Analyze strategy x model interactions"""
        text = []
        
        # Best model-strategy combinations
        interaction = self.df.groupby(['model', 'retrieval_strategy'])['f1_score'].mean().reset_index()
        interaction = interaction.sort_values('f1_score', ascending=False)
        
        text.append("Top 5 Model-Strategy Combinations (by F1 Score):\n")
        for idx, row in interaction.head(5).iterrows():
            text.append(f"- {row['model']} + {row['retrieval_strategy'].upper()}: F1={row['f1_score']:.4f}")
        
        # Worst combinations
        text.append("\nLowest Performing Combinations:\n")
        for idx, row in interaction.tail(5).iterrows():
            text.append(f"- {row['model']} + {row['retrieval_strategy'].upper()}: F1={row['f1_score']:.4f}")
        
        return "\n".join(text)
    
    def _generate_statistical_summary(self) -> str:
        """Generate statistical summary table"""
        text = []
        
        metrics = ['f1_score', 'context_precision', 'context_recall', 'faithfulness', 
                   'answer_relevancy', 'mrr', 'ndcg']
        
        text.append("\nOverall Metric Distribution:\n")
        text.append("| Metric | Mean | Std | Min | Max |")
        text.append("|--------|------|-----|-----|-----|")
        
        for metric in metrics:
            mean = self.df[metric].mean()
            std = self.df[metric].std()
            min_val = self.df[metric].min()
            max_val = self.df[metric].max()
            text.append(f"| {metric.replace('_', ' ').title()} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} |")
        
        return "\n".join(text)


def main():
    """Generate analysis"""
    analysis = RAGAnalysis()
    findings = analysis.generate_findings_document()
    print(findings)


if __name__ == "__main__":
    main()

