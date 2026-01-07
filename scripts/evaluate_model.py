"""Evaluation metrics and testing for earnings call analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import json
from pathlib import Path

from src.earnings_analysis.models import EarningsAnalyzer
from src.earnings_analysis.utils import Config, setup_logging, set_seed, get_logger

logger = get_logger(__name__)


class EvaluationMetrics:
    """Evaluation metrics for earnings call analysis."""
    
    def __init__(self) -> None:
        """Initialize evaluation metrics."""
        self.metrics_history = []
    
    def evaluate_sentiment(
        self, 
        y_true: List[str], 
        y_pred: List[str]
    ) -> Dict[str, float]:
        """Evaluate sentiment analysis performance.
        
        Args:
            y_true: True sentiment labels
            y_pred: Predicted sentiment labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert to consistent format
        y_true_clean = [label.upper() for label in y_true]
        y_pred_clean = [label.upper() for label in y_pred]
        
        metrics = {
            "accuracy": accuracy_score(y_true_clean, y_pred_clean),
            "precision": precision_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0),
            "recall": recall_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true_clean, y_pred_clean, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def evaluate_topic_coherence(
        self, 
        topics: List[Dict[str, List[Tuple[str, float]]]], 
        texts: List[str]
    ) -> Dict[str, float]:
        """Evaluate topic model coherence.
        
        Args:
            topics: List of topics with word-weight pairs
            texts: List of texts used for topic modeling
            
        Returns:
            Dictionary of coherence metrics
        """
        try:
            from gensim.models import CoherenceModel
            from gensim.corpora import Dictionary
            
            # Prepare texts for coherence evaluation
            processed_texts = []
            for text in texts:
                # Simple preprocessing
                words = text.lower().split()
                processed_texts.append(words)
            
            # Create dictionary and corpus
            dictionary = Dictionary(processed_texts)
            corpus = [dictionary.doc2bow(text) for text in processed_texts]
            
            # Extract topic words
            topic_words = []
            for topic in topics:
                words = [word for word, _ in topic.get('words', [])]
                topic_words.append(words)
            
            # Calculate coherence
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=processed_texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            
            coherence_score = coherence_model.get_coherence()
            
            return {
                "coherence_score": coherence_score,
                "num_topics": len(topics),
                "avg_words_per_topic": np.mean([len(topic.get('words', [])) for topic in topics])
            }
            
        except ImportError:
            logger.warning("Gensim not available for coherence evaluation")
            return {
                "coherence_score": 0.0,
                "num_topics": len(topics),
                "avg_words_per_topic": np.mean([len(topic.get('words', [])) for topic in topics])
            }
    
    def evaluate_key_phrase_extraction(
        self, 
        true_phrases: List[List[str]], 
        predicted_phrases: List[List[str]]
    ) -> Dict[str, float]:
        """Evaluate key phrase extraction performance.
        
        Args:
            true_phrases: True key phrases for each text
            predicted_phrases: Predicted key phrases for each text
            
        Returns:
            Dictionary of evaluation metrics
        """
        precisions = []
        recalls = []
        f1_scores = []
        
        for true_phrases_text, pred_phrases_text in zip(true_phrases, predicted_phrases):
            true_set = set(true_phrases_text)
            pred_set = set(pred_phrases_text)
            
            if len(pred_set) == 0:
                precision = 0.0
            else:
                precision = len(true_set.intersection(pred_set)) / len(pred_set)
            
            if len(true_set) == 0:
                recall = 0.0
            else:
                recall = len(true_set.intersection(pred_set)) / len(true_set)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        return {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1_score": np.mean(f1_scores),
            "std_precision": np.std(precisions),
            "std_recall": np.std(recalls),
            "std_f1_score": np.std(f1_scores)
        }
    
    def comprehensive_evaluation(
        self, 
        analyzer: EarningsAnalyzer, 
        test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation on test data.
        
        Args:
            analyzer: Trained earnings analyzer
            test_data: Test dataset
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive evaluation")
        
        results = {
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
            "num_test_samples": len(test_data),
            "model_info": {
                "sentiment_model": analyzer.sentiment_model,
                "topic_method": analyzer.topic_method,
                "num_topics": analyzer.num_topics
            }
        }
        
        # Extract transcripts and metadata
        transcripts = [item["transcript"] for item in test_data]
        metadata_list = [item.get("metadata", {}) for item in test_data]
        
        # Analyze all transcripts
        analysis_results = analyzer.analyze_batch(test_data)
        
        # Sentiment evaluation (if ground truth available)
        if all("sentiment_label" in item for item in test_data):
            true_sentiments = [item["sentiment_label"] for item in test_data]
            pred_sentiments = [result["sentiment"]["overall"]["label"] for result in analysis_results]
            
            results["sentiment_metrics"] = self.evaluate_sentiment(true_sentiments, pred_sentiments)
        
        # Topic evaluation
        try:
            analyzer.fit_topic_model(transcripts)
            topics = analyzer.get_topics()
            topic_predictions = analyzer.predict_topics(transcripts)
            
            results["topic_metrics"] = self.evaluate_topic_coherence(topics, transcripts)
            results["topic_predictions"] = topic_predictions
            
        except Exception as e:
            logger.warning(f"Topic evaluation failed: {e}")
            results["topic_metrics"] = {"error": str(e)}
        
        # Key phrase evaluation (if ground truth available)
        if all("key_phrases" in item for item in test_data):
            true_phrases = [item["key_phrases"] for item in test_data]
            pred_phrases = [result["key_phrases"]["financial_phrases"] for result in analysis_results]
            
            results["phrase_metrics"] = self.evaluate_key_phrase_extraction(true_phrases, pred_phrases)
        
        # Overall performance summary
        results["summary"] = self._create_summary(results)
        
        logger.info("Comprehensive evaluation completed")
        return results
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create evaluation summary.
        
        Args:
            results: Evaluation results
            
        Returns:
            Summary dictionary
        """
        summary = {
            "overall_score": 0.0,
            "component_scores": {}
        }
        
        # Calculate component scores
        if "sentiment_metrics" in results:
            sentiment_score = results["sentiment_metrics"].get("f1_score", 0.0)
            summary["component_scores"]["sentiment"] = sentiment_score
        
        if "topic_metrics" in results and "error" not in results["topic_metrics"]:
            topic_score = results["topic_metrics"].get("coherence_score", 0.0)
            summary["component_scores"]["topics"] = topic_score
        
        if "phrase_metrics" in results:
            phrase_score = results["phrase_metrics"].get("f1_score", 0.0)
            summary["component_scores"]["phrases"] = phrase_score
        
        # Calculate overall score
        if summary["component_scores"]:
            summary["overall_score"] = np.mean(list(summary["component_scores"].values()))
        
        return summary


def run_evaluation_example() -> None:
    """Run example evaluation on sample data."""
    # Setup
    setup_logging()
    set_seed(42)
    
    # Load sample data
    sample_data_path = Path("data/sample_transcripts.json")
    if not sample_data_path.exists():
        logger.error("Sample data not found. Run create_sample_data.py first.")
        return
    
    with open(sample_data_path, 'r') as f:
        sample_data = json.load(f)
    
    # Initialize analyzer
    config = Config()
    analyzer = EarningsAnalyzer(config=config)
    
    # Initialize evaluator
    evaluator = EvaluationMetrics()
    
    # Run evaluation
    results = evaluator.comprehensive_evaluation(analyzer, sample_data)
    
    # Save results
    results_path = Path("assets/evaluation_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS SUMMARY")
    print("="*50)
    print(f"Number of test samples: {results['num_test_samples']}")
    print(f"Overall score: {results['summary']['overall_score']:.3f}")
    
    if "sentiment_metrics" in results:
        print(f"Sentiment F1-score: {results['sentiment_metrics']['f1_score']:.3f}")
    
    if "topic_metrics" in results and "error" not in results["topic_metrics"]:
        print(f"Topic coherence: {results['topic_metrics']['coherence_score']:.3f}")
    
    if "phrase_metrics" in results:
        print(f"Phrase extraction F1-score: {results['phrase_metrics']['f1_score']:.3f}")
    
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    run_evaluation_example()
