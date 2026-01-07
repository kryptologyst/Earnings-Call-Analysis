"""Tests for earnings call analysis package."""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from earnings_analysis.models import EarningsAnalyzer
from earnings_analysis.data import TranscriptProcessor
from earnings_analysis.features import SentimentExtractor, TopicModeler, KeyPhraseExtractor
from earnings_analysis.utils import Config, set_seed


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.get("model.sentiment_model") == "finbert"
        assert config.get("evaluation.random_state") == 42
    
    def test_config_update(self):
        """Test configuration updates."""
        config = Config()
        config.update({"model": {"sentiment_model": "textblob"}})
        assert config.get("model.sentiment_model") == "textblob"


class TestTranscriptProcessor:
    """Test transcript processing."""
    
    def setup_method(self):
        """Setup test method."""
        self.processor = TranscriptProcessor()
    
    def test_clean_transcript(self):
        """Test transcript cleaning."""
        dirty_text = "CEO: Hello world.    CFO: How are you?"
        clean_text = self.processor.clean_transcript(dirty_text)
        assert "CEO:" not in clean_text
        assert "CFO:" not in clean_text
        assert "  " not in clean_text
    
    def test_validate_transcript(self):
        """Test transcript validation."""
        # Valid transcript
        valid_text = "This is a valid transcript with enough content to pass validation."
        assert self.processor.validate_transcript(valid_text) == True
        
        # Too short
        short_text = "Short"
        assert self.processor.validate_transcript(short_text) == False
        
        # Empty text
        assert self.processor.validate_transcript("") == False
    
    def test_extract_financial_terms(self):
        """Test financial term extraction."""
        text = "Revenue was $1.2B, up 15% from last quarter."
        terms = self.processor.extract_financial_terms(text)
        
        assert "monetary" in terms
        assert "$1.2B" in terms["monetary"]
        assert "15%" in terms["percentages"]


class TestSentimentExtractor:
    """Test sentiment extraction."""
    
    def setup_method(self):
        """Setup test method."""
        self.extractor = SentimentExtractor(use_finbert=False)  # Use TextBlob for testing
    
    def test_textblob_sentiment(self):
        """Test TextBlob sentiment analysis."""
        positive_text = "This is great news! We're doing very well."
        negative_text = "This is terrible. We're struggling badly."
        
        pos_result = self.extractor.analyze_sentiment_textblob(positive_text)
        neg_result = self.extractor.analyze_sentiment_textblob(negative_text)
        
        assert pos_result["label"] == "POSITIVE"
        assert neg_result["label"] == "NEGATIVE"
        assert pos_result["polarity"] > 0
        assert neg_result["polarity"] < 0


class TestTopicModeler:
    """Test topic modeling."""
    
    def setup_method(self):
        """Setup test method."""
        self.modeler = TopicModeler(method="lda", num_topics=3)
    
    def test_topic_modeling(self):
        """Test topic modeling."""
        texts = [
            "Apple reported strong revenue growth and increased iPhone sales.",
            "Microsoft's cloud business continues to expand rapidly.",
            "Tesla delivered record vehicle production and improved margins.",
            "Apple's services revenue grew significantly this quarter.",
            "Microsoft Azure revenue increased by 25% year-over-year.",
            "Tesla's energy storage business showed strong performance."
        ]
        
        self.modeler.fit(texts)
        topics = self.modeler.get_topics(num_words=5)
        
        assert len(topics) == 3
        assert all("words" in topic for topic in topics)
    
    def test_topic_prediction(self):
        """Test topic prediction."""
        texts = [
            "Apple reported strong revenue growth and increased iPhone sales.",
            "Microsoft's cloud business continues to expand rapidly.",
            "Tesla delivered record vehicle production and improved margins."
        ]
        
        self.modeler.fit(texts)
        predictions = self.modeler.predict_topics(texts)
        
        assert len(predictions) == 3
        assert all(isinstance(p, int) for p in predictions)


class TestKeyPhraseExtractor:
    """Test key phrase extraction."""
    
    def setup_method(self):
        """Setup test method."""
        self.extractor = KeyPhraseExtractor()
    
    def test_monetary_extraction(self):
        """Test monetary value extraction."""
        text = "Revenue was $1.2B, up 15% from last quarter. Profit margin improved to 25%."
        values = self.extractor.extract_monetary_values(text)
        
        assert len(values) > 0
        assert any(v["type"] == "dollar_amount" for v in values)
        assert any(v["type"] == "percentage" for v in values)
    
    def test_financial_phrases(self):
        """Test financial phrase extraction."""
        text = "Revenue growth was strong this quarter. Operating income increased significantly."
        phrases = self.extractor.extract_financial_phrases(text)
        
        assert len(phrases) > 0
        assert any("revenue" in p["phrase"].lower() for p in phrases)
    
    def test_n_grams(self):
        """Test n-gram extraction."""
        text = "Apple reported strong revenue growth this quarter."
        bigrams = self.extractor.extract_n_grams(text, n=2)
        
        assert len(bigrams) > 0
        assert any("revenue growth" in phrase for phrase, _ in bigrams)


class TestEarningsAnalyzer:
    """Test main earnings analyzer."""
    
    def setup_method(self):
        """Setup test method."""
        set_seed(42)
        self.analyzer = EarningsAnalyzer(
            sentiment_model="textblob",
            topic_method="lda",
            num_topics=3
        )
    
    def test_single_analysis(self):
        """Test single transcript analysis."""
        transcript = "Apple reported strong revenue growth of 15% this quarter. We're optimistic about future prospects."
        metadata = {
            "company": "Apple Inc.",
            "quarter": "Q1 2024",
            "date": "2024-01-15"
        }
        
        result = self.analyzer.analyze(transcript, metadata)
        
        assert "sentiment" in result
        assert "key_phrases" in result
        assert "financial_terms" in result
        assert result["company"] == "Apple Inc."
        assert result["word_count"] > 0
    
    def test_batch_analysis(self):
        """Test batch analysis."""
        transcripts = [
            {
                "transcript": "Apple reported strong revenue growth this quarter.",
                "metadata": {"company": "Apple", "quarter": "Q1"}
            },
            {
                "transcript": "Microsoft's cloud business continues to expand rapidly.",
                "metadata": {"company": "Microsoft", "quarter": "Q1"}
            }
        ]
        
        results = self.analyzer.analyze_batch(transcripts)
        
        assert len(results) == 2
        assert all("sentiment" in r for r in results)
        assert all("key_phrases" in r for r in results)
    
    def test_topic_modeling(self):
        """Test topic modeling functionality."""
        texts = [
            "Apple reported strong revenue growth and increased iPhone sales.",
            "Microsoft's cloud business continues to expand rapidly.",
            "Tesla delivered record vehicle production and improved margins."
        ]
        
        self.analyzer.fit_topic_model(texts)
        topics = self.analyzer.get_topics()
        
        assert len(topics) == 3
        assert all("words" in topic for topic in topics)
    
    def test_save_load_results(self):
        """Test saving and loading results."""
        transcript = "Apple reported strong revenue growth this quarter."
        metadata = {"company": "Apple", "quarter": "Q1"}
        
        result = self.analyzer.analyze(transcript, metadata)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.analyzer.save_results(result, temp_path)
            
            # Load results
            loaded_result = self.analyzer.load_results(temp_path)
            
            assert loaded_result["company"] == result["company"]
            assert loaded_result["sentiment"]["overall"]["label"] == result["sentiment"]["overall"]["label"]
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_analysis(self):
        """Test end-to-end analysis workflow."""
        set_seed(42)
        
        # Create sample data
        sample_data = [
            {
                "transcript": "Apple reported strong revenue growth of 15% this quarter. We're optimistic about future prospects.",
                "metadata": {
                    "company": "Apple Inc.",
                    "quarter": "Q1 2024",
                    "date": "2024-01-15",
                    "participants": ["CEO", "CFO"]
                }
            },
            {
                "transcript": "Microsoft's cloud business continues to expand rapidly. Azure revenue grew 25% year-over-year.",
                "metadata": {
                    "company": "Microsoft Corp.",
                    "quarter": "Q1 2024", 
                    "date": "2024-01-15",
                    "participants": ["CEO", "CFO"]
                }
            }
        ]
        
        # Initialize analyzer
        analyzer = EarningsAnalyzer(
            sentiment_model="textblob",
            topic_method="lda",
            num_topics=2
        )
        
        # Fit topic model
        texts = [item["transcript"] for item in sample_data]
        analyzer.fit_topic_model(texts)
        
        # Analyze transcripts
        results = analyzer.analyze_batch(sample_data)
        
        # Verify results
        assert len(results) == 2
        assert all("sentiment" in r for r in results)
        assert all("key_phrases" in r for r in results)
        assert all("financial_terms" in r for r in results)
        
        # Test topic prediction
        topics = analyzer.get_topics()
        assert len(topics) == 2
        
        predictions = analyzer.predict_topics(texts)
        assert len(predictions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
