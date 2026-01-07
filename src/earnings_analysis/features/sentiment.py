"""Sentiment analysis for earnings call transcripts."""

import torch
from typing import Dict, List, Optional, Union
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from textblob import TextBlob
from ..utils.logging import get_logger
from ..utils.seeding import get_device

logger = get_logger(__name__)


class SentimentExtractor:
    """Extract sentiment from earnings call transcripts.
    
    Supports multiple sentiment analysis methods including FinBERT
    and traditional NLP approaches.
    """
    
    def __init__(
        self, 
        model_name: str = "ProsusAI/finbert",
        use_finbert: bool = True,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize sentiment extractor.
        
        Args:
            model_name: Name of the sentiment model to use
            use_finbert: Whether to use FinBERT for sentiment analysis
            device: Device to run model on (auto-detected if None)
        """
        self.model_name = model_name
        self.use_finbert = use_finbert
        self.device = device or get_device()
        
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        
        if self.use_finbert:
            self._load_finbert_model()
    
    def _load_finbert_model(self) -> None:
        """Load FinBERT model and tokenizer."""
        try:
            logger.info(f"Loading FinBERT model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3  # positive, negative, neutral
            )
            
            self._model.to(self.device)
            self._model.eval()
            
            # Create pipeline for easier inference
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self.device.type == "cuda" else -1
            )
            
            logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}. Falling back to TextBlob.")
            self.use_finbert = False
    
    def analyze_sentiment_finbert(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment using FinBERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment label and confidence
        """
        if not self._pipeline:
            raise RuntimeError("FinBERT model not loaded")
        
        try:
            # Split long text into chunks if necessary
            max_length = 512
            if len(text) > max_length:
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                results = []
                
                for chunk in chunks:
                    result = self._pipeline(chunk)
                    results.append(result[0])
                
                # Average the results
                avg_score = np.mean([r['score'] for r in results])
                labels = [r['label'] for r in results]
                
                # Use most common label
                from collections import Counter
                most_common_label = Counter(labels).most_common(1)[0][0]
                
                return {
                    'label': most_common_label,
                    'score': float(avg_score),
                    'method': 'finbert'
                }
            else:
                result = self._pipeline(text)[0]
                return {
                    'label': result['label'],
                    'score': float(result['score']),
                    'method': 'finbert'
                }
                
        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
            return self.analyze_sentiment_textblob(text)
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment label and polarity
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            label = "POSITIVE"
        elif polarity < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        return {
            'label': label,
            'score': abs(polarity),
            'polarity': polarity,
            'method': 'textblob'
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment using the configured method.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if self.use_finbert and self._pipeline:
            return self.analyze_sentiment_finbert(text)
        else:
            return self.analyze_sentiment_textblob(text)
    
    def analyze_sentiment_by_speaker(
        self, 
        transcript: str, 
        speakers: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Union[str, float]]]:
        """Analyze sentiment by speaker.
        
        Args:
            transcript: Full transcript text
            speakers: List of speaker names to extract
            
        Returns:
            Dictionary mapping speakers to their sentiment analysis
        """
        if not speakers:
            # Extract speakers automatically
            import re
            speaker_pattern = r'^([A-Z]+):\s*(.*?)(?=^[A-Z]+:|$)'
            matches = re.findall(speaker_pattern, transcript, re.MULTILINE | re.DOTALL)
            speakers = list(set([match[0] for match in matches]))
        
        results = {}
        
        for speaker in speakers:
            # Extract text for this speaker
            pattern = f'^{speaker}:\\s*(.*?)(?=^[A-Z]+:|$)'
            matches = re.findall(pattern, transcript, re.MULTILINE | re.DOTALL)
            
            if matches:
                speaker_text = ' '.join(matches)
                results[speaker] = self.analyze_sentiment(speaker_text)
            else:
                results[speaker] = {
                    'label': 'NEUTRAL',
                    'score': 0.0,
                    'method': 'no_text'
                }
        
        return results
    
    def get_sentiment_summary(self, text: str) -> Dict[str, Union[str, float, int]]:
        """Get comprehensive sentiment summary.
        
        Args:
            text: Text to analyze
            
        Returns:
            Comprehensive sentiment analysis results
        """
        sentiment_result = self.analyze_sentiment(text)
        
        # Additional analysis
        blob = TextBlob(text)
        subjectivity = blob.sentiment.subjectivity
        
        # Count positive/negative words
        positive_words = len([word for word in blob.words if word in TextBlob().sentiment.polarity])
        
        return {
            **sentiment_result,
            'subjectivity': subjectivity,
            'text_length': len(text),
            'word_count': len(text.split()),
            'analysis_timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
