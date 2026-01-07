"""Key phrase extraction for earnings call transcripts."""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import Counter
from textblob import TextBlob
from ..utils.logging import get_logger

logger = get_logger(__name__)


class KeyPhraseExtractor:
    """Extract key phrases and financial metrics from earnings call transcripts.
    
    Uses multiple methods including regex patterns, TF-IDF, and financial dictionaries.
    """
    
    def __init__(self, min_phrase_length: int = 2, max_phrase_length: int = 4) -> None:
        """Initialize key phrase extractor.
        
        Args:
            min_phrase_length: Minimum phrase length in words
            max_phrase_length: Maximum phrase length in words
        """
        self.min_phrase_length = min_phrase_length
        self.max_phrase_length = max_phrase_length
        
        self._financial_patterns = self._compile_financial_patterns()
        self._financial_dictionary = self._load_financial_dictionary()
    
    def _compile_financial_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for financial term extraction."""
        return {
            "monetary_values": re.compile(r'\$[0-9,]+(?:\.[0-9]{1,2})?[BMK]?'),
            "percentages": re.compile(r'\b\d+(?:\.\d+)?%\b'),
            "ratios": re.compile(r'\b\d+(?:\.\d+)?[xX]\b'),
            "quarters": re.compile(r'\bQ[1-4]\s*\d{4}\b'),
            "years": re.compile(r'\b(20|19)\d{2}\b'),
            "financial_metrics": re.compile(
                r'\b(revenue|sales|profit|loss|earnings|margin|debt|equity|'
                r'assets|liabilities|cash flow|free cash flow|EBITDA|EPS|'
                r'ROE|ROA|P/E|P/B|dividend|yield)\b',
                re.IGNORECASE
            ),
            "growth_terms": re.compile(
                r'\b(growth|increase|decrease|decline|rise|fall|'
                r'expansion|contraction|improvement|deterioration)\b',
                re.IGNORECASE
            ),
            "time_periods": re.compile(
                r'\b(quarterly|annual|yearly|monthly|weekly|daily|'
                r'Q[1-4]|quarter|year|month|week|day)\b',
                re.IGNORECASE
            )
        }
    
    def _load_financial_dictionary(self) -> Dict[str, List[str]]:
        """Load financial terminology dictionary."""
        return {
            "revenue_terms": [
                "revenue", "sales", "income", "turnover", "gross revenue",
                "net revenue", "total revenue", "operating revenue"
            ],
            "profit_terms": [
                "profit", "earnings", "net income", "operating income",
                "gross profit", "net profit", "EBITDA", "EBIT"
            ],
            "cost_terms": [
                "cost", "expense", "expenditure", "spending", "outlay",
                "operating cost", "cost of goods sold", "SG&A"
            ],
            "asset_terms": [
                "assets", "property", "equipment", "inventory", "cash",
                "receivables", "investments", "fixed assets", "current assets"
            ],
            "liability_terms": [
                "debt", "liabilities", "borrowing", "loan", "credit",
                "payables", "accruals", "current liabilities", "long-term debt"
            ],
            "performance_terms": [
                "growth", "increase", "decrease", "improvement", "decline",
                "expansion", "contraction", "outperformance", "underperformance"
            ]
        }
    
    def extract_monetary_values(self, text: str) -> List[Dict[str, str]]:
        """Extract monetary values from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of monetary value dictionaries
        """
        monetary_values = []
        
        # Extract dollar amounts
        dollar_matches = self._financial_patterns["monetary_values"].findall(text)
        for match in dollar_matches:
            monetary_values.append({
                "value": match,
                "type": "dollar_amount",
                "context": self._get_context(text, match)
            })
        
        # Extract percentages
        percent_matches = self._financial_patterns["percentages"].findall(text)
        for match in percent_matches:
            monetary_values.append({
                "value": match,
                "type": "percentage",
                "context": self._get_context(text, match)
            })
        
        return monetary_values
    
    def _get_context(self, text: str, phrase: str, context_window: int = 50) -> str:
        """Get context around a phrase.
        
        Args:
            text: Full text
            phrase: Phrase to find context for
            context_window: Number of characters before and after
            
        Returns:
            Context string
        """
        start_idx = text.find(phrase)
        if start_idx == -1:
            return ""
        
        start_context = max(0, start_idx - context_window)
        end_context = min(len(text), start_idx + len(phrase) + context_window)
        
        return text[start_context:end_context].strip()
    
    def extract_financial_phrases(self, text: str) -> List[Dict[str, str]]:
        """Extract financial phrases using dictionary matching.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of financial phrase dictionaries
        """
        phrases = []
        text_lower = text.lower()
        
        for category, terms in self._financial_dictionary.items():
            for term in terms:
                if term in text_lower:
                    phrases.append({
                        "phrase": term,
                        "category": category,
                        "context": self._get_context(text, term)
                    })
        
        return phrases
    
    def extract_n_grams(self, text: str, n: int = 2) -> List[Tuple[str, int]]:
        """Extract n-grams from text.
        
        Args:
            text: Text to analyze
            n: N-gram size
            
        Returns:
            List of (n-gram, frequency) tuples
        """
        blob = TextBlob(text)
        words = blob.words
        
        n_grams = []
        for i in range(len(words) - n + 1):
            n_gram = ' '.join(words[i:i+n])
            n_grams.append(n_gram.lower())
        
        return Counter(n_grams).most_common()
    
    def extract_key_phrases_tfidf(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
        """Extract key phrases using TF-IDF.
        
        Args:
            texts: List of texts to analyze
            top_k: Number of top phrases to return
            
        Returns:
            List of (phrase, tfidf_score) tuples
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Preprocess texts
            processed_texts = []
            for text in texts:
                blob = TextBlob(text)
                processed_texts.append(' '.join(blob.words))
            
            # Create TF-IDF vectorizer with n-grams
            vectorizer = TfidfVectorizer(
                ngram_range=(self.min_phrase_length, self.max_phrase_length),
                stop_words='english',
                max_features=1000
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top phrases
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]
            top_phrases = [(feature_names[i], mean_scores[i]) for i in top_indices]
            
            return top_phrases
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple frequency counting")
            return self._extract_key_phrases_simple(texts, top_k)
    
    def _extract_key_phrases_simple(self, texts: List[str], top_k: int = 20) -> List[Tuple[str, int]]:
        """Extract key phrases using simple frequency counting.
        
        Args:
            texts: List of texts to analyze
            top_k: Number of top phrases to return
            
        Returns:
            List of (phrase, frequency) tuples
        """
        all_phrases = []
        
        for text in texts:
            # Extract n-grams of different sizes
            for n in range(self.min_phrase_length, self.max_phrase_length + 1):
                n_grams = self.extract_n_grams(text, n)
                all_phrases.extend([phrase for phrase, _ in n_grams])
        
        # Count frequencies
        phrase_counts = Counter(all_phrases)
        
        # Filter out very common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_phrases = {phrase: count for phrase, count in phrase_counts.items() 
                           if not any(word in stop_words for word in phrase.split())}
        
        return Counter(filtered_phrases).most_common(top_k)
    
    def extract_all_key_phrases(self, text: str) -> Dict[str, List]:
        """Extract all types of key phrases from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with different types of extracted phrases
        """
        return {
            "monetary_values": self.extract_monetary_values(text),
            "financial_phrases": self.extract_financial_phrases(text),
            "bigrams": [phrase for phrase, _ in self.extract_n_grams(text, 2)[:10]],
            "trigrams": [phrase for phrase, _ in self.extract_n_grams(text, 3)[:10]],
            "financial_metrics": self._financial_patterns["financial_metrics"].findall(text),
            "growth_terms": self._financial_patterns["growth_terms"].findall(text),
            "time_periods": self._financial_patterns["time_periods"].findall(text)
        }
    
    def get_phrase_importance(self, text: str, phrases: List[str]) -> Dict[str, float]:
        """Calculate importance scores for phrases.
        
        Args:
            text: Text to analyze
            phrases: List of phrases to score
            
        Returns:
            Dictionary mapping phrases to importance scores
        """
        blob = TextBlob(text)
        words = blob.words
        word_count = len(words)
        
        importance_scores = {}
        
        for phrase in phrases:
            phrase_words = phrase.split()
            phrase_length = len(phrase_words)
            
            # Count occurrences
            occurrences = text.lower().count(phrase.lower())
            
            # Calculate TF-IDF-like score
            tf = occurrences / word_count
            idf = np.log(word_count / max(occurrences, 1))
            tfidf_score = tf * idf
            
            # Bonus for financial terms
            financial_bonus = 1.0
            for category, terms in self._financial_dictionary.items():
                if any(term in phrase.lower() for term in terms):
                    financial_bonus = 1.5
                    break
            
            importance_scores[phrase] = tfidf_score * financial_bonus
        
        return importance_scores
