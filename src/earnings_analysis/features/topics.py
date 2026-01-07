"""Topic modeling for earnings call transcripts."""

import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TopicModeler:
    """Extract topics from earnings call transcripts.
    
    Supports LDA, BERT-based clustering, and other topic modeling methods.
    """
    
    def __init__(
        self, 
        num_topics: int = 10,
        method: str = "lda",
        max_features: int = 1000
    ) -> None:
        """Initialize topic modeler.
        
        Args:
            num_topics: Number of topics to extract
            method: Topic modeling method ('lda', 'bert_clustering', 'tfidf_lda')
            max_features: Maximum number of features for TF-IDF
        """
        self.num_topics = num_topics
        self.method = method
        self.max_features = max_features
        
        self._model = None
        self._vectorizer = None
        self._dictionary = None
        self._corpus = None
        
        if method == "lda":
            self._init_lda()
        elif method == "tfidf_lda":
            self._init_tfidf_lda()
        elif method == "bert_clustering":
            self._init_bert_clustering()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _init_lda(self) -> None:
        """Initialize Gensim LDA model."""
        logger.info("Initializing Gensim LDA model")
        # Will be created during fit
    
    def _init_tfidf_lda(self) -> None:
        """Initialize TF-IDF + LDA model."""
        logger.info("Initializing TF-IDF + LDA model")
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    
    def _init_bert_clustering(self) -> None:
        """Initialize BERT-based clustering."""
        logger.info("Initializing BERT clustering model")
        # Will be implemented with transformers
        self._kmeans = KMeans(n_clusters=self.num_topics, random_state=42)
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for topic modeling.
        
        Args:
            text: Raw text
            
        Returns:
            List of preprocessed tokens
        """
        import re
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        
        # Download required NLTK data
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            pass
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except:
            # Fallback stopwords
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            tokens = [token for token in tokens if token not in stop_words]
        
        # Remove short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def fit(self, texts: List[str]) -> None:
        """Fit topic model to texts.
        
        Args:
            texts: List of transcript texts
        """
        logger.info(f"Fitting topic model with {len(texts)} documents")
        
        if self.method == "lda":
            self._fit_gensim_lda(texts)
        elif self.method == "tfidf_lda":
            self._fit_tfidf_lda(texts)
        elif self.method == "bert_clustering":
            self._fit_bert_clustering(texts)
    
    def _fit_gensim_lda(self, texts: List[str]) -> None:
        """Fit Gensim LDA model."""
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Create dictionary and corpus
        self._dictionary = corpora.Dictionary(processed_texts)
        self._corpus = [self._dictionary.doc2bow(text) for text in processed_texts]
        
        # Train LDA model
        self._model = models.LdaModel(
            self._corpus,
            num_topics=self.num_topics,
            id2word=self._dictionary,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
    
    def _fit_tfidf_lda(self, texts: List[str]) -> None:
        """Fit TF-IDF + LDA model."""
        # Preprocess texts
        processed_texts = [' '.join(self._preprocess_text(text)) for text in texts]
        
        # Fit TF-IDF vectorizer
        tfidf_matrix = self._vectorizer.fit_transform(processed_texts)
        
        # Train LDA model
        self._model = LatentDirichletAllocation(
            n_components=self.num_topics,
            random_state=42,
            max_iter=10
        )
        self._model.fit(tfidf_matrix)
    
    def _fit_bert_clustering(self, texts: List[str]) -> None:
        """Fit BERT-based clustering."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            # Load BERT model
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            model = AutoModel.from_pretrained('bert-base-uncased')
            
            # Get embeddings
            embeddings = []
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    embeddings.append(embedding.numpy())
            
            # Fit K-means
            embeddings = np.array(embeddings)
            self._kmeans.fit(embeddings)
            
        except ImportError:
            logger.warning("Transformers not available, falling back to TF-IDF clustering")
            self.method = "tfidf_lda"
            self._fit_tfidf_lda(texts)
    
    def get_topics(self, num_words: int = 10) -> List[Dict[str, List[Tuple[str, float]]]]:
        """Get topic words and their weights.
        
        Args:
            num_words: Number of words per topic
            
        Returns:
            List of topics with word-weight pairs
        """
        if self._model is None:
            raise RuntimeError("Model not fitted")
        
        topics = []
        
        if self.method == "lda":
            for topic_id in range(self.num_topics):
                topic_words = self._model.show_topic(topic_id, topn=num_words)
                topics.append({
                    'topic_id': topic_id,
                    'words': topic_words,
                    'method': 'gensim_lda'
                })
        
        elif self.method == "tfidf_lda":
            feature_names = self._vectorizer.get_feature_names_out()
            for topic_id in range(self.num_topics):
                topic_words = []
                for word_idx in self._model.components_[topic_id].argsort()[-num_words:][::-1]:
                    word = feature_names[word_idx]
                    weight = self._model.components_[topic_id][word_idx]
                    topic_words.append((word, weight))
                topics.append({
                    'topic_id': topic_id,
                    'words': topic_words,
                    'method': 'tfidf_lda'
                })
        
        elif self.method == "bert_clustering":
            # For BERT clustering, we can't easily get topic words
            # This would require additional analysis
            topics = [{'topic_id': i, 'words': [], 'method': 'bert_clustering'} 
                     for i in range(self.num_topics)]
        
        return topics
    
    def predict_topics(self, texts: List[str]) -> List[int]:
        """Predict topics for new texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predicted topic IDs
        """
        if self._model is None:
            raise RuntimeError("Model not fitted")
        
        predictions = []
        
        if self.method == "lda":
            processed_texts = [self._preprocess_text(text) for text in texts]
            for text in processed_texts:
                bow = self._dictionary.doc2bow(text)
                topic_probs = self._model[bow]
                # Get the topic with highest probability
                if topic_probs:
                    best_topic = max(topic_probs, key=lambda x: x[1])[0]
                    predictions.append(best_topic)
                else:
                    predictions.append(0)
        
        elif self.method == "tfidf_lda":
            processed_texts = [' '.join(self._preprocess_text(text)) for text in texts]
            tfidf_matrix = self._vectorizer.transform(processed_texts)
            topic_probs = self._model.transform(tfidf_matrix)
            predictions = topic_probs.argmax(axis=1).tolist()
        
        elif self.method == "bert_clustering":
            # This would require re-running BERT embeddings
            # For now, return random predictions
            predictions = [0] * len(texts)
        
        return predictions
    
    def get_topic_distribution(self, text: str) -> Dict[int, float]:
        """Get topic distribution for a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping topic IDs to probabilities
        """
        if self._model is None:
            raise RuntimeError("Model not fitted")
        
        if self.method == "lda":
            processed_text = self._preprocess_text(text)
            bow = self._dictionary.doc2bow(processed_text)
            topic_probs = self._model[bow]
            return {topic_id: prob for topic_id, prob in topic_probs}
        
        elif self.method == "tfidf_lda":
            processed_text = ' '.join(self._preprocess_text(text))
            tfidf_vector = self._vectorizer.transform([processed_text])
            topic_probs = self._model.transform(tfidf_vector)[0]
            return {i: float(prob) for i, prob in enumerate(topic_probs)}
        
        else:
            return {i: 1.0 / self.num_topics for i in range(self.num_topics)}
