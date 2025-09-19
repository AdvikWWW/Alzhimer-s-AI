import spacy
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from transformers import pipeline

logger = logging.getLogger(__name__)

class LexicalSemanticAnalyzer:
    """
    Lexical-semantic analysis based on Favaro et al. (2023)
    Implements lexical diversity, semantic coherence, and syntactic complexity measures
    """
    
    def __init__(self):
        # Initialize spaCy for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize NLTK components
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {str(e)}")
            self.stop_words = set()
        
        # Initialize sentence transformer for semantic analysis
        self.sentence_model = None
        self._initialize_sentence_model()
        
        # Initialize sentiment analysis (optional)
        self.sentiment_analyzer = None
        self._initialize_sentiment_analyzer()
    
    def _initialize_sentence_model(self):
        """Initialize sentence transformer model"""
        try:
            logger.info("Loading sentence transformer model")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {str(e)}")
            self.sentence_model = None
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analysis pipeline"""
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except Exception as e:
            logger.warning(f"Failed to initialize sentiment analyzer: {str(e)}")
            self.sentiment_analyzer = None
    
    def analyze_lexical_semantic_features(self, transcription_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive lexical-semantic analysis of transcribed speech
        """
        try:
            transcript = transcription_result.get("transcript_text", "")
            word_timestamps = transcription_result.get("word_timestamps", [])
            
            if not transcript.strip():
                return self._empty_analysis()
            
            logger.info(f"Analyzing lexical-semantic features for transcript: {len(transcript)} characters")
            
            # Preprocess text
            doc = self.nlp(transcript) if self.nlp else None
            sentences = self._extract_sentences(transcript)
            words = self._extract_words(transcript)
            
            # 1. Lexical diversity analysis
            lexical_diversity = self._analyze_lexical_diversity(words, transcript)
            
            # 2. Semantic coherence analysis
            semantic_coherence = self._analyze_semantic_coherence(sentences)
            
            # 3. Syntactic complexity analysis
            syntactic_complexity = self._analyze_syntactic_complexity(doc, sentences)
            
            # 4. Word frequency and familiarity
            word_frequency = self._analyze_word_frequency(words)
            
            # 5. Discourse analysis
            discourse_features = self._analyze_discourse_features(doc, sentences)
            
            # 6. Semantic embeddings
            semantic_embeddings = self._extract_semantic_embeddings(sentences)
            
            # Combine all features
            analysis_result = {
                **lexical_diversity,
                **semantic_coherence,
                **syntactic_complexity,
                **word_frequency,
                **discourse_features,
                **semantic_embeddings,
                "analysis_metadata": {
                    "transcript_length": len(transcript),
                    "sentence_count": len(sentences),
                    "word_count": len(words),
                    "analysis_method": "comprehensive_lexical_semantic"
                }
            }
            
            logger.info("Lexical-semantic analysis completed")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Lexical-semantic analysis failed: {str(e)}")
            return self._empty_analysis()
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        try:
            if self.nlp:
                doc = self.nlp(text)
                return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                # Fallback: simple sentence splitting
                sentences = re.split(r'[.!?]+', text)
                return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Sentence extraction failed: {str(e)}")
            return [text]  # Return whole text as single sentence
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text, removing punctuation and stopwords"""
        try:
            # Remove punctuation and convert to lowercase
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Filter out stopwords and very short words
            filtered_words = [
                word for word in words 
                if word not in self.stop_words and len(word) > 2
            ]
            
            return filtered_words
        except Exception as e:
            logger.error(f"Word extraction failed: {str(e)}")
            return text.lower().split()
    
    def _analyze_lexical_diversity(self, words: List[str], full_text: str) -> Dict[str, Any]:
        """
        Analyze lexical diversity measures
        Based on Favaro et al. (2023) lexical diversity metrics
        """
        try:
            if not words:
                return {
                    "total_words": 0,
                    "unique_words": 0,
                    "type_token_ratio": 0.0,
                    "moving_average_ttr": 0.0
                }
            
            total_words = len(words)
            unique_words = len(set(words))
            
            # Type-Token Ratio (TTR)
            ttr = unique_words / total_words if total_words > 0 else 0.0
            
            # Moving Average TTR (MATTR) - calculated over windows of 50 words
            window_size = 50
            mattr_values = []
            
            if total_words >= window_size:
                for i in range(total_words - window_size + 1):
                    window_words = words[i:i + window_size]
                    window_unique = len(set(window_words))
                    window_ttr = window_unique / window_size
                    mattr_values.append(window_ttr)
                
                mattr = np.mean(mattr_values) if mattr_values else ttr
            else:
                mattr = ttr
            
            # Additional lexical measures
            word_lengths = [len(word) for word in words]
            avg_word_length = np.mean(word_lengths) if word_lengths else 0.0
            
            # Lexical sophistication (proportion of low-frequency words)
            # This is a simplified measure - in practice, you'd use frequency databases
            long_words = [word for word in words if len(word) > 6]
            lexical_sophistication = len(long_words) / total_words if total_words > 0 else 0.0
            
            return {
                "total_words": total_words,
                "unique_words": unique_words,
                "type_token_ratio": float(ttr),
                "moving_average_ttr": float(mattr),
                "average_word_length": float(avg_word_length),
                "lexical_sophistication": float(lexical_sophistication)
            }
            
        except Exception as e:
            logger.error(f"Lexical diversity analysis failed: {str(e)}")
            return {
                "total_words": 0,
                "unique_words": 0,
                "type_token_ratio": 0.0,
                "moving_average_ttr": 0.0
            }
    
    def _analyze_semantic_coherence(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Analyze semantic coherence and topic consistency
        """
        try:
            if len(sentences) < 2 or not self.sentence_model:
                return {
                    "semantic_coherence_score": None,
                    "topic_drift_score": None,
                    "semantic_similarity_matrix": None
                }
            
            # Generate sentence embeddings
            embeddings = self.sentence_model.encode(sentences)
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(embeddings)
            
            # Semantic coherence: average similarity between adjacent sentences
            adjacent_similarities = []
            for i in range(len(sentences) - 1):
                similarity = similarity_matrix[i][i + 1]
                adjacent_similarities.append(similarity)
            
            semantic_coherence = np.mean(adjacent_similarities) if adjacent_similarities else 0.0
            
            # Topic drift: measure how similarity changes over time
            if len(adjacent_similarities) > 1:
                # Calculate the trend in similarities (negative slope indicates drift)
                x = np.arange(len(adjacent_similarities))
                slope = np.polyfit(x, adjacent_similarities, 1)[0]
                topic_drift = abs(slope)  # Absolute value of slope
            else:
                topic_drift = 0.0
            
            # Overall semantic consistency (average of all pairwise similarities)
            upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            overall_consistency = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
            
            return {
                "semantic_coherence_score": float(semantic_coherence),
                "topic_drift_score": float(topic_drift),
                "overall_semantic_consistency": float(overall_consistency),
                "semantic_similarity_matrix": similarity_matrix.tolist()
            }
            
        except Exception as e:
            logger.error(f"Semantic coherence analysis failed: {str(e)}")
            return {
                "semantic_coherence_score": None,
                "topic_drift_score": None,
                "semantic_similarity_matrix": None
            }
    
    def _analyze_syntactic_complexity(self, doc, sentences: List[str]) -> Dict[str, Any]:
        """
        Analyze syntactic complexity measures
        """
        try:
            if not doc or not sentences:
                return {
                    "mean_sentence_length": 0.0,
                    "syntactic_complexity_score": 0.0,
                    "pos_tag_distribution": {}
                }
            
            # Sentence length statistics
            sentence_lengths = [len(sent.split()) for sent in sentences]
            mean_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0.0
            
            # POS tag distribution
            pos_tags = [token.pos_ for token in doc if not token.is_space]
            pos_distribution = dict(Counter(pos_tags))
            
            # Syntactic complexity measures
            # 1. Clause density (approximate using conjunctions and relative pronouns)
            complex_structures = [
                token for token in doc 
                if token.pos_ in ['SCONJ', 'CCONJ'] or token.dep_ in ['relcl', 'advcl', 'ccomp']
            ]
            clause_density = len(complex_structures) / len(sentences) if sentences else 0.0
            
            # 2. Dependency tree depth (average)
            sentence_depths = []
            for sent in doc.sents:
                depths = [self._get_token_depth(token) for token in sent]
                avg_depth = np.mean(depths) if depths else 0.0
                sentence_depths.append(avg_depth)
            
            avg_dependency_depth = np.mean(sentence_depths) if sentence_depths else 0.0
            
            # 3. Readability scores using textstat
            full_text = " ".join(sentences)
            flesch_reading_ease = textstat.flesch_reading_ease(full_text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(full_text)
            
            # Combine into overall syntactic complexity score
            # Normalize and combine different measures
            complexity_score = (
                min(clause_density / 2.0, 1.0) * 0.3 +  # Clause density (normalized)
                min(avg_dependency_depth / 5.0, 1.0) * 0.3 +  # Dependency depth
                max(0, (flesch_kincaid_grade - 5) / 15.0) * 0.4  # Reading grade level
            )
            
            return {
                "mean_sentence_length": float(mean_sentence_length),
                "syntactic_complexity_score": float(complexity_score),
                "clause_density": float(clause_density),
                "avg_dependency_depth": float(avg_dependency_depth),
                "flesch_reading_ease": float(flesch_reading_ease),
                "flesch_kincaid_grade": float(flesch_kincaid_grade),
                "pos_tag_distribution": pos_distribution
            }
            
        except Exception as e:
            logger.error(f"Syntactic complexity analysis failed: {str(e)}")
            return {
                "mean_sentence_length": 0.0,
                "syntactic_complexity_score": 0.0,
                "pos_tag_distribution": {}
            }
    
    def _get_token_depth(self, token) -> int:
        """Calculate the depth of a token in the dependency tree"""
        depth = 0
        current = token
        while current.head != current:  # Until we reach the root
            depth += 1
            current = current.head
            if depth > 20:  # Prevent infinite loops
                break
        return depth
    
    def _analyze_word_frequency(self, words: List[str]) -> Dict[str, Any]:
        """
        Analyze word frequency and familiarity measures
        """
        try:
            if not words:
                return {
                    "word_frequency_score": 0.0,
                    "age_of_acquisition_score": None
                }
            
            # Word frequency analysis (simplified)
            # In practice, you'd use established frequency databases like SUBTLEX
            word_counts = Counter(words)
            
            # High-frequency words (appearing more than once)
            high_freq_words = [word for word, count in word_counts.items() if count > 1]
            word_frequency_score = len(high_freq_words) / len(set(words)) if words else 0.0
            
            # Age of acquisition (simplified approximation based on word length)
            # In practice, you'd use AoA databases
            avg_word_length = np.mean([len(word) for word in words]) if words else 0.0
            age_of_acquisition_score = min(avg_word_length / 8.0, 1.0)  # Normalize to 0-1
            
            return {
                "word_frequency_score": float(word_frequency_score),
                "age_of_acquisition_score": float(age_of_acquisition_score),
                "vocabulary_size": len(set(words)),
                "most_frequent_words": dict(word_counts.most_common(10))
            }
            
        except Exception as e:
            logger.error(f"Word frequency analysis failed: {str(e)}")
            return {
                "word_frequency_score": 0.0,
                "age_of_acquisition_score": None
            }
    
    def _analyze_discourse_features(self, doc, sentences: List[str]) -> Dict[str, Any]:
        """
        Analyze discourse-level features
        """
        try:
            if not doc or not sentences:
                return {
                    "idea_density": 0.0,
                    "propositional_density": 0.0
                }
            
            # Idea density: ratio of propositions to total words
            # Approximate using content words (nouns, verbs, adjectives, adverbs)
            content_words = [
                token for token in doc 
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop
            ]
            
            total_words = len([token for token in doc if not token.is_space and not token.is_punct])
            idea_density = len(content_words) / total_words if total_words > 0 else 0.0
            
            # Propositional density: ratio of propositions to sentences
            # Approximate using main verbs and clauses
            main_verbs = [token for token in doc if token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'ccomp', 'xcomp']]
            propositional_density = len(main_verbs) / len(sentences) if sentences else 0.0
            
            # Discourse markers
            discourse_markers = [
                'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
                'consequently', 'meanwhile', 'additionally', 'similarly', 'conversely'
            ]
            
            text_lower = " ".join(sentences).lower()
            discourse_marker_count = sum(1 for marker in discourse_markers if marker in text_lower)
            discourse_marker_density = discourse_marker_count / len(sentences) if sentences else 0.0
            
            return {
                "idea_density": float(idea_density),
                "propositional_density": float(propositional_density),
                "discourse_marker_density": float(discourse_marker_density),
                "content_word_ratio": float(len(content_words) / total_words if total_words > 0 else 0.0)
            }
            
        except Exception as e:
            logger.error(f"Discourse analysis failed: {str(e)}")
            return {
                "idea_density": 0.0,
                "propositional_density": 0.0
            }
    
    def _extract_semantic_embeddings(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Extract sentence embeddings for downstream analysis
        """
        try:
            if not sentences or not self.sentence_model:
                return {
                    "sentence_embeddings": None,
                    "embedding_statistics": {}
                }
            
            # Generate embeddings
            embeddings = self.sentence_model.encode(sentences)
            
            # Calculate embedding statistics
            embedding_mean = np.mean(embeddings, axis=0)
            embedding_std = np.std(embeddings, axis=0)
            embedding_variance = np.var(embeddings, axis=0)
            
            # Semantic diversity (average pairwise distance)
            if len(embeddings) > 1:
                pairwise_distances = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        distance = np.linalg.norm(embeddings[i] - embeddings[j])
                        pairwise_distances.append(distance)
                
                semantic_diversity = np.mean(pairwise_distances) if pairwise_distances else 0.0
            else:
                semantic_diversity = 0.0
            
            return {
                "sentence_embeddings": embeddings.tolist(),
                "embedding_statistics": {
                    "mean_embedding": embedding_mean.tolist(),
                    "embedding_dimensionality": embeddings.shape[1],
                    "semantic_diversity": float(semantic_diversity),
                    "embedding_variance_mean": float(np.mean(embedding_variance))
                }
            }
            
        except Exception as e:
            logger.error(f"Semantic embedding extraction failed: {str(e)}")
            return {
                "sentence_embeddings": None,
                "embedding_statistics": {}
            }
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis result"""
        return {
            "total_words": 0,
            "unique_words": 0,
            "type_token_ratio": 0.0,
            "moving_average_ttr": 0.0,
            "semantic_coherence_score": None,
            "topic_drift_score": None,
            "mean_sentence_length": 0.0,
            "syntactic_complexity_score": 0.0,
            "word_frequency_score": 0.0,
            "age_of_acquisition_score": None,
            "idea_density": 0.0,
            "pos_tag_distribution": {},
            "sentence_embeddings": None,
            "semantic_similarity_matrix": None,
            "analysis_metadata": {
                "transcript_length": 0,
                "sentence_count": 0,
                "word_count": 0,
                "analysis_method": "empty"
            }
        }
