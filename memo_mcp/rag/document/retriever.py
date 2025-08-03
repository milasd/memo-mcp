import logging
import re
from datetime import date, datetime
from typing import Any

from memo_mcp.rag.config.rag_config import RAGConfig
from memo_mcp.rag.vector.embeddings import EmbeddingManager
from memo_mcp.rag.vector.vector_store import VectorStore


class DocumentRetriever:
    """
    Document retrieval with advanced filtering and ranking.
    I arbitrarily set these for a preliminary version, but they can be changed later.

    Provides semantic search with optional date filtering and result reranking.
    """

    def __init__(
        self,
        config: RAGConfig,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStore,
    ):
        self.config = config
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

    def retrieve(
        self,
        query_text: str,
        top_k: int,
        date_filter: tuple[date, date] | None = None,
        similarity_threshold: float = 0.0,
        enable_reranking: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query_text: Search query
            top_k: Number of results to return
            date_filter: Optional date range (start_date, end_date)
            similarity_threshold: Minimum similarity score
            enable_reranking: Whether to apply reranking

        Returns:
            List of relevant documents with metadata and scores
        """
        self.logger.debug(f"Retrieving documents for query: '{query_text}'")

        processed_query = self._preprocess_query(query_text)
        query_embedding = self.embedding_manager.embed_text(processed_query)

        # Search vector store (get more results for filtering/reranking)
        search_k = min(top_k * 3, 50)  # Get extra results for filtering
        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=search_k,
            similarity_threshold=similarity_threshold
            * 0.8,  # Lower threshold for initial search
        )

        if not raw_results:
            return []

        if date_filter:
            raw_results = self._apply_date_filter(raw_results, date_filter)

        if enable_reranking:
            raw_results = self._rerank_results(raw_results, query_text, processed_query)

        # Apply final similarity threshold
        filtered_results = [
            result
            for result in raw_results
            if result["similarity_score"] >= similarity_threshold
        ]

        # Limit to requested number of results
        final_results = filtered_results[:top_k]
        # Enhance results with additional metadata
        enhanced_results = self._enhance_results(final_results, query_text)

        self.logger.debug(f"Retrieved {len(enhanced_results)} documents")
        return enhanced_results

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query for better matching."""
        query = query.strip()

        expansions = {"w/": "with", "w/o": "without", "b/c": "because", "&": "and"}

        for abbrev, expansion in expansions.items():
            query = re.sub(
                rf"\b{re.escape(abbrev)}\b", expansion, query, flags=re.IGNORECASE
            )

        # Handle date expressions
        query = self._expand_date_expressions(query)

        return query

    def _expand_date_expressions(self, query: str) -> str:
        """Expand date expressions in query for better matching."""
        # Convert relative dates
        today = datetime.now().date()

        replacements = {
            r"\btoday\b": today.isoformat(),
            r"\byesterday\b": (
                today.replace(day=today.day - 1) if today.day > 1 else today
            ).isoformat(),
            r"\bthis week\b": f"{today.isoformat()} week",
            r"\blast week\b": f"{today.isoformat()} last week",
            r"\bthis month\b": f"{today.year}-{today.month:02d}",
            r"\blast month\b": f"{today.year}-{today.month - 1:02d}"
            if today.month > 1
            else f"{today.year - 1}-12",
        }

        for pattern, replacement in replacements.items():
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)

        return query

    def _apply_date_filter(
        self, results: list[dict[str, Any]], date_filter: tuple[date, date]
    ) -> list[dict[str, Any]]:
        """Filter results by date range."""
        start_date, end_date = date_filter
        filtered_results = []

        for result in results:
            try:
                # Parse date from metadata
                doc_date_str = result["metadata"].date_created
                doc_date = datetime.fromisoformat(doc_date_str).date()

                if start_date <= doc_date <= end_date:
                    filtered_results.append(result)

            except (ValueError, AttributeError) as e:
                self.logger.debug(f"Failed to parse date for filtering: {e}")
                # Include result if date parsing fails
                filtered_results.append(result)

        return filtered_results

    def _rerank_results(
        self, results: list[dict[str, Any]], original_query: str, processed_query: str
    ) -> list[dict[str, Any]]:
        """Rerank results using additional signals."""
        if len(results) <= 1:
            return results

        # Calculate additional relevance signals
        for result in results:
            text = result["text"]
            metadata = result["metadata"]

            # Keyword overlap score
            keyword_score = self._calculate_keyword_overlap(processed_query, text)

            # Date relevance score (more recent = higher score)
            date_score = self._calculate_date_relevance(metadata.date_created)

            # Length penalty (very short or very long chunks get penalty)
            length_score = self._calculate_length_score(text)

            # Position bonus (earlier chunks in document might be more important)
            position_score = self._calculate_position_score(
                metadata.chunk_index, metadata.total_chunks
            )

            # Combine scores
            combined_score = (
                result["similarity_score"] * 0.6  # Semantic similarity (primary)
                + keyword_score * 0.2  # Keyword overlap
                + date_score * 0.1  # Recency
                + length_score * 0.05  # Length appropriateness
                + position_score * 0.05  # Position in document
            )

            result["combined_score"] = combined_score
            result["keyword_score"] = keyword_score
            result["date_score"] = date_score

        # Sort by combined score
        results.sort(key=lambda x: x["combined_score"], reverse=True)

        return results

    def _calculate_keyword_overlap(self, query: str, text: str) -> float:
        """Calculate keyword overlap between query and text."""
        # Extract keywords from query (remove stop words)
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
            "would",
            "i",
            "you",
            "me",
        }

        query_words = {
            word.lower()
            for word in re.findall(r"\b\w+\b", query)
            if word.lower() not in stop_words and len(word) > 2
        }

        if not query_words:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for word in query_words if word in text_lower)

        return matches / len(query_words)

    def _calculate_date_relevance(self, date_str: str) -> float:
        """Calculate relevance based on distance from note to current day."""
        try:
            doc_date = datetime.fromisoformat(date_str).date()
            today = datetime.now().date()
            days_diff = (today - doc_date).days

            # Decay function: more recent = higher score
            if days_diff <= 7:
                return 1.0
            elif days_diff <= 30:
                return 0.8
            elif days_diff <= 90:
                return 0.6
            elif days_diff <= 365:
                return 0.4
            else:
                return 0.2

        except (ValueError, AttributeError):
            return 0.5  # Default score if date parsing fails

    def _calculate_length_score(self, text: str) -> float:
        """Calculate score based on text length."""
        length = len(text)

        # Optimal length is around 200-800 characters
        if 200 <= length <= 800:
            return 1.0
        elif 100 <= length < 200 or 800 < length <= 1200:
            return 0.8
        elif 50 <= length < 100 or 1200 < length <= 2000:
            return 0.6
        else:
            return 0.4

    def _calculate_position_score(self, chunk_index: int, total_chunks: int) -> float:
        """Calculate score based on position in document."""
        if total_chunks <= 1:
            return 1.0

        # Earlier chunks get slightly higher scores
        position_ratio = chunk_index / (total_chunks - 1)

        if position_ratio <= 0.3:  # First third
            return 1.0
        elif position_ratio <= 0.6:  # Middle third
            return 0.8
        else:  # Last third
            return 0.6

    def _enhance_results(
        self, results: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """Enhance results with additional metadata and highlighting."""
        enhanced = []

        for result in results:
            enhanced_result = result.copy()

            # Add highlighted text
            enhanced_result["highlighted_text"] = self._highlight_query_terms(
                result["text"], query
            )

            # Add relevance explanation
            enhanced_result["relevance_explanation"] = (
                self._generate_relevance_explanation(result)
            )

            # Add document summary if this is the first chunk
            if result["metadata"].chunk_index == 0:
                enhanced_result["is_document_start"] = True
                enhanced_result["document_summary"] = self._generate_document_summary(
                    result["text"]
                )
            else:
                enhanced_result["is_document_start"] = False

            enhanced.append(enhanced_result)

        return enhanced

    def _highlight_query_terms(self, text: str, query: str) -> str:
        """Highlight query terms in text."""
        # Extract meaningful terms from query
        terms = re.findall(r"\b\w{3,}\b", query.lower())

        highlighted = text
        for term in terms:
            # Case-insensitive highlighting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f"**{term.upper()}**", highlighted)

        return highlighted

    def _generate_relevance_explanation(self, result: dict[str, Any]) -> str:
        """Generate explanation for why this result is relevant."""
        explanations = []

        similarity_score = result.get("similarity_score", 0)
        keyword_score = result.get("keyword_score", 0)
        date_score = result.get("date_score", 0)

        if similarity_score > 0.8:
            explanations.append("high semantic similarity")
        elif similarity_score > 0.6:
            explanations.append("good semantic similarity")

        if keyword_score > 0.5:
            explanations.append("strong keyword match")
        elif keyword_score > 0.2:
            explanations.append("keyword match")

        if date_score > 0.8:
            explanations.append("recent content")

        if result["metadata"].chunk_index == 0:
            explanations.append("document beginning")

        return ", ".join(explanations) if explanations else "semantic match"

    def _generate_document_summary(self, text: str) -> str:
        """Generate a brief summary of document content."""
        # Simple extractive summary - take first sentence or two
        sentences = re.split(r"[.!?]+", text)

        summary_parts: list[str] = []
        char_count = 0
        max_chars = 200

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if char_count + len(sentence) > max_chars and summary_parts:
                break

            summary_parts.append(sentence)
            char_count += len(sentence)

        summary = ". ".join(summary_parts)
        if summary and not summary.endswith("."):
            summary += "..."

        return summary or text[:200] + "..."


class QueryExpander:
    """
    Expands queries with related terms and synonyms for better retrieval.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Simple synonym dictionary - could be enhanced with external resources
        self.synonyms = {
            "happy": ["joyful", "pleased", "content", "glad"],
            "sad": ["unhappy", "depressed", "down", "melancholy"],
            "work": ["job", "career", "employment", "profession"],
            "meeting": ["conference", "discussion", "session", "gathering"],
            "idea": ["concept", "thought", "notion", "suggestion"],
            "problem": ["issue", "challenge", "difficulty", "trouble"],
            "solution": ["answer", "resolution", "fix", "remedy"],
        }

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        words = query.lower().split()
        expanded_words = []

        for word in words:
            expanded_words.append(word)

            # Add synonyms if available
            if word in self.synonyms:
                expanded_words.extend(self.synonyms[word][:2])  # Limit to 2 synonyms

        return " ".join(expanded_words)


class ResultAggregator:
    """
    Aggregates and deduplicates results from multiple search strategies.
    """

    @staticmethod
    def merge_results(
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]] | None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Merge results from different search strategies."""
        all_results = {}

        # Add semantic results
        for result in semantic_results:
            key = f"{result['metadata'].file_path}_{result['metadata'].chunk_index}"
            if key not in all_results:
                all_results[key] = result.copy()
                all_results[key]["search_methods"] = ["semantic"]
            else:
                all_results[key]["similarity_score"] = max(
                    all_results[key]["similarity_score"], result["similarity_score"]
                )

        # Add keyword results if provided
        if keyword_results:
            for result in keyword_results:
                key = f"{result['metadata'].file_path}_{result['metadata'].chunk_index}"
                if key not in all_results:
                    all_results[key] = result.copy()
                    all_results[key]["search_methods"] = ["keyword"]
                else:
                    # Boost score for multi-method matches
                    all_results[key]["similarity_score"] *= 1.2
                    all_results[key]["search_methods"].append("keyword")

        # Sort by score and return top results
        merged = list(all_results.values())
        merged.sort(
            key=lambda x: x.get("combined_score", x["similarity_score"]), reverse=True
        )

        return merged[:top_k]
