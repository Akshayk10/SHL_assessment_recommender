"""
retrieval_hybrid.py
-------------------
Novel Dual-Stage Hybrid Retrieval with Reciprocal Rank Fusion (RRF).

BM25 excels at:   exact tech terms (Java 8, .NET, OPQ32r, Verify G+)
Dense excels at:  semantic concepts (stakeholder management, leadership)
RRF fuses both:   no score normalization needed, robust to distribution shifts
Metadata rerank:  boosts results matching user's preferred test types / remote
"""

import re
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Retriever (Exact Keyword Matching)
# ─────────────────────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    BM25Okapi-based retrieval for exact keyword matching.
    Critical for tech assessments where names matter (Java 8, .NET, OPQ32r).
    """

    def __init__(self):
        self.catalog: List[Dict] = []
        self._bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text preserving tech terms.
        Keeps: alphanumeric, dots (.NET), hashes (C#), plus (+).
        """
        tokens = re.findall(r"[\w.#+]+", text.lower())
        return tokens if tokens else ["<empty>"]

    def _make_doc(self, item: Dict) -> str:
        """Build a searchable text blob for one catalog entry."""
        parts = [
            item.get("name", ""),
            item.get("name", ""),          # Repeat name for higher BM25 weight
            item.get("test_type_label", ""),
            item.get("description", "")[:300],
            " ".join(item.get("all_test_types", [])),
        ]
        if item.get("remote_testing"):
            parts.append("remote testing")
        if item.get("adaptive"):
            parts.append("adaptive irt")
        return " ".join(parts)

    def build(self, catalog: List[Dict]):
        """Build BM25 index from catalog. Called once at startup."""
        from rank_bm25 import BM25Okapi

        self.catalog = catalog
        corpus = [self._tokenize(self._make_doc(item)) for item in catalog]
        self._bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built: {len(catalog)} documents")

    def search(self, query: str, k: int = 20) -> List[Dict]:
        """Return top-k results with bm25_score field."""
        if self._bm25 is None:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                item = self.catalog[int(idx)].copy()
                item["bm25_score"] = float(scores[idx])
                results.append(item)

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Re-ranker
# ─────────────────────────────────────────────────────────────────────────────

class MetadataReranker:
    """
    Re-rank fused candidates using structured metadata preferences.
    Boosts: preferred test types, remote/adaptive, keyword matches in name.
    """

    def rerank(
        self,
        candidates: List[Dict],
        preferred_types: Optional[List[str]] = None,
        remote: Optional[bool] = None,
        boost_keywords: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Apply metadata-based re-ranking on top of RRF scores."""
        for item in candidates:
            score = item.get("rrf_score", 0.01)
            all_types = set(item.get("all_test_types", [item.get("test_type", "")]))

            # Boost: preferred test types (user said "personality" → boost P types)
            if preferred_types:
                if any(t in all_types for t in preferred_types):
                    score *= 1.5

            # Boost: remote testing preference
            if remote is True and item.get("remote_testing"):
                score *= 1.25

            # Boost: keyword appears literally in assessment name (exact product match)
            if boost_keywords:
                name_lower = item.get("name", "").lower()
                for kw in boost_keywords:
                    if len(kw) >= 3 and kw.lower() in name_lower:
                        score *= 1.4
                        break  # One boost per item max

            item["final_score"] = score

        return sorted(candidates, key=lambda x: x.get("final_score", 0), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    k: int = 60,
) -> List[Dict]:
    """
    Fuse multiple ranked lists using RRF.
    Score = Σ  1 / (k + rank_i)  for each list where item appears.
    k=60 is the standard RRF constant (from the original 2009 paper).
    """
    rrf_scores: Dict[str, float] = {}
    url_to_item: Dict[str, Dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            url = item.get("url", "")
            if not url:
                continue
            rrf_scores[url] = rrf_scores.get(url, 0.0) + 1.0 / (k + rank)
            url_to_item[url] = item

    sorted_urls = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

    result = []
    for url in sorted_urls:
        item = url_to_item[url].copy()
        item["rrf_score"] = rrf_scores[url]
        result.append(item)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Retriever
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 (keyword) + FAISS dense (semantic) via RRF,
    then applies metadata re-ranking for structure-aware boosting.

    Designed to maximize Recall@10 across diverse query types:
      - "Java developer"    → BM25 finds exact "Java 8 (New)" assessment
      - "stakeholder comms" → Dense finds personality/competency assessments
      - "personality tests" → Metadata re-ranker boosts P-type assessments
    """

    def __init__(self):
        self.bm25 = BM25Retriever()
        self.reranker = MetadataReranker()
        self._built = False

    def build(self, catalog: List[Dict]):
        """Build all indexes. Called once at startup."""
        self.bm25.build(catalog)
        self._built = True
        logger.info("HybridRetriever fully built")

    def retrieve(
        self,
        query: str,
        k: int = 10,
        preferred_types: Optional[List[str]] = None,
        remote: Optional[bool] = None,
        boost_keywords: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Full pipeline: BM25 → Dense FAISS → RRF → Metadata Rerank → top-k.
        """
        # Stage 1: BM25 keyword retrieval (fast, exact)
        bm25_results = self.bm25.search(query, k=20)
        logger.debug(f"BM25: {len(bm25_results)} results")

        # Stage 2: Dense FAISS semantic retrieval
        dense_results = []
        try:
            from retrieval_rag import retrieve_assessments
            dense_results = retrieve_assessments(query, k=20)
            logger.debug(f"Dense: {len(dense_results)} results")
        except Exception as exc:
            logger.warning(f"Dense retrieval failed, using BM25 only: {exc}")

        # Stage 3: Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion([bm25_results, dense_results])
        logger.debug(f"After RRF fusion: {len(fused)} candidates")

        # Stage 4: Metadata-aware re-ranking
        reranked = self.reranker.rerank(
            fused,
            preferred_types=preferred_types,
            remote=remote,
            boost_keywords=boost_keywords,
        )

        return reranked[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Singleton & Convenience API
# ─────────────────────────────────────────────────────────────────────────────

_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Singleton: build once, reuse across requests."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
        from retrieval_rag import get_retrieval_system
        system = get_retrieval_system()  # loads catalog
        _hybrid_retriever.build(system.catalog)
    return _hybrid_retriever


def hybrid_retrieve_assessments(
    query: str,
    k: int = 10,
    preferred_types: Optional[List[str]] = None,
    remote: Optional[bool] = None,
    boost_keywords: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Convenience wrapper.  Drop-in replacement for retrieve_assessments().
    """
    retriever = get_hybrid_retriever()
    return retriever.retrieve(
        query,
        k=k,
        preferred_types=preferred_types,
        remote=remote,
        boost_keywords=boost_keywords,
    )
