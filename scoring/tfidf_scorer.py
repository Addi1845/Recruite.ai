# scoring/tfidf_scorer.py
"""
TF-IDF cosine-similarity scorer — pure Python, zero external deps.

Measures lexical overlap between the Job Description and each candidate's
combined profile text.  Designed for speed: the JD vector and IDF table are
pre-computed once at module load; per-candidate work is just tokenise →
count → dot-product.

Constraints
-----------
* No sklearn, torch, or network calls.
* Only Python stdlib + math.
* Must handle 100K candidates in < 60 s.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

# ═══════════════════════════════════════════════════════════════════════════════
#  Job Description (embedded constant)
# ═══════════════════════════════════════════════════════════════════════════════

JD_TEXT = '''
Senior AI Engineer - Search, Ranking & Retrieval

About the Role:
We are hiring a Senior AI Engineer to own and evolve our search, ranking, and retrieval stack.
You will design and ship embedding-based retrieval, hybrid search, learning-to-rank models,
and evaluation frameworks. You will work across the full ML lifecycle from feature engineering
through production deployment and A/B testing.

Must-Have Skills:
- Sentence Transformers, BGE, E5 or similar bi-encoder embedding models
- Vector databases: Pinecone, Weaviate, Qdrant, Milvus, or FAISS
- Hybrid search combining dense retrieval with BM25 or sparse methods
- Learning to Rank: LambdaRank, XGBoost/LightGBM rankers, NDCG/MRR optimization
- Production ML deployment, latency optimization, A/B testing frameworks
- Strong Python, NLP fundamentals

Nice-to-Have:
- LoRA/QLoRA fine-tuning of embedding models
- RAG pipeline design and optimization
- Recommendation systems
- Distributed systems for large-scale indexing

Experience: 5-9 years preferred, 4-10 acceptable
Location: Pune or Noida (hybrid), open to relocation within India
Notice Period: Prefer sub-30 days, can buy out up to 30 days
'''

# ═══════════════════════════════════════════════════════════════════════════════
#  Stopwords — top ~100 English function words
# ═══════════════════════════════════════════════════════════════════════════════

_STOPWORDS: frozenset = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "d", "did", "didn", "do", "does", "doesn", "doing", "don",
    "down", "during", "each", "few", "for", "from", "further", "get",
    "got", "had", "has", "hasn", "have", "haven", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "isn", "it", "its", "itself", "just", "ll", "m",
    "me", "might", "more", "most", "must", "mustn", "my", "myself", "no",
    "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
    "same", "shan", "she", "should", "shouldn", "so", "some", "such", "t",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "ve", "very", "was", "wasn", "we", "were",
    "weren", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "won", "would", "wouldn", "you", "your",
    "yours", "yourself", "yourselves",
})

# Pre-compiled regex: split on anything that is NOT alphanumeric.
_SPLIT_RE = re.compile(r"[^a-z0-9]+")


# ═══════════════════════════════════════════════════════════════════════════════
#  Tokenisation
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenise(text: str) -> List[str]:
    """Lowercase → split on non-alnum → drop stopwords & 1-char tokens."""
    return [
        tok
        for tok in _SPLIT_RE.split(text.lower())
        if tok and len(tok) > 1 and tok not in _STOPWORDS
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  IDF computation  (built from JD sentences — see docstring note)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_idf(text: str) -> Dict[str, float]:
    """Build IDF from pseudo-documents (sentences of *text*).

    Since we cannot pre-scan 100K candidates, we split the JD itself into
    sentence-level "documents" and compute IDF over those.  Words that
    appear in every sentence get a lower weight; words unique to one
    sentence get a higher weight.  This is a pragmatic approximation:
    the scorer's job is to measure *how much JD vocabulary a candidate
    echoes*, so JD-internal distribution is a reasonable prior.
    """
    # Split on sentence-ending punctuation or newline-dash (bullet items).
    sentences = re.split(r"[.\n]", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    n_docs = len(sentences)
    if n_docs == 0:
        return {}

    # Document-frequency: in how many sentences does each term appear?
    df: Counter = Counter()
    for sent in sentences:
        unique_tokens = set(_tokenise(sent))
        for tok in unique_tokens:
            df[tok] += 1

    # IDF with +1 smoothing to avoid log(0) and reduce blow-up for hapax.
    idf: Dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((n_docs + 1) / (freq + 1)) + 1.0

    return idf


# ═══════════════════════════════════════════════════════════════════════════════
#  TF-IDF vector construction
# ═══════════════════════════════════════════════════════════════════════════════

def _tfidf_vector(
    tokens: List[str],
    idf: Dict[str, float],
) -> Dict[str, float]:
    """Return a sparse TF-IDF vector (term → weight) with sublinear TF.

    Sublinear TF:  tf = 1 + log(raw_count)  if raw_count > 0, else 0.
    """
    tf_raw = Counter(tokens)
    vec: Dict[str, float] = {}
    for term, count in tf_raw.items():
        if term in idf:
            tf = 1.0 + math.log(count) if count > 0 else 0.0
            vec[term] = tf * idf[term]
    return vec


def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors stored as dicts.

    Iterates over the *smaller* vector for speed.
    """
    if not a or not b:
        return 0.0

    # Always iterate over the smaller dict.
    if len(a) > len(b):
        a, b = b, a

    dot = 0.0
    for term, w_a in a.items():
        w_b = b.get(term)
        if w_b is not None:
            dot += w_a * w_b

    if dot == 0.0:
        return 0.0

    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    denom = norm_a * norm_b

    if denom == 0.0:
        return 0.0

    return dot / denom


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-level pre-computation  (runs once at import time)
# ═══════════════════════════════════════════════════════════════════════════════

_JD_IDF: Dict[str, float] = _build_idf(JD_TEXT)
_JD_TOKENS: List[str] = _tokenise(JD_TEXT)
_JD_VEC: Dict[str, float] = _tfidf_vector(_JD_TOKENS, _JD_IDF)


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_candidate_text(candidate: dict) -> str:
    """Concatenate all scoreable text fields from a candidate dict.

    Fields pulled:
        • profile.summary
        • profile.headline
        • career_history[].title
        • career_history[].description
        • skills[].name
    """
    parts: List[str] = []

    profile = candidate.get("profile") or {}
    summary = profile.get("summary") or ""
    headline = profile.get("headline") or ""
    if summary:
        parts.append(summary)
    if headline:
        parts.append(headline)

    for job in candidate.get("career_history") or []:
        title = job.get("title") or ""
        desc = job.get("description") or ""
        if title:
            parts.append(title)
        if desc:
            parts.append(desc)

    for skill in candidate.get("skills") or []:
        name = skill.get("name") or "" if isinstance(skill, dict) else str(skill)
        if name:
            parts.append(name)

    return " ".join(parts)


def compute_tfidf_score(candidate: dict) -> float:
    """Compute TF-IDF cosine similarity between the JD and *candidate*.

    Parameters
    ----------
    candidate : dict
        A candidate record with optional keys ``profile``, ``career_history``,
        and ``skills``.

    Returns
    -------
    float
        Cosine similarity in [0.0, 1.0].  Higher means the candidate's text
        more closely mirrors the JD's vocabulary.
    """
    text = _extract_candidate_text(candidate)
    if not text:
        return 0.0

    tokens = _tokenise(text)
    if not tokens:
        return 0.0

    # Cap at 800 tokens for speed
    if len(tokens) > 800:
        tokens = tokens[:800]

    candidate_vec = _tfidf_vector(tokens, _JD_IDF)
    return _cosine_similarity(_JD_VEC, candidate_vec)
