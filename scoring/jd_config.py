# scoring/jd_config.py

# Skills with importance weights (1-5 scale)
# These come from careful JD reading, NOT from ChatGPT pattern matching
#
# IMPORTANT: Matching uses word-boundary-aware logic in skills_scorer.py
# Short keys like "ml", "nlp" use exact word matching (not substring)
CORE_SKILLS = {
    # === MUST HAVES (weight 4-5) ===
    # Embeddings & retrieval (the heart of the role)
    "sentence-transformers": 5.0, "sentence transformers": 5.0,
    "bge": 4.5, "e5 embedding": 4.5,
    "embedding": 4.0, "embeddings": 4.0,
    "dense retrieval": 4.5, "semantic search": 4.5,
    "hybrid search": 5.0, "hybrid retrieval": 5.0,
    "bm25": 4.0,

    # Vector databases
    "pinecone": 4.5, "weaviate": 4.5, "qdrant": 4.5,
    "milvus": 4.5, "faiss": 4.5, "elasticsearch": 4.0,
    "opensearch": 4.0, "vector database": 4.5, "vector db": 4.5,
    "vector store": 4.0,

    # Ranking & evaluation (explicit in JD)
    "ndcg": 5.0, "mrr": 4.5,
    "ranking": 4.5, "ranking system": 4.5,
    "learning to rank": 5.0, "ltr": 5.0, "lambdarank": 4.5,
    "a/b test": 4.0, "a/b testing": 4.0,
    "evaluation framework": 4.5,
    "information retrieval": 4.5,

    # Strong Python signals
    "python": 3.5,

    # NLP core
    "nlp": 4.0, "natural language processing": 4.0,
    "text classification": 3.0, "named entity recognition": 3.0,

    # === NICE TO HAVES (weight 2.5-3.5) ===
    "lora": 3.5, "qlora": 3.5, "peft": 3.5,
    "fine-tuning": 3.0, "fine tuning": 3.0, "finetuning": 3.0,
    "rag": 3.5, "retrieval augmented generation": 3.5,
    "xgboost": 2.5, "lightgbm": 2.5, "gradient boosting": 2.5,
    "recommendation system": 3.0, "recommender": 3.0,
    "search engine": 3.5, "search ranking": 4.0,
    "reranking": 4.0, "re-ranking": 4.0,
    "transformers": 3.0,
    "distributed systems": 2.5,
    "mlops": 2.5,

    # === WEAK SIGNALS (weight 1-2) ===
    "machine learning": 1.5,
    "deep learning": 1.5,
    "llm": 2.0, "large language model": 2.0,
    "pytorch": 2.0, "tensorflow": 1.5,
    "data science": 1.0,
}

# Short JD skill keys that need exact word-boundary matching (not substring).
# These are checked in skills_scorer.py to prevent false positives like
# "HTML" matching "ml", or "Google Maps" matching "map".
EXACT_MATCH_SKILLS = {
    "bge", "mrr", "ltr", "nlp", "rag", "llm", "ml",
    "peft", "lora", "qlora", "bm25", "ndcg", "faiss",
}

# Companies that = automatic full career penalty (JD explicitly mentions bad fit)
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl technologies", "tech mahindra",
    "mphasis", "hexaware", "mindtree"
}

# Titles that indicate role coherence with the JD
GOOD_TITLES = [
    "ml engineer", "machine learning engineer", "ai engineer",
    "applied scientist", "applied ml", "nlp engineer",
    "search engineer", "ranking engineer", "retrieval engineer",
    "data scientist",  # acceptable
    "software engineer",  # acceptable if ML work evident
]

# Bad title patterns — use word-boundary matching to avoid false positives.
# "hr" should match "HR Manager" but not "Scheduler" or "Chrome Developer".
BAD_TITLE_PATTERNS = [
    "human resources", "marketing", "content writer",
    "graphic designer", "business analyst", "customer support",
    "project manager", "scrum master",
]
# These short patterns are checked with word-boundary logic in skills_scorer.py
BAD_TITLE_EXACT_WORDS = {"hr", "sales"}

# Location preferences (India cities)
PREFERRED_CITIES = {
    "pune", "noida", "delhi", "new delhi", "ncr", "delhi ncr",
    "gurugram", "gurgaon", "hyderabad", "mumbai", "bangalore",
    "bengaluru", "navi mumbai"
}

# Production deployment vocabulary in career descriptions
# NOTE: Only counted when co-occurring with ML/AI context words
PRODUCTION_SIGNALS = [
    "deployed", "serving", "inference",
    "latency", "throughput", "scaled to",
    "million users", "billion", "real-time", "real time",
    "index refresh", "retrieval quality", "embedding drift",
    "live system"
]

# Context words that must appear near "production" to count it as ML production
# This prevents "production tooling" (manufacturing) from counting
ML_CONTEXT_SIGNALS = [
    "model", "ml", "machine learning", "ai", "nlp", "embedding",
    "retrieval", "search", "ranking", "recommendation", "inference",
    "pipeline", "feature", "prediction", "training", "neural",
    "vector", "transformer", "bert", "gpt", "llm", "deep learning"
]

# Research-only red flags
RESEARCH_ONLY_SIGNALS = [
    "research lab", "academic research", "phd thesis",
    "published paper", "arxiv", "conference paper",
    "nips", "neurips", "icml", "iclr"  # acceptable if also has production
]

# Max achievable score for normalization
# Using a lower multiplier to allow top candidates to reach meaningful scores
MAX_SKILLS_SCORE = sum(v for v in CORE_SKILLS.values() if v >= 4.0) * 0.5
