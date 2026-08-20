"""Topic taxonomy.

Each topic carries three things the rest of the system needs:

* **subtopics** — the diversity lever. Asking a 3B model for "40 operating
  systems questions" yields the same eight questions with different wording.
  Asking it for "one question about priority inversion, difficulty 4" does not.
  This is the concrete mechanism behind plan.md §2.2.
* **vocabulary** — technical terms fed to Whisper as `initial_prompt`
  (plan.md §2.4.1). Without it `small.en` normalises jargon into common English
  and the evaluator grades a misheard answer, which is the worst failure the
  system has.
* **mode** — how answers here must be judged (plan.md §2.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Mode


@dataclass(frozen=True)
class Topic:
    key: str
    label: str
    mode: Mode
    subtopics: tuple[str, ...]
    vocabulary: tuple[str, ...] = ()
    blurb: str = ""
    needs_resume: bool = False

    @property
    def vocab_prompt(self) -> str:
        return ", ".join(self.vocabulary)


TOPICS: dict[str, Topic] = {}


def _add(t: Topic) -> Topic:
    TOPICS[t.key] = t
    return t


# --------------------------------------------------------------------------- #
# Technical — factual
# --------------------------------------------------------------------------- #

_add(Topic(
    key="dsa", label="Data Structures & Algorithms", mode=Mode.FACTUAL,
    blurb="Complexity, core structures, and the reasoning behind algorithm choice.",
    subtopics=(
        "time and space complexity analysis", "arrays versus linked lists",
        "hash tables and collision handling", "binary search and its variants",
        "sorting algorithm tradeoffs", "binary trees and traversals",
        "balanced trees and when they matter", "heaps and priority queues",
        "graph representations", "breadth-first versus depth-first search",
        "shortest path algorithms", "dynamic programming versus greedy",
        "recursion and stack depth", "two pointers and sliding window",
        "tries and prefix matching", "union-find",
        "amortised analysis", "in-place algorithms and space tradeoffs",
    ),
    vocabulary=(
        "big O notation", "amortised", "hash table", "collision", "chaining",
        "open addressing", "binary search tree", "AVL", "red-black tree",
        "heap", "trie", "adjacency list", "Dijkstra", "memoization",
        "dynamic programming", "quicksort", "mergesort", "union-find",
    ),
))

_add(Topic(
    key="os", label="Operating Systems", mode=Mode.FACTUAL,
    blurb="Processes, concurrency, memory, and scheduling.",
    subtopics=(
        "processes versus threads", "context switching costs",
        "mutexes versus semaphores", "deadlock conditions and prevention",
        "race conditions and atomicity", "virtual memory and paging",
        "page replacement policies", "thrashing", "CPU scheduling algorithms",
        "priority inversion", "user space versus kernel space", "system calls",
        "interrupts", "memory fragmentation", "copy-on-write",
        "inter-process communication", "file descriptors", "zombie and orphan processes",
    ),
    vocabulary=(
        "mutex", "semaphore", "deadlock", "livelock", "race condition",
        "context switch", "virtual memory", "paging", "page fault", "TLB",
        "thrashing", "preemptive", "round robin", "priority inversion",
        "kernel space", "system call", "copy-on-write", "IPC", "spinlock",
    ),
))

_add(Topic(
    key="dbms", label="Databases", mode=Mode.FACTUAL,
    blurb="Relational modelling, transactions, indexing, and query performance.",
    subtopics=(
        "normalisation and its tradeoffs", "primary and foreign keys",
        "ACID properties", "isolation levels and anomalies",
        "how B-tree indexes work", "when an index hurts", "composite index ordering",
        "query execution plans", "joins and join algorithms",
        "transactions and rollback", "optimistic versus pessimistic locking",
        "SQL versus NoSQL tradeoffs", "sharding and partitioning",
        "replication and read replicas", "the N+1 query problem",
        "denormalisation for read performance", "connection pooling",
    ),
    vocabulary=(
        "normalisation", "denormalisation", "ACID", "atomicity", "isolation level",
        "read committed", "repeatable read", "serializable", "phantom read",
        "B-tree", "composite index", "query plan", "hash join", "nested loop join",
        "sharding", "replication", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    ),
))

_add(Topic(
    key="networking", label="Networking", mode=Mode.FACTUAL,
    blurb="Protocols, the request lifecycle, and what breaks between machines.",
    subtopics=(
        "TCP versus UDP", "the TCP handshake", "what happens when you type a URL",
        "DNS resolution", "HTTP methods and idempotency", "HTTP status codes",
        "HTTPS and TLS", "cookies versus tokens", "CORS",
        "REST versus RPC", "WebSockets versus polling", "load balancing",
        "caching layers and cache headers", "CDNs", "latency versus bandwidth",
        "network partitions", "rate limiting",
    ),
    vocabulary=(
        "TCP", "UDP", "handshake", "DNS", "HTTP", "HTTPS", "TLS", "idempotent",
        "CORS", "REST", "WebSocket", "load balancer", "CDN", "latency",
        "bandwidth", "packet loss", "rate limiting", "reverse proxy", "nginx",
    ),
))

_add(Topic(
    key="python", label="Python", mode=Mode.FACTUAL,
    blurb="Language semantics, the data model, and idiomatic use.",
    # Ordered roughly by depth, because the Python bank is hand-written
    # (scripts/build_python_bank.py) and deliberately bottom-heavy: a screening
    # round spends most of its time checking that the basics are solid.
    subtopics=(
        # basics
        "list versus tuple", "dictionaries", "sets", "identity versus equality",
        "mutability of strings", "dynamic typing", "slicing", "None",
        "list comprehensions", "loops", "modules and imports", "docstrings",
        # everyday
        "shallow versus deep copy", "args and kwargs", "decorators",
        "generators", "method types", "default mutable arguments",
        "context managers", "iterators versus iterables", "scope",
        "sort versus sorted", "lambda", "f-strings", "truthiness",
        # depth
        "the GIL", "threading versus multiprocessing", "yield",
        "dunder methods", "closures", "duck typing", "method resolution order",
        "composition versus inheritance", "garbage collection",
        "attribute lookup", "equality and hashing",
        # stretch
        "asyncio", "slots", "reference cycles", "metaclasses", "descriptors",
        "profiling", "parallelism in practice", "import system",
    ),
    vocabulary=(
        "GIL", "global interpreter lock", "generator", "yield", "decorator",
        "context manager", "comprehension", "kwargs", "dunder", "iterable",
        "iterator", "closure", "asyncio", "coroutine", "multiprocessing",
        "reference counting", "garbage collection", "duck typing", "mutable",
        # Terms the hand-written bank actually asks about. Whisper normalises
        # these into ordinary English without the bias prompt — "dunder" alone
        # comes back as "under" often enough to break grading.
        "immutable", "tuple", "hashable", "slicing", "lambda", "f-string",
        "enumerate", "metaclass", "descriptor", "slots", "monkey patching",
        "method resolution order", "shallow copy", "deep copy", "pickling",
    ),
))

# --------------------------------------------------------------------------- #
# Applied Science
#
# The Amazon Applied Scientist loop is what these exist for, and it is not one
# subject. It is ML breadth, ML depth, statistics and experimentation, an ML
# system design round, and Leadership Principles — so each of those gets its own
# topic rather than being folded into a single "machine learning" bucket that
# would ask a bandit question and a backprop question with the same rubric.
#
# Statistics has its own entry deliberately. It is the round most candidates
# skip preparing for, and at Amazon "how would you measure whether this model
# actually helped" is asked as often as anything about architectures.
# --------------------------------------------------------------------------- #

_add(Topic(
    key="ml", label="Machine Learning", mode=Mode.FACTUAL,
    blurb="Classical ML: the bias-variance story, and why models fail in production.",
    subtopics=(
        # foundations
        "supervised versus unsupervised", "training, validation and test splits",
        "overfitting and underfitting", "the bias-variance tradeoff",
        "cross-validation", "feature scaling", "categorical encoding",
        "handling missing data", "the curse of dimensionality",
        # models
        "linear regression assumptions", "logistic regression",
        "regularisation, L1 versus L2", "decision trees",
        "random forests versus boosting", "gradient boosting",
        "support vector machines", "k-nearest neighbours", "naive Bayes",
        "k-means clustering", "principal component analysis",
        # evaluation
        "precision versus recall", "the ROC curve and AUC",
        "choosing a metric for imbalanced data", "confusion matrices",
        "calibration of predicted probabilities",
        # practice
        "class imbalance", "data leakage", "feature importance",
        "train-serve skew", "concept drift", "hyperparameter search",
        "ensembling", "interpretability versus accuracy",
    ),
    vocabulary=(
        "overfitting", "underfitting", "bias-variance", "regularisation",
        "L1", "L2", "lasso", "ridge", "cross-validation", "k-fold",
        "gradient boosting", "XGBoost", "random forest", "bagging", "boosting",
        "SVM", "kernel", "k-means", "PCA", "eigenvector", "precision", "recall",
        "F1", "ROC", "AUC", "ROC AUC", "PR curve", "confusion matrix",
        "class imbalance", "SMOTE", "data leakage", "concept drift",
        "hyperparameter", "feature engineering", "one-hot encoding",
        "multicollinearity", "calibration", "Platt scaling", "SHAP",
    ),
))

_add(Topic(
    key="deep_learning", label="Deep Learning", mode=Mode.FACTUAL,
    blurb="Networks, how they train, and why training goes wrong.",
    subtopics=(
        "what a neuron computes", "activation functions",
        "why non-linearity is needed", "backpropagation",
        "gradient descent variants", "learning rate and schedules",
        "the vanishing gradient problem", "exploding gradients and clipping",
        "batch normalisation", "layer normalisation", "dropout",
        "weight initialisation", "loss functions for classification",
        "cross-entropy versus mean squared error", "the softmax function",
        "convolutional layers", "pooling", "receptive field",
        "recurrent networks", "LSTM and GRU gating",
        "residual connections", "attention", "transformers",
        "transfer learning", "fine-tuning versus feature extraction",
        "data augmentation", "batch size effects", "mixed precision training",
        "distributed training", "catastrophic forgetting",
        "overfitting in deep networks", "early stopping",
    ),
    vocabulary=(
        "backpropagation", "gradient descent", "SGD", "Adam", "AdamW",
        "momentum", "learning rate", "vanishing gradient", "exploding gradient",
        "gradient clipping", "batch normalisation", "layer normalisation",
        "dropout", "ReLU", "GELU", "sigmoid", "tanh", "softmax",
        "cross-entropy", "logits", "epoch", "mini-batch", "convolution",
        "pooling", "receptive field", "LSTM", "GRU", "residual connection",
        "skip connection", "attention", "self-attention", "transformer",
        "encoder", "decoder", "embedding", "fine-tuning", "transfer learning",
        "Xavier initialisation", "He initialisation", "mixed precision",
    ),
))

_add(Topic(
    key="nlp", label="Natural Language Processing", mode=Mode.FACTUAL,
    blurb="From tokenisation to transformers, and what breaks in between.",
    subtopics=(
        "tokenisation", "subword tokenisation and byte pair encoding",
        "stemming versus lemmatisation", "stop words",
        "bag of words and TF-IDF", "n-gram language models",
        "word embeddings", "word2vec versus GloVe",
        "static versus contextual embeddings", "out-of-vocabulary words",
        "sequence labelling and named entity recognition",
        "the encoder-decoder architecture", "attention in sequence models",
        "positional encoding", "multi-head attention",
        "BERT and masked language modelling", "autoregressive language models",
        "the pretrain then fine-tune recipe", "sentence embeddings",
        "semantic search and vector similarity", "text classification",
        "evaluating generation with BLEU and ROUGE", "perplexity",
        "handling long documents", "multilingual models",
        "bias in language models", "text preprocessing pitfalls",
    ),
    vocabulary=(
        "tokenisation", "tokeniser", "byte pair encoding", "BPE", "WordPiece",
        "SentencePiece", "subword", "stemming", "lemmatisation", "stop words",
        "TF-IDF", "bag of words", "n-gram", "embedding", "word2vec", "GloVe",
        "skip-gram", "CBOW", "contextual embedding", "BERT", "GPT",
        "masked language modelling", "autoregressive", "encoder-decoder",
        "seq2seq", "attention", "self-attention", "multi-head attention",
        "positional encoding", "perplexity", "BLEU", "ROUGE", "cosine similarity",
        "named entity recognition", "NER", "part of speech", "corpus",
    ),
))

_add(Topic(
    key="genai", label="Generative AI & LLMs", mode=Mode.FACTUAL,
    blurb="How large models are trained, adapted, served, and evaluated.",
    subtopics=(
        "how an LLM generates text", "temperature and top-p sampling",
        "greedy versus beam search decoding", "the context window",
        "prompt engineering", "few-shot prompting",
        "chain-of-thought prompting", "hallucination and its causes",
        "retrieval augmented generation", "chunking strategy for retrieval",
        "vector databases and approximate nearest neighbours",
        "embedding models for retrieval", "reranking",
        "fine-tuning versus prompting versus retrieval",
        "parameter-efficient fine-tuning", "LoRA and adapters",
        "instruction tuning", "reinforcement learning from human feedback",
        "reward models", "quantisation", "KV caching",
        "inference latency and throughput", "batching for serving",
        "evaluating generative output", "LLM-as-a-judge",
        "guardrails and prompt injection", "agents and tool use",
        "cost control in production", "diffusion models",
        "distillation into smaller models",
    ),
    vocabulary=(
        "LLM", "large language model", "token", "context window", "prompt",
        "few-shot", "zero-shot", "chain of thought", "hallucination",
        "temperature", "top-p", "nucleus sampling", "greedy decoding",
        "beam search", "RAG", "retrieval augmented generation", "chunking",
        "vector database", "approximate nearest neighbour", "reranker",
        "fine-tuning", "LoRA", "PEFT", "adapter", "instruction tuning",
        "RLHF", "reward model", "DPO", "quantisation", "KV cache",
        "distillation", "diffusion", "prompt injection", "guardrail",
        "embedding", "cosine similarity", "throughput", "time to first token",
    ),
))

_add(Topic(
    key="rag", label="Retrieval & RAG", mode=Mode.FACTUAL,
    blurb="Getting the right context in front of the model, and knowing when you did.",
    # Split out of `genai` rather than duplicated. GenAI owns the model — how it
    # decodes, how it is adapted, how it is served. This owns everything on the
    # retrieval side, from chunking to whether the answer was actually grounded
    # in what came back. The introductory questions moved here too, so one topic
    # tells the whole story instead of both telling half of it badly.
    subtopics=(
        "what retrieval augmented generation is", "when retrieval beats fine-tuning",
        "chunking strategy", "chunk size tradeoffs", "chunk overlap",
        "embedding models for retrieval", "vector similarity",
        "approximate nearest neighbour search", "vector index tradeoffs",
        "lexical versus semantic retrieval", "hybrid search",
        "reranking", "cross-encoders versus bi-encoders",
        "query rewriting", "handling multi-hop questions",
        "metadata filtering", "access control in retrieval",
        "context ordering and lost in the middle", "how much context to pass",
        "grounding and citations", "detecting ungrounded answers",
        "evaluating retrieval separately", "recall at k and precision at k",
        "building a retrieval evaluation set", "index freshness and updates",
        "deduplication in the corpus", "handling tables and structured documents",
        "caching in a retrieval pipeline", "retrieval latency budget",
        "common RAG failure modes",
    ),
    vocabulary=(
        "RAG", "retrieval augmented generation", "retriever", "chunking",
        "chunk", "overlap", "embedding", "vector database", "vector index",
        "cosine similarity", "dot product", "approximate nearest neighbour",
        "ANN", "HNSW", "IVF", "recall at k", "precision at k", "BM25",
        "lexical search", "semantic search", "hybrid search", "reciprocal rank fusion",
        "reranker", "cross-encoder", "bi-encoder", "query rewriting",
        "multi-hop", "grounding", "citation", "hallucination", "context window",
        "lost in the middle", "metadata filter", "top k", "corpus",
    ),
))

_add(Topic(
    key="model_eval", label="Model Evaluation", mode=Mode.FACTUAL,
    blurb="How you find out whether a model is actually good, and whether you can trust the number.",
    # Deliberately about *method*, not about metric definitions. "What is
    # recall" belongs in Machine Learning; "your holdout says 94% and you do not
    # believe it, what do you check" belongs here. That distinction is what
    # keeps this from being a second copy of the ML bank.
    subtopics=(
        "choosing a baseline", "what a metric is a proxy for",
        "train validation test discipline", "cross-validation strategy",
        "splitting time series data", "grouped and stratified splits",
        "detecting leakage in an evaluation", "the test set as a scarce resource",
        "overfitting to a validation set", "error analysis",
        "slicing performance by segment", "worst-group performance",
        "comparing two models honestly", "significance of a model comparison",
        "confidence intervals on a metric", "evaluating imbalanced problems",
        "threshold selection", "calibration checks",
        "evaluating ranking systems", "evaluating recommenders",
        "evaluating generative output", "human evaluation",
        "inter-annotator agreement", "label noise in the test set",
        "benchmark contamination", "offline versus online evaluation",
        "regression test sets", "evaluating in production",
        "the cost of evaluation", "reproducibility of results",
    ),
    vocabulary=(
        "baseline", "holdout", "validation set", "test set", "cross-validation",
        "k-fold", "stratified", "grouped split", "leakage", "error analysis",
        "slice", "segment", "worst-group", "confidence interval",
        "statistical significance", "bootstrap", "paired test", "threshold",
        "calibration", "reliability diagram", "NDCG", "MRR", "recall at k",
        "hit rate", "offline metric", "online metric", "proxy metric",
        "human evaluation", "inter-annotator agreement", "Cohen's kappa",
        "benchmark contamination", "regression test", "golden set",
        "shadow evaluation", "backtesting",
    ),
))

_add(Topic(
    key="stats", label="Statistics & Experimentation", mode=Mode.FACTUAL,
    blurb="Probability, inference, and proving a model actually helped.",
    subtopics=(
        "probability versus likelihood", "conditional probability",
        "Bayes theorem", "expectation and variance",
        "the normal distribution", "the central limit theorem",
        "the law of large numbers", "sampling and sampling bias",
        "confidence intervals", "hypothesis testing",
        "p-values and what they do not mean", "type one and type two errors",
        "statistical power", "multiple comparisons",
        "A/B test design", "choosing a success metric",
        "guardrail metrics", "sample size and minimum detectable effect",
        "novelty and primacy effects", "network effects in experiments",
        "sequential testing and peeking", "correlation versus causation",
        "confounding variables", "observational causal inference",
        "difference in differences", "maximum likelihood estimation",
        "the bootstrap", "regression to the mean", "Simpson's paradox",
        "long tail and heavy-tailed distributions",
    ),
    vocabulary=(
        "probability", "likelihood", "Bayes theorem", "prior", "posterior",
        "expectation", "variance", "standard deviation", "standard error",
        "normal distribution", "Gaussian", "central limit theorem",
        "confidence interval", "hypothesis test", "null hypothesis",
        "p-value", "type one error", "type two error", "statistical power",
        "A/B test", "control group", "treatment group", "randomisation",
        "minimum detectable effect", "guardrail metric", "sample size",
        "Bonferroni", "multiple comparisons", "bootstrap", "maximum likelihood",
        "confounder", "causal inference", "difference in differences",
        "Simpson's paradox", "regression to the mean", "heteroscedasticity",
    ),
))

# --------------------------------------------------------------------------- #
# Technical — open-ended
# --------------------------------------------------------------------------- #

_add(Topic(
    key="ml_design", label="ML System Design", mode=Mode.OPEN_ENDED,
    blurb="End-to-end ML systems: data, training, serving, and what happens after launch.",
    subtopics=(
        "design a product recommendation system",
        "design a search ranking system",
        "design a fraud detection system",
        "design a demand forecasting system",
        "design a customer review summariser",
        "design a support ticket routing system",
        "design a product-matching deduplication system",
        "design an ads click-through prediction system",
        "design a document question-answering assistant",
        "design a content moderation system",
        "framing a business problem as an ML problem",
        "choosing offline and online metrics",
        "building the training dataset and labels",
        "handling delayed or missing labels",
        "feature stores and train-serve consistency",
        "batch versus real-time inference",
        "model monitoring and drift detection",
        "retraining cadence and rollback",
        "cold start for new users or items",
        "cost and latency budgets at scale",
    ),
    vocabulary=(
        "training pipeline", "inference pipeline", "feature store",
        "train-serve skew", "offline metric", "online metric", "proxy metric",
        "labelling", "ground truth", "cold start", "candidate generation",
        "retrieval and ranking", "two-tower model", "embedding index",
        "model registry", "shadow deployment", "canary", "A/B test",
        "drift detection", "retraining", "batch inference", "real-time inference",
        "latency budget", "throughput", "click-through rate", "human in the loop",
    ),
))

_add(Topic(
    key="system_design", label="System Design", mode=Mode.OPEN_ENDED,
    blurb="Architecture, tradeoffs, and how systems fail at scale.",
    subtopics=(
        "design a URL shortener", "design a rate limiter",
        "design a news feed", "design a chat application",
        "design a file storage service", "design a ride-hailing dispatch system",
        "design a notification service", "design a search autocomplete",
        "caching strategy and invalidation", "database choice for a given workload",
        "handling traffic spikes", "designing for failure and retries",
        "consistency versus availability tradeoffs", "API design and versioning",
        "queueing and asynchronous processing", "observability and alerting",
    ),
    vocabulary=(
        "horizontal scaling", "vertical scaling", "load balancer", "sharding",
        "consistent hashing", "cache invalidation", "eventual consistency",
        "CAP theorem", "message queue", "Kafka", "idempotency", "throughput",
        "p99 latency", "circuit breaker", "backpressure", "microservices",
    ),
))

# --------------------------------------------------------------------------- #
# Behavioural
# --------------------------------------------------------------------------- #

_add(Topic(
    key="behavioural", label="Behavioural", mode=Mode.BEHAVIOURAL,
    blurb="Past experience, judged on structure and specificity — not correctness.",
    subtopics=(
        "a conflict with a teammate", "a project that failed",
        "a time you received hard feedback", "a deadline you missed",
        "convincing someone who disagreed with you",
        "a time you had to learn something quickly",
        "a mistake you made in production", "working with an unclear requirement",
        "prioritising when everything was urgent", "a time you disagreed with a decision",
        "helping someone who was struggling", "taking ownership of something unglamorous",
        "a technical decision you later regretted", "handling an interruption mid-project",
        "your proudest piece of work", "a time you asked for help",
    ),
    vocabulary=(
        "stakeholder", "deadline", "sprint", "retrospective", "code review",
        "escalate", "scope", "requirement", "on-call", "postmortem",
    ),
))

_add(Topic(
    key="resume", label="Résumé & Projects", mode=Mode.BEHAVIOURAL,
    blurb="Questions about your actual projects. Requires your CV.",
    needs_resume=True,
    subtopics=(
        "why you chose that technology", "the hardest bug in that project",
        "what you would rebuild differently", "how you tested it",
        "what you personally owned versus the team", "how it performed under load",
        "what you learned from it", "what you would add next",
    ),
    vocabulary=(),
))


# --------------------------------------------------------------------------- #

APPLIED_SCIENCE = ("ml", "deep_learning", "nlp", "genai", "rag", "model_eval",
                   "stats", "ml_design")
TECHNICAL = ("dsa", "os", "dbms", "networking", "python", "system_design")
BEHAVIOURAL = ("behavioural",)

# Weights for the "Random Interview" button (plan.md R3).
#
# Shaped like an Amazon Applied Scientist loop rather than a generic screen:
# ML breadth and depth carry the most weight, statistics gets a real share
# because it is the round candidates most often under-prepare, and behavioural
# stays high because Leadership Principles are weighted heavily there and are
# not a formality. Coding is present but is not the centre of gravity.
RANDOM_MIX: dict[str, float] = {
    "ml": 0.15, "deep_learning": 0.11, "stats": 0.10,
    "model_eval": 0.09, "nlp": 0.07, "genai": 0.07, "rag": 0.06,
    "ml_design": 0.09, "python": 0.06, "dsa": 0.05,
    "behavioural": 0.15,
}


def get(key: str) -> Topic:
    if key not in TOPICS:
        raise KeyError(f"unknown topic {key!r}; known: {sorted(TOPICS)}")
    return TOPICS[key]


def vocabulary_for(keys: list[str]) -> str:
    """Combined STT bias prompt for a set of topics, deduped and length-capped.

    Whisper's `initial_prompt` competes with the audio for the model's attention,
    so an unbounded term dump degrades transcription instead of helping it.
    """
    seen: list[str] = []
    for k in keys:
        for term in TOPICS[k].vocabulary:
            if term not in seen:
                seen.append(term)
    return ", ".join(seen[:60])


def selectable() -> list[dict]:
    """Topic list for the UI picker."""
    return [
        {"key": t.key, "label": t.label, "mode": str(t.mode),
         "blurb": t.blurb, "needs_resume": t.needs_resume,
         "subtopics": len(t.subtopics)}
        for t in TOPICS.values()
    ]
