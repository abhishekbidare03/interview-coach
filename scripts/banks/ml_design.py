r"""ML System Design — the open-ended round.

Graded as OPEN_ENDED, so the expected points are *considerations a strong answer
raises*, not facts it states. There is no correct architecture for "design a
recommender"; there is a set of things you are negligent not to mention.

Fewer entries than the factual topics on purpose. Each one is a fifteen-minute
conversation rather than a thirty-second answer, so a bank of eighty would never
be reached.
"""

from __future__ import annotations

ENTRIES = [

    # ------------------------------------------------------------------ 2 --- #

    (2, "framing a business problem as an ML problem",
     "A product manager asks you to reduce customer churn. How do you start?",
     "The first move is turning a business goal into a prediction problem with a "
     "decision attached. What counts as churn and over what horizon, what action "
     "follows a prediction, and whether the model is even the constraint. A "
     "perfect churn model is worthless if there is no retention lever to pull.",
     ["defined churn precisely, with a time horizon",
      "identified the action a prediction triggers",
      "questioned whether prediction is the actual bottleneck",
      "chose a metric tied to business value, not model accuracy"],
     ["Suppose there is no retention lever. What then?"]),

    (2, "choosing offline and online metrics",
     "How do you choose what to measure for a new ML feature?",
     "Two layers. An offline metric to iterate on quickly, and an online metric "
     "that reflects real value, with the offline one justified as a proxy for "
     "it. Then guardrails for what must not regress. The proxy relationship is "
     "the part people skip, and it is the part that breaks.",
     ["separated offline iteration metrics from online outcome metrics",
      "justified the offline metric as a proxy for the online one",
      "chose a metric tied to user or business value",
      "included guardrail metrics that must not regress"],
     []),

    (2, "building the training dataset and labels",
     "Where does the training data come from for a new model?",
     "That is usually the hardest part of the design. Either labels already "
     "exist as a by-product of the product, or they have to be created — "
     "annotation, weak supervision, or a heuristic. Each has a cost and a bias, "
     "and logged behavioural labels carry the bias of whatever system produced "
     "them.",
     ["identified a concrete source of labels",
      "considered annotation cost and label quality",
      "recognised bias in implicitly logged labels",
      "considered weak supervision or heuristics to bootstrap"],
     []),

    # ------------------------------------------------------------------ 3 --- #

    (3, "design a product recommendation system",
     "Design a system that recommends products to customers on a retail site.",
     "The usual shape is two stages: cheap candidate generation to cut millions "
     "of items to a few hundred, then an expensive ranker over those. Beyond "
     "that it is about the signals available, the cold start problem for new "
     "users and items, and the fact that the logs you train on came from the "
     "previous recommender.",
     ["proposed a candidate generation and ranking split",
      "identified the signals and features available",
      "addressed cold start for new users or items",
      "recognised feedback loops from training on its own logs",
      "discussed latency budget at serving time"],
     ["How would you stop it recommending only what is already popular?"]),

    (3, "design a search ranking system",
     "Design search ranking for a large product catalogue.",
     "Retrieval then ranking, with lexical and semantic retrieval combined so "
     "exact identifiers and paraphrases both work. The interesting parts are "
     "what the training signal is, that clicks are biased by position, and that "
     "relevance is not the only objective the business has.",
     ["separated retrieval from ranking",
      "combined lexical and semantic retrieval",
      "derived a training signal from user interactions",
      "accounted for position and presentation bias in clicks",
      "balanced relevance against other business objectives"],
     []),

    (3, "design a fraud detection system",
     "Design a system to detect fraudulent transactions.",
     "Extreme class imbalance and an adversary who adapts, so the metric cannot "
     "be accuracy and the model cannot be static. Latency is tight because the "
     "decision blocks a payment, labels arrive late through chargebacks, and the "
     "cost of a false positive on a real customer is very different from a "
     "missed fraud.",
     ["addressed severe class imbalance",
      "chose metrics reflecting asymmetric error costs",
      "handled delayed labels from chargebacks",
      "accounted for an adapting adversary and drift",
      "respected a real-time latency budget"],
     ["How do you handle the label delay when retraining?"]),

    (3, "design a customer review summariser",
     "Design a system that summarises product reviews for shoppers.",
     "A generative problem where correctness is not verifiable by a single "
     "answer, so evaluation dominates the design. Faithfulness to the source "
     "reviews matters more than fluency, coverage should be representative "
     "rather than cherry-picked, and cost per product matters when there are "
     "millions of them.",
     ["prioritised faithfulness to the source reviews",
      "proposed how to evaluate without a single correct answer",
      "addressed representative coverage rather than cherry-picking",
      "considered cost, batching or caching at catalogue scale",
      "handled products with very few reviews"],
     []),

    (3, "design a demand forecasting system",
     "Design a system to forecast demand for inventory planning.",
     "A time series problem where the evaluation must respect time — no random "
     "splits, and backtesting on rolling origins. Seasonality and promotions "
     "dominate the signal, the error cost is asymmetric between overstock and "
     "stockout, and a forecast without an uncertainty range is not actionable.",
     ["used time-based splits and backtesting, not random splits",
      "handled seasonality, trend and promotional spikes",
      "recognised asymmetric costs of over and under forecasting",
      "produced uncertainty intervals, not point estimates",
      "addressed cold start for new products"],
     []),

    (3, "model monitoring and drift detection",
     "The model is deployed. What do you monitor, and what alerts you?",
     "Three layers: that the system is healthy, that the inputs still look like "
     "training data, and that the outputs are still good. The third is the "
     "hardest because labels lag, so prediction distribution and business "
     "metrics act as the early warning while true performance catches up.",
     ["monitored operational health such as latency and errors",
      "monitored input feature distributions for drift",
      "monitored the prediction distribution",
      "tracked business metrics as an early signal for delayed labels",
      "defined what triggers retraining or rollback"],
     []),

    # ------------------------------------------------------------------ 4 --- #

    (4, "design a document question-answering assistant",
     "Design an assistant that answers questions over internal company documents.",
     "Retrieval quality decides everything, so I would measure it separately "
     "from answer quality. Then access control, because the retriever must never "
     "surface a document the asker cannot see; freshness as documents change; "
     "and an explicit path for the model to say it does not know rather than "
     "inventing an answer.",
     ["measured retrieval quality separately from answer quality",
      "enforced per-user access control in retrieval",
      "handled document updates and index freshness",
      "grounded answers with citations to the source",
      "gave the system a way to decline when unsure"],
     ["How would you evaluate this before launch?"]),

    (4, "feature stores and train-serve consistency",
     "How do you make sure training and serving compute the same features?",
     "By not computing them twice. One definition, used by both paths, is the "
     "structural fix — that is what a feature store is for. Beyond that, the "
     "training data must reflect what was actually knowable at prediction time, "
     "which means point-in-time correct joins rather than the latest value.",
     ["defined features once and shared the definition",
      "identified train-serve skew as the failure mode",
      "used point-in-time correct joins for training data",
      "avoided using values unavailable at prediction time",
      "proposed monitoring to detect divergence"],
     []),

    (4, "batch versus real-time inference",
     "How would you decide between batch and real-time inference?",
     "From how fresh the prediction has to be and what the request pattern is. "
     "If the inputs change slowly and the population is enumerable, batch is far "
     "cheaper and simpler. Real time is for when the input arrives with the "
     "request. Precomputing plus a light real-time layer often gets both.",
     ["decided based on required freshness of the prediction",
      "considered cost and operational complexity",
      "batch suits slowly-changing, enumerable populations",
      "real-time is needed when input arrives with the request",
      "considered a hybrid with precomputed components"],
     []),

    (4, "retraining cadence and rollback",
     "How often should a model be retrained, and how do you deploy safely?",
     "Cadence should follow measured drift rather than a calendar. On "
     "deployment, the new model earns its way in — shadow traffic to compare "
     "without risk, then a small percentage, then a full experiment — with an "
     "automatic rollback trigger and the old version kept ready.",
     ["tied retraining frequency to observed drift, not a fixed schedule",
      "validated a new model against the current one before launch",
      "used shadow or canary deployment",
      "defined automatic rollback criteria",
      "kept model versions and their training data reproducible"],
     []),

    (4, "design a content moderation system",
     "Design a system to moderate user-generated content at scale.",
     "Volume forces automation but the error costs are asymmetric and political, "
     "so it should be a triage system rather than a binary classifier — "
     "confident cases automated, uncertain ones queued for humans. Policy "
     "changes over time, adversaries adapt, and appeals are part of the system, "
     "not an afterthought.",
     ["designed a human-in-the-loop triage rather than full automation",
      "used confidence thresholds to route uncertain cases",
      "recognised asymmetric and context-dependent error costs",
      "accounted for adversarial adaptation and policy change",
      "included an appeals or correction path that feeds back labels"],
     []),

    # ------------------------------------------------------------------ 5 --- #

    (5, "cold start for new users or items",
     "How would you handle cold start for both new users and new items?",
     "They are different problems. A new item has content but no interactions, "
     "so content features and a deliberate exploration budget get it seen. A new "
     "user has no history, so you fall back on context and population priors and "
     "adapt quickly within the session. Both are exploration problems, and "
     "exploration costs money you have to be willing to spend.",
     ["separated new-item from new-user cold start",
      "used content or metadata features for new items",
      "used context and population priors for new users",
      "allocated explicit exploration to gather interaction data",
      "acknowledged the short-term cost of exploring"],
     []),

    (5, "cost and latency budgets at scale",
     "Your model is accurate but too slow and too expensive to launch. What now?",
     "First establish what the budget actually is and what the accuracy is "
     "worth, because the answer may be that a cheaper model is the right "
     "product. Then the standard levers: cascade so the expensive model only "
     "sees hard cases, precompute what is predictable, cache repeats, and "
     "distil or quantise. Measure the accuracy cost of each.",
     ["quantified the latency and cost budget explicitly",
      "weighed the business value of the extra accuracy",
      "proposed a cascade so few requests reach the expensive model",
      "considered precomputation, caching or distillation",
      "measured the accuracy lost to each optimisation"],
     []),

    (5, "handling delayed or missing labels",
     "How do you evaluate a model whose true labels arrive months later?",
     "You cannot wait, so you need a proxy you have validated against the "
     "eventual outcome on historical data. Meanwhile the mature cohorts give you "
     "real performance on a lag, and monitoring watches inputs and predictions "
     "for change. The dangerous move is treating the proxy as the objective "
     "without ever checking it still tracks.",
     ["identified an early proxy signal for the delayed outcome",
      "validated that proxy against historical mature labels",
      "evaluated fully-labelled older cohorts on a lag",
      "monitored input and prediction drift in the meantime",
      "planned to re-check that the proxy still tracks the outcome"],
     []),
]
