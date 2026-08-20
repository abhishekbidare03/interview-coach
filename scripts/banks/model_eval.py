r"""Model Evaluation.

Deliberately about *method*, not about metric definitions. "What is recall"
belongs in the Machine Learning bank; "your holdout says ninety-four percent and
you do not believe it — what do you check" belongs here. Keeping that line is
what stops this becoming a second copy of the ML bank.

The through-line is that a number is a claim, and most of this discipline is
about whether the claim survives contact with how it was produced: what it was
compared against, how the split was made, how many times the test set has been
looked at, and whether the average is hiding a segment where the model is
useless.
"""

from __future__ import annotations

ENTRIES = [

    # ------------------------------------------------------------------ 1 --- #

    (1, "choosing a baseline",
     "Why does an evaluation need a baseline?",
     "Because a number on its own means nothing. Ninety percent accuracy is "
     "excellent or embarrassing depending on what predicting the majority class "
     "would have got you. The baseline is what tells you whether the model added "
     "anything at all.",
     ["a metric alone has no interpretation",
      "the baseline shows what is achievable trivially",
      "it reveals whether the model adds value",
      "examples are majority class or a simple heuristic"],
     ["What baseline would you use for a regression problem?"]),

    (1, "train validation test discipline",
     "What is each of the training, validation and test sets for?",
     "Training fits the parameters. Validation guides your choices — "
     "hyperparameters, which model, when to stop. Test estimates how it will "
     "perform on data nobody has used for anything, which only holds if you have "
     "genuinely left it alone.",
     ["training data fits the model parameters",
      "validation guides tuning and model choice",
      "test gives an unbiased final estimate",
      "the test set must not influence any decision"],
     []),

    (1, "what a metric is a proxy for",
     "What does it mean to call a metric a proxy?",
     "That you are measuring something convenient in place of what you actually "
     "care about. Click-through rate stands in for usefulness, accuracy stands in "
     "for value delivered. The gap between the proxy and the real goal is where "
     "models that look good go wrong.",
     ["the metric substitutes for the real objective",
      "it is chosen because it is measurable",
      "the proxy and the goal can diverge",
      "optimising the proxy can hurt the real outcome"],
     []),

    (1, "error analysis",
     "What is error analysis?",
     "Actually reading the cases the model got wrong, rather than only looking at "
     "the aggregate. You sort them into groups and count, and that usually tells "
     "you what to fix far faster than any metric does — often the errors are one "
     "or two systematic problems, not a spread.",
     ["examine individual wrong predictions",
      "group the errors into categories and count them",
      "it reveals systematic failures an aggregate hides",
      "it directs what to fix next"],
     []),

    (1, "offline versus online evaluation",
     "What is the difference between offline and online evaluation?",
     "Offline scores a model against historical data, which is fast and cheap and "
     "lets you iterate. Online measures the effect on real users and real "
     "behaviour, which is slow and expensive and is the only one that tells you "
     "whether it actually helped.",
     ["offline uses historical data and is fast to run",
      "online measures real user behaviour",
      "offline supports rapid iteration",
      "only online measures real impact"],
     []),

    (1, "slicing performance by segment",
     "Why look at performance on subgroups instead of just overall?",
     "Because an average can be good while the model is useless for a whole "
     "group. Ninety-five percent overall might be ninety-nine on the common case "
     "and sixty on new users. If those users matter, the aggregate has hidden "
     "the only thing worth knowing.",
     ["a good average can hide a failing subgroup",
      "aggregate performance is weighted by group size",
      "small but important groups get averaged away",
      "measure each meaningful segment separately"],
     []),

    (1, "the test set as a scarce resource",
     "Why should you look at the test set as little as possible?",
     "Because every look leaks it into your decisions. If you evaluate on test, "
     "change something, and evaluate again, you have started tuning on it — and "
     "the number stops being an estimate of unseen performance and becomes "
     "another training score.",
     ["repeated use leaks it into your decisions",
      "it stops being an unbiased estimate",
      "tuning against it is a form of overfitting",
      "use validation data for iteration instead"],
     []),

    # ------------------------------------------------------------------ 2 --- #

    (2, "splitting time series data",
     "How should you split data when the task involves time?",
     "By time, always — train on the past and test on the future. A random split "
     "lets the model learn from data recorded after what it is predicting, which "
     "is leakage and produces a score you can never reproduce in production.",
     ["split chronologically, not randomly",
      "train on earlier data and test on later",
      "a random split leaks future information",
      "the offline score would not hold in production"],
     ["What is backtesting?"]),

    (2, "grouped and stratified splits",
     "When is a plain random split the wrong thing to do?",
     "When rows are not independent. If one patient, user or document produces "
     "several rows, a random split puts some in train and some in test, and the "
     "model scores well by recognising the entity rather than the pattern. You "
     "split by group instead.",
     ["rows from the same entity are not independent",
      "a random split puts the same entity on both sides",
      "the model recognises the entity, not the pattern",
      "split by group so entities do not cross the boundary"],
     []),

    (2, "detecting leakage in an evaluation",
     "Your model scores 99 percent on the holdout. What do you check first?",
     "Leakage, before celebrating. Almost always a feature encodes the answer — "
     "something recorded after the outcome, an identifier correlated with the "
     "label, or preprocessing fitted before the split. Ninety-nine percent on a "
     "hard problem is a bug report, not a result.",
     ["suspect leakage rather than genuine performance",
      "look for features computed after the outcome",
      "check preprocessing was fitted after the split",
      "check for identifiers correlated with the label"],
     []),

    (2, "overfitting to a validation set",
     "Can you overfit to a validation set without ever training on it?",
     "Yes, and it is the common way. Every time you pick the better of two "
     "options by validation score you are fitting your choices to that data. "
     "After enough decisions the validation score is optimistic, which is why a "
     "final untouched test set exists.",
     ["repeated model selection fits choices to the validation data",
      "no gradient update is needed for this to happen",
      "the validation score becomes optimistic",
      "a separate untouched test set protects against it"],
     []),

    (2, "cross-validation strategy",
     "When would you use cross-validation instead of a single holdout?",
     "When the dataset is small enough that a single split is mostly luck. "
     "Rotating folds uses every row for both training and evaluation and gives "
     "you a spread as well as a mean. The cost is training as many times as you "
     "have folds.",
     ["use it when a single split is too noisy",
      "small datasets benefit most",
      "it gives a variance estimate as well as a mean",
      "the cost is training once per fold"],
     []),

    (2, "comparing two models honestly",
     "Model A scores 0.81 and model B scores 0.82. Is B better?",
     "Not on that evidence. The difference could easily be noise from the "
     "particular split or the random seed. I would want the same folds for both, "
     "a spread across seeds, and a confidence interval on the difference before "
     "calling it a win.",
     ["a small gap can be sampling noise",
      "evaluate both on identical splits",
      "repeat across seeds or folds",
      "look at the interval on the difference, not the point estimates"],
     []),

    (2, "threshold selection",
     "Where should the decision threshold come from?",
     "From the relative cost of the two error types, not from a default of one "
     "half. You pick it on validation data by asking what a false positive costs "
     "against a false negative, and the answer is a business question rather than "
     "a modelling one.",
     ["it should reflect the cost of each error type",
      "one half is a default, not a decision",
      "choose it on validation data, not test",
      "the costs come from the business context"],
     []),

    (2, "evaluating imbalanced problems",
     "How do you evaluate a model when positives are one percent of the data?",
     "Not with accuracy, which a trivial model wins. I would look at precision "
     "and recall on the rare class, use a precision-recall curve rather than ROC "
     "since the false positive rate is diluted by the huge negative class, and "
     "report performance at the operating point that will actually be used.",
     ["accuracy is meaningless at that base rate",
      "measure precision and recall on the rare class",
      "prefer a precision-recall curve over ROC",
      "report at the operating point you will deploy"],
     []),

    (2, "label noise in the test set",
     "What happens if some of your test labels are wrong?",
     "You get a ceiling on the measured score that has nothing to do with the "
     "model, and comparisons between good models become noise. Worse, a model "
     "that is genuinely right on a mislabelled case is punished for it — so past "
     "a point you are ranking models by how well they reproduce your errors.",
     ["it caps the achievable measured score",
      "correct predictions get counted as wrong",
      "it obscures differences between strong models",
      "audit a sample of labels before trusting the number"],
     []),

    (2, "regression test sets",
     "What is a regression test set for a model?",
     "A fixed set of cases, including every failure you have previously fixed, "
     "that you re-run on every change. It is how you find out that improving one "
     "thing broke another — which happens constantly with models and is "
     "invisible in an aggregate metric.",
     ["a fixed set of cases run on every change",
      "it includes previously fixed failures",
      "it catches regressions an aggregate metric hides",
      "the set must stay stable to be comparable"],
     []),

    # ------------------------------------------------------------------ 3 --- #

    (3, "worst-group performance",
     "Why might you optimise worst-group performance instead of the average?",
     "Because the average is a business decision disguised as a metric — it "
     "accepts being bad for a minority as long as you are good for the majority. "
     "If the model gates something that matters, or the weak group is one you "
     "have obligations to, the worst slice is the number that governs.",
     ["the average trades minority performance for majority",
      "an important group can be systematically failed",
      "worst-group performance bounds the harm",
      "it matters for fairness and for high-stakes decisions"],
     []),

    (3, "significance of a model comparison",
     "How would you test whether one model is genuinely better than another?",
     "By comparing them on the same examples and testing the paired differences, "
     "which removes the variance from example difficulty. Bootstrapping the test "
     "set gives an interval on the gap. Comparing two independent averages throws "
     "away the pairing and needs far more data.",
     ["evaluate both models on identical examples",
      "test the paired per-example differences",
      "pairing removes variance due to example difficulty",
      "bootstrap the test set for an interval on the gap"],
     []),

    (3, "confidence intervals on a metric",
     "Your test set has 500 examples and accuracy is 90 percent. How precise is that?",
     "Not very. With five hundred examples the interval on ninety percent is "
     "roughly plus or minus two and a half points, so anything inside about five "
     "points is indistinguishable. That is worth knowing before declaring a "
     "one-point improvement.",
     ["the estimate has meaningful uncertainty",
      "the interval width depends on test set size",
      "at this size it is a few percentage points",
      "differences inside the interval cannot be distinguished"],
     []),

    (3, "calibration checks",
     "How would you check whether a model's probabilities are trustworthy?",
     "Bucket the predictions by score and compare the predicted probability in "
     "each bucket to the observed frequency. Plotted, that is a reliability "
     "diagram, and a well calibrated model sits on the diagonal. Ranking metrics "
     "tell you nothing about this.",
     ["group predictions into score buckets",
      "compare predicted probability to observed frequency",
      "plot it as a reliability diagram",
      "AUC and ranking metrics do not capture calibration"],
     []),

    (3, "evaluating ranking systems",
     "Why is accuracy the wrong metric for a ranking system?",
     "Because rank position matters and accuracy ignores it. Getting the right "
     "item into position one is worth far more than position twenty, and a "
     "classification metric treats those identically. Ranking metrics discount by "
     "position for exactly that reason.",
     ["position in the ranking matters",
      "accuracy treats all positions identically",
      "users mostly see the top few results",
      "ranking metrics discount gains by position"],
     ["Which ranking metric would you actually use?"]),

    (3, "evaluating recommenders",
     "What makes recommender evaluation harder than classification?",
     "You only observe feedback on what was shown, and what was shown came from "
     "the previous system. So the logs are biased by that policy, and offline "
     "accuracy rewards agreeing with it. There is also no signal for the good "
     "recommendation nobody ever saw.",
     ["feedback exists only for items that were shown",
      "the logs are biased by the previous recommender",
      "offline metrics reward reproducing that policy",
      "unshown but good items generate no signal"],
     []),

    (3, "human evaluation",
     "When do you need human evaluation, and what makes it reliable?",
     "When there is no automatic metric that captures quality — generation, "
     "relevance, tone. It is reliable when the rubric is specific enough that two "
     "people agree, when raters see items in random order, and when you measure "
     "agreement rather than assume it.",
     ["needed where no automatic metric captures quality",
      "requires a specific written rubric",
      "randomise order to avoid position bias",
      "measure inter-annotator agreement"],
     []),

    (3, "benchmark contamination",
     "What is benchmark contamination?",
     "When the evaluation data has appeared in the model's training data, so the "
     "score measures memorisation rather than ability. It is endemic with models "
     "trained on the open web, and it is why a strong public benchmark result may "
     "say nothing about your task.",
     ["evaluation data appeared in training data",
      "the score reflects memorisation, not generalisation",
      "common for models trained on web-scale corpora",
      "use private or newly-created evaluation data"],
     []),

    (3, "evaluating generative output",
     "How do you evaluate output that has many valid answers?",
     "Stop trying to produce one number. Break quality into dimensions you can "
     "judge separately — factual accuracy, relevance, format, safety — score each "
     "against a rubric, and validate any automated judge against human ratings "
     "before trusting it at scale.",
     ["decompose quality into separate dimensions",
      "score each against an explicit rubric",
      "reference-overlap metrics penalise valid paraphrase",
      "validate an automated judge against human labels"],
     []),

    (3, "reproducibility of results",
     "Someone cannot reproduce your reported number. What are the usual causes?",
     "Uncontrolled randomness — seeds for initialisation, shuffling and "
     "augmentation — a different data version or split, a preprocessing step "
     "applied inconsistently, or a different library version. It is why the split "
     "and the seed belong with the result.",
     ["random seeds not fixed or not recorded",
      "a different data version or split",
      "inconsistent preprocessing",
      "record the data version, split, seed and environment"],
     []),

    # ------------------------------------------------------------------ 4 --- #

    (4, "offline versus online evaluation",
     "Offline metrics improved but the online test was flat. What explains that?",
     "Several things, and they are worth separating. The offline metric may be a "
     "poor proxy for the online one. The offline data came from the old policy, "
     "so it rewards agreeing with it. The test may be underpowered. Or serving "
     "features differ from training features.",
     ["the offline metric may not proxy the online outcome",
      "offline data is biased by the previous policy",
      "the online test may lack power to detect the effect",
      "train-serve differences can eat the gain"],
     []),

    (4, "error analysis",
     "How would you turn error analysis into a plan rather than a list?",
     "By counting. I would sample errors, label each with a cause, and tally — "
     "which turns an anecdote into a distribution. Then estimate the achievable "
     "gain per category and weigh it against effort. Usually two categories "
     "account for most of the loss and the rest is not worth touching.",
     ["sample errors and label each with a cause",
      "count the categories to get a distribution",
      "estimate the recoverable gain per category",
      "prioritise by gain against effort"],
     []),

    (4, "the test set as a scarce resource",
     "You have run hundreds of experiments against one test set. What is it worth now?",
     "Much less than it says. Selecting the best of hundreds of results against "
     "the same data means the winner is partly chosen for fitting that sample — "
     "the reported score is biased upward and the gap to second place is mostly "
     "noise. I would want a fresh held-out set before believing it.",
     ["selecting a winner across many runs biases the estimate",
      "the reported score is optimistic",
      "the margin over other candidates is largely noise",
      "confirm on a fresh untouched set"],
     []),

    (4, "evaluating in production",
     "How do you evaluate a model in production when labels arrive late?",
     "You need something to watch in the meantime. Input and prediction "
     "distributions catch drift immediately. A validated early proxy tracks "
     "quality on a shorter lag. And the fully-labelled older cohorts give real "
     "performance, just delayed — so you evaluate on a rolling window that is "
     "always behind.",
     ["monitor input and prediction distributions immediately",
      "use an early proxy validated against the true outcome",
      "evaluate mature cohorts on a lag",
      "re-check that the proxy still tracks the outcome"],
     []),

    (4, "inter-annotator agreement",
     "Your raters agree only 60 percent of the time. What does that tell you?",
     "That the task as written is ambiguous, not that the raters are bad. No "
     "model can be measured more precisely than the labels, so that agreement is "
     "a ceiling on any score derived from them. The fix is the rubric and the "
     "examples, not more raters.",
     ["the task definition is ambiguous",
      "agreement bounds how precisely anything can be measured",
      "it caps the meaningful score on that data",
      "fix the rubric and adjudicate disagreements"],
     []),

    (4, "what a metric is a proxy for",
     "How would you catch a metric being gamed by the model?",
     "Watch the guardrails and the distribution, not just the headline. A gamed "
     "metric usually rises alongside something degrading — length, repetition, "
     "hedging, one segment carrying the whole gain. I would also inspect the top "
     "scoring outputs directly, because gaming is obvious to a reader long before "
     "it is obvious in aggregate.",
     ["monitor guardrail metrics alongside the target",
      "watch for a shift in the output distribution",
      "check whether the gain is concentrated in one segment",
      "read the highest-scoring outputs directly"],
     []),

    # ------------------------------------------------------------------ 5 --- #

    (5, "building an evaluation from scratch",
     "There is no evaluation set. Build one.",
     "From real traffic, because the distribution has to be right. I would sample "
     "across the query types that occur, deliberately over-sample the hard and "
     "rare cases, label with a written rubric and measure agreement on a shared "
     "subset. Then freeze it, keep a separate untouched slice, and add every "
     "production failure to it permanently.",
     ["sample from real traffic for a correct distribution",
      "cover the range of cases and over-sample hard ones",
      "label against a written rubric and check agreement",
      "freeze the set and hold part of it back",
      "add production failures as permanent cases"],
     []),

    (5, "comparing two models honestly",
     "A colleague says their model is better. What do you ask for?",
     "The same evaluation, not a better number. Identical test data and splits, "
     "the same metric at the same operating point, per-segment results, and a "
     "confidence interval on the difference. Then I would ask how many times the "
     "test set has been used, because that decides what the number is worth.",
     ["identical test data and split for both",
      "the same metric at the same operating point",
      "an interval on the difference, not two point estimates",
      "per-segment results, not only the average",
      "how many times the test set has been reused"],
     []),

    (5, "the cost of evaluation",
     "Your evaluation takes six hours. How does that change what you do?",
     "It changes the shape of the work more than the numbers. A slow evaluation "
     "means fewer iterations, so I would build a fast noisy subset for the inner "
     "loop and reserve the full run for candidates that clear it — after checking "
     "the two rank changes the same way. Anything else and you evaluate less "
     "often, which is worse.",
     ["slow evaluation reduces iteration count",
      "build a fast approximate subset for the inner loop",
      "reserve the full run for serious candidates",
      "verify the fast proxy ranks models the same way"],
     []),
]
