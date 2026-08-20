r"""Machine Learning — classical ML breadth.

The round this prepares for is "ML breadth": rapid-fire fundamentals, where the
interviewer is checking that the floor is solid before anyone talks about
transformers. Most of it is difficulty 1 and 2 on purpose.

The bias here is towards questions that have a *wrong* answer people actually
give — "accuracy is the metric", "more features is better", "the model with the
best AUC wins" — because those are what a screen is looking for.
"""

from __future__ import annotations

ENTRIES = [

    # ------------------------------------------------------------------ 1 --- #

    (1, "supervised versus unsupervised",
     "What's the difference between supervised and unsupervised learning?",
     "Supervised learning trains on examples that carry a label, so the model "
     "learns a mapping from inputs to a known answer. Unsupervised learning has "
     "no labels and looks for structure in the data itself, like clusters or a "
     "lower-dimensional representation.",
     ["supervised learning uses labelled examples",
      "it learns a mapping from input to a known target",
      "unsupervised learning has no labels",
      "it finds structure such as clusters or components"],
     ["Where does reinforcement learning sit in that split?"]),

    (1, "overfitting and underfitting",
     "What is overfitting?",
     "Overfitting is when a model learns the noise in the training data as if it "
     "were signal. It scores very well on data it has seen and much worse on "
     "data it has not, because what it memorised does not generalise.",
     ["the model fits noise as if it were signal",
      "training performance is much better than test performance",
      "it fails to generalise to unseen data",
      "usually from too much capacity or too little data"],
     ["What would you do about it?"]),

    (1, "training, validation and test splits",
     "Why do you hold out a test set instead of scoring on training data?",
     "Because training performance measures memorisation, not generalisation. "
     "A model can score perfectly on data it was fitted to and be useless on "
     "anything new. The test set is the only honest estimate of how it behaves "
     "on data it has never seen.",
     ["training scores measure memorisation, not generalisation",
      "the test set estimates performance on unseen data",
      "the test set must not influence training or tuning",
      "a separate validation set is used for tuning choices"],
     ["What's the validation set for, if you already have a test set?"]),

    (1, "supervised versus unsupervised",
     "What's the difference between classification and regression?",
     "Both are supervised, but the target differs. Classification predicts a "
     "discrete label, like spam or not spam. Regression predicts a continuous "
     "number, like a price. That difference drives the loss function and the "
     "metrics you can use.",
     ["classification predicts a discrete label",
      "regression predicts a continuous value",
      "they use different loss functions",
      "they need different evaluation metrics"],
     []),

    (1, "choosing a metric for imbalanced data",
     "When is accuracy a misleading metric?",
     "Whenever the classes are imbalanced. If one percent of transactions are "
     "fraud, a model that predicts 'never fraud' is ninety-nine percent "
     "accurate and completely worthless. Accuracy hides which class the errors "
     "fall on, which is usually the thing you care about.",
     ["accuracy misleads when classes are imbalanced",
      "a trivial majority-class model can score highly",
      "it hides which class the errors fall on",
      "precision, recall or AUC are better in that case"],
     ["So what would you look at instead?"]),

    (1, "feature scaling",
     "What is a feature, and what makes a good one?",
     "A feature is an input variable the model learns from. A good one carries "
     "signal about the target, is available at prediction time, and is not "
     "simply a restatement of the label. That last point is where most feature "
     "bugs come from.",
     ["a feature is an input variable used for prediction",
      "it should carry signal about the target",
      "it must be available at prediction time",
      "it must not leak the label"],
     []),

    (1, "hyperparameter search",
     "What's the difference between a parameter and a hyperparameter?",
     "Parameters are learned from the data during training, like the weights of "
     "a linear model. Hyperparameters are set before training and control how "
     "learning happens, like the learning rate or the depth of a tree. You tune "
     "hyperparameters on validation data, never on the test set.",
     ["parameters are learned from data during training",
      "hyperparameters are set before training",
      "hyperparameters control the learning process",
      "they are tuned on validation data"],
     []),

    (1, "cross-validation",
     "What is cross-validation?",
     "You split the training data into several folds, train on all but one and "
     "evaluate on the one held out, then rotate. Averaging the folds gives a "
     "more stable estimate than a single split, which matters most when the "
     "dataset is small enough that one split is luck.",
     ["the data is split into several folds",
      "each fold is held out in turn for evaluation",
      "results are averaged across folds",
      "it gives a more stable estimate than one split"],
     ["When would cross-validation be the wrong thing to do?"]),

    # ------------------------------------------------------------------ 2 --- #

    (2, "the bias-variance tradeoff",
     "What is the bias-variance tradeoff?",
     "Bias is error from the model being too simple to capture the pattern. "
     "Variance is error from it being so flexible that it changes a lot with the "
     "particular training sample. Reducing one usually raises the other, so the "
     "job is finding the capacity that minimises their total.",
     ["bias is error from an overly simple model",
      "variance is error from sensitivity to the training sample",
      "reducing one typically increases the other",
      "the goal is minimising their combined error"],
     ["Which one does a deep decision tree suffer from?"]),

    (2, "regularisation, L1 versus L2",
     "What does regularisation do?",
     "It adds a penalty on model complexity to the training objective, so the "
     "model is pushed to fit the data without using larger coefficients than it "
     "needs. That trades a little training accuracy for better generalisation — "
     "it deliberately raises bias to cut variance.",
     ["it penalises model complexity during training",
      "it discourages large coefficients",
      "it trades training fit for generalisation",
      "it reduces variance at the cost of some bias"],
     []),

    (2, "regularisation, L1 versus L2",
     "What's the practical difference between L1 and L2 regularisation?",
     "L1 penalises the absolute size of coefficients and can drive them exactly "
     "to zero, so it performs feature selection and gives a sparse model. L2 "
     "penalises squared size, shrinking coefficients smoothly towards zero "
     "without eliminating them, which handles correlated features more gracefully.",
     ["L1 penalises absolute coefficient size",
      "L1 can drive coefficients to exactly zero",
      "L1 therefore does feature selection",
      "L2 shrinks coefficients smoothly without zeroing them"],
     ["Which would you reach for with a thousand correlated features?"]),

    (2, "precision versus recall",
     "When would you optimise for precision rather than recall?",
     "When a false positive is more costly than a false negative. Flagging a "
     "legitimate transaction as fraud annoys a real customer, so precision "
     "matters there. Screening for a serious disease is the reverse — you would "
     "rather investigate a few extra people than miss a case.",
     ["precision matters when false positives are costly",
      "recall matters when false negatives are costly",
      "gave a concrete example of the tradeoff",
      "the choice comes from the business cost of each error"],
     ["How would you pick the threshold in practice?"]),

    (2, "confusion matrices",
     "What does a confusion matrix tell you that accuracy does not?",
     "It breaks the errors down by type — true and false positives, true and "
     "false negatives — so you can see which class the model gets wrong and in "
     "which direction. Accuracy collapses all of that into one number and hides "
     "the asymmetry that usually matters.",
     ["it separates errors by type",
      "it shows false positives and false negatives separately",
      "it reveals which class is being got wrong",
      "accuracy collapses this into a single number"],
     []),

    (2, "feature scaling",
     "Which models need feature scaling, and which do not?",
     "Anything driven by distances or by gradient magnitude needs it — k-nearest "
     "neighbours, support vector machines, k-means, and neural networks. Tree "
     "based models do not, because a split compares a feature to a threshold and "
     "that comparison is unaffected by the scale.",
     ["distance-based models need scaling",
      "gradient-based models converge better with scaling",
      "tree-based models do not need it",
      "trees split on thresholds, which scaling does not change"],
     []),

    (2, "categorical encoding",
     "How would you handle a categorical feature with very many levels?",
     "One-hot encoding explodes the dimensionality, so for high cardinality I "
     "would use target or frequency encoding, or a learned embedding. Target "
     "encoding has to be fitted inside the cross-validation folds, otherwise it "
     "leaks the label into the features.",
     ["one-hot encoding explodes dimensionality at high cardinality",
      "target or frequency encoding is a compact alternative",
      "an embedding can be learned instead",
      "target encoding must be fitted within folds to avoid leakage"],
     ["What exactly leaks if you fit it on the whole training set?"]),

    (2, "random forests versus boosting",
     "What's the difference between bagging and boosting?",
     "Bagging trains many models independently on bootstrap samples and averages "
     "them, which reduces variance. Boosting trains models in sequence, each one "
     "focusing on what the previous ones got wrong, which mainly reduces bias. "
     "Bagging parallelises; boosting cannot.",
     ["bagging trains models independently and averages them",
      "bagging mainly reduces variance",
      "boosting trains models sequentially on prior errors",
      "boosting mainly reduces bias"],
     ["Which of the two is easier to overfit with?"]),

    (2, "class imbalance",
     "How would you approach a problem where one class is one percent of the data?",
     "First change the metric, because accuracy is meaningless there. Then I "
     "would try class weights in the loss, resampling the training set, or "
     "adjusting the decision threshold. Resampling must happen inside the "
     "training folds only, never before the split.",
     ["change the evaluation metric first",
      "use class weights in the loss",
      "resample by oversampling or undersampling",
      "tune the decision threshold rather than using one half"],
     ["Why must resampling happen after the split, not before?"]),

    (2, "handling missing data",
     "How do you deal with missing values in a feature?",
     "It depends on why they are missing. If it is random, imputing with a "
     "median or a model is reasonable. If the absence itself carries signal — a "
     "field a customer chose not to fill in — then I would add an indicator "
     "column, because imputing it away destroys real information.",
     ["the right approach depends on why data is missing",
      "impute with a statistic or a model when missing at random",
      "missingness itself can carry signal",
      "add an indicator column for missingness"],
     []),

    # ------------------------------------------------------------------ 3 --- #

    (3, "decision trees",
     "How does a decision tree decide where to split?",
     "It searches candidate splits and picks the one that most reduces impurity "
     "in the resulting children — Gini or entropy for classification, variance "
     "for regression. It repeats greedily at each node, which is why a tree "
     "finds a locally good structure rather than a globally optimal one.",
     ["it picks the split that most reduces impurity",
      "impurity is measured by Gini or entropy",
      "regression trees reduce variance instead",
      "the search is greedy and local, not globally optimal"],
     ["What stops it from splitting until every leaf is pure?"]),

    (3, "gradient boosting",
     "What's the difference between a random forest and gradient boosting?",
     "A random forest grows deep trees independently on bootstrap samples and "
     "averages them, so it fights variance. Gradient boosting grows shallow "
     "trees in sequence, each fitted to the residual error of the ensemble so "
     "far, so it fights bias. Boosting is usually more accurate and much easier "
     "to overfit.",
     ["a forest trains independent deep trees and averages",
      "boosting trains shallow trees sequentially",
      "each boosted tree fits the residual error so far",
      "boosting is more accurate but easier to overfit"],
     ["What are the main knobs for stopping a boosted model overfitting?"]),

    (3, "the ROC curve and AUC",
     "What does the ROC curve show, and what does AUC mean?",
     "The curve plots the true positive rate against the false positive rate as "
     "you sweep the decision threshold. The area underneath is the probability "
     "that the model ranks a random positive above a random negative — so it "
     "measures ranking quality, independent of any one threshold.",
     ["it plots true positive rate against false positive rate",
      "the curve is traced by sweeping the threshold",
      "AUC is the probability a positive outranks a negative",
      "it measures ranking, independent of a chosen threshold"],
     ["When would a precision-recall curve be the better choice?"]),

    (3, "data leakage",
     "What is data leakage, and how would you catch it?",
     "Leakage is when information that would not exist at prediction time gets "
     "into training — a feature computed after the outcome, or scaling fitted on "
     "the whole dataset before splitting. The tell is validation performance "
     "that looks too good and collapses in production.",
     ["information unavailable at prediction time enters training",
      "common causes are future features or preprocessing before the split",
      "the symptom is unrealistically good validation scores",
      "performance collapses in production"],
     ["How would you build a split for time-series data to avoid it?"]),

    (3, "k-means clustering",
     "How does k-means work, and what's its main weakness?",
     "It assigns each point to the nearest of k centroids, recomputes the "
     "centroids as the mean of their members, and repeats until stable. The main "
     "weaknesses are that you must choose k up front, it assumes roughly "
     "spherical clusters of similar size, and the result depends on the "
     "initialisation.",
     ["points are assigned to the nearest centroid",
      "centroids are recomputed as the mean of their members",
      "k must be chosen in advance",
      "it assumes spherical clusters and is sensitive to initialisation"],
     ["How would you choose k?"]),

    (3, "principal component analysis",
     "What does principal component analysis actually do?",
     "It finds the directions along which the data varies most and re-expresses "
     "the data in those directions, ordered by how much variance each explains. "
     "Keeping the first few gives a lower-dimensional representation that "
     "retains most of the variance, at the cost of interpretable features.",
     ["it finds the directions of maximum variance",
      "components are orthogonal and ordered by variance explained",
      "keeping the top components reduces dimensionality",
      "the new features are no longer interpretable"],
     ["Why do you have to scale the features before running it?"]),

    (3, "linear regression assumptions",
     "What assumptions does linear regression make?",
     "Linearity between the features and the target, independent errors, "
     "constant error variance, and roughly normal errors if you want valid "
     "confidence intervals. Strongly correlated features do not break the "
     "predictions but make the individual coefficients unstable and unreadable.",
     ["a linear relationship between features and target",
      "errors are independent",
      "errors have constant variance",
      "multicollinearity destabilises the coefficients"],
     []),

    (3, "logistic regression",
     "Why is logistic regression called regression when it classifies?",
     "Because it is a linear regression on the log-odds. The linear combination "
     "of features predicts the log-odds of the positive class, and the logistic "
     "function squashes that into a probability between zero and one. "
     "Classification comes from applying a threshold afterwards.",
     ["it fits a linear model of the log-odds",
      "the logistic function maps that to a probability",
      "the output is a probability, not a class",
      "classification comes from thresholding that probability"],
     []),

    (3, "choosing a metric for imbalanced data",
     "Why might a precision-recall curve beat ROC on an imbalanced problem?",
     "Because the false positive rate has the large negative class in its "
     "denominator, so a huge number of false positives barely moves it. ROC AUC "
     "can look excellent while the model is unusable. Precision uses predicted "
     "positives instead, so it reacts to exactly the failure you care about.",
     ["the false positive rate is diluted by a large negative class",
      "ROC AUC can look good on an unusable model",
      "precision is computed over predicted positives",
      "the PR curve reflects performance on the rare class"],
     []),

    # ------------------------------------------------------------------ 4 --- #

    (4, "calibration of predicted probabilities",
     "What does it mean for a classifier to be well calibrated?",
     "That its predicted probabilities match observed frequencies — of the cases "
     "it scores at seventy percent, about seventy percent really are positive. "
     "Ranking metrics like AUC say nothing about this, and it matters the moment "
     "the score feeds a threshold, a cost calculation, or a downstream system.",
     ["predicted probabilities match observed frequencies",
      "AUC is unaffected by calibration",
      "it matters when the probability drives a decision",
      "fixed with Platt scaling or isotonic regression"],
     ["Which common model families tend to be poorly calibrated?"]),

    (4, "concept drift",
     "How would you detect concept drift in a deployed model?",
     "By monitoring rather than waiting for complaints. I would track the "
     "distribution of the input features and of the predictions against the "
     "training baseline, and track live performance wherever labels arrive. "
     "Feature drift is visible immediately; a change in the input-output "
     "relationship only shows once labels come back.",
     ["monitor input feature distributions against a baseline",
      "monitor the prediction distribution",
      "track live metrics where labels are available",
      "distinguished data drift from a change in the true relationship"],
     ["What would you do once you detect it?"]),

    (4, "train-serve skew",
     "Why might a model with better offline metrics do worse in production?",
     "Usually because the offline setup did not match reality. The features "
     "computed at serving time differ from the training ones, the evaluation "
     "split leaked, the offline metric is a poor proxy for the business "
     "outcome, or the model changes user behaviour and so changes the data it "
     "then sees.",
     ["train-serve skew in how features are computed",
      "leakage or an unrealistic evaluation split",
      "the offline metric is a poor proxy for the real objective",
      "the model itself changes the data distribution"],
     []),

    (4, "gradient boosting",
     "How is gradient boosting related to gradient descent?",
     "It is gradient descent performed in function space rather than parameter "
     "space. Each new tree is fitted to the negative gradient of the loss with "
     "respect to the current predictions, and adding it takes a step downhill. "
     "The learning rate is the size of that step.",
     ["it is gradient descent in function space",
      "each tree fits the negative gradient of the loss",
      "adding the tree takes a step down the loss surface",
      "the learning rate scales each step"],
     []),

    (4, "support vector machines",
     "What is the kernel trick?",
     "Some problems that are not separable in the original space become "
     "separable in a higher-dimensional one. The kernel trick computes inner "
     "products in that space without ever constructing the coordinates, so you "
     "get the power of the mapping at the cost of a similarity function.",
     ["data can be separable in a higher-dimensional space",
      "the kernel computes inner products in that space",
      "the mapping is never explicitly computed",
      "it avoids the cost of the high-dimensional representation"],
     []),

    (4, "feature importance",
     "When is feature importance from a tree model misleading?",
     "When features are correlated, importance is split arbitrarily between "
     "them, so a genuinely important signal looks weak. Impurity-based "
     "importance also favours high-cardinality features regardless of signal. "
     "Permutation importance on held-out data is more trustworthy, and SHAP "
     "gives per-prediction attribution.",
     ["correlated features split importance between them",
      "impurity importance is biased toward high-cardinality features",
      "it describes the model, not the underlying causal effect",
      "permutation importance or SHAP is more reliable"],
     []),

    # ------------------------------------------------------------------ 5 --- #

    (5, "hyperparameter search",
     "How would you decide whether to invest in more data or a better model?",
     "Plot a learning curve. If validation error is still falling as training "
     "size grows, more data will help. If it has flattened well above the "
     "training error, the gap is variance and regularisation or more data helps. "
     "If both curves have converged at a poor level, the model itself is the "
     "limit and more data will not save it.",
     ["plot performance against training set size",
      "a still-falling curve means more data helps",
      "a large train-validation gap points to variance",
      "converged curves at poor performance mean the model is the limit"],
     []),

    (5, "train-serve skew",
     "Your offline metric improved but the online test showed nothing. What now?",
     "First I would check the gap is real and not a power problem — the test may "
     "be underpowered for the effect size. Then I would look for whether the "
     "offline metric is a poor proxy for the online one, whether serving "
     "features differ from training features, and whether the change only "
     "affects a segment the overall metric averages away.",
     ["check whether the experiment had the power to detect the effect",
      "question whether the offline metric proxies the online one",
      "check for train-serve feature differences",
      "look for segment-level effects hidden by the average"],
     []),

    (5, "class imbalance",
     "How would you build a model when labels are expensive and scarce?",
     "I would get the most out of few labels: start with a simple model and "
     "strong features, use active learning to spend the labelling budget on the "
     "cases the model is least sure about, and exploit the unlabelled data "
     "through pretraining or semi-supervised methods. Weak supervision from "
     "heuristics can bootstrap a first training set.",
     ["prefer a simple model with strong features when data is scarce",
      "use active learning to label the most informative examples",
      "exploit unlabelled data via pretraining or semi-supervision",
      "use weak supervision or heuristics to bootstrap labels"],
     []),
    # --------------------------------------------------------------------- #
    # Added in the expansion pass. All ground floor and lower middle, because
    # that is where a breadth round spends its time and where the original set
    # was thinnest.
    # --------------------------------------------------------------------- #

    (1, "features and labels",
     "What's the difference between a feature and a label?",
     "The label is what you are trying to predict; the features are what you "
     "predict it from. Getting that backwards is the most common source of "
     "leakage, because something derived from the label looks like a very good "
     "feature right up until production.",
     ["the label is the target being predicted",
      "features are the inputs used to predict it",
      "only labelled data can train a supervised model",
      "a feature derived from the label causes leakage"],
     []),

    (1, "what a model learns",
     "What does a machine learning model actually learn?",
     "A set of parameters that map inputs to outputs, chosen because they "
     "minimise error on the training examples. It is not learning rules someone "
     "wrote down - it is finding a function that fits the data it was shown, "
     "which is why it fails on data unlike that.",
     ["it learns parameters that map inputs to outputs",
      "they are chosen to minimise error on training data",
      "the rules are not explicitly programmed",
      "it generalises only to data resembling the training set"],
     []),

    (1, "more data",
     "Is more training data always better?",
     "More of the right data is. More of the same data adds little once the "
     "curve flattens, and more biased data makes the bias more confident. What "
     "helps is data covering cases the model currently gets wrong, which is why "
     "error analysis usually beats collecting indiscriminately.",
     ["more data helps most while the learning curve is still rising",
      "duplicated or similar data adds little",
      "biased data reinforces the bias",
      "coverage of failure cases matters more than volume"],
     []),

    (1, "one-hot encoding",
     "What is one-hot encoding and why is it needed?",
     "Turning a category into a set of binary columns, one per value. It is "
     "needed because most models do arithmetic on their inputs, and numbering "
     "categories one, two, three would imply an ordering and a spacing that do "
     "not exist.",
     ["each category becomes its own binary column",
      "models require numeric input",
      "integer codes would imply a false ordering",
      "it adds one column per distinct value"],
     ["What happens when the column has thousands of values?"]),

    (1, "linear and non-linear models",
     "What does it mean for a model to be linear?",
     "That its prediction is a weighted sum of the features. It can only "
     "represent a straight-line relationship, so if the truth curves or the "
     "features interact it will underfit no matter how much data you give it. "
     "The upside is that the coefficients are readable.",
     ["the prediction is a weighted sum of features",
      "it can only fit linear relationships",
      "it underfits curved or interacting patterns",
      "the coefficients are directly interpretable"],
     []),

    (2, "the curse of dimensionality",
     "What is the curse of dimensionality?",
     "As you add features the space grows exponentially, so your data becomes "
     "sparse in it and every point ends up roughly equally far from every other. "
     "Distance stops being meaningful, which breaks anything relying on it, and "
     "you need far more data to cover the space.",
     ["the volume of the space grows exponentially with features",
      "data becomes sparse in high dimensions",
      "distances between points become similar and less meaningful",
      "distance-based methods degrade"],
     []),

    (2, "k-nearest neighbours",
     "How does k-nearest neighbours work, and what does it cost?",
     "It stores the training data and, to predict, finds the k closest examples "
     "and takes their majority or average. Training is free and prediction is "
     "expensive, which is the reverse of most models, and it needs scaled "
     "features because it is entirely distance based.",
     ["it finds the k closest training examples",
      "it predicts by majority vote or average",
      "training is trivial but prediction is expensive",
      "it requires feature scaling"],
     []),

    (2, "naive Bayes",
     "What is naive about naive Bayes?",
     "The assumption that features are independent given the class, which is "
     "almost never true. It still works surprisingly well, especially on text, "
     "because it only needs the ranking of class probabilities to come out right, "
     "not their actual values.",
     ["it assumes features are conditionally independent",
      "that assumption is usually false",
      "it often still classifies well despite this",
      "the probabilities it outputs are poorly calibrated"],
     []),

    (2, "parametric versus non-parametric",
     "What's the difference between a parametric and a non-parametric model?",
     "A parametric model has a fixed number of parameters regardless of data "
     "size - a linear model has one weight per feature however many rows you "
     "have. A non-parametric one grows with the data, like a tree that gets "
     "deeper or nearest neighbours that stores everything.",
     ["a parametric model has a fixed parameter count",
      "it does not grow with the training set size",
      "a non-parametric model grows with the data",
      "non-parametric models are more flexible and need more data"],
     []),

    (2, "shuffling and ordering",
     "Why does the order of your training data matter?",
     "Because most training is done in batches, and if the data arrives sorted by "
     "label or by time each batch is unrepresentative. The model chases whatever "
     "the current batch looks like. Shuffling fixes it - except with time series, "
     "where shuffling destroys the split.",
     ["batches from ordered data are unrepresentative",
      "gradient updates chase the current batch",
      "shuffling makes each batch resemble the whole",
      "time-ordered data must not be shuffled across the split"],
     []),

    (3, "ensembling",
     "Why does averaging several models usually beat any one of them?",
     "Because their errors are not identical. Where models are wrong in "
     "different directions, averaging cancels part of the error and keeps the "
     "signal they agree on. The gain comes from the diversity, so an ensemble of "
     "near-identical models buys almost nothing.",
     ["individual models make different errors",
      "averaging cancels uncorrelated error",
      "the shared signal survives the average",
      "the benefit depends on the models being diverse"],
     []),

    (3, "outliers",
     "How would you deal with outliers in your training data?",
     "Find out what they are first. A measurement error should be removed or "
     "corrected; a genuine rare case usually should not, because that is the "
     "tail you may care most about. If they are real but destabilising, cap them "
     "or use a loss that is less sensitive to large errors.",
     ["determine whether they are errors or genuine rare cases",
      "errors can be removed or corrected",
      "genuine extremes may be the cases that matter",
      "cap values or use a robust loss instead of deleting"],
     []),

    (3, "feature selection",
     "Why remove features rather than let the model ignore them?",
     "Because models do not ignore them cleanly. Extra features add variance, "
     "invite overfitting, cost compute at serving time, and give more places for "
     "leakage or drift to enter. Fewer, better features also make the model "
     "easier to explain and to monitor.",
     ["irrelevant features add variance and overfitting risk",
      "each feature is a maintenance and serving cost",
      "more features mean more opportunity for leakage or drift",
      "simpler models are easier to explain and monitor"],
     []),

    (3, "choosing a model family",
     "Tabular data, fifty thousand rows. Where do you start and why?",
     "Gradient boosted trees, with a linear or logistic baseline first for "
     "reference. Boosting handles mixed types, missing values and non-linear "
     "interactions without much feature engineering, and on tabular data of that "
     "size it reliably beats a neural network for a fraction of the effort.",
     ["start with a simple baseline for reference",
      "gradient boosted trees are the strong default on tabular data",
      "they handle mixed types and interactions with little preprocessing",
      "neural networks rarely win on tabular data at this scale"],
     []),

    (3, "regularisation and data size",
     "How does the right amount of regularisation change as you get more data?",
     "It falls. Regularisation is there to stop the model reading noise as "
     "signal, and more data makes the signal clearer, so less constraint is "
     "needed. A strength tuned on a small sample will underfit once the dataset "
     "grows, which is why it gets retuned rather than fixed.",
     ["more data needs less regularisation",
      "regularisation compensates for limited data",
      "a fixed strength underfits as data grows",
      "it should be retuned when the dataset changes"],
     []),
]
