r"""Statistics & Experimentation.

The round candidates most often skip preparing for, and the one that separates
an Applied Scientist from someone who can fine-tune a model. "How would you know
whether this actually helped" is asked at Amazon as often as anything about
architectures, and the wrong answers here are confident ones — p-values read as
the probability the hypothesis is true, peeking at a running experiment,
declaring a win from a segment found after the fact.
"""

from __future__ import annotations

ENTRIES = [

    # ------------------------------------------------------------------ 1 --- #

    (1, "expectation and variance",
     "What do the mean and the variance tell you about a distribution?",
     "The mean is where the distribution sits — its centre of mass. The variance "
     "is how spread out it is around that centre. Two datasets can share a mean "
     "and behave completely differently, which is why reporting an average alone "
     "is rarely enough.",
     ["the mean is the centre of the distribution",
      "the variance measures spread around the mean",
      "standard deviation is its square root, in original units",
      "equal means can hide very different spreads"],
     []),

    (1, "the normal distribution",
     "Why does the normal distribution turn up so often?",
     "Because of the central limit theorem: when you add up many independent "
     "small effects, the total tends towards a normal shape whatever the "
     "individual effects looked like. Most measurements are sums of many small "
     "causes, so the bell curve appears everywhere.",
     ["sums of many independent effects tend to normality",
      "this is the central limit theorem",
      "it holds regardless of the underlying distribution",
      "many real measurements are such sums"],
     ["Where does that reasoning break down?"]),

    (1, "conditional probability",
     "What does conditional probability mean?",
     "The probability of one event given that another has already happened. It "
     "restricts attention to the cases where the condition holds and asks how "
     "often the event occurs within those. It is generally not the same as the "
     "probability the other way round.",
     ["the probability of an event given another has occurred",
      "it restricts the sample space to the condition",
      "it differs from the unconditional probability",
      "it is not symmetric between the two events"],
     []),

    (1, "sampling and sampling bias",
     "What is sampling bias?",
     "When the way you collected the sample makes it unrepresentative of the "
     "population you want to talk about. Surveying only people who answered a "
     "survey tells you about people who answer surveys. No amount of extra data "
     "fixes it, because the bias is in the collection, not the size.",
     ["the sample is not representative of the target population",
      "it comes from how the data was collected",
      "more data does not remove it",
      "conclusions drawn from it do not generalise"],
     []),

    (1, "correlation versus causation",
     "Why does correlation not imply causation?",
     "Because two things can move together for reasons other than one causing "
     "the other. A third variable may drive both, the causation may run the "
     "other way, or it may be coincidence in a large search. Establishing "
     "causation needs an intervention or a design that rules those out.",
     ["a third variable may cause both",
      "the direction of causation may be reversed",
      "it can be coincidence, especially across many comparisons",
      "causation requires an experiment or a causal design"],
     []),

    (1, "A/B test design",
     "What is an A/B test?",
     "A controlled experiment where users are randomly assigned to a control "
     "group that sees the current experience and a treatment group that sees the "
     "change. Random assignment is what makes the two groups comparable, so a "
     "difference in outcomes can be attributed to the change.",
     ["users are randomly assigned to control and treatment",
      "control sees the existing experience",
      "randomisation makes the groups comparable",
      "the difference in outcomes is attributed to the change"],
     ["What exactly does randomisation protect you from?"]),

    (1, "hypothesis testing",
     "What is a null hypothesis?",
     "The default position that there is no effect — the two groups come from "
     "the same distribution. A test asks how surprising the observed data would "
     "be if that were true. You either reject it or fail to reject it; you never "
     "prove it.",
     ["the default assumption of no effect",
      "the test measures how surprising the data is under it",
      "you either reject it or fail to reject it",
      "failing to reject is not proof it is true"],
     []),

    (1, "choosing a success metric",
     "What makes a good success metric for an experiment?",
     "It should be tied to the outcome you actually care about, sensitive enough "
     "to move in the time available, and hard to game. It also needs to be "
     "chosen before the experiment runs, because picking it afterwards turns any "
     "noise into a result.",
     ["it reflects the outcome you actually care about",
      "it is sensitive enough to move during the test",
      "it is difficult to game",
      "it must be chosen before the experiment starts"],
     []),

    # ------------------------------------------------------------------ 2 --- #

    (2, "p-values and what they do not mean",
     "What does a p-value of 0.03 actually mean?",
     "That if there were genuinely no effect, you would see data at least this "
     "extreme about three percent of the time. It is a statement about the data "
     "under the null hypothesis. It is not the probability that the null is "
     "true, and not the probability your result is real.",
     ["the probability of data this extreme if the null were true",
      "it is conditional on the null hypothesis",
      "it is not the probability the null is true",
      "it says nothing about the size of the effect"],
     []),

    (2, "confidence intervals",
     "Why report a confidence interval rather than just a p-value?",
     "Because the interval carries the effect size and the uncertainty together. "
     "A p-value only tells you whether you cleared a threshold. An interval tells "
     "you whether the plausible effects are large enough to matter, which is the "
     "actual decision.",
     ["it shows the size of the effect, not just significance",
      "it communicates the uncertainty around the estimate",
      "a p-value only reports a threshold decision",
      "it supports a judgement about practical importance"],
     []),

    (2, "type one and type two errors",
     "What's the difference between a type one and a type two error?",
     "A type one error is a false positive — you declare an effect that is not "
     "there. A type two error is a false negative — a real effect you failed to "
     "detect. Tightening the significance threshold reduces the first and "
     "increases the second.",
     ["type one is a false positive",
      "type two is a false negative",
      "the significance level controls type one error",
      "the two trade off against each other"],
     ["Which is worse in a product launch decision?"]),

    (2, "statistical power",
     "What is statistical power?",
     "The probability that a test detects an effect that genuinely exists. It "
     "rises with the sample size and the size of the effect, and falls as the "
     "data gets noisier. An underpowered test that finds nothing tells you "
     "almost nothing.",
     ["the probability of detecting a real effect",
      "it increases with sample size",
      "it increases with the size of the true effect",
      "an underpowered null result is uninformative"],
     []),

    (2, "sample size and minimum detectable effect",
     "How do you decide how long to run an experiment?",
     "You work backwards from the smallest effect worth acting on. Given that, "
     "the variance of the metric, and the power and significance you want, the "
     "required sample size follows — and traffic tells you how long that takes. "
     "You also run at least a full week to cover weekly seasonality.",
     ["start from the minimum effect worth detecting",
      "compute the sample size from variance, power and significance",
      "convert sample size to duration using traffic",
      "cover full weekly cycles to avoid day-of-week effects"],
     []),

    (2, "guardrail metrics",
     "What is a guardrail metric?",
     "A metric you do not expect to improve but refuse to damage — latency, "
     "crash rate, unsubscribes. It catches the case where the headline number "
     "goes up because the change quietly hurt something else. You monitor it "
     "rather than optimise it.",
     ["a metric you protect rather than improve",
      "it catches harm hidden behind a headline win",
      "examples include latency, errors or churn",
      "a regression in it can block a launch"],
     []),

    (2, "Bayes theorem",
     "What does Bayes theorem let you do?",
     "It flips a conditional probability around: given how likely the evidence is "
     "under each hypothesis, plus how likely each hypothesis was beforehand, it "
     "gives the probability of the hypothesis after seeing the evidence. The "
     "prior is what makes rare conditions produce so many false alarms.",
     ["it reverses a conditional probability",
      "it combines a prior with the likelihood of the evidence",
      "it yields the posterior probability of the hypothesis",
      "a low prior makes positive results mostly false alarms"],
     ["A test is 99 percent accurate for a disease affecting one in ten thousand. "
      "What's the chance a positive is real?"]),

    (2, "expectation and variance",
     "What's the difference between standard deviation and standard error?",
     "Standard deviation describes the spread of individual values in the data. "
     "Standard error describes the spread of an estimate, like the sample mean, "
     "across repeated samples. The standard error shrinks as the sample grows; "
     "the standard deviation does not.",
     ["standard deviation is the spread of the data",
      "standard error is the spread of an estimate",
      "standard error shrinks as sample size grows",
      "standard deviation does not shrink with more data"],
     []),

    (2, "multiple comparisons",
     "What goes wrong if you test twenty metrics in one experiment?",
     "At a five percent significance level you expect roughly one false positive "
     "by chance alone, even with no real effect anywhere. Unless you correct for "
     "the number of comparisons or pre-register one primary metric, you will "
     "find a winner in pure noise.",
     ["each test carries its own false positive rate",
      "the chance of at least one false positive grows with the count",
      "correct with Bonferroni or false discovery rate control",
      "or pre-register a single primary metric"],
     []),

    # ------------------------------------------------------------------ 3 --- #

    (3, "sequential testing and peeking",
     "Why is checking an experiment every day and stopping at significance wrong?",
     "Because you are running a new test at every look, so the real false "
     "positive rate is far above the nominal one. The metric wanders, and "
     "stopping the first time it crosses the line systematically selects the "
     "moments when noise favoured you.",
     ["each look is another chance to cross the threshold",
      "the true false positive rate is much higher than nominal",
      "stopping at a crossing selects favourable noise",
      "use a fixed horizon or a sequential testing method"],
     []),

    (3, "novelty and primacy effects",
     "What are novelty and primacy effects in an experiment?",
     "Novelty is users engaging with a change simply because it is new, which "
     "fades. Primacy is the reverse — existing users perform worse at first "
     "because they have to relearn something. Both mean the effect measured in "
     "the first days is not the long-run effect.",
     ["novelty is a temporary lift from a change being new",
      "primacy is a temporary drop while users adapt",
      "both distort early measurements",
      "run longer or analyse new users separately"],
     []),

    (3, "confounding variables",
     "What is a confounder, and how does randomisation handle it?",
     "A confounder is a variable that affects both the treatment someone gets "
     "and the outcome, creating an association that is not causal. Randomisation "
     "handles it by making assignment independent of everything else, including "
     "confounders you never thought to measure.",
     ["a variable affecting both treatment and outcome",
      "it creates a non-causal association",
      "randomisation makes assignment independent of it",
      "it works even for unmeasured confounders"],
     []),

    (3, "Simpson's paradox",
     "What is Simpson's paradox?",
     "When a trend that holds in every subgroup reverses once the groups are "
     "pooled. It happens when group membership is related to both the variable "
     "and the outcome, and the groups have very different sizes. The lesson is "
     "that the aggregate can point the opposite way to the truth.",
     ["a trend within subgroups reverses when they are combined",
      "it arises from an uneven confounding variable",
      "group sizes differ substantially",
      "the aggregate can mislead about every subgroup"],
     []),

    (3, "the bootstrap",
     "What is the bootstrap, and when would you use it?",
     "You resample your data with replacement many times, recompute the "
     "statistic on each resample, and use the spread of those values as the "
     "sampling distribution. It gives you a confidence interval when the "
     "statistic has no clean analytic formula — a median, a ratio, a percentile.",
     ["resample the data with replacement many times",
      "recompute the statistic on each resample",
      "the spread estimates the sampling distribution",
      "useful when no analytic formula exists"],
     []),

    (3, "maximum likelihood estimation",
     "What is maximum likelihood estimation?",
     "You choose the parameter values that make the observed data most probable "
     "under your model. It is the principle behind fitting most standard models, "
     "and minimising cross-entropy loss is exactly maximum likelihood for a "
     "classifier.",
     ["choose parameters that maximise the probability of the data",
      "the likelihood is a function of parameters, not data",
      "usually optimised via the log-likelihood",
      "cross-entropy minimisation is maximum likelihood"],
     []),

    (3, "network effects in experiments",
     "When does randomising individual users break an A/B test?",
     "When users influence each other. If treated users change the experience of "
     "control users — a marketplace where they compete for the same inventory, "
     "or a social feature that spreads — the control group is contaminated and "
     "the measured difference understates or distorts the true effect.",
     ["when subjects interact, control is contaminated",
      "common in marketplaces and social products",
      "the measured effect is biased",
      "randomise clusters or regions instead of individuals"],
     []),

    (3, "regression to the mean",
     "What is regression to the mean, and how does it fool people?",
     "Extreme measurements are extreme partly by luck, so the next measurement "
     "tends to be closer to average even with no intervention. Target your worst "
     "performing group, do anything at all, and it improves — which is why "
     "before-and-after comparisons without a control are so misleading.",
     ["extreme values are partly due to chance",
      "subsequent measurements drift towards the average",
      "improvement can appear without any real effect",
      "it is why a control group is necessary"],
     []),

    # ------------------------------------------------------------------ 4 --- #

    (4, "observational causal inference",
     "You cannot randomise. How do you still estimate a causal effect?",
     "You look for something that mimics randomisation. Difference in "
     "differences if there is a comparison group and a before period, an "
     "instrument that shifts treatment but not the outcome directly, a "
     "discontinuity at a threshold, or matching on observed confounders — which "
     "is the weakest, because it only handles what you measured.",
     ["difference in differences with a comparison group",
      "instrumental variables",
      "regression discontinuity at a cutoff",
      "matching or propensity scores, limited to observed confounders"],
     []),

    (4, "difference in differences",
     "What assumption does difference in differences rely on?",
     "Parallel trends: that without the intervention, the treated and control "
     "groups would have moved the same way. It is not testable directly, so you "
     "support it by showing the two tracked each other before the intervention "
     "and by checking nothing else changed at the same time.",
     ["the parallel trends assumption",
      "both groups would have moved similarly without treatment",
      "it cannot be tested directly",
      "supported by checking pre-intervention trends match"],
     []),

    (4, "sample size and minimum detectable effect",
     "Your experiment is underpowered and you cannot get more traffic. Options?",
     "Reduce the variance rather than chase the sample. Use a less noisy metric, "
     "use pre-experiment data to adjust the outcome, or stratify the "
     "randomisation. Failing that, run longer, accept a larger minimum "
     "detectable effect, or accept that this change cannot be measured and "
     "decide on other grounds.",
     ["reduce metric variance rather than only adding users",
      "use pre-experiment covariates to adjust the estimate",
      "stratify randomisation to balance known factors",
      "or accept a larger detectable effect and say so"],
     []),

    (4, "multiple comparisons",
     "A launch is flat overall but strong for new users. What do you conclude?",
     "Not much, if that segment was found after the fact. Slicing after the "
     "result guarantees some segment looks good. I would treat it as a "
     "hypothesis, not a finding, and run a follow-up experiment targeted at that "
     "segment with the metric pre-registered.",
     ["a segment found after the fact is not evidence",
      "slicing many ways guarantees some look significant",
      "treat it as a hypothesis to test, not a result",
      "confirm with a pre-registered follow-up"],
     []),

    (4, "choosing a success metric",
     "How would you measure whether a new ranking model actually improved things?",
     "Offline metrics only rank candidates for the online test, so the real "
     "answer is an experiment. I would pick one primary online metric tied to "
     "user value rather than to engagement alone, add guardrails for latency and "
     "for the long tail of queries, and be careful that clicks measure exposure "
     "as much as quality.",
     ["decide with an online experiment, not offline metrics",
      "pre-register one primary metric tied to user value",
      "add guardrail metrics such as latency",
      "account for position and exposure bias in click data"],
     []),

    # ------------------------------------------------------------------ 5 --- #

    (5, "long tail and heavy-tailed distributions",
     "Your metric is a heavy-tailed average. Why is that a problem, and what do you do?",
     "A few enormous values dominate the mean, so the variance is huge and the "
     "central limit theorem needs a much bigger sample than usual. I would cap "
     "or winsorise the extreme values, analyse a transformed or rank-based "
     "version, or switch to a quantile that is not driven by the tail — and say "
     "which decision the metric is meant to support.",
     ["a few extreme values dominate the mean",
      "variance is large so tests are underpowered",
      "cap, winsorise or transform the values",
      "or report a median or quantile instead"],
     []),

    (5, "A/B test design",
     "How would you design an experiment for a change that only affects rare events?",
     "The rarity is the whole problem: the metric has very low base rate, so "
     "power is terrible. I would move the measurement upstream to a more "
     "frequent proxy that is causally linked, restrict the experiment to the "
     "population that can actually experience the event, and run it long enough "
     "to accumulate events rather than users.",
     ["a rare outcome gives very low statistical power",
      "use a more frequent upstream proxy metric",
      "restrict to the population that can experience the event",
      "power the test on event count, not user count"],
     []),

    (5, "correlation versus causation",
     "How would you tell whether a model's recommendations cause purchases?",
     "Correlation is guaranteed here — recommendations go to people already "
     "likely to buy. I would randomly withhold recommendations from a holdout "
     "group, which measures the incremental effect directly. Where a full "
     "holdout is too costly, a small permanent one or a geographic split gives "
     "the same comparison.",
     ["recommendations are targeted, so correlation is expected",
      "randomly withhold from a holdout group",
      "this measures the incremental effect",
      "a small or geographic holdout limits the business cost"],
     []),
]
