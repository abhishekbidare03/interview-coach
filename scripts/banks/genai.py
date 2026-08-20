r"""Generative AI and LLMs.

Weighted towards the questions an Applied Scientist actually gets asked about
production systems — when retrieval beats fine-tuning, what makes a model
hallucinate, how you evaluate output that has no single right answer — rather
than towards naming the latest model. Anything tied to a specific vendor or
release would be stale within months, so nothing here is.
"""

from __future__ import annotations

ENTRIES = [

    # ------------------------------------------------------------------ 1 --- #

    (1, "how an LLM generates text",
     "How does a large language model produce text?",
     "One token at a time. At each step it produces a probability distribution "
     "over the whole vocabulary for what comes next, samples one, appends it to "
     "the input, and repeats. Everything it writes is conditioned on everything "
     "before it, including its own output.",
     ["it predicts a distribution over the next token",
      "one token is chosen and appended",
      "the process repeats autoregressively",
      "each token is conditioned on all previous text"],
     []),

    (1, "the context window",
     "What is a context window?",
     "The maximum number of tokens the model can attend to at once — the prompt "
     "plus what it has generated so far. Anything beyond it has to be dropped or "
     "summarised. It is a hard architectural limit, not a preference.",
     ["the maximum tokens the model can process at once",
      "it covers the prompt and the generated output together",
      "content beyond it must be dropped or summarised",
      "it is a hard limit of the model"],
     []),

    (1, "temperature and top-p sampling",
     "What does the temperature setting do?",
     "It reshapes the probability distribution before sampling. Low temperature "
     "sharpens it, so the model almost always picks its top choice and output is "
     "repetitive but reliable. High temperature flattens it, giving more variety "
     "and more mistakes. Zero is effectively deterministic.",
     ["it scales the distribution before sampling",
      "low temperature makes output more deterministic",
      "high temperature increases variety",
      "high temperature also increases errors"],
     []),

    (1, "prompt engineering",
     "What makes a prompt work better?",
     "Being specific about the task, the format you want back, and the audience. "
     "Giving an example or two usually helps more than any amount of "
     "instruction. And telling the model what to do beats telling it what not to "
     "do.",
     ["state the task and desired output format explicitly",
      "provide examples of the wanted behaviour",
      "give relevant context rather than assuming it",
      "positive instructions work better than prohibitions"],
     []),

    (1, "hallucination and its causes",
     "What is a hallucination in this context?",
     "Output that is fluent and confident and simply not true. It happens "
     "because the model is trained to produce plausible continuations, not to "
     "verify facts — nothing in the objective distinguishes a true statement "
     "from a likely-sounding one.",
     ["confident output that is factually wrong",
      "the model optimises plausibility, not truth",
      "nothing in training verifies facts",
      "it is more likely on rare or unseen topics"],
     ["What would you do to reduce it?"]),

    (1, "few-shot prompting",
     "What's the difference between zero-shot and few-shot prompting?",
     "Zero-shot gives the model only an instruction. Few-shot includes several "
     "worked examples in the prompt first. The examples usually help most by "
     "pinning down the output format and the level of detail, which instructions "
     "alone often fail to convey.",
     ["zero-shot provides only an instruction",
      "few-shot includes worked examples in the prompt",
      "examples clarify the expected format",
      "no weights change in either case"],
     []),

    # ------------------------------------------------------------------ 2 --- #

    (2, "fine-tuning versus prompting versus retrieval",
     "When would you fine-tune a model rather than use retrieval?",
     "Fine-tuning teaches behaviour — a format, a tone, a task the model does "
     "badly. Retrieval supplies knowledge. If the problem is that the model does "
     "not know your facts, fine-tuning is the wrong tool, because facts baked "
     "into weights go stale and cannot be cited.",
     ["fine-tuning is for behaviour, format or style",
      "retrieval is for supplying knowledge",
      "facts in weights become stale and cannot be cited",
      "retrieval updates without retraining"],
     ["Could you need both at once?"]),

    (2, "temperature and top-p sampling",
     "What does top-p sampling do differently from temperature?",
     "Top-p truncates rather than reshapes. It keeps the smallest set of tokens "
     "whose probabilities add up to p and samples only from those, so the tail "
     "is cut off entirely. That prevents the rare nonsense token that "
     "temperature alone still leaves reachable.",
     ["it keeps the smallest set of tokens summing to p",
      "sampling happens only within that set",
      "the low-probability tail is excluded",
      "the size of the candidate set varies by step"],
     []),

    (2, "quantisation",
     "What is quantisation, and what does it cost you?",
     "Storing weights at lower precision — eight or four bits instead of "
     "sixteen. Memory drops roughly proportionally and inference gets faster, "
     "which is often what makes a model fit on the hardware at all. The cost is "
     "a small accuracy loss that grows as precision falls.",
     ["weights are stored at lower numerical precision",
      "memory use drops substantially",
      "inference is faster and fits smaller hardware",
      "accuracy degrades as precision decreases"],
     []),

    (2, "parameter-efficient fine-tuning",
     "What is parameter-efficient fine-tuning?",
     "Adapting a model by training a small number of extra parameters while the "
     "original weights stay frozen. You get most of the benefit of fine-tuning "
     "for a fraction of the memory, and you can keep many task-specific adapters "
     "against one shared base model.",
     ["most of the original weights stay frozen",
      "a small number of new parameters are trained",
      "memory and compute requirements drop sharply",
      "many adapters can share one base model"],
     ["How does LoRA achieve that specifically?"]),

    (2, "guardrails and prompt injection",
     "What is prompt injection?",
     "When text the model reads — a retrieved document, a web page, a user field "
     "— contains instructions, and the model follows them instead of yours. The "
     "root cause is that instructions and data arrive in the same channel, so "
     "the model has no reliable way to tell them apart.",
     ["untrusted content contains instructions the model follows",
      "instructions and data share the same channel",
      "it can override the intended system behaviour",
      "retrieved or user-supplied text is the usual vector"],
     []),

    # ------------------------------------------------------------------ 3 --- #

    (3, "LoRA and adapters",
     "How does LoRA work?",
     "Instead of updating a weight matrix directly, it learns a low-rank pair of "
     "small matrices whose product is added to it. The original weights never "
     "move, the trained parameters are a tiny fraction of the total, and the "
     "update can be merged in at the end so inference costs nothing extra.",
     ["it learns a low-rank update to a weight matrix",
      "the original weights stay frozen",
      "trainable parameters are a small fraction of the total",
      "the update can be merged for inference"],
     []),

    (3, "reinforcement learning from human feedback",
     "What problem does RLHF solve that supervised fine-tuning does not?",
     "Supervised fine-tuning needs a written example of the ideal answer, and "
     "for open-ended tasks nobody can write that. People can, however, reliably "
     "say which of two answers is better. RLHF turns those comparisons into a "
     "reward model and optimises the policy against it.",
     ["ideal answers are hard to write for open-ended tasks",
      "people can compare two outputs more reliably",
      "a reward model is trained on those preferences",
      "the model is then optimised against that reward"],
     ["What goes wrong if the reward model is imperfect?"]),

    (3, "KV caching",
     "What does the KV cache do, and why does it matter?",
     "It stores the keys and values already computed for previous tokens, so "
     "generating each new token does not recompute attention over the whole "
     "sequence. Without it generation is quadratic in output length. The cost is "
     "memory that grows with sequence length and batch size.",
     ["it reuses keys and values from earlier tokens",
      "it avoids recomputing attention over the full prefix",
      "it makes per-token generation much cheaper",
      "it consumes memory proportional to sequence length"],
     []),

    (3, "inference latency and throughput",
     "Why is the first token slower than the rest?",
     "The first token requires processing the entire prompt in one go, which is "
     "compute-bound and grows with prompt length. After that each token only "
     "processes one new position against the cache, which is memory-bandwidth "
     "bound and roughly constant.",
     ["the first token processes the whole prompt",
      "prefill cost grows with prompt length",
      "later tokens reuse the cache and add one position",
      "prefill is compute-bound, decoding memory-bound"],
     []),

    (3, "LLM-as-a-judge",
     "What are the risks of using an LLM to grade another model's output?",
     "It has known biases — towards longer answers, towards its own style, and "
     "towards whichever option came first. It also cannot reliably catch errors "
     "it would make itself. It is usable, but only after you have measured its "
     "agreement with human ratings on the task at hand.",
     ["it favours longer or more verbose answers",
      "position and self-preference bias affect comparisons",
      "it misses errors it would make itself",
      "validate agreement against human labels first"],
     []),

    (3, "chain-of-thought prompting",
     "Why does asking a model to reason step by step improve accuracy?",
     "Because each generated token is a fixed amount of computation, and a hard "
     "problem answered in one token gets one step of it. Writing intermediate "
     "steps lets the model spend more computation and condition each step on the "
     "last, which is what turns a guess into a derivation.",
     ["intermediate tokens give the model more computation",
      "each step conditions on the previous ones",
      "it decomposes a hard problem into easier steps",
      "it helps most on multi-step reasoning tasks"],
     []),

    # ------------------------------------------------------------------ 4 --- #

    (4, "reward models",
     "What is reward hacking in RLHF?",
     "The policy learns to maximise the reward model rather than to be good. "
     "Since the reward model is only an approximation of human preference, "
     "optimising it hard enough finds its blind spots — verbosity, flattery, "
     "confident hedging — and quality falls while the measured reward rises.",
     ["the policy exploits flaws in the reward model",
      "the reward model only approximates human preference",
      "measured reward rises while real quality falls",
      "constrained by KL penalties or fresh preference data"],
     []),

    (4, "agents and tool use",
     "What makes an agent that calls tools hard to make reliable?",
     "Errors compound. Each step has some chance of choosing the wrong tool or "
     "malformed arguments, so a ten-step chain with ninety-five percent per-step "
     "reliability succeeds barely half the time. Add that failures are hard to "
     "detect mid-run and cost grows with every retry.",
     ["per-step errors compound over a long chain",
      "high per-step accuracy still gives poor end-to-end success",
      "failures are hard to detect during execution",
      "latency and cost grow with steps and retries"],
     []),

    (4, "distillation into smaller models",
     "How would you get a small model to behave like a large one?",
     "Distillation. Use the large model to generate outputs on your actual "
     "traffic distribution, then fine-tune the small model on those. You are "
     "buying back most of the quality on your specific task, not in general — "
     "which is fine, because that is the only distribution you serve.",
     ["generate training data with the larger model",
      "fine-tune the smaller model on those outputs",
      "use the real task distribution, not generic data",
      "quality is recovered on that task, not universally"],
     []),

    (4, "cost control in production",
     "Your LLM feature works but costs too much. What do you do?",
     "Look at where the tokens go first. Usually the prompt is bloated — trim "
     "context, cache the shared prefix, retrieve fewer and better chunks. Then "
     "route by difficulty so a small model handles the easy majority, and cache "
     "answers to repeated questions outright.",
     ["measure where tokens are actually being spent",
      "reduce prompt size and retrieved context",
      "cache repeated queries or shared prompt prefixes",
      "route easy requests to a smaller model"],
     []),

    (4, "guardrails and prompt injection",
     "How would you defend a RAG system against prompt injection?",
     "Treat retrieved text as data that can be hostile. Keep it clearly "
     "separated from instructions, do not give the model authority it does not "
     "need, and validate any action it proposes before executing it. The "
     "durable defence is limiting what a compromised response can actually do, "
     "not trying to filter every malicious phrasing.",
     ["treat retrieved content as untrusted",
      "separate instructions from data in the prompt",
      "restrict the model's permissions and available actions",
      "validate proposed actions rather than trusting output"],
     []),

    # ------------------------------------------------------------------ 5 --- #

    (5, "fine-tuning versus prompting versus retrieval",
     "Walk me through deciding between prompting, RAG, and fine-tuning.",
     "I would ask what the model is missing. Missing knowledge means retrieval. "
     "Missing behaviour or format means fine-tuning. Missing instruction means "
     "the prompt. In practice I would start with the cheapest that could work, "
     "measure against a fixed evaluation set, and escalate only when the numbers "
     "say the simpler option is not enough.",
     ["diagnose whether knowledge, behaviour or instruction is missing",
      "retrieval for knowledge, fine-tuning for behaviour",
      "start with the cheapest approach that could work",
      "escalate based on a fixed evaluation set"],
     []),

    (5, "inference latency and throughput",
     "How would you serve an LLM feature under a strict latency budget?",
     "Stream the response so time to first token is what the user feels, not "
     "total generation. Cut the prompt, since prefill scales with it. Use a "
     "smaller or quantised model, cap the output length, and batch requests to "
     "trade a little latency for a lot of throughput. Then cache anything "
     "repeated.",
     ["stream output so first token arrives quickly",
      "shorten the prompt to reduce prefill cost",
      "use a smaller or quantised model",
      "batch requests and cache repeated queries"],
     []),
    # --------------------------------------------------------------------- #
    # Added in the expansion pass: more of the ground floor, because the round
    # opens there. Retrieval questions live in `rag.py` now, and evaluation in
    # `model_eval.py` — this topic owns the model itself.
    # --------------------------------------------------------------------- #

    (1, "tokens",
     "What is a token, and why does it matter?",
     "The unit a model actually reads and writes - usually a word fragment "
     "rather than a word. It matters because everything is priced, limited and "
     "measured in tokens, and because a word you think of as one thing may be "
     "three tokens, which is why models are bad at counting letters.",
     ["a token is a word fragment, not always a whole word",
      "models read and generate token by token",
      "context limits and cost are counted in tokens",
      "token boundaries explain some odd model behaviour"],
     ["Roughly how many tokens is a page of English?"]),

    (1, "system prompts",
     "What is a system prompt?",
     "Instructions given to the model ahead of the conversation that set its "
     "role, tone and rules. It is not privileged in any deep sense - it is text "
     "in the same context window - but models are trained to weight it more "
     "heavily than what a user says later.",
     ["instructions that precede the conversation",
      "it sets role, tone and constraints",
      "models are trained to weight it heavily",
      "it is still text in the context, not a hard guarantee"],
     []),

    (1, "statelessness",
     "Does a language model remember your previous messages?",
     "Not by itself. Each request is independent, and the appearance of memory "
     "comes from the application resending the conversation every time. That is "
     "why a long chat costs more per message and eventually runs out of context.",
     ["the model itself holds no state between calls",
      "the conversation is resent with each request",
      "cost grows as the conversation lengthens",
      "the context window eventually limits it"],
     []),

    (1, "knowledge cutoff",
     "Why does a language model not know about recent events?",
     "Its knowledge comes from a training corpus collected up to some date, and "
     "nothing after that is in the weights. Retraining is expensive, so the "
     "cutoff stays until the next version - which is why current information has "
     "to come from the prompt.",
     ["knowledge comes from a fixed training corpus",
      "the corpus was collected up to a cutoff date",
      "the weights do not update after training",
      "current information must be supplied in the prompt"],
     []),

    (2, "greedy versus beam search decoding",
     "What's the difference between greedy decoding and beam search?",
     "Greedy takes the single most likely token at each step, which is fast and "
     "can commit early to a bad path. Beam search keeps several candidate "
     "sequences alive and picks the best complete one, which is better for "
     "translation but tends to produce bland text for open-ended generation.",
     ["greedy takes the top token at each step",
      "beam search keeps several candidate sequences",
      "beam search scores whole sequences, not single steps",
      "beam search suits constrained tasks but flattens open generation"],
     []),

    (2, "stop sequences and output length",
     "How do you stop a model generating past the part you want?",
     "A stop sequence - a string that ends generation when produced - plus a "
     "maximum token cap as a backstop. The cap alone truncates mid-sentence, "
     "which is why both are used. Asking in the prompt is not a control, because "
     "nothing enforces it.",
     ["a stop sequence ends generation when emitted",
      "a maximum token limit acts as a backstop",
      "a cap alone can truncate mid-sentence",
      "prompt instructions alone do not enforce length"],
     []),

    (2, "determinism",
     "Why does the same prompt give different answers each time?",
     "Because generation samples from a distribution rather than picking the "
     "argmax, so any temperature above zero introduces randomness. Even at zero "
     "you can see small differences, from batching and floating point order on "
     "the server rather than from the model itself.",
     ["output is sampled from a probability distribution",
      "temperature above zero introduces randomness",
      "temperature zero is close to deterministic",
      "batching and floating point can still cause small variation"],
     []),

    (2, "structured output",
     "How do you get reliable JSON out of a language model?",
     "By constraining it rather than asking nicely. Constrained decoding that "
     "only permits valid tokens for the schema is the strong version; a schema in "
     "the prompt plus validation and a retry is the weak one. Either way you "
     "validate, because a prompt instruction is not a guarantee.",
     ["constrain decoding to the schema where possible",
      "supply the schema explicitly in the prompt",
      "validate the output rather than trusting it",
      "retry or repair on a parse failure"],
     []),

    (2, "model size tradeoffs",
     "What do you actually get from a larger model?",
     "Better reasoning on hard and unusual inputs, and more reliable instruction "
     "following. What you pay is latency, cost and memory, all of which rise "
     "steeply. For a narrow, well-specified task a small model often matches it, "
     "which is why the size is worth measuring rather than assuming.",
     ["better performance on harder or rarer inputs",
      "more reliable instruction following",
      "latency, cost and memory all rise",
      "a smaller model often suffices for a narrow task"],
     []),

    (3, "prompt caching",
     "What is prompt caching, and when does it help?",
     "The server keeps the computed state for a prompt prefix, so a later request "
     "sharing that prefix skips recomputing it. It helps most when a long fixed "
     "instruction block or document precedes a short varying question - which is "
     "why the static part of a prompt should come first.",
     ["the computed state for a shared prefix is reused",
      "it avoids recomputing the prefix on each request",
      "it helps when a long static prefix precedes short input",
      "put the unchanging part of the prompt first"],
     []),

    (3, "speculative decoding",
     "What is speculative decoding?",
     "A small fast model drafts several tokens ahead and the large model verifies "
     "them in one pass, accepting the prefix it agrees with. Since verification "
     "is parallel and generation is not, you get several tokens for roughly the "
     "cost of one - with identical output.",
     ["a small model drafts tokens ahead",
      "the large model verifies them in one pass",
      "accepted drafts skip sequential generation steps",
      "the output distribution is unchanged"],
     []),

    (3, "long inputs",
     "The document you need to process is larger than the context window. Options?",
     "Split and combine, or shrink. Chunk it and process each piece then merge, "
     "summarise progressively so earlier content survives in compressed form, or "
     "retrieve only the parts relevant to the question. Which one depends on "
     "whether the task needs the whole document or one part of it.",
     ["split into chunks and combine the results",
      "summarise progressively to compress earlier content",
      "retrieve only the relevant sections",
      "the choice depends on whether the task is local or global"],
     []),

    (3, "mixture of experts",
     "What is a mixture of experts model?",
     "One where each layer holds many expert subnetworks and a router sends each "
     "token to only a couple of them. Total parameters are huge but the compute "
     "per token stays small. The catch is that all the experts must still be "
     "resident in memory.",
     ["many expert subnetworks per layer",
      "a router activates only a few per token",
      "compute per token is far below total parameters",
      "all experts still occupy memory"],
     []),

    (3, "why models are confidently wrong",
     "Why can a model be more confident when it is wrong?",
     "Because its confidence reflects how typical the text looks, not whether it "
     "is true. A plausible-sounding fabrication about a rare topic can be exactly "
     "the pattern the model has seen most, so it scores high while being wrong - "
     "which is why token probability is a poor confidence signal.",
     ["confidence reflects textual plausibility, not truth",
      "fluent fabrication can score highly",
      "rare topics are where this is worst",
      "token probability is a poor signal of correctness"],
     []),
]
