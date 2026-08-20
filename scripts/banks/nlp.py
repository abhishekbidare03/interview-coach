r"""Natural Language Processing.

Deliberately spans the whole history rather than starting at transformers. A
screen still asks what TF-IDF is, and the candidates who only know the last two
years tend to fall over on tokenisation and evaluation — the unglamorous parts
that actually break production systems.
"""

from __future__ import annotations

ENTRIES = [

    # ------------------------------------------------------------------ 1 --- #

    (1, "tokenisation",
     "What is tokenisation?",
     "Splitting text into the units a model actually operates on. Those units "
     "might be words, characters, or subword pieces. Every downstream step "
     "depends on it, so a tokeniser that splits differently at training and "
     "serving time quietly breaks the whole system.",
     ["splitting text into units the model processes",
      "units may be words, subwords or characters",
      "it is the first step in any text pipeline",
      "the same tokeniser must be used at training and inference"],
     []),

    (1, "bag of words and TF-IDF",
     "What does a bag-of-words representation throw away?",
     "Word order, and with it grammar and most of the meaning that depends on "
     "structure. It keeps only which words appeared and how often, so 'the dog "
     "bit the man' and 'the man bit the dog' become identical vectors.",
     ["it discards word order",
      "it keeps only word counts or presence",
      "sentences with different meaning can be identical",
      "syntax and structure are lost"],
     []),

    (1, "stemming versus lemmatisation",
     "What's the difference between stemming and lemmatisation?",
     "Stemming chops affixes off with rules, so it is fast but can produce "
     "non-words. Lemmatisation uses a vocabulary and the word's part of speech "
     "to return the real dictionary form. Lemmatisation is more accurate and "
     "considerably slower.",
     ["stemming strips affixes using rules",
      "stemming can produce non-words",
      "lemmatisation returns a real dictionary form",
      "lemmatisation is more accurate but slower"],
     []),

    (1, "word embeddings",
     "What is a word embedding?",
     "A dense vector representing a word, learned so that words used in similar "
     "contexts end up near each other. Unlike a one-hot vector it is compact and "
     "carries similarity — the geometry of the space encodes something about "
     "meaning.",
     ["a dense vector representation of a word",
      "learned from the contexts a word appears in",
      "similar words are close in the space",
      "far more compact than one-hot encoding"],
     []),

    (1, "stop words",
     "What are stop words, and should you always remove them?",
     "Very common words like 'the' and 'is' that carry little topical meaning. "
     "Removing them helped bag-of-words models by cutting noise. For modern "
     "contextual models it is usually harmful, because those words carry "
     "grammatical signal the model uses.",
     ["very common words with little topical content",
      "removed to reduce noise in count-based models",
      "removal can destroy meaning, as in negation",
      "modern contextual models generally keep them"],
     []),

    (1, "text classification",
     "How would you build a simple text classifier?",
     "Start with a strong baseline: TF-IDF features into logistic regression. It "
     "trains in seconds, is easy to inspect, and is surprisingly hard to beat on "
     "small datasets. Only once that is measured would I try a fine-tuned "
     "transformer and see whether the gain justifies the cost.",
     ["begin with a simple, strong baseline",
      "TF-IDF features with a linear model",
      "measure before reaching for a larger model",
      "fine-tuned transformers usually win on larger datasets"],
     []),

    (1, "sequence labelling and named entity recognition",
     "What is named entity recognition?",
     "Finding and classifying the spans in text that refer to real-world things "
     "— people, organisations, locations, dates. It is a sequence labelling "
     "problem, because the label belongs to a span of tokens rather than to the "
     "document.",
     ["identifying spans that refer to real-world entities",
      "classifying them into types such as person or location",
      "it is a token-level sequence labelling task",
      "entities can span several tokens"],
     []),

    # ------------------------------------------------------------------ 2 --- #

    (2, "bag of words and TF-IDF",
     "What problem does TF-IDF solve that raw word counts do not?",
     "Raw counts are dominated by words that appear everywhere, which carry no "
     "discriminating information. TF-IDF scales each count down by how many "
     "documents contain the word, so terms common across the corpus are damped "
     "and terms distinctive to a document stand out.",
     ["raw counts are dominated by ubiquitous words",
      "inverse document frequency down-weights common terms",
      "distinctive terms get higher weight",
      "it improves retrieval and classification over raw counts"],
     []),

    (2, "subword tokenisation and byte pair encoding",
     "Why do modern models use subword tokenisation?",
     "It is the compromise between a word vocabulary, which cannot handle "
     "anything unseen, and characters, which make sequences far too long. "
     "Subwords keep frequent words whole and break rare ones into known pieces, "
     "so nothing is ever out of vocabulary.",
     ["word-level vocabularies cannot handle unseen words",
      "character-level sequences become very long",
      "frequent words stay whole, rare words are split",
      "it eliminates out-of-vocabulary tokens"],
     ["Roughly how does byte pair encoding build that vocabulary?"]),

    (2, "word2vec versus GloVe",
     "How does word2vec learn its embeddings?",
     "By training a shallow network on a fake task: predict a word from its "
     "neighbours, or the neighbours from the word. Nobody wants the predictions "
     "— the weights learned along the way are the embeddings, and they capture "
     "context because that is what the task rewarded.",
     ["it trains on a prediction task over context windows",
      "skip-gram predicts context from a word, CBOW the reverse",
      "the learned weights are the embeddings",
      "the prediction task itself is discarded"],
     []),

    (2, "static versus contextual embeddings",
     "What do contextual embeddings give you that word2vec does not?",
     "A different vector for a word depending on the sentence it is in. Word2vec "
     "assigns 'bank' one vector averaging every sense of it. A contextual model "
     "produces one vector for a river bank and another for a savings bank, "
     "because it encodes the surrounding words.",
     ["static embeddings give one vector per word type",
      "that vector blends all senses of the word",
      "contextual embeddings depend on the surrounding text",
      "polysemous words get different vectors per context"],
     []),

    (2, "the encoder-decoder architecture",
     "What does an encoder-decoder architecture do?",
     "The encoder reads the whole input and builds a representation of it. The "
     "decoder generates the output one token at a time, conditioned on that "
     "representation and on what it has produced so far. It is the standard "
     "shape for translation and summarisation.",
     ["the encoder builds a representation of the input",
      "the decoder generates output tokens one at a time",
      "generation is conditioned on the encoded input",
      "it suits translation and summarisation"],
     []),

    (2, "semantic search and vector similarity",
     "How does semantic search differ from keyword search?",
     "Keyword search matches the literal terms, so a query phrased differently "
     "from the document finds nothing. Semantic search embeds both into a vector "
     "space and ranks by similarity, so it can match meaning across different "
     "wording — at the cost of sometimes matching things that merely feel related.",
     ["keyword search matches literal terms",
      "semantic search compares embeddings",
      "it matches meaning across different wording",
      "it can retrieve loosely related but wrong results"],
     ["Why do production systems often run both?"]),

    (2, "evaluating generation with BLEU and ROUGE",
     "What do BLEU and ROUGE measure, and what is their weakness?",
     "Both compare generated text to reference text by overlapping n-grams — "
     "BLEU leans towards precision for translation, ROUGE towards recall for "
     "summarisation. The weakness is that a correct paraphrase using different "
     "words scores badly, so they measure surface overlap rather than meaning.",
     ["both measure n-gram overlap with a reference",
      "BLEU is precision-oriented, ROUGE recall-oriented",
      "a valid paraphrase can score poorly",
      "they measure surface form rather than meaning"],
     []),

    (2, "perplexity",
     "What is perplexity?",
     "A measure of how surprised a language model is by text it did not train "
     "on. Lower means the model assigned higher probability to what actually "
     "came next. It is useful for comparing language models on the same data, "
     "but it does not tell you whether output is useful.",
     ["it measures how well a model predicts unseen text",
      "lower perplexity means better prediction",
      "it is derived from the likelihood of the held-out text",
      "it does not measure usefulness or factual accuracy"],
     []),

    # ------------------------------------------------------------------ 3 --- #

    (3, "positional encoding",
     "Why do transformers need positional encodings?",
     "Because self-attention treats the input as a set — it computes the same "
     "thing regardless of the order of the tokens. Without an explicit signal "
     "for position, the model could not distinguish two sentences with the same "
     "words in a different order.",
     ["self-attention is permutation invariant",
      "it has no inherent notion of token order",
      "position must be injected explicitly",
      "otherwise reordered sentences look identical"],
     []),

    (3, "multi-head attention",
     "What does multi-head attention buy you over a single attention head?",
     "Each head learns its own projection, so different heads can attend to "
     "different kinds of relationship at once — one tracking syntax, another "
     "coreference. A single head has to average all of that into one attention "
     "pattern, and the average is worse than the parts.",
     ["each head learns its own projection",
      "different heads capture different relationships",
      "they attend in parallel and are concatenated",
      "a single head must average all relations together"],
     []),

    (3, "BERT and masked language modelling",
     "How does BERT's training objective differ from GPT's?",
     "BERT masks tokens in the middle and predicts them using context from both "
     "sides, which makes it good at understanding but not at generating. GPT "
     "predicts the next token from the left only, which makes generation natural "
     "but means each token sees no future context.",
     ["BERT predicts masked tokens using both directions",
      "GPT predicts the next token from left context only",
      "bidirectional context suits understanding tasks",
      "the autoregressive objective suits generation"],
     []),

    (3, "the pretrain then fine-tune recipe",
     "Why does the pretrain-then-fine-tune recipe work so well?",
     "Pretraining uses enormous amounts of unlabelled text to learn general "
     "language structure, which is the expensive part. Fine-tuning then adapts "
     "that to a specific task with a comparatively tiny labelled set, because "
     "the model only has to learn the task, not the language.",
     ["pretraining uses abundant unlabelled text",
      "it learns general language representations",
      "fine-tuning adapts them with little labelled data",
      "the expensive general learning is done once and reused"],
     []),

    (3, "sentence embeddings",
     "Why not just average word embeddings to represent a sentence?",
     "You can, and it is a reasonable baseline, but averaging discards order and "
     "lets common words dominate. Two sentences with opposite meaning but shared "
     "vocabulary land in nearly the same place. Models trained directly on "
     "sentence similarity do much better.",
     ["averaging discards word order",
      "frequent words dominate the average",
      "opposite meanings can produce similar vectors",
      "models trained for sentence similarity perform better"],
     []),

    (3, "handling long documents",
     "How would you classify a document longer than the model's context window?",
     "Either shorten the input or aggregate over pieces. Truncation is the cheap "
     "option and often fine when the signal is at the start. Otherwise split "
     "into chunks, encode each, and pool the results — or use a model built for "
     "long contexts if the signal is genuinely spread throughout.",
     ["truncate when the signal is concentrated early",
      "split into chunks and encode each",
      "pool or aggregate the chunk representations",
      "or use a long-context architecture"],
     []),

    (3, "out-of-vocabulary words",
     "How do modern models handle a word they have never seen?",
     "Subword tokenisation means they never truly see an unknown word — it gets "
     "split into known pieces, right down to bytes if necessary. The model then "
     "composes a representation from those pieces, which works well for "
     "morphology and badly for genuinely novel proper nouns.",
     ["subword tokenisers split unknown words into known pieces",
      "byte-level fallback guarantees coverage",
      "the representation is composed from the pieces",
      "it works better for morphology than for novel names"],
     []),

    (3, "text preprocessing pitfalls",
     "What preprocessing mistakes quietly damage an NLP pipeline?",
     "Lowercasing when case carries meaning, stripping punctuation that marks "
     "sentence boundaries or negation, removing stop words before a contextual "
     "model, and — most damaging — preprocessing differently at training and "
     "serving time.",
     ["lowercasing when case is meaningful",
      "removing punctuation that carries structure or negation",
      "stripping stop words before a contextual model",
      "inconsistent preprocessing between training and serving"],
     []),

    # ------------------------------------------------------------------ 4 --- #

    (4, "attention in sequence models",
     "Why was attention added to sequence-to-sequence models originally?",
     "Because the encoder had to compress an entire sentence into one fixed "
     "vector, and that bottleneck got worse as sentences got longer. Attention "
     "let the decoder look back at every encoder state and weight them per "
     "output token, removing the bottleneck.",
     ["a fixed-size vector was an information bottleneck",
      "performance degraded on longer sequences",
      "attention lets the decoder access all encoder states",
      "weights are computed per output token"],
     []),

    (4, "bias in language models",
     "Where does bias in a language model come from, and what can you do?",
     "From the training corpus, which reflects who wrote it and what they wrote. "
     "It shows up as skewed associations and uneven quality across groups. "
     "Mitigations exist at every stage — curating data, tuning the objective, "
     "filtering outputs — but the first requirement is measuring it per group "
     "rather than reporting one aggregate number.",
     ["it originates in the training corpus",
      "it appears as skewed associations or uneven quality",
      "measure performance separately per group",
      "mitigate through data curation, training or output filtering"],
     []),

    (4, "multilingual models",
     "What is hard about a single model serving many languages?",
     "Capacity is shared, so adding languages can degrade the ones already "
     "there. Data is wildly unbalanced, so low-resource languages get little "
     "signal. And the tokeniser is usually fitted mostly on high-resource text, "
     "so other languages consume far more tokens for the same content.",
     ["model capacity is shared across languages",
      "training data is highly imbalanced",
      "low-resource languages perform worse",
      "tokenisation is inefficient for under-represented scripts"],
     []),

    (4, "evaluating generation with BLEU and ROUGE",
     "How would you evaluate a summarisation system properly?",
     "Overlap metrics only as a cheap regression check. Beyond that I would "
     "measure the dimensions that actually matter separately — is it faithful to "
     "the source, does it cover the key points, is it readable — with human "
     "ratings on a sample and a model-based faithfulness check at scale. "
     "Hallucination will not show up in ROUGE at all.",
     ["overlap metrics are a weak proxy",
      "evaluate faithfulness to the source separately",
      "evaluate coverage of key content",
      "use human judgement on a sample, plus automated checks at scale"],
     []),

    # ------------------------------------------------------------------ 5 --- #

    (5, "semantic search and vector similarity",
     "How would you build search over a few million documents?",
     "A hybrid. Lexical retrieval catches exact terms, identifiers and rare "
     "words that embeddings blur; dense retrieval catches paraphrase. I would "
     "combine both candidate sets, rerank the top few hundred with a cross-"
     "encoder, and tune the cheap first stage on recall and the reranker on "
     "precision.",
     ["combine lexical and dense retrieval",
      "lexical handles exact and rare terms",
      "dense retrieval handles paraphrase",
      "rerank the merged candidates with a stronger model"],
     []),

    (5, "the pretrain then fine-tune recipe",
     "You have ten thousand labelled examples for a text task. What do you do?",
     "Fine-tune a pretrained encoder — that is comfortably enough data for it "
     "and it will beat anything trained from scratch. I would still build a "
     "TF-IDF baseline first to know what the gain actually is, hold out a proper "
     "test set, and check whether a smaller distilled model gets close enough to "
     "be worth the serving cost.",
     ["fine-tune a pretrained model rather than training from scratch",
      "build a simple baseline first for comparison",
      "ten thousand examples is enough for fine-tuning",
      "consider a smaller model for serving cost"],
     []),
    # --------------------------------------------------------------------- #
    # Added in the expansion pass: more of the ground floor and lower middle.
    # --------------------------------------------------------------------- #

    (1, "corpora",
     "What is a corpus, and why does its composition matter?",
     "The body of text a model is trained or evaluated on. Composition matters "
     "because everything the model knows about language comes from it - the "
     "domain, the register, the era and the demographics of whoever wrote it all "
     "end up baked into the model's behaviour.",
     ["a corpus is the body of text used for training or evaluation",
      "the model's knowledge comes entirely from it",
      "domain and register carry through to model behaviour",
      "gaps or skews in the corpus become model biases"],
     []),

    (1, "vectorising text",
     "Why does text have to be turned into numbers at all?",
     "Because models do arithmetic, and words are not numbers. Every NLP system "
     "starts by mapping text to a numeric representation - counts, or learned "
     "vectors - and the quality of that representation puts a ceiling on "
     "everything downstream.",
     ["models operate on numbers, not characters",
      "text is mapped to a numeric representation",
      "counts and learned embeddings are two approaches",
      "the representation limits downstream performance"],
     []),

    (1, "language models",
     "What is a language model, at its simplest?",
     "Something that assigns probabilities to sequences of words - in practice, "
     "predicting what comes next given what came before. Everything from an old "
     "n-gram model to a modern transformer is doing that same job at different "
     "levels of sophistication.",
     ["it assigns probability to sequences of text",
      "in practice it predicts the next unit given context",
      "n-gram models and transformers share this objective",
      "it needs no labelled data to train"],
     []),

    (1, "text normalisation",
     "What is text normalisation, and when can it hurt?",
     "Standardising text before processing - lowercasing, stripping punctuation, "
     "collapsing whitespace. It helps count-based methods by merging variants. "
     "It hurts when the thing you removed carried meaning: case distinguishes "
     "names, and punctuation carries negation and sentence structure.",
     ["standardising text so variants collapse together",
      "it reduces sparsity for count-based methods",
      "case can carry meaning, as with proper nouns",
      "punctuation can carry structure or negation"],
     []),

    (2, "vocabulary size tradeoffs",
     "What do you trade when choosing a tokeniser's vocabulary size?",
     "A large vocabulary keeps more words whole, so sequences are shorter and "
     "cheaper, but the embedding table grows and rare tokens are seen too seldom "
     "to learn well. A small one splits more aggressively, giving longer "
     "sequences but better-trained pieces.",
     ["a large vocabulary gives shorter sequences",
      "it also grows the embedding table",
      "rare tokens get too little training signal",
      "a small vocabulary means longer sequences to process"],
     []),

    (2, "negation",
     "Why do simple text models get negation wrong?",
     "Because bag-of-words style representations drop word order, and 'not good' "
     "keeps the strong positive word. Stop word removal often deletes the "
     "negation outright. Contextual models handle it far better because the "
     "representation of a word depends on what surrounds it.",
     ["order-free representations lose the scope of negation",
      "the sentiment-bearing word survives the negation",
      "stop word removal can delete the negation itself",
      "contextual models handle it much better"],
     []),

    (2, "zero-shot classification",
     "What is zero-shot text classification?",
     "Classifying into categories the model was never trained on, by describing "
     "the labels in natural language instead of learning them from examples. It "
     "is enormously useful when labels change or data is scarce, and usually "
     "weaker than a supervised model where you have real training data.",
     ["classifying into unseen categories",
      "labels are described in natural language",
      "no task-specific training examples are needed",
      "typically weaker than a supervised model with real data"],
     []),

    (2, "text augmentation",
     "How do you augment text data, and what makes it risky?",
     "Synonym replacement, back-translation, paraphrasing with a model. The risk "
     "is that text is fragile in a way images are not - flipping an image keeps "
     "its label, but swapping one word can invert the meaning entirely, so the "
     "augmented example is now mislabelled.",
     ["synonym substitution, back-translation or paraphrasing",
      "the transformation must preserve the label",
      "small word changes can invert meaning",
      "text is less robust to augmentation than images"],
     []),

    (2, "sequence length",
     "Why does sequence length matter so much in NLP?",
     "Because cost scales with it, and for attention it scales with the square of "
     "it. It also has a hard limit: anything past the model's maximum length is "
     "truncated, so a long document silently loses its ending unless you handle "
     "it deliberately.",
     ["compute and memory grow with sequence length",
      "attention cost grows quadratically",
      "models have a maximum length",
      "input past the limit is truncated, often silently"],
     []),

    (3, "named entity recognition in practice",
     "What makes named entity recognition hard on real text?",
     "Ambiguity and novelty. The same string is a person, a place or a company "
     "depending on context; entities span several tokens and the boundaries are "
     "arguable; and new names appear constantly, so the model must generalise "
     "from form and context rather than recall a list.",
     ["the same string can be different entity types",
      "context determines the correct type",
      "entity boundaries span multiple tokens and are ambiguous",
      "new entities appear constantly and cannot be memorised"],
     []),

    (3, "numbers and tokenisation",
     "Why are language models unreliable with numbers?",
     "Partly because tokenisation splits them arbitrarily - the same quantity can "
     "be one token or four depending on the digits - so there is no consistent "
     "representation of magnitude. And the training objective rewards plausible "
     "continuations, which for arithmetic is not the same as correct ones.",
     ["tokenisation splits numbers inconsistently",
      "there is no stable representation of magnitude",
      "the objective rewards plausibility, not correctness",
      "delegating to a tool is more reliable than generating"],
     []),

    (3, "domain shift in text",
     "A model trained on news does badly on customer support chats. Why?",
     "Different distribution in almost every respect - vocabulary, spelling, "
     "length, register, and the way meaning is carried. Chats are short, "
     "misspelled and full of domain shorthand the model never saw. It is domain "
     "shift, and the fix is in-domain data rather than a bigger model.",
     ["vocabulary and register differ between domains",
      "informal text has spelling and grammar variation",
      "the training distribution does not match the target",
      "in-domain data or fine-tuning is the fix"],
     []),

    (3, "multi-label classification",
     "How does multi-label text classification differ from multi-class?",
     "Multi-class picks exactly one label, so the outputs compete through a "
     "softmax. Multi-label allows any number, so each label gets its own "
     "independent probability and its own threshold. Using softmax for a "
     "multi-label problem forces a choice the task does not want.",
     ["multi-class assigns exactly one label",
      "multi-label allows several labels at once",
      "multi-label uses independent probabilities per label",
      "softmax is wrong for multi-label because outputs compete"],
     []),
]
