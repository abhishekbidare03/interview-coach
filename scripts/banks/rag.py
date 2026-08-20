r"""Retrieval and RAG.

Split out of `genai` rather than duplicated. GenAI owns the model — decoding,
adaptation, serving. This owns everything on the retrieval side, and it owns it
from the introductory question upward so one topic tells the whole story.

The bias is towards the things that actually break a RAG system in production,
because they are not the things people prepare for. Almost nobody's RAG problem
is the language model; it is chunks that do not stand alone, an evaluation that
never measured retrieval separately from generation, or an index that quietly
went stale.
"""

from __future__ import annotations

ENTRIES = [

    # ------------------------------------------------------------------ 1 --- #

    (1, "what retrieval augmented generation is",
     "What is retrieval augmented generation?",
     "You search a document store for passages relevant to the question and put "
     "them into the prompt, so the model answers from supplied text rather than "
     "from memory. It lets a fixed model use current or private data, and lets "
     "you show where an answer came from.",
     ["relevant documents are retrieved for the query",
      "they are inserted into the prompt as context",
      "the model answers from supplied text rather than memory",
      "it supports fresh or private data without retraining"],
     ["What are the two stages that make up a RAG system?"]),

    (1, "when retrieval beats fine-tuning",
     "Why not just fine-tune the model on your documents instead?",
     "Because fine-tuning teaches behaviour, not facts. Knowledge baked into "
     "weights goes stale the moment a document changes, cannot be cited, and "
     "cannot be permission-checked per user. Retrieval updates by reindexing, "
     "which is minutes rather than a training run.",
     ["fine-tuning suits behaviour, retrieval suits knowledge",
      "facts in weights go stale and need retraining",
      "retrieved answers can be cited to a source",
      "retrieval can respect per-user access control"],
     []),

    (1, "chunking strategy",
     "Why are documents split into chunks rather than embedded whole?",
     "Two reasons. A single embedding of a long document averages everything it "
     "says, so it matches nothing specifically. And the context window is "
     "limited, so you want to pass the relevant section rather than the entire "
     "file.",
     ["one embedding of a long document blurs its content",
      "smaller chunks give more precise matching",
      "the context window limits how much can be passed",
      "chunks let you pass only the relevant part"],
     []),

    (1, "vector similarity",
     "How is similarity between a question and a chunk actually measured?",
     "Both are turned into vectors by the same embedding model, and similarity "
     "is the cosine of the angle between them — how aligned they point, "
     "independent of length. Closer vectors mean text used in more similar "
     "contexts.",
     ["both are embedded by the same model",
      "similarity is usually cosine of the angle between vectors",
      "it ignores vector magnitude",
      "closer vectors mean more similar meaning"],
     ["What breaks if the query and documents use different embedding models?"]),

    (1, "grounding and citations",
     "Why does a RAG system show citations?",
     "So the answer can be checked. The model can still misread or overstate "
     "what it was given, and a citation lets a reader go and verify the claim "
     "against the source. It also makes wrong answers diagnosable — you can see "
     "whether retrieval or generation failed.",
     ["it lets a reader verify the claim",
      "the model can still misstate what it retrieved",
      "it builds trust in the answer",
      "it makes failures diagnosable"],
     []),

    (1, "common RAG failure modes",
     "Your RAG system gives a confidently wrong answer. Where do you look first?",
     "Retrieval, because most of these are retrieval failures. If the passage "
     "containing the answer never arrived, the model falls back on memory and "
     "invents something. Checking what was actually retrieved tells you "
     "immediately which half of the system failed.",
     ["check what was actually retrieved first",
      "missing context makes the model fall back on memory",
      "most failures are retrieval failures, not generation failures",
      "it separates a retrieval problem from a generation problem"],
     []),

    (1, "how much context to pass",
     "Why not just retrieve the top fifty chunks and pass them all?",
     "Cost and accuracy both get worse. You pay for every token, latency rises, "
     "and the signal gets buried — the model has to find the relevant passage "
     "among forty-nine irrelevant ones, which it does less reliably than you "
     "would expect.",
     ["every extra chunk costs tokens and latency",
      "irrelevant context dilutes the relevant passage",
      "accuracy can fall as context grows",
      "there is a limited context window"],
     []),

    # ------------------------------------------------------------------ 2 --- #

    (2, "chunk size tradeoffs",
     "What goes wrong if chunks are too small, or too large?",
     "Too small and a chunk loses the context that makes it meaningful — a "
     "sentence referring to 'this policy' with the policy name in the previous "
     "paragraph. Too large and the embedding averages several topics, so it "
     "matches everything weakly and nothing well.",
     ["small chunks lose the context that makes them meaningful",
      "references to earlier text become unresolvable",
      "large chunks average several topics into one embedding",
      "large chunks match weakly and waste context"],
     ["So how do you pick a size?"]),

    (2, "chunk overlap",
     "Why do people overlap their chunks?",
     "Because a hard split lands in the middle of a sentence or an explanation "
     "roughly as often as not, and both halves become useless. Overlapping by a "
     "sentence or two means whichever chunk is retrieved carries the whole "
     "thought. It costs some duplicated storage.",
     ["a hard boundary can split a sentence or idea",
      "overlap ensures the whole thought is in one chunk",
      "it improves the chance of retrieving complete context",
      "the cost is duplicated storage and possible duplicate hits"],
     []),

    (2, "chunking strategy",
     "How would you split a document that has headings and sections?",
     "On its own structure, not on a character count. Sections and paragraphs "
     "are boundaries the author already put there, and they usually correspond "
     "to a single idea. I would also carry the heading into each chunk, so a "
     "chunk retrieved alone still says what it is about.",
     ["split on the document's own structure",
      "sections and paragraphs are natural idea boundaries",
      "prefer structure over a fixed character count",
      "keep the heading with the chunk text"],
     []),

    (2, "lexical versus semantic retrieval",
     "What does keyword search do better than vector search?",
     "Exact matches. Product codes, error numbers, names, rare technical terms — "
     "an embedding blurs those into whatever it thinks they resemble, while "
     "lexical search matches them precisely. It also handles words the embedding "
     "model never saw in training.",
     ["exact matches on identifiers and codes",
      "rare or novel terms an embedding blurs",
      "it needs no training on the corpus",
      "results are easy to explain"],
     ["So why not just use keyword search?"]),

    (2, "hybrid search",
     "What is hybrid search?",
     "Running lexical and vector retrieval together and combining their results. "
     "Each covers the other's weakness — lexical catches exact terms, vector "
     "catches paraphrase — and merging them, usually by fusing the two rankings, "
     "beats either alone on most real corpora.",
     ["it runs lexical and vector retrieval together",
      "the two candidate sets are merged",
      "each covers the other's failure mode",
      "it usually outperforms either method alone"],
     []),

    (2, "embedding models for retrieval",
     "How does a vector search find the relevant chunks at query time?",
     "Every chunk is embedded once, up front, and stored in an index. The "
     "question is embedded with the same model, and the index returns the "
     "nearest vectors. Only the query is embedded at request time, which is why "
     "the search is fast.",
     ["chunks are embedded once and indexed in advance",
      "the query is embedded with the same model",
      "the index returns the nearest vectors",
      "only the query embedding is computed at request time"],
     []),

    (2, "reranking",
     "What does a reranker add on top of vector search?",
     "Accuracy at the top of the list. Vector search compares two embeddings "
     "computed independently, which only captures coarse similarity. A reranker "
     "reads the query and the chunk together and scores them jointly, so it "
     "catches relevance the embedding missed.",
     ["it scores query and chunk together, not independently",
      "joint scoring is more accurate than embedding similarity",
      "it reorders the top candidates",
      "it is applied to a shortlist, not the whole corpus"],
     ["Why not rerank the whole corpus?"]),

    (2, "metadata filtering",
     "What is metadata filtering in a retrieval system?",
     "Restricting the search to chunks matching structured criteria — a date "
     "range, a document type, a product line — before or during the vector "
     "search. It is how you answer 'what changed this year' without hoping the "
     "embedding understood the year.",
     ["restricts search using structured fields",
      "applied alongside or before similarity search",
      "handles constraints embeddings represent poorly",
      "examples are date, source or document type"],
     []),

    (2, "evaluating retrieval separately",
     "Why evaluate retrieval separately from the final answer?",
     "Because otherwise you cannot tell which half is broken. A bad answer might "
     "be a retrieval miss or a generation failure, and the fixes are completely "
     "different. Measuring whether the right chunk was in the results tells you "
     "which one to work on.",
     ["a bad answer can come from either stage",
      "the fixes for each are different",
      "measure whether the right chunk was retrieved",
      "it localises the failure before you tune anything"],
     []),

    (2, "recall at k and precision at k",
     "What does recall at k measure in retrieval?",
     "The fraction of the relevant documents that appear in the top k results. "
     "For RAG it is the metric that matters most in the first stage: if the "
     "answer is not in what you retrieved, nothing downstream can recover it.",
     ["the share of relevant documents found in the top k",
      "it measures the first stage's coverage",
      "if the answer is absent, generation cannot recover",
      "the reranker is what turns recall into precision"],
     []),

    (2, "index freshness and updates",
     "A document changes. What has to happen for RAG to notice?",
     "Its chunks have to be re-chunked, re-embedded and replaced in the index — "
     "and the old ones deleted, which is the step people forget. A stale chunk "
     "left behind is worse than a missing one, because it gets retrieved and "
     "cited with confidence.",
     ["the document must be re-chunked and re-embedded",
      "the index entries must be updated",
      "stale chunks must be deleted, not just added over",
      "a stale chunk is retrieved and cited as if current"],
     []),

    # ------------------------------------------------------------------ 3 --- #

    (3, "approximate nearest neighbour search",
     "Why is vector search approximate rather than exact?",
     "Exact search compares the query to every vector, which is linear in corpus "
     "size and far too slow past a few hundred thousand. Approximate methods "
     "search a graph or a set of clusters instead, trading a small amount of "
     "recall for orders of magnitude in speed.",
     ["exact search must compare against every vector",
      "that is too slow at scale",
      "approximate methods search a graph or clusters",
      "they trade a little recall for a large speed gain"],
     ["What knob controls that tradeoff?"]),

    (3, "cross-encoders versus bi-encoders",
     "What's the difference between a bi-encoder and a cross-encoder?",
     "A bi-encoder embeds query and document separately, so documents can be "
     "indexed ahead of time and search is fast. A cross-encoder feeds both "
     "through the model together, which is much more accurate because the two "
     "can attend to each other, and far too slow to precompute.",
     ["a bi-encoder embeds query and document independently",
      "that allows the corpus to be indexed in advance",
      "a cross-encoder processes both together",
      "cross-encoders are more accurate but cannot be precomputed"],
     []),

    (3, "query rewriting",
     "Why rewrite the user's query before retrieving?",
     "Because the raw question is often a poor search key. Follow-ups depend on "
     "the conversation — 'what about the second one' retrieves nothing — and "
     "short questions lack the vocabulary of the documents. Rewriting resolves "
     "references and expands the query into something searchable.",
     ["a raw question may be a poor search key",
      "conversational follow-ups lack standalone context",
      "rewriting resolves references from the conversation",
      "it can expand terms to match document vocabulary"],
     []),

    (3, "context ordering and lost in the middle",
     "Does the order of retrieved chunks in the prompt matter?",
     "Yes, more than people expect. Models attend most reliably to the beginning "
     "and end of a long context and are measurably weaker in the middle. So the "
     "strongest evidence should go at the edges, and stuffing more chunks in can "
     "push the good one into the dead zone.",
     ["position within the context affects how well it is used",
      "the beginning and end are attended to more reliably",
      "material in the middle is used less well",
      "put the strongest evidence at the edges"],
     []),

    (3, "detecting ungrounded answers",
     "How would you detect that an answer is not supported by the retrieved text?",
     "Check the claim against the context rather than trusting the model. That "
     "can be an entailment model, or a second model asked whether each statement "
     "follows from the passages, or requiring citations and verifying the cited "
     "span actually says it. All three catch the same failure.",
     ["verify claims against the retrieved passages",
      "use an entailment or faithfulness check",
      "require citations and validate the cited span",
      "do not rely on the model's own confidence"],
     []),

    (3, "handling multi-hop questions",
     "Why does a question needing two documents often fail?",
     "Because one retrieval pass matches the question as a whole, and neither "
     "document looks like the whole question. The second fact is usually only "
     "findable once you know the first, so a single pass cannot get there. It "
     "needs a decomposed or iterative retrieval.",
     ["a single retrieval matches the question as a whole",
      "neither document individually matches it well",
      "the second fact depends on knowing the first",
      "it needs decomposition or iterative retrieval"],
     []),

    (3, "building a retrieval evaluation set",
     "How would you build an evaluation set for retrieval?",
     "Take real questions, and for each one record which chunks actually contain "
     "the answer. That is the labour, and there is no way around it. Real "
     "traffic gives the right distribution; generating questions from your own "
     "chunks is a fast start but tends to produce questions that are too easy.",
     ["collect real user questions",
      "label which chunks genuinely contain the answer",
      "use real traffic so the distribution is right",
      "generated questions are a starting point but skew easy"],
     []),

    (3, "access control in retrieval",
     "How do you stop a RAG system leaking documents a user cannot see?",
     "Filter at retrieval time, using the asker's permissions as a constraint on "
     "the search. Filtering after retrieval is too late if anything was already "
     "summarised, and asking the model to respect permissions in the prompt is "
     "not a control at all.",
     ["apply permission filters during retrieval",
      "the filter must use the asking user's identity",
      "post-filtering is too late once content is in the prompt",
      "prompt instructions are not an access control"],
     []),

    (3, "deduplication in the corpus",
     "What happens when your corpus contains many near-duplicate documents?",
     "The top results fill up with the same content repeated, so the effective "
     "context is one passage taking five slots. It crowds out genuinely "
     "different evidence and makes multi-source answers impossible. You "
     "deduplicate at index time, or diversify at retrieval time.",
     ["duplicates crowd out the top results",
      "effective context shrinks to one distinct passage",
      "it prevents drawing on multiple sources",
      "fix by deduplicating the index or diversifying results"],
     []),

    (3, "handling tables and structured documents",
     "Why do tables and spreadsheets retrieve badly?",
     "Because chunking flattens them. A row loses its column headers, numbers "
     "lose the thing they measure, and the embedding of a grid of digits carries "
     "almost no meaning. They usually need converting to text that states the "
     "relationship, or querying structurally rather than by similarity.",
     ["chunking separates cells from their headers",
      "numbers lose the meaning their labels gave them",
      "embeddings represent numeric content poorly",
      "convert to descriptive text or query structurally instead"],
     []),

    # ------------------------------------------------------------------ 4 --- #

    (4, "vector index tradeoffs",
     "What do you trade when tuning an approximate index?",
     "Recall against latency and memory. Searching more of the graph or more "
     "clusters finds more true neighbours and costs more time; a denser index "
     "is faster to search but larger in memory and slower to build. The right "
     "point is set by measuring recall against exact search.",
     ["higher recall costs query latency",
      "index structure trades memory against speed",
      "build time rises with index quality",
      "tune by measuring recall against exact search"],
     []),

    (4, "retrieval latency budget",
     "Where does the time go in a RAG request, and what would you cut?",
     "Embedding the query, the index search, the reranker, then generation — and "
     "generation usually dominates, with the reranker second. I would measure "
     "before cutting, then shrink the reranked shortlist, cache embeddings for "
     "repeated queries, and overlap retrieval with prompt assembly.",
     ["named the stages: embed, search, rerank, generate",
      "generation usually dominates the total",
      "reranking cost scales with shortlist size",
      "cache or parallelise rather than guessing"],
     []),

    (4, "evaluating retrieval separately",
     "Retrieval recall is 95 percent but answers are still wrong. What now?",
     "Then the problem is downstream of retrieval. Either the right chunk is "
     "present but buried among distractors, or it is present and the model is "
     "not using it. I would check where in the context it landed, cut the "
     "number of chunks, and test whether the answer improves with only the gold "
     "passage.",
     ["the failure is in generation, not retrieval",
      "the right chunk may be buried among distractors",
      "check its position within the context",
      "test with only the correct passage to isolate the cause"],
     []),

    (4, "caching in a retrieval pipeline",
     "What can you cache in a RAG system, and what is the risk?",
     "Query embeddings, retrieval results for repeated questions, and whole "
     "answers. The risk is staleness — a cached answer keeps being served after "
     "the underlying document changed, and it is invisible because the system "
     "looks like it is working. Cache keys have to include an index version.",
     ["query embeddings and retrieval results can be cached",
      "whole answers can be cached for repeat questions",
      "the main risk is serving stale results after reindexing",
      "invalidate on index version or content change"],
     []),

    (4, "common RAG failure modes",
     "Users say the system 'does not know' things that are in the documents. Diagnose it.",
     "That phrasing points at retrieval, not the model. I would check whether "
     "those documents are indexed at all, whether their chunks are retrievable "
     "for the words users actually use, and whether a metadata or permission "
     "filter is silently excluding them. Vocabulary mismatch is the usual "
     "culprit.",
     ["confirm the documents are actually in the index",
      "check chunk retrievability for the user's phrasing",
      "check filters or permissions are not excluding them",
      "vocabulary mismatch between query and document is common"],
     []),

    # ------------------------------------------------------------------ 5 --- #

    (5, "building a retrieval evaluation set",
     "How would you know a change to chunking made things better?",
     "By holding the rest fixed and measuring retrieval alone on a labelled set "
     "— recall at k before and after, on the same questions. End-to-end answer "
     "quality is too noisy and too slow to attribute. I would also check it did "
     "not regress on the queries that were already working.",
     ["measure retrieval in isolation, not end to end",
      "use a fixed labelled question set",
      "compare recall at k before and after",
      "check for regressions on previously working queries"],
     []),

    (5, "hybrid search",
     "Design retrieval for a corpus mixing policy documents, code, and chat logs.",
     "They do not share a retrieval strategy, so I would not force one. Code and "
     "identifiers need lexical matching, policy prose suits dense retrieval, chat "
     "logs need conversation-aware chunking and heavy time weighting. Route or "
     "run per-source retrievers and merge, keeping the source as metadata so it "
     "can be filtered and cited.",
     ["different content types need different retrieval strategies",
      "lexical matching for code and identifiers",
      "dense retrieval for prose",
      "merge per-source results and keep source metadata"],
     []),

    (5, "index freshness and updates",
     "How would you keep an index fresh over millions of frequently-edited documents?",
     "Incrementally, driven by change events rather than by periodic full "
     "rebuilds — reindex what changed, and delete what was removed. Full rebuilds "
     "are too slow and leave a long staleness window. I would track indexing lag "
     "as a monitored metric, because staleness is invisible in the answers.",
     ["reindex incrementally on change events",
      "avoid periodic full rebuilds at that scale",
      "handle deletions explicitly, not just updates",
      "monitor indexing lag as an explicit metric"],
     []),
]
