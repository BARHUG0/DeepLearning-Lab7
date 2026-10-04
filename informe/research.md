# Embeddings: API research and methodological decisions

## PyTorch tools

`nn.Embedding(num_embeddings, embedding_dim)` stores one trainable row per token
ID. `padding_idx` fixes a reserved row and prevents its gradient updates.
`sparse=True` produces sparse gradients, useful for large vocabularies, but
requires a compatible optimizer such as SGD or SparseAdam. Two independent
tables represent SGNS input words and output contexts.

`nn.Embedding.from_pretrained(weights, freeze=True)` initializes a table from an
existing two-dimensional tensor. `freeze=False` enables fine-tuning. The
pretrained dimension and row-to-token mapping must be preserved.

`nn.EmbeddingBag` aggregates variable-length sequences without constructing the
full intermediate embedding tensor. `mode="mean"` averages nonpadding rows,
`"sum"` adds them, and `"max"` takes componentwise maxima. `offsets` identifies
where each document begins in a flattened ID array. With
`include_last_offset=True`, a final offset gives the flattened array length.
The classifier uses mean aggregation followed by a linear layer, ReLU, dropout,
and a four-class output layer. Padding ID zero is ignored. A document containing
only unknown tokens receives a zero aggregate instead of an undefined mean.

## Gensim tools

`Word2Vec` uses `sg=1` for skip-gram, `vector_size` for the vector dimension,
`window` for the maximum context distance, `negative` for negative samples,
`sample` for subsampling, `min_count` for vocabulary filtering, `alpha` and
`min_alpha` for the learning-rate schedule, and `epochs` for corpus passes.
`ns_exponent=0.75` defines the smoothed negative distribution.
`shrink_windows=False` matches the custom fixed context window.
`build_vocab_from_freq` explicitly constructs the shared vocabulary.
`train(..., compute_loss=True)` exposes the accumulated training loss, which is
not directly interchangeable with the custom mean loss per positive pair.

`KeyedVectors` retains vectors without the training state. `add_vectors`
constructs an evaluation model, `save` and `load` persist it, and
`save_word2vec_format` exports text vectors. `most_similar` performs normalized
cosine retrieval; `similarity` compares two words. The assignment supplies
benchmarks through `gensim.test.utils.datapath`. This implementation verifies
their availability and uses explicit shared-question filtering rather than
allowing separate coverage denominators.

`gensim.downloader.load("glove-wiki-gigaword-100")` loads the original 100D
Wikipedia/Gigaword vectors. These original vectors are never modified.
Classification fine-tuning occurs in a separate PyTorch table.

## Learning objectives

CBOW predicts a central word from aggregated context. Skip-gram predicts
contexts from one central word. SGNS replaces a full-vocabulary softmax with
binary positive-versus-noise objectives:

$$
\mathcal{L} =
-\log \sigma(u_c^\top v_o)
-\sum_{j=1}^{K}\log \sigma(-u_c^\top v_{n_j}).
$$

Input vectors $u$ and output vectors $v$ have different roles. Using one table
would impose symmetry not required by the prediction task. The exported model
uses the input table; the output table remains in training checkpoints.

Word2Vec learns through local predictive examples. GloVe instead fits weighted
logarithms of global word co-occurrence counts. This comparison does not isolate
the method because the GloVe training corpus is much larger and separately
tokenized.

## Preprocessing and fair comparison

The tokenizer lowercases text, standardizes curly apostrophes, repairs
WikiText's `@-@`, `@,@`, and `@.@` markers, preserves internal apostrophes and
numeric forms, and drops stand-alone punctuation. Hyphenated compounds split
into words. Headings are excluded from body-token counts and training.
Sentence boundaries use punctuation followed by whitespace. Abbreviations can
therefore be split imperfectly, a disclosed limitation rather than an assumed
linguistically complete sentence segmenter.

Lowercasing matches the uncased GloVe vocabulary. Numeric strings are retained
rather than mapped to an artificial token absent from GloVe. Punctuation
differences and contractions are quantified through coverage and the tokenizer
comparison. AG News reuses exactly this tokenizer and normalization.

The vocabulary uses frequency at least five in the first 20-million-token
subset. It remains fixed for corpus-size and hyperparameter comparisons.
The smallest training subset extends slightly beyond 20 million normalized
tokens to retain at least 20 million tokens after vocabulary filtering.
Unknown corpus words are removed before contexts are constructed in both
training implementations. Negative draws follow retained-vocabulary counts
raised to the power of 0.75. Frequent words are independently retained with

$$
p_{\mathrm{keep}}(w)=
\min\left(1,
\left[\sqrt{f(w)/t}+1\right]t/f(w)\right).
$$

This reduces dominance by function words, decreases pair counts, and increases
the relative contribution of informative rarer words.

SGNS uses summed minibatch gradients with SGD. This preserves the per-pair
learning-rate scale but differs from gensim's sequential asynchronous updates.
Custom negative samples can coincide with the positive context; gensim skips
those collisions. Multithreaded gensim is not bitwise deterministic.

## Selection, arithmetic, and held-out evaluation

All embedding configurations are selected by 3CosAdd accuracy on one fixed
shared question set, with WordSim-353 Spearman as the tie-breaker.
SimLex-999 is not read during selection. Each benchmark reports coverage.

3CosAdd normalizes each source vector, constructs $b-a+c$, normalizes that
result, and ranks candidate cosine similarities. The three query words are
excluded unless an experiment explicitly disables exclusion.
3CosMul uses shifted cosine similarities:

$$
\operatorname{score}(x)=
\frac{[1+\cos(x,b)][1+\cos(x,c)]}
{2[1+\cos(x,a)]+\epsilon}.
$$

The implemented denominator adds $10^{-6}$ after shifting cosines to $[0,1]$.
Individual analogy ranks cover the model's entire vocabulary; systematic
benchmark ranks use the fixed 30,000-word intersection.

The absent assignment section 5.3 is operationalized as the cosine between
raw difference vectors $v_b-v_a$ and $v_d-v_c$ for independent examples of the
same relationship. Shared-word comparisons are excluded. Category counts and
the overall mean accompany the result.

Classifier selection uses only a seeded stratified training/validation split.
Training fractions are nested within each class. Each selected initialization
and training fraction is evaluated once on official test, with results cached.
The full-data selected models also receive confusion matrices.

## Sources

1. PyTorch, Embedding:
   <https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html>
2. PyTorch, EmbeddingBag:
   <https://docs.pytorch.org/docs/2.13/generated/torch.nn.EmbeddingBag.html>
3. Gensim, Word2Vec:
   <https://radimrehurek.com/gensim/models/word2vec.html>
4. Mikolov et al., Distributed Representations of Words and Phrases, 2013:
   <https://arxiv.org/abs/1310.4546>
5. Pennington et al., GloVe: Global Vectors for Word Representation, 2014:
   <https://nlp.stanford.edu/pubs/glove.pdf>
6. Original GloVe models, corpus and vocabulary metadata:
   <https://nlp.stanford.edu/projects/glove/>
7. Hill et al., SimLex-999:
   <https://fh295.github.io/simlex.html>
8. Levy and Goldberg, Linguistic Regularities in Sparse and Explicit Word
   Representations, 2014:
   <https://aclanthology.org/W14-1618/>
