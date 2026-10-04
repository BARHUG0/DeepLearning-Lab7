import re
from collections import Counter

from .common import ARTIFACTS, FIGURES, RESULTS, SEED, directories, load_json, save_json

import numpy as np

WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)*|[0-9]+(?:[.,][0-9]+)*")
ARTICLE_PATTERN = re.compile(r"^\s*=\s+[^=].*?\s+=\s*$")
HEADING_PATTERN = re.compile(r"^\s*=.*=\s*$")
SENTENCE_PATTERN = re.compile(r"[.!?]\s+")
THRESHOLDS = (20_000_000, 30_000_000, 40_000_000)
MIN_COUNT = 5
SAMPLE = 1e-4
EXAMPLES = [
    "I can't believe it's already January.",
    "They're buying state-of-the-art computers.",
    "Dr. Smith lives in the U.S.A.",
    "The price is $12.50, not $1,250.",
    "King's horses aren't running.",
    "COVID-19 changed 2020–2021.",
    "Email alice@example.com or visit https://example.com.",
    "“Hello!” she said... then left.",
    "He'll re-enter the twenty-first competition.",
    "New York's capital isn't New York City.",
    "WikiText writes well @-@ known and 1 @,@ 000.",
]


def tokenize(text):
    text = text.lower().replace("’", "'")
    for source, target in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", ".")):
        text = text.replace(source, target)
    return WORD_PATTERN.findall(text)


def sentences(text):
    if not text.strip() or HEADING_PATTERN.match(text):
        return []
    return [tokens for part in SENTENCE_PATTERN.split(text) if (tokens := tokenize(part))]


def keep_probabilities(counts, sample=SAMPLE):
    counts = np.asarray(counts, dtype=np.float64)
    if sample <= 0:
        return np.ones_like(counts)
    frequencies = counts / counts.sum()
    return np.minimum(1.0, (np.sqrt(frequencies / sample) + 1) * sample / frequencies)


def pair_count(lengths, window):
    lengths = np.asarray(lengths, dtype=np.int64)
    return int(sum(2 * np.maximum(lengths - distance, 0).sum() for distance in range(1, window + 1)))


def compare_tokenizers():
    from nltk.tokenize import TreebankWordTokenizer

    library = TreebankWordTokenizer()
    return [
        {"text": text, "custom": tokenize(text), "nltk": library.tokenize(text.lower())}
        for text in EXAMPLES
    ]


def prepare():
    from datasets import load_dataset
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directories()
    if (RESULTS / "preprocessing.json").exists():
        print("Preprocessing already complete.", flush=True)
        return load_json(RESULTS / "preprocessing.json")
    corpus = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    full_counts = Counter()
    selected_counts = Counter()
    snapshots = {}
    split_stats = {}
    selected_tokens = 0
    selected_sentences = 0
    sentence_path = ARTIFACTS / "sentences.txt"
    with sentence_path.open("w", encoding="utf-8") as output:
        for split, data in corpus.items():
            stats = {"lines": len(data), "nonempty_lines": 0, "articles": 0, "tokens": 0}
            for batch in data.iter(batch_size=1000):
                for text in batch["text"]:
                    stats["nonempty_lines"] += bool(text.strip())
                    stats["articles"] += bool(ARTICLE_PATTERN.match(text))
                    for words in sentences(text):
                        stats["tokens"] += len(words)
                        if split != "train":
                            continue
                        full_counts.update(words)
                        if selected_tokens >= THRESHOLDS[-1]:
                            continue
                        output.write(" ".join(words) + "\n")
                        selected_counts.update(words)
                        selected_tokens += len(words)
                        selected_sentences += 1
                        for threshold in THRESHOLDS:
                            if selected_tokens >= threshold and str(threshold) not in snapshots:
                                snapshots[str(threshold)] = {
                                    "normalized_tokens": selected_tokens,
                                    "sentences": selected_sentences,
                                    "counts": dict(selected_counts) if threshold == THRESHOLDS[0] else None,
                                }
            stats["fingerprint"] = data._fingerprint
            split_stats[split] = stats
            print(f"{split}: {stats}", flush=True)
    if selected_tokens < THRESHOLDS[-1]:
        raise ValueError("The normalized training corpus does not contain 40 million tokens.")
    smallest = Counter(snapshots[str(THRESHOLDS[0])].pop("counts"))
    words = sorted(
        (word for word, count in smallest.items() if count >= MIN_COUNT),
        key=lambda word: (-smallest[word], word),
    )
    indices = {word: index for index, word in enumerate(words)}
    ends = []
    encoded_count = 0
    actual_counts = np.zeros(len(words), dtype=np.int64)
    subset_info = {}
    with sentence_path.open(encoding="utf-8") as source, (ARTIFACTS / "tokens.bin").open("wb") as target:
        for sentence_number, line in enumerate(source, start=1):
            ids = np.asarray([indices[word] for word in line.split() if word in indices], dtype=np.uint32)
            ids.tofile(target)
            np.add.at(actual_counts, ids, 1)
            encoded_count += len(ids)
            ends.append(encoded_count)
            for threshold, snapshot in snapshots.items():
                if sentence_number == snapshot["sentences"]:
                    subset_info[threshold] = {
                        **snapshot,
                        "retained_tokens": encoded_count,
                        "encoded_end": encoded_count,
                    }
    np.save(ARTIFACTS / "sentence_ends.npy", np.asarray(ends, dtype=np.int64))
    np.save(ARTIFACTS / "counts.npy", actual_counts)
    save_json(ARTIFACTS / "vocabulary.json", words)
    np.save(ARTIFACTS / "full_frequencies.npy", np.asarray([count for _, count in full_counts.most_common()]))
    save_json(ARTIFACTS / "full_counts.json", dict(full_counts))
    save_json(ARTIFACTS / "selected_counts.json", dict(selected_counts))
    threshold_results = []
    for threshold in (5, 10, 20):
        retained = sum(count for count in smallest.values() if count >= threshold)
        threshold_results.append({
            "min_count": threshold,
            "vocabulary": sum(count >= threshold for count in smallest.values()),
            "unknown_percent": 100 * (1 - retained / sum(smallest.values())),
        })
    frequencies = np.asarray([count for _, count in full_counts.most_common()])
    coverage = {str(k): float(frequencies[:k].sum() / frequencies.sum()) for k in (10, 1000, 30000)}
    probabilities = keep_probabilities(actual_counts)
    ids, sentence_ids, retained_counts = subsampled_tokens(THRESHOLDS[-1], SAMPLE, SEED)
    lengths = np.bincount(sentence_ids, minlength=len(ends))
    before_lengths = np.diff(np.r_[0, ends])
    subsampling = {}
    for word in ("the", "of", "and"):
        index = indices[word]
        subsampling[word] = {
            "theoretical_discard_percent": float(100 * (1 - probabilities[index])),
            "observed_discard_percent": float(100 * (1 - retained_counts[index] / actual_counts[index])),
        }
    slope = float(np.polyfit(np.log10(np.arange(1, 30001)), np.log10(frequencies[:30000]), 1)[0])
    result = {
        "splits": split_stats,
        "subsets": subset_info,
        "vocabulary_size": len(words),
        "min_count": MIN_COUNT,
        "sample": SAMPLE,
        "coverage": coverage,
        "threshold_comparison": threshold_results,
        "subsampling": subsampling,
        "pairs_before_subsampling": pair_count(before_lengths, 5),
        "pairs_after_subsampling": pair_count(lengths, 5),
        "retained_after_subsampling": len(ids),
        "zipf_slope_top_30000": slope,
        "tokenizers": compare_tokenizers(),
        "article_definition": "Top-level WikiText title line, excluding subsection headers.",
        "token_definition": "Normalized body words, headings and punctuation-only tokens excluded.",
    }
    save_json(RESULTS / "preprocessing.json", result)
    plt.figure(figsize=(5, 3))
    plt.loglog(np.arange(1, len(frequencies) + 1), frequencies, linewidth=0.8)
    plt.xlabel("Rank")
    plt.ylabel("Token frequency")
    plt.tight_layout()
    plt.savefig(FIGURES / "zipf.png", dpi=160)
    plt.close()
    print(f"Vocabulary {len(words):,}; pairs {result['pairs_after_subsampling']:,}", flush=True)
    return result


def corpus_arrays(tokens):
    metadata = load_json(RESULTS / "preprocessing.json") if (RESULTS / "preprocessing.json").exists() else None
    ends = np.load(ARTIFACTS / "sentence_ends.npy")
    if metadata is None:
        stop = int(ends[-1])
    else:
        subset = metadata["subsets"][str(tokens)]
        ends = ends[:subset["sentences"]]
        stop = subset["encoded_end"]
    ids = np.memmap(ARTIFACTS / "tokens.bin", mode="r", dtype=np.uint32, shape=(int(np.load(ARTIFACTS / "sentence_ends.npy")[-1]),))
    return ids[:stop], ends


def subsampled_tokens(tokens, sample, seed):
    ids, ends = corpus_arrays(tokens)
    counts = np.bincount(ids, minlength=len(load_json(ARTIFACTS / "vocabulary.json")))
    probabilities = keep_probabilities(counts, sample)
    rng = np.random.default_rng(seed)
    mask = rng.random(len(ids)) < probabilities[ids]
    sentence_ids = np.repeat(np.arange(len(ends), dtype=np.int32), np.diff(np.r_[0, ends]))
    filtered = np.asarray(ids[mask], dtype=np.int64)
    retained_counts = np.bincount(filtered, minlength=len(counts))
    return filtered, sentence_ids[mask], retained_counts


def pair_batches(ids, sentence_ids, window, batch_size=2048, chunk_size=65536, seed=SEED):
    rng = np.random.default_rng(seed)
    for start in range(0, len(ids), chunk_size):
        positions = np.arange(start, min(start + chunk_size, len(ids)))
        centers, contexts = [], []
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            neighbors = positions + offset
            valid = (neighbors >= 0) & (neighbors < len(ids))
            left, right = positions[valid], neighbors[valid]
            valid = sentence_ids[left] == sentence_ids[right]
            centers.append(ids[left[valid]])
            contexts.append(ids[right[valid]])
        center = np.concatenate(centers)
        context = np.concatenate(contexts)
        order = rng.permutation(len(center))
        for offset in range(0, len(order), batch_size):
            batch = order[offset:offset + batch_size]
            yield center[batch], context[batch]


class EncodedSentences:
    def __init__(self, tokens):
        self.tokens = tokens
        self.words = load_json(ARTIFACTS / "vocabulary.json")

    def __iter__(self):
        ids, ends = corpus_arrays(self.tokens)
        start = 0
        for end in ends:
            if end > start:
                yield [self.words[index] for index in ids[start:end]]
            start = end
