from collections import defaultdict

from .common import ARTIFACTS, RESULTS, load_json

import numpy as np
import torch
from scipy.stats import spearmanr

SEMANTIC_CATEGORIES = {
    "capital-common-countries", "capital-world", "currency", "city-in-state", "family",
}
CONTROL_WORDS = ("king", "france", "computer", "good", "january", "run")
OWN_ANALOGIES = [
    ("gender", "man", "woman", "king", "queen"),
    ("gender", "boy", "girl", "father", "mother"),
    ("gender", "brother", "sister", "uncle", "aunt"),
    ("capital", "france", "paris", "italy", "rome"),
    ("capital", "germany", "berlin", "spain", "madrid"),
    ("capital", "japan", "tokyo", "china", "beijing"),
    ("nationality", "france", "french", "germany", "german"),
    ("nationality", "italy", "italian", "spain", "spanish"),
    ("nationality", "japan", "japanese", "china", "chinese"),
    ("comparative", "good", "better", "bad", "worse"),
    ("comparative", "small", "smaller", "large", "larger"),
    ("comparative", "fast", "faster", "slow", "slower"),
    ("past-tense", "walk", "walked", "play", "played"),
    ("past-tense", "go", "went", "run", "ran"),
    ("past-tense", "see", "saw", "eat", "ate"),
    ("plural", "dog", "dogs", "cat", "cats"),
    ("plural", "car", "cars", "book", "books"),
    ("plural", "child", "children", "man", "men"),
]


def load_analogy_questions():
    from gensim.test.utils import datapath

    category = None
    questions = []
    with open(datapath("questions-words.txt"), encoding="utf-8") as source:
        for line in source:
            if line.startswith(":"):
                category = line[1:].strip()
            else:
                words = line.lower().split()
                if len(words) == 4:
                    questions.append((category, *words))
    return questions


def similarity_benchmark(vectors, name="wordsim353.tsv"):
    from gensim.test.utils import datapath

    scores, similarities = [], []
    total = 0
    with open(datapath(name), encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                value = float(fields[2])
            except ValueError:
                continue
            total += 1
            a, b = fields[0].lower(), fields[1].lower()
            if a in vectors and b in vectors:
                scores.append(value)
                similarities.append(float(vectors.similarity(a, b)))
    correlation = float(spearmanr(scores, similarities).statistic) if len(scores) >= 3 else None
    return {
        "spearman": correlation if correlation is not None and np.isfinite(correlation) else None,
        "evaluated": len(scores),
        "total": total,
        "coverage": len(scores) / total if total else 0,
    }


def shared_words(models, limit=30000):
    counts = load_json(ARTIFACTS / "selected_counts.json")
    words = sorted(counts, key=lambda word: (-counts[word], word))
    return [word for word in words if all(word in model for model in models)][:limit]


def normalized_tensor(vectors, words, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    matrix = torch.tensor(np.asarray([vectors[word] for word in words]), device=device)
    return torch.nn.functional.normalize(matrix, dim=1)


def analogy_scores(matrix, a, b, c, method="3CosAdd"):
    if method == "3CosAdd":
        query = torch.nn.functional.normalize(matrix[b] - matrix[a] + matrix[c], dim=-1)
        return query @ matrix.T
    if method == "3CosMul":
        positive_b = (matrix[b] @ matrix.T + 1) / 2
        positive_c = (matrix[c] @ matrix.T + 1) / 2
        negative_a = (matrix[a] @ matrix.T + 1) / 2
        return positive_b * positive_c / (negative_a + 1e-6)
    raise ValueError(f"Unknown analogy method: {method}")


def analogia(vectors, a, b, c, k=5, exclude=True):
    if any(word not in vectors for word in (a, b, c)):
        return []
    words = vectors.index_to_key
    matrix = normalized_tensor(vectors, words)
    indices = [vectors.key_to_index[word] for word in (a, b, c)]
    scores = analogy_scores(matrix, *indices)
    if exclude:
        scores[indices] = -torch.inf
    values, positions = torch.topk(scores, min(k, len(words) - 3 if exclude else len(words)))
    return [(words[position], float(value)) for position, value in zip(positions.tolist(), values.tolist())]


def analogy_benchmark(vectors, words, methods=("3CosAdd",), batch_size=128):
    questions = load_analogy_questions()
    indices = {word: index for index, word in enumerate(words)}
    covered = [row for row in questions if all(word in indices for word in row[1:])]
    matrix = normalized_tensor(vectors, words)
    result = {"shared_vocabulary": len(words), "evaluated": len(covered), "total": len(questions)}
    result["coverage"] = len(covered) / len(questions)
    for method in methods:
        category_counts = defaultdict(lambda: [0, 0])
        for start in range(0, len(covered), batch_size):
            batch = covered[start:start + batch_size]
            ids = torch.tensor(
                [[indices[word] for word in row[1:]] for row in batch], device=matrix.device,
            )
            scores = analogy_scores(matrix, ids[:, 0], ids[:, 1], ids[:, 2], method)
            scores.scatter_(1, ids[:, :3], -torch.inf)
            correct = (scores.argmax(dim=1) == ids[:, 3]).tolist()
            for row, success in zip(batch, correct):
                category_counts[row[0]][0] += int(success)
                category_counts[row[0]][1] += 1
        categories = {}
        for category in dict.fromkeys(row[0] for row in questions):
            success, total = category_counts[category]
            available = sum(row[0] == category for row in questions)
            categories[category] = {
                "correct": success, "evaluated": total, "total": available,
                "accuracy": success / total if total else None,
            }
        aggregates = {}
        for group in ("semantic", "syntactic", "total"):
            rows = [
                value for category, value in categories.items()
                if group == "total"
                or (group == "semantic" and category in SEMANTIC_CATEGORIES)
                or (group == "syntactic" and category not in SEMANTIC_CATEGORIES)
            ]
            success = sum(row["correct"] for row in rows)
            total = sum(row["evaluated"] for row in rows)
            aggregates[group] = success / total if total else None
        result[method] = {"categories": categories, **aggregates}
    return result


def neighbors(vectors):
    return {
        word: [(other, float(score)) for other, score in vectors.most_similar(word, topn=5)]
        if word in vectors else []
        for word in CONTROL_WORDS
    }


def individual_analogies(vectors):
    words = vectors.index_to_key
    matrix = normalized_tensor(vectors, words)
    rows = []
    for category, a, b, c, expected in OWN_ANALOGIES:
        row = {"type": category, "a": a, "b": b, "c": c, "expected": expected}
        missing = [word for word in (a, b, c, expected) if word not in vectors]
        if missing:
            rows.append({**row, "missing": missing})
            continue
        indices = [vectors.key_to_index[word] for word in (a, b, c)]
        target = vectors.key_to_index[expected]
        scores = analogy_scores(matrix, *indices)
        row["answer_cosine"] = float(scores[target])
        values, positions = torch.topk(scores, 5)
        row["without_exclusion"] = [(words[p], float(v)) for p, v in zip(positions.tolist(), values.tolist())]
        scores[indices] = -torch.inf
        row["answer_rank"] = int((scores > scores[target]).sum().item() + 1)
        values, positions = torch.topk(scores, 5)
        row["top5"] = [(words[p], float(v)) for p, v in zip(positions.tolist(), values.tolist())]
        reference = vectors.most_similar(positive=[b, c], negative=[a], topn=5)
        row["gensim_match"] = (
            [word for word, _ in row["top5"]] == [word for word, _ in reference]
            and np.allclose([score for _, score in row["top5"]], [score for _, score in reference], atol=2e-5)
        )
        if not row["gensim_match"]:
            raise AssertionError(f"3CosAdd differs from gensim: {a}, {b}, {c}")
        rows.append(row)
    return rows


def parallelism(vectors):
    pairs = defaultdict(list)
    for category, a, b, c, d in OWN_ANALOGIES:
        for pair in ((a, b), (c, d)):
            if pair not in pairs[category] and all(word in vectors for word in pair):
                pairs[category].append(pair)
    results = {}
    all_values = []
    for category, examples in pairs.items():
        values = []
        for i, (a, b) in enumerate(examples):
            for c, d in examples[i + 1:]:
                if len({a, b, c, d}) != 4:
                    continue
                first, second = vectors[b] - vectors[a], vectors[d] - vectors[c]
                denominator = np.linalg.norm(first) * np.linalg.norm(second)
                if denominator > 0:
                    values.append(float(first @ second / denominator))
        results[category] = {
            "comparisons": len(values),
            "mean_cosine": float(np.mean(values)) if values else None,
            "word_pairs": examples,
        }
        all_values.extend(values)
    return {
        "categories": results, "comparisons": len(all_values),
        "mean_cosine": float(np.mean(all_values)) if all_values else None,
    }
