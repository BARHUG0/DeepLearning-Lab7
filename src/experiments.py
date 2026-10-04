import time

from .common import ARTIFACTS, DOWNLOADS, FIGURES, RESULTS, SEED, directories, load_json, save_json
from .evaluation import (
    analogy_benchmark, individual_analogies, neighbors, parallelism,
    shared_words, similarity_benchmark,
)
from .preprocessing import EncodedSentences, corpus_arrays
from .sgns import CONFIGURATIONS, select_best, train_configuration

import numpy as np


def load_glove():
    from gensim.models import KeyedVectors

    path = ARTIFACTS / "glove.kv"
    if path.exists():
        return KeyedVectors.load(str(path), mmap="r")
    import gensim.downloader as api

    api.BASE_DIR = str(DOWNLOADS / "gensim")
    vectors = api.load("glove-wiki-gigaword-100")
    vectors.save(str(path))
    return vectors


def load_models():
    from gensim.models import KeyedVectors

    return {
        "sgns": KeyedVectors.load(str(ARTIFACTS / "best_sgns.kv"), mmap="r"),
        "gensim": KeyedVectors.load(str(ARTIFACTS / "gensim.kv"), mmap="r"),
        "glove": load_glove(),
    }


def train_gensim(best, glove):
    from gensim.models import Word2Vec

    if (RESULTS / "gensim.json").exists():
        return load_json(RESULTS / "gensim.json")
    config = best["configuration"]
    words = load_json(ARTIFACTS / "vocabulary.json")
    ids, ends = corpus_arrays(config["tokens"])
    counts = np.bincount(ids, minlength=len(words))
    model = Word2Vec(
        vector_size=config["dimension"], sg=1, hs=0, negative=config["negatives"],
        window=config["window"], min_count=config["min_count"],
        sample=config["sample"], alpha=config["learning_rate"],
        min_alpha=config["min_learning_rate"], ns_exponent=0.75,
        workers=4, seed=SEED, sorted_vocab=0, shrink_windows=False,
        batch_words=10000,
    )
    model.build_vocab_from_freq(dict(zip(words, counts.tolist())))
    if model.wv.index_to_key != words:
        raise AssertionError("Gensim vocabulary differs from the PyTorch vocabulary.")
    candidates = shared_words([model.wv, glove])
    sentences = EncodedSentences(config["tokens"])
    history = {
        "configuration": {**config, "workers": 4, "ns_exponent": 0.75, "shrink_windows": False},
        "epochs": [], "actual_normalized_tokens": best["actual_normalized_tokens"],
        "retained_tokens": len(ids),
        "gpu_peak_bytes": None,
        "optimization_difference": (
            "Gensim uses asynchronous CPU online SGD and skips negative targets equal to "
            "the positive context; PyTorch uses minibatch summed gradients and permits "
            "negative collisions. Seeds do not make multithreaded gensim bitwise deterministic."
        ),
    }
    for epoch in range(config["epochs"]):
        started = time.perf_counter()
        initial_alpha = config["learning_rate"] + epoch / config["epochs"] * (config["min_learning_rate"] - config["learning_rate"])
        final_alpha = config["learning_rate"] + (epoch + 1) / config["epochs"] * (config["min_learning_rate"] - config["learning_rate"])
        effective, raw = model.train(
            sentences, total_examples=int(np.count_nonzero(np.diff(np.r_[0, ends]))),
            total_words=len(ids), epochs=1, start_alpha=initial_alpha,
            end_alpha=final_alpha, compute_loss=True,
        )
        elapsed = time.perf_counter() - started
        history["epochs"].append({
            "epoch": epoch + 1, "loss_total": float(model.get_latest_training_loss()),
            "effective_words": effective, "raw_words": raw,
            "training_seconds": elapsed, "gpu_peak_bytes": None,
            "analogies": analogy_benchmark(model.wv, candidates),
            "wordsim": similarity_benchmark(model.wv), "neighbors": neighbors(model.wv),
        })
        print(f"gensim epoch {epoch + 1}: {elapsed:.1f}s", flush=True)
    history["total_training_seconds"] = sum(row["training_seconds"] for row in history["epochs"])
    model.wv.save(str(ARTIFACTS / "gensim.kv"))
    save_json(RESULTS / "gensim.json", history)
    return history


def train_embeddings():
    directories()
    glove = load_glove()
    for config in CONFIGURATIONS:
        train_configuration(config, glove)
    best = select_best()
    train_gensim(best, glove)
    plot_training(glove)


def plot_training(glove):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gensim.models import KeyedVectors

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for config in CONFIGURATIONS:
        history = load_json(RESULTS / f"sgns_{config.name}.json")
        epochs = [row["epoch"] for row in history["epochs"]]
        axes[0].plot(epochs, [row["loss_per_pair"] for row in history["epochs"]], marker=".", label=config.name)
        axes[1].plot(epochs, [row["analogies"]["3CosAdd"]["total"] for row in history["epochs"]], marker=".", label=config.name)
    axes[0].set_ylabel("SGNS loss per positive pair")
    axes[1].set_ylabel("Shared-question analogy accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "embedding_training.png", dpi=170)
    plt.close(fig)
    reference = KeyedVectors.load(str(ARTIFACTS / "sgns_reference_epoch3.kv"), mmap="r")
    candidates = shared_words([reference, glove])
    glove_accuracy = analogy_benchmark(glove, candidates)["3CosAdd"]["total"]
    plt.figure(figsize=(5, 3))
    for epoch in (1, 2, 3):
        sizes, accuracies = [], []
        for name in ("corpus_20m", "corpus_30m", "reference"):
            history = load_json(RESULTS / f"sgns_{name}.json")
            sizes.append(history["actual_normalized_tokens"] / 1e6)
            accuracies.append(history["epochs"][epoch - 1]["analogies"]["3CosAdd"]["total"])
        plt.plot(sizes, accuracies, marker="o", label=f"Epoch {epoch}")
    plt.axhline(glove_accuracy, linestyle="--", color="black", label="GloVe")
    plt.xlabel("Normalized training tokens, millions")
    plt.ylabel("Shared-question 3CosAdd accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "corpus_size.png", dpi=170)
    plt.close()


def arithmetic():
    if (RESULTS / "arithmetic.json").exists():
        return load_json(RESULTS / "arithmetic.json")
    models = load_models()
    candidates = shared_words(list(models.values()))
    result = {
        "shared_words": candidates,
        "vocabulary_shortfall": max(0, 30000 - len(candidates)),
        "models": {},
    }
    for name, vectors in models.items():
        result["models"][name] = {
            "vocabulary": len(vectors), "dimension": vectors.vector_size,
            "analogies": analogy_benchmark(vectors, candidates, ("3CosAdd", "3CosMul")),
            "individual": individual_analogies(vectors),
            "parallelism": parallelism(vectors),
            "wordsim": similarity_benchmark(vectors),
        }
        print(f"Arithmetic evaluation complete: {name}", flush=True)
    save_json(RESULTS / "arithmetic.json", result)
    thematic_tsne(models["sgns"])
    return result


def thematic_tsne(vectors):
    from sklearn.manifold import TSNE
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = load_json(ARTIFACTS / "selected_counts.json")
    ordered = sorted(counts, key=lambda word: (-counts[word], word))
    seeds = {
        "countries": ["france", "germany", "italy", "spain", "japan", "china"],
        "numbers": ["one", "two", "three", "four", "five", "six"],
        "animals": ["dog", "cat", "horse", "bird", "fish", "lion"],
        "verbs": ["run", "walk", "eat", "play", "see", "write"],
        "technology": ["computer", "software", "internet", "hardware", "network", "program"],
    }
    labels = {}
    groups = {}
    for category, anchors in seeds.items():
        related = {}
        for anchor in anchors:
            if anchor in vectors:
                related[anchor] = 2.0
                for word, similarity in vectors.most_similar(anchor, topn=200):
                    if word.isalpha() and word not in labels:
                        related[word] = max(related.get(word, -1), similarity)
        chosen = [word for word in sorted(related, key=related.get, reverse=True) if word not in labels][:100]
        groups[category] = {"anchors": anchors, "words": chosen}
        labels.update({word: category for word in chosen})
    words = list(labels)
    embedding = np.asarray([vectors.get_vector(word, norm=True) for word in words])
    coordinates = TSNE(
        n_components=2, perplexity=30, random_state=SEED,
        init="pca", learning_rate="auto",
    ).fit_transform(embedding)
    fig, axis = plt.subplots(figsize=(7, 4))
    for category in seeds:
        mask = np.asarray([labels[word] == category for word in words])
        axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=8, alpha=0.65, label=category)
    for i, word in enumerate(words):
        if any(word in anchors for anchors in seeds.values()):
            axis.annotate(word, coordinates[i], fontsize=6)
    axis.legend(fontsize=7)
    axis.set_title("SGNS thematic neighborhoods: exploratory t-SNE")
    axis.set_xticks([])
    axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURES / "tsne.png", dpi=170)
    plt.close(fig)
    save_json(RESULTS / "tsne.json", {
        "word_count": len(words), "groups": groups,
        "selection": "Nonoverlapping nearest neighbors of thematic anchors; not independent evidence of clustering.",
        "coordinates": coordinates.tolist(),
    })
