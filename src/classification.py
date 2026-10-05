import copy
import hashlib
import time
import warnings
from collections import Counter

from .common import ARTIFACTS, DATASET_REVISIONS, FIGURES, RESULTS, SEED, load_json, save_json, seed_all
from .preprocessing import tokenize

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

FRACTIONS = (0.01, 0.1, 0.5, 1.0)
LEARNING_RATES = (0.001, 0.003)
MAX_EPOCHS = 12
PATIENCE = 3


def metrics(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision), "recall_macro": float(recall), "f1_macro": float(f1),
    }


def prepare_news():
    if (ARTIFACTS / "news_train.pt").exists():
        return torch.load(ARTIFACTS / "news_train.pt", weights_only=False)
    from datasets import load_dataset

    data = load_dataset(
        "fancyzhx/ag_news", split="train", revision=DATASET_REVISIONS["fancyzhx/ag_news"],
    )
    texts = [tokenize(text) for text in data["text"]]
    labels = np.asarray(data["label"], dtype=np.int64)
    train, validation = train_test_split(
        np.arange(len(labels)), test_size=0.1, stratify=labels, random_state=SEED,
    )
    result = {"texts": texts, "labels": labels, "train": train, "validation": validation}
    torch.save(result, ARTIFACTS / "news_train.pt")
    fingerprint = hashlib.sha256(train.tobytes() + validation.tobytes()).hexdigest()
    save_json(RESULTS / "news_split.json", {
        "dataset_fingerprint": data._fingerprint,
        "split_hash": fingerprint, "training_documents": len(train),
        "validation_documents": len(validation),
        "training_class_counts": np.bincount(labels[train]).tolist(),
        "validation_class_counts": np.bincount(labels[validation]).tolist(),
        "seed": SEED,
    })
    return result


def fraction_indices(train, labels, fraction):
    rng = np.random.default_rng(SEED)
    indices = []
    for category in range(4):
        group = rng.permutation(train[labels[train] == category])
        indices.extend(group[:max(1, int(round(len(group) * fraction)))].tolist())
    return np.asarray(indices, dtype=np.int64)


def collate_bags(batch):
    lengths = [len(ids) for ids, _ in batch]
    offsets = torch.tensor(np.r_[0, np.cumsum(lengths)], dtype=torch.long)
    flat = torch.tensor([index for ids, _ in batch for index in ids], dtype=torch.long)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    return flat, offsets, labels


class BagClassifier(nn.Module):
    def __init__(self, weights, freeze=False):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(
            torch.as_tensor(weights, dtype=torch.float32),
            freeze=freeze, mode="mean", padding_idx=0, include_last_offset=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(weights.shape[1], 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 4),
        )

    def forward(self, ids, offsets):
        return self.mlp(self.embedding(ids, offsets))


def initialization(name, vectors, news):
    seed_all()
    if name == "random":
        counts = Counter(word for index in news["train"] for word in news["texts"][index])
        words = sorted(counts, key=lambda word: (-counts[word], word))
        dimension = vectors.vector_size
        weights = np.random.default_rng(SEED).normal(0, 0.1, (len(words) + 1, dimension)).astype(np.float32)
    else:
        words = list(vectors.index_to_key)
        weights = np.vstack([np.zeros((1, vectors.vector_size), dtype=np.float32), vectors.vectors])
    weights[0] = 0
    mapping = {word: index + 1 for index, word in enumerate(words)}
    encoded = [[mapping.get(word, 0) for word in text] for text in news["texts"]]
    oov = {}
    for split in ("train", "validation"):
        total = sum(len(news["texts"][index]) for index in news[split])
        unknown = sum(word not in mapping for index in news[split] for word in news["texts"][index])
        oov[split] = {"unknown_tokens": unknown, "total_tokens": total, "percent": 100 * unknown / total}
    return weights, mapping, encoded, oov


def loader(encoded, labels, indices, shuffle=False):
    data = [(encoded[index], int(labels[index])) for index in indices]
    return DataLoader(
        data, batch_size=256, shuffle=shuffle, collate_fn=collate_bags,
        generator=torch.Generator().manual_seed(SEED),
    )


def evaluate_classifier(model, data, device):
    model.eval()
    loss, count, predictions, labels = 0., 0, [], []
    with torch.no_grad():
        for ids, offsets, target in data:
            logits = model(ids.to(device), offsets.to(device))
            loss += float(nn.functional.cross_entropy(logits, target.to(device), reduction="sum"))
            count += len(target)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(target.tolist())
    return loss / count, metrics(labels, predictions), labels, predictions


def train_neural(name, weights, encoded, news, fraction, freeze, learning_rate):
    variant = "frozen" if freeze else "finetuned"
    if name == "random":
        variant = "trainable"
    key = f"{name}_{variant}_f{fraction:g}_lr{learning_rate:g}"
    result_path = RESULTS / f"classifier_{key}.json"
    checkpoint_path = ARTIFACTS / f"classifier_{key}.pt"
    if result_path.exists():
        return load_json(result_path)
    seed_all()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BagClassifier(weights.copy(), freeze).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_indices = fraction_indices(news["train"], news["labels"], fraction)
    train_data = loader(encoded, news["labels"], train_indices, shuffle=True)
    validation_data = loader(encoded, news["labels"], news["validation"])
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    history, best_f1, best_epoch, stale, best_state = [], -1, 0, 0, None
    started = time.perf_counter()
    for epoch in range(MAX_EPOCHS):
        model.train()
        loss_sum, documents = 0., 0
        epoch_started = time.perf_counter()
        for ids, offsets, labels in train_data:
            optimizer.zero_grad(set_to_none=True)
            output = model(ids.to(device), offsets.to(device))
            loss = nn.functional.cross_entropy(output, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            documents += len(labels)
        validation_loss, scores, _, _ = evaluate_classifier(model, validation_data, device)
        history.append({
            "epoch": epoch + 1, "training_loss": loss_sum / documents,
            "validation_loss": validation_loss, **scores,
            "seconds": time.perf_counter() - epoch_started,
        })
        if scores["f1_macro"] > best_f1 + 1e-5:
            best_f1, best_epoch, stale = scores["f1_macro"], epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    elapsed = time.perf_counter() - started
    torch.save(best_state, checkpoint_path)
    result = {
        "key": key, "initialization": name, "variant": variant, "fraction": fraction,
        "learning_rate": learning_rate, "dimension": weights.shape[1],
        "training_documents": len(train_indices), "best_epoch": best_epoch,
        "validation": {key: history[best_epoch - 1][key] for key in (
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
        )},
        "epochs": history, "training_seconds": elapsed,
        "total_parameters": total_parameters, "trainable_parameters": trainable_parameters,
        "checkpoint": checkpoint_path.name,
    }
    save_json(result_path, result)
    print(f"{key}: validation F1={best_f1:.4f}, {elapsed:.1f}s", flush=True)
    del model, optimizer, best_state
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def train_baseline(news, fraction):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib

    final_path = RESULTS / f"baseline_f{fraction:g}.json"
    if final_path.exists():
        cached = load_json(final_path)
        if cached.get("optimizer_block_size") == 25:
            return cached
    indices = fraction_indices(news["train"], news["labels"], fraction)
    train_texts = [" ".join(news["texts"][index]) for index in indices]
    validation_texts = [" ".join(news["texts"][index]) for index in news["validation"]]
    train_labels, validation_labels = news["labels"][indices], news["labels"][news["validation"]]
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False,
        ngram_range=(1, 2), max_features=100000, min_df=2, sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    validation_matrix = vectorizer.transform(validation_texts)
    candidates = []
    for c in (1., 4.):
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=25, warm_start=True, random_state=SEED)
        history, best_f1, best_state, stale = [], -1., None, 0
        candidate_started = time.perf_counter()
        for iteration in range(30):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(train_matrix, train_labels)
            train_probability = model.predict_proba(train_matrix)
            validation_probability = model.predict_proba(validation_matrix)
            scores = metrics(validation_labels, validation_probability.argmax(axis=1))
            history.append({
                "optimization_block": iteration + 1,
                "training_loss": float(log_loss(train_labels, train_probability)),
                "validation_loss": float(log_loss(validation_labels, validation_probability)),
                **scores,
            })
            if scores["f1_macro"] > best_f1 + 1e-5:
                best_f1, best_state, stale = scores["f1_macro"], copy.deepcopy(model), 0
                best_block = iteration + 1
            else:
                stale += 1
            if stale >= PATIENCE:
                break
        candidates.append({
            "model": best_state, "C": c, "history": history, "best_block": best_block,
            "seconds": time.perf_counter() - candidate_started, "f1": best_f1,
        })
    selected = max(candidates, key=lambda row: row["f1"])
    model = selected.pop("model")
    checkpoint = f"baseline_f{fraction:g}.joblib"
    joblib.dump({"vectorizer": vectorizer, "model": model}, ARTIFACTS / checkpoint)
    scores = selected["history"][selected["best_block"] - 1]
    result = {
        "key": f"baseline_f{fraction:g}", "initialization": "tfidf",
        "fraction": fraction, "C": selected["C"], "best_block": selected["best_block"],
        "validation": {key: scores[key] for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
        "epochs": selected["history"], "training_documents": len(indices),
        "total_parameters": int(model.coef_.size + model.intercept_.size),
        "trainable_parameters": int(model.coef_.size + model.intercept_.size),
        "training_seconds": time.perf_counter() - started, "checkpoint": checkpoint,
        "curve_unit": "Warm-start L-BFGS blocks, at most 25 optimizer iterations each.",
        "optimizer_block_size": 25,
        "search": [{"C": row["C"], "validation_f1": row["f1"], "seconds": row["seconds"]} for row in candidates],
    }
    save_json(final_path, result)
    print(f"TF-IDF fraction {fraction}: validation F1={result['validation']['f1_macro']:.4f}", flush=True)
    return result


def classification():
    from .experiments import load_models

    if (RESULTS / "classification_selection.json").exists():
        return load_json(RESULTS / "classification_selection.json")
    news = prepare_news()
    models = load_models()
    selections, oov_rates, trials = {}, {}, []
    for fraction in FRACTIONS:
        selected = train_baseline(news, fraction)
        selections[f"tfidf_f{fraction:g}"] = selected
    for name in ("random", "sgns", "gensim", "glove"):
        vectors = models["sgns"] if name == "random" else models[name]
        weights, mapping, encoded, oov = initialization(name, vectors, news)
        oov_rates[name] = oov
        save_json(ARTIFACTS / f"classifier_mapping_{name}.json", mapping)
        for fraction in FRACTIONS:
            candidates = []
            for freeze in ((False,) if name == "random" else (True, False)):
                for rate in LEARNING_RATES:
                    trial = train_neural(name, weights, encoded, news, fraction, freeze, rate)
                    candidates.append(trial)
                    trials.append(trial)
            selections[f"{name}_f{fraction:g}"] = max(candidates, key=lambda row: row["validation"]["f1_macro"])
        del weights, encoded
    result = {
        "split": load_json(RESULTS / "news_split.json"), "fractions": FRACTIONS,
        "selection_metric": "Validation macro F1; ties keep the first prespecified configuration.",
        "selected": selections, "trials": trials, "oov": oov_rates,
        "total_neural_search_seconds": sum(row["training_seconds"] for row in trials),
    }
    save_json(RESULTS / "classification_selection.json", result)
    plot_classification(result)
    return result


def plot_classification(result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_models = [result["selected"]["tfidf_f1"]]
    for name in ("random", "sgns", "gensim", "glove"):
        for variant in (("trainable",) if name == "random" else ("frozen", "finetuned")):
            candidates = [
                row for row in result["trials"]
                if row["initialization"] == name and row["fraction"] == 1 and row["variant"] == variant
            ]
            selected_models.append(max(candidates, key=lambda row: row["validation"]["f1_macro"]))
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for axis, selected in zip(axes.flat, selected_models):
        history = selected["epochs"]
        x = np.arange(1, len(history) + 1)
        axis.plot(x, [row["training_loss"] for row in history], label="train")
        axis.plot(x, [row["validation_loss"] for row in history], label="validation")
        axis.set_title(f"{selected['initialization']} {selected.get('variant', 'logistic')}", fontsize=9)
        axis.set_xlabel("Epoch or optimizer block", fontsize=8)
        axis.set_ylabel("Cross-entropy", fontsize=8)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "classification_training.png", dpi=170)
    plt.close(fig)


def final_evaluation():
    from .experiments import load_models
    from .evaluation import similarity_benchmark
    from datasets import load_dataset
    import joblib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selection_path = RESULTS / "classification_selection.json"
    if (RESULTS / "final_evaluation.json").exists():
        recorded = load_json(RESULTS / "final_evaluation.json")
        current_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        if recorded["selection_sha256"] != current_hash:
            raise RuntimeError("Selections changed after locked test evaluation. Use a separate fresh experiment folder.")
        print("Final evaluation already locked; reusing recorded test results.", flush=True)
        return recorded
    selection = load_json(selection_path)
    models = load_models()
    data = load_dataset(
        "fancyzhx/ag_news", split="test", revision=DATASET_REVISIONS["fancyzhx/ag_news"],
    )
    texts = [tokenize(text) for text in data["text"]]
    labels = np.asarray(data["label"], dtype=np.int64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lock = {
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "test_fingerprint": data._fingerprint, "test_documents": len(labels),
    }
    progress_path = RESULTS / "final_progress.json"
    result = load_json(progress_path) if progress_path.exists() else {**lock, "models": {}, "simlex": {}}
    if result["selection_sha256"] != lock["selection_sha256"]:
        raise RuntimeError("Validation selections changed after final evaluation began.")
    for key, selected in selection["selected"].items():
        if key in result["models"]:
            continue
        name = selected["initialization"]
        if name == "tfidf":
            baseline = joblib.load(ARTIFACTS / selected["checkpoint"])
            predictions = baseline["model"].predict(
                baseline["vectorizer"].transform([" ".join(text) for text in texts]),
            )
            scores = metrics(labels, predictions)
            oov = None
        else:
            mapping = load_json(ARTIFACTS / f"classifier_mapping_{name}.json")
            encoded = [[mapping.get(word, 0) for word in text] for text in texts]
            state = torch.load(ARTIFACTS / selected["checkpoint"], weights_only=True, map_location="cpu")
            weights = state["embedding.weight"].numpy()
            model = BagClassifier(weights, selected["variant"] == "frozen").to(device)
            model.load_state_dict(state)
            test_loader = loader(encoded, labels, np.arange(len(labels)))
            _, scores, _, predictions = evaluate_classifier(model, test_loader, device)
            total = sum(map(len, texts))
            unknown = sum(word not in mapping for text in texts for word in text)
            oov = {"unknown_tokens": unknown, "total_tokens": total, "percent": 100 * unknown / total}
            del model, state
        matrix = confusion_matrix(labels, predictions).tolist()
        result["models"][key] = {
            **scores, "selected_configuration": selected["key"], "oov": oov,
            "confusion_matrix": matrix, "fraction": selected["fraction"],
        }
        save_json(progress_path, result)
        print(f"Test {key}: F1={scores['f1_macro']:.4f}", flush=True)
    for name, vectors in models.items():
        if name not in result["simlex"]:
            result["simlex"][name] = similarity_benchmark(vectors, "simlex999.txt")
            save_json(progress_path, result)
    save_json(RESULTS / "final_evaluation.json", result)
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
    for axis, name in zip(axes, ("tfidf", "random", "sgns", "gensim", "glove")):
        matrix = np.asarray(result["models"][f"{name}_f1"]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues")
        for i in range(4):
            for j in range(4):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=6)
        axis.set_title(name)
        axis.set_xticks(range(4), ["W", "S", "B", "T"])
        axis.set_yticks(range(4), ["W", "S", "B", "T"])
        axis.set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrices.png", dpi=170)
    plt.close(fig)
    plt.figure(figsize=(5, 3))
    for name in ("tfidf", "random", "sgns", "gensim", "glove"):
        plt.plot(
            FRACTIONS, [result["models"][f"{name}_f{fraction:g}"]["f1_macro"] for fraction in FRACTIONS],
            marker="o", label=name,
        )
    plt.xscale("log")
    plt.xlabel("Fraction of AG News training data")
    plt.ylabel("Official test macro F1")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "data_efficiency.png", dpi=170)
    plt.close()
    return result
