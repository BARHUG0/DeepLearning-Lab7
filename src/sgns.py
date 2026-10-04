import time
from dataclasses import asdict, dataclass

from .common import ARTIFACTS, RESULTS, SEED, load_json, save_json, seed_all
from .preprocessing import SAMPLE, corpus_arrays, pair_batches, pair_count, subsampled_tokens
from .evaluation import analogy_benchmark, neighbors, shared_words, similarity_benchmark

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Configuration:
    name: str
    tokens: int = 40_000_000
    dimension: int = 100
    window: int = 5
    negatives: int = 5
    sample: float = SAMPLE
    min_count: int = 5
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0001
    epochs: int = 3
    batch_size: int = 2048


CONFIGURATIONS = [
    Configuration("reference"),
    Configuration("dimension_50", dimension=50),
    Configuration("dimension_300", dimension=300),
    Configuration("corpus_20m", tokens=20_000_000),
    Configuration("corpus_30m", tokens=30_000_000),
    Configuration("window_2", window=2),
    Configuration("negatives_10", negatives=10),
]


class SGNS(nn.Module):
    def __init__(self, vocabulary_size, dimension):
        super().__init__()
        self.input = nn.Embedding(vocabulary_size, dimension, sparse=True)
        self.output = nn.Embedding(vocabulary_size, dimension, sparse=True)
        nn.init.uniform_(self.input.weight, -0.5 / dimension, 0.5 / dimension)
        nn.init.zeros_(self.output.weight)

    def forward(self, centers, contexts, negatives):
        source = self.input(centers)
        positive = (source * self.output(contexts)).sum(dim=-1)
        negative = torch.bmm(self.output(negatives), source.unsqueeze(-1)).squeeze(-1)
        return -F.logsigmoid(positive) - F.logsigmoid(-negative).sum(dim=-1)

    def keyed_vectors(self, words):
        from gensim.models import KeyedVectors

        vectors = KeyedVectors(vector_size=self.input.embedding_dim)
        vectors.add_vectors(words, self.input.weight.detach().cpu().numpy())
        return vectors


def negative_distribution(counts):
    weights = np.asarray(counts, dtype=np.float64) ** 0.75
    return weights / weights.sum()


def train_configuration(config, glove):
    seed_all()
    words = load_json(ARTIFACTS / "vocabulary.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SGNS(len(words), config.dimension).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    raw_ids, ends = corpus_arrays(config.tokens)
    counts = np.bincount(raw_ids, minlength=len(words))
    distribution = torch.tensor(negative_distribution(counts), device=device, dtype=torch.float32)
    candidates = shared_words([model.keyed_vectors(words), glove])
    history_path = RESULTS / f"sgns_{config.name}.json"
    checkpoint_path = ARTIFACTS / f"sgns_{config.name}.pt"
    history = load_json(history_path) if history_path.exists() else {
        "configuration": asdict(config), "epochs": [], "selection_words": candidates,
        "actual_normalized_tokens": load_json(RESULTS / "preprocessing.json")["subsets"][str(config.tokens)]["normalized_tokens"],
        "retained_tokens": len(raw_ids),
        "pairs_before_subsampling": pair_count(np.diff(np.r_[0, ends]), config.window),
    }
    if history["epochs"]:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if checkpoint["epoch"] != len(history["epochs"]):
            raise RuntimeError("Checkpoint/history epoch mismatch; recovery requires inspection.")
    for epoch in range(len(history["epochs"]), config.epochs):
        seed_all(SEED + epoch)
        ids, sentence_ids, _ = subsampled_tokens(config.tokens, config.sample, SEED + epoch)
        lengths = np.bincount(sentence_ids, minlength=len(ends))
        expected_pairs = pair_count(lengths, config.window)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        processed = 0
        last_log = started
        model.train()
        for centers, contexts in pair_batches(ids, sentence_ids, config.window, config.batch_size, seed=SEED + epoch):
            center = torch.from_numpy(centers).to(device)
            context = torch.from_numpy(contexts).to(device)
            negatives = torch.multinomial(
                distribution, len(center) * config.negatives, replacement=True,
            ).reshape(len(center), config.negatives)
            progress = (epoch + processed / expected_pairs) / config.epochs
            learning_rate = config.learning_rate + progress * (config.min_learning_rate - config.learning_rate)
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            losses = model(center, context, negatives)
            losses.sum().backward()
            optimizer.step()
            loss_sum += losses.detach().sum()
            processed += len(center)
            if time.perf_counter() - last_log >= 30:
                print(
                    f"{config.name} epoch {epoch + 1}: {processed:,}/{expected_pairs:,} "
                    f"pairs; {processed / (time.perf_counter() - started):,.0f} pairs/s",
                    flush=True,
                )
                last_log = time.perf_counter()
        if processed != expected_pairs:
            raise AssertionError(f"Pair count mismatch: {processed} != {expected_pairs}")
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        vectors = model.keyed_vectors(words)
        if not np.isfinite(vectors.vectors).all():
            raise FloatingPointError("Nonfinite embeddings.")
        evaluation_started = time.perf_counter()
        row = {
            "epoch": epoch + 1,
            "loss_per_pair": float(loss_sum.cpu()) / processed,
            "training_seconds": elapsed,
            "gpu_peak_bytes": peak,
            "pairs": processed,
            "subsampled_tokens": len(ids),
            "learning_rate_end": learning_rate,
            "analogies": analogy_benchmark(vectors, candidates),
            "wordsim": similarity_benchmark(vectors),
            "neighbors": neighbors(vectors),
        }
        row["evaluation_seconds"] = time.perf_counter() - evaluation_started
        torch.save({"model": model.state_dict(), "epoch": epoch + 1}, checkpoint_path)
        vectors.save(str(ARTIFACTS / f"sgns_{config.name}_epoch{epoch + 1}.kv"))
        history["epochs"].append(row)
        history["total_training_seconds"] = sum(item["training_seconds"] for item in history["epochs"])
        save_json(history_path, history)
        print(
            f"{config.name} epoch {epoch + 1}: loss={row['loss_per_pair']:.4f}, "
            f"analogy={row['analogies']['3CosAdd']['total']:.4f}, "
            f"WordSim={row['wordsim']['spearman']:.4f}, seconds={elapsed:.1f}",
            flush=True,
        )
    return history


def select_best():
    candidates = []
    for config in CONFIGURATIONS:
        path = RESULTS / f"sgns_{config.name}.json"
        if not path.exists():
            raise RuntimeError(f"Missing experiment: {config.name}")
        history = load_json(path)
        if len(history["epochs"]) != config.epochs:
            raise RuntimeError(f"Incomplete experiment: {config.name}")
        for epoch in history["epochs"]:
            candidates.append((
                epoch["analogies"]["3CosAdd"]["total"],
                epoch["wordsim"]["spearman"],
                config.name, epoch["epoch"], history,
            ))
    _, _, name, epoch, history = max(candidates, key=lambda row: row[:2])
    from gensim.models import KeyedVectors

    vectors = KeyedVectors.load(str(ARTIFACTS / f"sgns_{name}_epoch{epoch}.kv"))
    vectors.save(str(ARTIFACTS / "best_sgns.kv"), separately=[])
    result = {
        "name": name, "epoch": epoch,
        "configuration": {**history["configuration"], "epochs": epoch},
        "actual_normalized_tokens": history["actual_normalized_tokens"],
        "retained_tokens": history["retained_tokens"],
        "training_seconds_to_checkpoint": sum(row["training_seconds"] for row in history["epochs"][:epoch]),
        "search_training_seconds": sum(load_json(RESULTS / f"sgns_{config.name}.json")["total_training_seconds"] for config in CONFIGURATIONS),
        "criterion": "Fixed-shared-question 3CosAdd accuracy, then WordSim-353 Spearman.",
    }
    save_json(RESULTS / "best_sgns.json", result)
    return result


def pilot():
    seed_all()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SGNS(2000, 100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.025)
    centers = torch.randint(0, 2000, (2048,), device=device)
    contexts = torch.randint(0, 2000, (2048,), device=device)
    negatives = torch.randint(0, 2000, (2048, 5), device=device)
    initial = float(model(centers, contexts, negatives).mean().detach())
    started = time.perf_counter()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        model(centers, contexts, negatives).sum().backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final = float(model(centers, contexts, negatives).mean().detach())
    assert np.isfinite(final) and final < initial
    result = {
        "device": device, "initial_loss": initial, "final_loss": final,
        "steps": 200, "batch_size": 2048, "seconds": elapsed,
        "synthetic_pairs_per_second": 200 * 2048 / elapsed,
        "assignment_experiment": False,
    }
    save_json(RESULTS / "pilot.json", result)
    print(result, flush=True)
