import importlib.metadata
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "informe" / "artifacts"
RESULTS = ROOT / "informe" / "results"
FIGURES = ROOT / "informe" / "figures"
DOWNLOADS = ROOT / "downloads"
SEED = 23556

os.environ["HF_HOME"] = str(DOWNLOADS / "huggingface")
os.environ["HF_DATASETS_CACHE"] = str(DOWNLOADS / "huggingface" / "datasets")
os.environ["GENSIM_DATA_DIR"] = str(DOWNLOADS / "gensim")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def directories():
    for directory in (ARTIFACTS, RESULTS, FIGURES, DOWNLOADS):
        directory.mkdir(parents=True, exist_ok=True)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def seed_all(seed=SEED):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))


def environment():
    import platform
    import torch

    packages = [
        "torch", "numpy", "scipy", "pandas", "matplotlib", "datasets",
        "gensim", "nltk", "scikit-learn", "nbformat", "nbclient",
    ]
    return {
        "seed": SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_threads": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available() else None
        ),
    }
