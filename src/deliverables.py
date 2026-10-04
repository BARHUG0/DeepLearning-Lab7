import argparse
import hashlib
import sys

from .common import ARTIFACTS, FIGURES, RESULTS, ROOT, load_json, save_json

import pandas as pd


def embedding_rows():
    arithmetic = load_json(RESULTS / "arithmetic.json")
    best = load_json(RESULTS / "best_sgns.json")
    reference = load_json(RESULTS / "gensim.json")
    final = load_json(RESULTS / "final_evaluation.json")
    rows = []
    for name in ("sgns", "gensim", "glove"):
        model = arithmetic["models"][name]
        add, mul = model["analogies"]["3CosAdd"], model["analogies"]["3CosMul"]
        classification = final["models"][f"{name}_f1"]
        rows.append({
            "model": name,
            "corpus_tokens": 6_000_000_000 if name == "glove" else best["actual_normalized_tokens"],
            "vocabulary": model["vocabulary"], "dimension": model["dimension"],
            "training_seconds": None if name == "glove" else (
                best["training_seconds_to_checkpoint"] if name == "sgns" else reference["total_training_seconds"]
            ),
            "hardware": (
                "Original 100D release: exact CPU model and wall time not reported"
                if name == "glove" else
                load_json(RESULTS / "environment.json")["gpu"] if name == "sgns"
                else "CPU, four workers"
            ),
            "add_semantic": add["semantic"], "add_syntactic": add["syntactic"], "add_total": add["total"],
            "mul_semantic": mul["semantic"], "mul_syntactic": mul["syntactic"], "mul_total": mul["total"],
            "analogy_coverage": model["analogies"]["coverage"],
            "wordsim": model["wordsim"]["spearman"],
            "wordsim_coverage": model["wordsim"]["coverage"],
            "simlex": final["simlex"][name]["spearman"],
            "simlex_coverage": final["simlex"][name]["coverage"],
            "parallelism": model["parallelism"]["mean_cosine"],
            "ag_news_oov_percent": classification["oov"]["percent"],
            "test_f1": classification["f1_macro"],
        })
    return rows


def iteration_rows():
    from .sgns import CONFIGURATIONS

    rows = []
    for config in CONFIGURATIONS:
        history = load_json(RESULTS / f"sgns_{config.name}.json")
        for epoch in history["epochs"]:
            rows.append({
                **history["configuration"], "actual_tokens": history["actual_normalized_tokens"],
                "epoch": epoch["epoch"], "loss": epoch["loss_per_pair"],
                "semantic": epoch["analogies"]["3CosAdd"]["semantic"],
                "syntactic": epoch["analogies"]["3CosAdd"]["syntactic"],
                "accuracy": epoch["analogies"]["3CosAdd"]["total"],
                "wordsim": epoch["wordsim"]["spearman"],
                "seconds": epoch["training_seconds"],
                "peak_gpu_mb": epoch["gpu_peak_bytes"] / 2**20 if epoch["gpu_peak_bytes"] is not None else None,
                "pairs": epoch["pairs"],
            })
    return rows


def category_rows():
    arithmetic = load_json(RESULTS / "arithmetic.json")
    categories = arithmetic["models"]["sgns"]["analogies"]["3CosAdd"]["categories"]
    rows = []
    for category in categories:
        row = {
            "category": category,
            "evaluated": categories[category]["evaluated"],
            "total": categories[category]["total"],
        }
        for name, label in (("sgns", "S"), ("gensim", "G"), ("glove", "V")):
            for method, abbreviation in (("3CosAdd", "Add"), ("3CosMul", "Mul")):
                row[f"{label}-{abbreviation}"] = arithmetic["models"][name]["analogies"][method]["categories"][category]["accuracy"]
        rows.append(row)
    return rows


def validation_rows():
    selection = load_json(RESULTS / "classification_selection.json")
    rows = [selection["selected"]["tfidf_f1"]]
    for name in ("random", "sgns", "gensim", "glove"):
        for variant in (("trainable",) if name == "random" else ("frozen", "finetuned")):
            trials = [
                row for row in selection["trials"]
                if row["initialization"] == name and row["fraction"] == 1 and row["variant"] == variant
            ]
            rows.append(max(trials, key=lambda row: row["validation"]["f1_macro"]))
    return [{
        "model": row["initialization"], "variant": row.get("variant", "logistic"),
        **row["validation"], "parameters": row["total_parameters"],
        "trainable": row["trainable_parameters"], "seconds": row["training_seconds"],
    } for row in rows]


def markdown_table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ])


def number(value, digits=3):
    return "NR" if value is None else f"{value:.{digits}f}"


def report_text():
    prep = load_json(RESULTS / "preprocessing.json")
    arithmetic = load_json(RESULTS / "arithmetic.json")
    best = load_json(RESULTS / "best_sgns.json")
    selection = load_json(RESULTS / "classification_selection.json")
    final = load_json(RESULTS / "final_evaluation.json")
    iterations = iteration_rows()
    models = embedding_rows()
    validation = validation_rows()
    train = prep["splits"]["train"]
    reference = [row for row in iterations if row["name"] == "reference"]
    dimension = {name: max(row["accuracy"] for row in iterations if row["name"] == name) for name in (
        "dimension_50", "reference", "dimension_300",
    )}
    corpus = {name: max(row["accuracy"] for row in iterations if row["name"] == name) for name in (
        "corpus_20m", "corpus_30m", "reference",
    )}
    best_categories = sorted(
        [(category, value["accuracy"]) for category, value in arithmetic["models"]["sgns"]["analogies"]["3CosAdd"]["categories"].items()
         if value["accuracy"] is not None], key=lambda row: row[1], reverse=True,
    )
    individual = []
    for name in ("sgns", "gensim", "glove"):
        row = arithmetic["models"][name]["individual"][0]
        individual.append([
            name, "; ".join(f"{word} {score:.2f}" for word, score in row.get("top5", [])),
            row.get("answer_rank", "OOV"), number(row.get("answer_cosine")),
            row.get("without_exclusion", [("OOV", 0)])[0][0],
        ])
    epoch_table = []
    for name in dict.fromkeys(row["name"] for row in iterations):
        rows = [row for row in iterations if row["name"] == name]
        selected = max(rows, key=lambda row: (row["accuracy"], row["wordsim"]))
        epoch_table.append([
            name.replace("dimension_", "dim_"), f"{selected['actual_tokens'] / 1e6:.0f}",
            selected["dimension"], selected["window"], selected["negatives"], selected["epoch"],
            number(selected["loss"]), number(selected["accuracy"]), number(selected["wordsim"]),
            f"{sum(row['seconds'] for row in rows):.0f}", f"{max(row['peak_gpu_mb'] for row in rows):.0f}",
        ])
    category_table = []
    for row in category_rows():
        category_table.append([
            row["category"].replace("gram", "g").replace("capital-common-countries", "capital-common"),
            f"{row['evaluated']}/{row['total']}",
            *[number(row[key]) for key in ("S-Add", "G-Add", "V-Add", "S-Mul", "G-Mul", "V-Mul")],
        ])
    data_efficiency = [
        [name, selection["selected"][f"{name}_f1"].get("variant", "logistic"),
         number(final["models"][f"{name}_f1"]["accuracy"]),
         number(final["models"][f"{name}_f1"]["f1_macro"]),
         number(final["models"][f"{name}_f0.01"]["f1_macro"])]
        for name in ("tfidf", "random", "sgns", "gensim", "glove")
    ]
    smaller_window = max(row["accuracy"] for row in iterations if row["name"] == "window_2")
    more_negatives = max(row["accuracy"] for row in iterations if row["name"] == "negatives_10")
    frozen_choice = "; ".join(
        f"{name}: {selection['selected'][f'{name}_f1']['variant']} at 100%, "
        f"{selection['selected'][f'{name}_f0.01']['variant']} at 1%"
        for name in ("sgns", "gensim", "glove")
    )
    best_test = max(
        ((name, final["models"][f"{name}_f1"]["f1_macro"]) for name in ("tfidf", "random", "sgns", "gensim", "glove")),
        key=lambda row: row[1],
    )
    dimensions = {row["model"]: row["dimension"] for row in models}
    dimension_caveat = (
        "All embedding initializations have 100 dimensions."
        if len(set(dimensions.values())) == 1 else
        "The selected SGNS/gensim dimension differs from GloVe's 100D; classification differences therefore include a dimension confound. Random vectors match SGNS."
    )
    loss_observation = (
        f"In the reference run, loss changes from {reference[1]['loss']:.3f} at epoch two "
        f"to {reference[2]['loss']:.3f} at epoch three, while analogy accuracy changes "
        f"from {reference[1]['accuracy']:.3f} to {reference[2]['accuracy']:.3f}."
    )
    sg = next(row for row in models if row["model"] == "sgns")
    ge = next(row for row in models if row["model"] == "gensim")
    gl = next(row for row in models if row["model"] == "glove")
    window_final = next(row for row in iterations if row["name"] == "window_2" and row["epoch"] == 3)
    negative_final = next(row for row in iterations if row["name"] == "negatives_10" and row["epoch"] == 3)
    reference_final = reference[-1]
    empirical_discussion = (
        f"SGNS semantic/syntactic accuracy is {sg['add_semantic']:.3f}/{sg['add_syntactic']:.3f}. "
        f"The overall GloVe minus SGNS gap is {gl['add_total'] - sg['add_total']:+.3f}; "
        f"gensim minus SGNS is {ge['add_total'] - sg['add_total']:+.3f}. "
        f"At epoch three, window 2 versus 5 changes semantic accuracy "
        f"from {reference_final['semantic']:.3f} to {window_final['semantic']:.3f} "
        f"and syntactic accuracy from {reference_final['syntactic']:.3f} "
        f"to {window_final['syntactic']:.3f}. Ten versus five negatives changes "
        f"overall accuracy from {reference_final['accuracy']:.3f} "
        f"to {negative_final['accuracy']:.3f}."
    )
    return f"""---
title: "NLP end-to-end and embeddings"
subtitle: "WikiText-103, SGNS, gensim, GloVe and AG News"
format:
  pdf:
    pdf-engine: xelatex
    documentclass: article
    papersize: a4
    fontsize: 9pt
    geometry: margin=1.35cm
    keep-tex: false
    number-sections: false
    toc: false
header-includes:
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{H}}
---

## 1. Research and preprocessing

PyTorch `Embedding` maps IDs to rows through `num_embeddings` and `embedding_dim`.
`padding_idx` reserves an untrained zero row; `sparse=True` saves gradient storage
with a compatible optimizer. `from_pretrained` imports weights and `freeze`
controls adaptation. `EmbeddingBag` uses flattened IDs and offsets for `mean`,
`sum` or `max`; the classifier uses mean. CBOW predicts a center from context;
skip-gram predicts context from a center. SGNS retains separate input and output
tables, exporting input vectors. Word2Vec is predictive; GloVe fits weighted
global log co-occurrences. Gensim uses `sg=1`, `negative`, `sample`, `alpha`,
`epochs` and fixed windows; `KeyedVectors` supports persistence and cosine lookup.

WikiText train contains **{train['articles']:,} articles, {train['lines']:,} lines
and {train['tokens']:,} normalized body tokens**. Validation has
{prep['splits']['validation']['tokens']:,} tokens and test
{prep['splits']['test']['tokens']:,}; neither trains embeddings. Article counts
use top-level titles. Headings and standalone punctuation are excluded.
Lowercase words retain apostrophes and numeric strings, repair WikiText markers,
and split hyphens. This matches uncased GloVe without inventing a numeric token.
Sentence splitting remains approximate for abbreviations.

The notebook compares 11 sentences with NLTK Treebank: `can't` stays one word
but Treebank yields `ca` and `n't`; compounds split only in the custom tokenizer;
abbreviations lose dots; numeric separators stay internal. All AG News stages
reuse this exact tokenizer.

{markdown_table(["min_count", "Vocabulary", "Unknown tokens %"], [
    [row["min_count"], f"{row['vocabulary']:,}", number(row["unknown_percent"])]
    for row in prep["threshold_comparison"]
])}

The vocabulary is fixed from the 20M subset. Top 10/1,000/30,000 words cover
{100 * prep['coverage']['10']:.2f}% / {100 * prep['coverage']['1000']:.2f}% /
{100 * prep['coverage']['30000']:.2f}% of train tokens. The top-30,000 log-log slope
is {prep['zipf_slope_top_30000']:.3f}: Zipf-like, not an exact slope of −1.
With $t=10^{{-4}}$, observed discard rates for the/of/and are
{prep['subsampling']['the']['observed_discard_percent']:.2f}% /
{prep['subsampling']['of']['observed_discard_percent']:.2f}% /
{prep['subsampling']['and']['observed_discard_percent']:.2f}%.
This reduces function-word dominance. Fixed-window pairs fall from
{prep['pairs_before_subsampling']:,} to {prep['pairs_after_subsampling']:,} per epoch
in the 40M subset; contexts never cross sentence boundaries.

![Observed frequency versus rank.](figures/zipf.png){{width=65% height=1.55in}}

\\newpage

## 2. SGNS implementation and controlled iterations

Two sparse embedding tables minimize positive and negative logistic losses.
Noise probability is proportional to frequency raised to 0.75. Batched summed
gradients use SGD, linearly decaying learning rate 0.025 to 0.0001 over three
epochs, `min_count=5`, subsampling $10^{{-4}}$ and batch size 2,048.
Gensim trains identical encoded sentences, vocabulary and equivalent settings.
It differs through asynchronous online updates and skipped negative collisions.
Loss values across implementations or different negative counts are not directly
comparable.

{markdown_table(["Run", "M tok", "Dim", "Win", "Neg", "Best ep", "Loss", "Add", "WS", "Sec", "GPU MB"], epoch_table)}

Metrics in the table refer to each run's best epoch; time sums all three epochs,
and memory is peak PyTorch allocation, not total GPU process memory.
The notebook retains every epoch's semantic/syntactic/total accuracy,
WordSim-353 Spearman, loss, timings and five neighbors of king, france, computer,
good, january and run.

![Loss and fixed-question analogy accuracy for seven runs.](figures/embedding_training.png){{width=100% height=2.0in}}

The selected model is **{best['name']}, epoch {best['epoch']}**.
Selection uses only shared-question 3CosAdd accuracy then WordSim-353.
SimLex and AG News test are held out. Best analogy accuracy for dimensions
50/100/300 is {dimension['dimension_50']:.3f}/{dimension['reference']:.3f}/
{dimension['dimension_300']:.3f}. For 20M/30M/40M tokens it is
{corpus['corpus_20m']:.3f}/{corpus['corpus_30m']:.3f}/{corpus['reference']:.3f}.
Window two yields {smaller_window:.3f}; ten negatives yield {more_negatives:.3f}.
These are descriptive single-seed comparisons, not statistical significance tests.
{loss_observation} A lower negative-sampling loss is not an unconditional
quality guarantee: the objective and benchmarks measure different properties.
Cross-negative-count losses also change their number of terms.

\\newpage

## 3. Vector arithmetic, coverage and parallelism

Custom 3CosAdd uses unit source vectors and normalized $b-a+c$, excluding query
words. Its top five match gensim numerically. The notebook evaluates 18 original
analogies across six types, retaining all similarities and correct-answer ranks.
For man → woman, king → queen:

{markdown_table(["Model", "Top five with cosine", "Queen rank", "Cosine", "First, no exclusion"], individual)}

The benchmark uses the same {len(arithmetic['shared_words']):,} most frequent
shared words. Coverage is
{arithmetic['models']['sgns']['analogies']['evaluated']:,}/19,544 questions.
S/G/V denote SGNS/gensim/GloVe. Every category reports evaluated/total:

{markdown_table(["Category", "Coverage", "S Add", "G Add", "V Add", "S Mul", "G Mul", "V Mul"], category_table)}

The missing section 5.3 is interpreted as cosine parallelism of raw difference
vectors from independent pairs of the same relation. Shared-word comparisons
are excluded. Mean cosine and comparison counts are
{'; '.join(f"{name}: {arithmetic['models'][name]['parallelism']['mean_cosine']:.3f}, "
           f"{arithmetic['models'][name]['parallelism']['comparisons']} comparisons"
           for name in ('sgns', 'gensim', 'glove'))}.

The strongest SGNS category is {best_categories[0][0]} at {best_categories[0][1]:.3f};
the weakest is {best_categories[-1][0]} at {best_categories[-1][1]:.3f}.
Query words can dominate without exclusion, and difference vectors are not
perfectly parallel. Consequently the familiar king−man+woman expression is
approximate retrieval, not equality. A 500-word thematic t-SNE is in the notebook;
neighborhood-based theme selection itself favors visible grouping.

\\newpage

## 4. End-to-end AG News classification

The official 120,000 training documents split into 108,000 train and 12,000
stratified validation documents. TF-IDF fits train only. Mean EmbeddingBag feeds
a 128-unit ReLU/dropout MLP; missing words map to ignored padding, with a zero
representation for all-unknown documents. Two learning rates and validation
macro F1 select epochs and frozen/fine-tuned variants. TF-IDF tunes two values
of C using warm-start logistic regression; its loss curve uses optimizer blocks.
{dimension_caveat}

Validation results below select the better learning rate within each variant.
P/R/F denotes macro precision/recall/F1; parameter counts include embedding rows.

{markdown_table(["Initialization", "Variant", "Acc", "P/R/F", "Total/trainable", "Seconds"], [
    [row["model"], row["variant"], number(row["accuracy"]),
     "/".join(number(row[key]) for key in ("precision_macro", "recall_macro", "f1_macro")),
     f"{row['parameters']:,}/{row['trainable']:,}", f"{row['seconds']:.1f}"]
    for row in validation
])}

The best initialization-specific configuration is selected on validation before
official test. Every selected fraction/configuration receives one cached test
evaluation. No test score changes model settings.

{markdown_table(["Initialization", "100% variant", "Test accuracy", "Test F1, 100%", "Test F1, 1%"], data_efficiency)}

![Official test confusion matrices. W/S/B/T: world, sports, business, science/technology.](figures/confusion_matrices.png){{width=100% height=1.55in}}

Variant selections: {frozen_choice}.
The highest full-data test F1 belongs to {best_test[0]} at {best_test[1]:.3f}.
The 1% experiment contains 1,080 training documents, with validation unchanged.
Coverage and limited labeled data explain why adaptation benefits can differ
between initialization and training fractions.

\\newpage

## 5. Comparison, data efficiency and conclusions

{markdown_table(["Model", "Tokens", "Vocabulary", "Dim", "Train sec", "WordSim", "SimLex"], [
    [row["model"], f"{row['corpus_tokens']:,}", f"{row['vocabulary']:,}", row["dimension"],
     number(row["training_seconds"], 1), number(row["wordsim"]), number(row["simlex"])]
    for row in models
])}

{markdown_table(["Model", "Add S/T/all", "Mul S/T/all", "Parallelism", "AG OOV %", "Test F1"], [
    [row["model"], "/".join(number(row[key]) for key in ("add_semantic", "add_syntactic", "add_total")),
     "/".join(number(row[key]) for key in ("mul_semantic", "mul_syntactic", "mul_total")),
     number(row["parallelism"]), number(row["ag_news_oov_percent"]), number(row["test_f1"])]
    for row in models
])}

S/T denotes semantic/syntactic. SGNS uses the RTX 4050 Laptop 6 GB; gensim uses
four CPU workers. The authors report the GloVe corpus as 6B tokens and 400,000
words. Exact released-100D wall time and CPU model are not specified; NR is
preferable to substituting a different-dimensional timing example.
Similarity coverage is retained in the accompanying comparison CSV and notebook.

![Controlled corpus-size experiment with GloVe reference.](figures/corpus_size.png){{width=72% height=1.55in}}

![Official test F1 versus labeled training fraction.](figures/data_efficiency.png){{width=72% height=1.55in}}

{empirical_discussion}

The corpus-size experiment measures a within-SGNS scaling trend, not a causal
decomposition of the GloVe gap. GloVe differs in corpus composition, tokenization,
objective and training budget. Window and negative-count comparisons must be
read together with category-specific scores; neither guarantees syntactic or
semantic improvement in every category. Gensim discrepancies remain consistent
with the documented optimization differences rather than proof that either
objective is implemented incorrectly. Single-seed results, approximate sentence
splitting and fixed vocabulary limit generalization.

**Sources:** Mikolov et al. 2013, arXiv:1310.4546; Pennington et al. 2014,
EMNLP, nlp.stanford.edu/pubs/glove.pdf; Stanford original GloVe release;
PyTorch Embedding/EmbeddingBag and gensim Word2Vec documentation.
Full links and API research are in `research.md` and the notebook.
Best original vectors: `informe/artifacts/best_sgns.kv`.
"""


def notebook_document():
    import nbformat

    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [
        markdown(
            "# NLP end-to-end and embeddings\n\n"
            "This notebook presents the complete pipeline and its recorded results. "
            "Reusable implementations live in `src/`; displayed source establishes the "
            "training and evaluation methods. Markdown cells explain decisions. "
            "The default execution reuses checkpoints and does not reopen final test."
        ),
        code(
            "from pathlib import Path\nimport sys, inspect\n"
            "ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n"
            "sys.path.insert(0, str(ROOT))\n"
            "import pandas as pd\n"
            "from IPython.display import display, Image, Markdown, HTML\n"
            "from src.common import ARTIFACTS, RESULTS, FIGURES, load_json\n"
            "from src.deliverables import embedding_rows, iteration_rows, category_rows, validation_rows\n"
            "from src.preprocessing import tokenize, sentences, keep_probabilities, pair_batches, prepare\n"
            "from src.sgns import SGNS, CONFIGURATIONS, train_configuration, pilot\n"
            "from src.experiments import train_embeddings, arithmetic\n"
            "from src.evaluation import analogia, analogy_benchmark, parallelism\n"
            "from src.classification import BagClassifier, classification, final_evaluation\n"
            "from src.checks import main as run_checks\n"
            "pd.set_option('display.max_columns', 25)\n"
            "pd.set_option('display.max_colwidth', 130)\n"
            "RUN_TRAINING = False\n"
            "display(pd.Series(load_json(RESULTS / 'environment.json')))"
        ),
        markdown("## 2. Corpus exploration and preprocessing\n\nArticle counts use top-level title markers; body-token counts exclude headings. Token counts before vocabulary filtering and after filtering are reported separately."),
        code("prep = load_json(RESULTS / 'preprocessing.json')\ndisplay(pd.DataFrame(prep['splits']).T)\ndisplay(pd.DataFrame(prep['subsets']).T)"),
        markdown("### Normalization, tokens and IDs\n\nLowercasing matches uncased GloVe. Internal apostrophes and numeric separators are retained; punctuation is removed and hyphens split. WikiText markers are repaired. Unknown corpus words are dropped before generating contexts."),
        code("print(inspect.getsource(tokenize))\nprint(inspect.getsource(sentences))\nwords = load_json(ARTIFACTS / 'vocabulary.json')\nword_to_id = {word: i for i, word in enumerate(words)}\nexample = tokenize(\"The king's well-known computer costs 1 @,@ 000.\")\ndisplay(pd.DataFrame({'token': example, 'id': [word_to_id.get(word, -1) for word in example]}))"),
        code("display(pd.DataFrame(prep['tokenizers']))"),
        markdown("### Zipf and vocabulary coverage\n\nThe log-log slope is an empirical approximation, not a proof of an exact Zipf law. Coverage uses all normalized training-body tokens; threshold comparison uses the smallest nested subset."),
        code("display(Image(filename=str(FIGURES / 'zipf.png')))\ndisplay(pd.Series(prep['coverage'], name='token_coverage'))\nprint('Top-30000 slope:', prep['zipf_slope_top_30000'])\ndisplay(pd.DataFrame(prep['threshold_comparison']))"),
        markdown("### Subsampling and skip-gram pairs\n\nThe reported realized discard rates use the seed shown in environment metadata. Pair counts include both context directions and never cross sentence boundaries."),
        code("print(inspect.getsource(keep_probabilities))\nprint(inspect.getsource(pair_batches))\ndisplay(pd.DataFrame(prep['subsampling']).T)\ndisplay(pd.Series({key: prep[key] for key in ('pairs_before_subsampling', 'pairs_after_subsampling', 'retained_after_subsampling')}))"),
        markdown((ROOT / "informe" / "research.md").read_text(encoding="utf-8")),
        markdown("## 4. Custom SGNS and reference implementations\n\nNo gensim Word2Vec call appears inside the custom SGNS implementation. The input and output tables are distinct, negative draws follow count^0.75, and only input vectors are exported."),
        code("print(inspect.getsource(SGNS))\nprint(inspect.getsource(train_configuration))\nrun_checks()\ndisplay(pd.Series(load_json(RESULTS / 'pilot.json')))"),
        markdown("### Reproduction switches\n\nThe following opt-in cells execute the actual corpus and training stages. They remain disabled during presentation execution. Cached completed experiments are reused."),
        code("if RUN_TRAINING:\n    prepare()\n    pilot()\n    train_embeddings()"),
        code("iterations = pd.DataFrame(iteration_rows())\ndisplay(iterations)\ndisplay(pd.Series(load_json(RESULTS / 'best_sgns.json')))\ndisplay(Image(filename=str(FIGURES / 'embedding_training.png')))\ndisplay(Image(filename=str(FIGURES / 'corpus_size.png')))"),
        markdown("### Five neighbors for fixed control words at every epoch\n\nThese outputs diagnose semantic changes without affecting the prespecified selection metric."),
        code(
            "neighbor_rows = []\n"
            "for config in CONFIGURATIONS:\n"
            "    history = load_json(RESULTS / f'sgns_{config.name}.json')\n"
            "    for epoch in history['epochs']:\n"
            "        for word, values in epoch['neighbors'].items():\n"
            "            neighbor_rows.append({'model': config.name, 'epoch': epoch['epoch'], 'word': word, 'top5': values})\n"
            "gensim_history = load_json(RESULTS / 'gensim.json')\n"
            "for epoch in gensim_history['epochs']:\n"
            "    for word, values in epoch['neighbors'].items():\n"
            "        neighbor_rows.append({'model': 'gensim', 'epoch': epoch['epoch'], 'word': word, 'top5': values})\n"
            "display(HTML(pd.DataFrame(neighbor_rows).to_html(index=False)))\n"
            "display(pd.DataFrame([{k: v for k, v in epoch.items() if k not in ('neighbors', 'analogies')} for epoch in gensim_history['epochs']]))"
        ),
        markdown("## 5. Vector arithmetic\n\n3CosAdd uses normalized source vectors, normalizes the resulting query and excludes its three words. The custom results are checked against gensim. Individual ranks use the complete original model vocabulary."),
        code("print(inspect.getsource(analogia))\nif RUN_TRAINING:\n    arithmetic()\narithmetic_results = load_json(RESULTS / 'arithmetic.json')"),
        code(
            "for name, model in arithmetic_results['models'].items():\n"
            "    print(name)\n"
            "    display(HTML(pd.DataFrame(model['individual']).to_html(index=False)))\n"
            "print('Shared vocabulary:', len(arithmetic_results['shared_words']))\n"
            "display(pd.DataFrame(category_rows()))\n"
            "display(pd.DataFrame({name: {'coverage': model['analogies']['coverage'], 'evaluated': model['analogies']['evaluated'], 'total': model['analogies']['total'], '3CosAdd': model['analogies']['3CosAdd']['total'], '3CosMul': model['analogies']['3CosMul']['total']} for name, model in arithmetic_results['models'].items()}).T)"
        ),
        markdown("### Parallelism and the missing section 5.3\n\nIndependent raw difference vectors from the same relationship type are compared by cosine. Pairs sharing words are excluded. Without query exclusion, a high-ranking query word demonstrates why vector arithmetic should not be interpreted as exact equality."),
        code("print(inspect.getsource(parallelism))\nfor name, model in arithmetic_results['models'].items():\n    print(name, model['parallelism']['mean_cosine'], model['parallelism']['comparisons'])\n    display(pd.DataFrame(model['parallelism']['categories']).T)"),
        markdown("### Thematic t-SNE\n\nFive groups each contribute 100 nonoverlapping words from anchor neighborhoods. Such selection favors visible clustering; this visualization is exploratory, not independent evidence of semantic separation."),
        code("display(Image(filename=str(FIGURES / 'tsne.png')))\ndisplay(pd.DataFrame(load_json(RESULTS / 'tsne.json')['groups']).T)"),
        markdown("## 6. AG News end-to-end classification\n\nThe same tokenizer converts text to IDs, EmbeddingBag averages vectors, and an MLP predicts the class. A stratified 10% validation split is fixed. All TF-IDF fitting, tuning and epoch selection use training/validation only."),
        code("print(inspect.getsource(BagClassifier))\nif RUN_TRAINING:\n    classification()\nselection = load_json(RESULTS / 'classification_selection.json')\ndisplay(pd.Series(selection['split']))\ndisplay(pd.DataFrame(validation_rows()))"),
        code("display(pd.DataFrame([{'initialization': name, 'split': split, **values} for name, splits in selection['oov'].items() for split, values in splits.items()]))\ndisplay(Image(filename=str(FIGURES / 'classification_training.png')))"),
        markdown("### Training and validation curves for every classifier variant\n\nPer-trial histories retain all four validation metrics, times and both losses. The figure above includes both frozen and fine-tuned full-data variants, random embeddings and TF-IDF."),
        code(
            "trial_epochs = [{'model': row['key'], **epoch} for row in selection['trials'] for epoch in row['epochs']]\n"
            "display(HTML(pd.DataFrame(trial_epochs).to_html(index=False)))\n"
            "display(pd.DataFrame([{'key': key, 'variant': row.get('variant', 'logistic'), 'fraction': row['fraction'], **row['validation']} for key, row in selection['selected'].items()]))"
        ),
        markdown("### Locked final evaluation\n\nThe final stage is opt-in, checks the selection hash, and caches each selected test result. SimLex-999 is evaluated only here for the original embeddings, not their classification-tuned copies."),
        code(
            "if RUN_TRAINING:\n"
            "    final_evaluation()\n"
            "final = load_json(RESULTS / 'final_evaluation.json')\n"
            "display(pd.DataFrame(final['models']).T)\n"
            "display(pd.DataFrame(final['simlex']).T)\n"
            "display(Image(filename=str(FIGURES / 'confusion_matrices.png')))\n"
            "display(Image(filename=str(FIGURES / 'data_efficiency.png')))"
        ),
        markdown("## 7. Full comparison\n\nThe GloVe corpus and vocabulary come from the original release. Its exact 100D training time and CPU model are unavailable. Every metric includes its appropriate coverage denominator in the saved JSON."),
        code("comparison = pd.DataFrame(embedding_rows())\ndisplay(comparison)\nprint('Best original vectors:', ARTIFACTS / 'best_sgns.kv')"),
        markdown("## 8. Discussion and conclusions\n\nThe following report provides the numerical interpretation, including dimension, context, negatives, corpus size, arithmetic limitations and performance at 1% labeled data."),
        code(
            "report = (ROOT / 'informe' / 'report.md').read_text(encoding='utf-8')\n"
            "pages = report.split('\\\\newpage')\n"
            "display(Markdown('### Dimensions, corpus size and loss\\n\\nThe selected model is' + pages[1].split('The selected model is')[1]))\n"
            "display(Markdown('### Analogy strengths and limits\\n\\nThe strongest SGNS category is' + pages[2].split('The strongest SGNS category is')[1]))\n"
            "display(Markdown('### Freezing, adaptation and labeled data\\n\\nVariant selections:' + pages[3].split('Variant selections:')[1]))\n"
            "display(Markdown('### Direct comparison and limitations\\n\\nSGNS semantic/syntactic' + pages[4].split('SGNS semantic/syntactic')[1]))"
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Lab7 existing Python environment", "language": "python", "name": "lab7"}
    return notebook


def execute_notebook(notebook):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    kernel_directory = ARTIFACTS / "kernels" / "lab7"
    kernel_directory.mkdir(parents=True, exist_ok=True)
    save_json(kernel_directory / "kernel.json", {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Lab7 existing Python environment", "language": "python",
    })
    manager = KernelManager(
        kernel_name="lab7",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_directory.parent)]),
    )
    client = NotebookClient(
        notebook, km=manager, timeout=180, resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    path = ROOT / "notebooks" / "nlp_embeddings.ipynb"
    path.parent.mkdir(exist_ok=True)
    nbformat.write(notebook, path)
    client.execute()
    nbformat.write(notebook, path)
    return path


def export_tables():
    tables = {
        "embedding_comparison": embedding_rows(),
        "embedding_iterations": iteration_rows(),
        "analogy_categories": category_rows(),
        "classification_validation": validation_rows(),
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(RESULTS / f"{name}.csv", index=False)


def validate_deliverables(require_pdf=True):
    import nbformat
    from gensim.models import KeyedVectors

    best = load_json(RESULTS / "best_sgns.json")
    arithmetic = load_json(RESULTS / "arithmetic.json")
    final = load_json(RESULTS / "final_evaluation.json")
    from .sgns import CONFIGURATIONS
    from .classification import FRACTIONS

    for config in CONFIGURATIONS:
        history = load_json(RESULTS / f"sgns_{config.name}.json")
        assert len(history["epochs"]) == config.epochs
        assert history["actual_normalized_tokens"] >= 20_000_000
        assert history["retained_tokens"] >= 20_000_000
        assert all(len(row["neighbors"]) == 6 for row in history["epochs"])
    assert len(arithmetic["shared_words"]) == 30000
    for name, row in arithmetic["models"].items():
        assert len(row["individual"]) >= 15
        assert len(row["analogies"]["3CosAdd"]["categories"]) == 14
        assert all(example.get("gensim_match", True) for example in row["individual"])
        assert final["simlex"][name]["total"] == 999
        assert row["wordsim"]["total"] == 353
    for name in ("tfidf", "random", "sgns", "gensim", "glove"):
        assert all(f"{name}_f{fraction:g}" in final["models"] for fraction in FRACTIONS)
    path = ROOT / "notebooks" / "nlp_embeddings.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.get("execution_count") is not None for cell in notebook.cells if cell.cell_type == "code")
    assert not any(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs)
    vectors = KeyedVectors.load(str(ARTIFACTS / "best_sgns.kv"))
    assert vectors.vector_size == best["configuration"]["dimension"]
    checks = {
        "complete_embedding_runs": len(CONFIGURATIONS),
        "test_configurations": len(final["models"]),
        "executed_notebook_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
        "exported_vectors": len(vectors),
        "vectors_sha256": hashlib.sha256((ARTIFACTS / "best_sgns.kv").read_bytes()).hexdigest(),
    }
    if require_pdf:
        from pypdf import PdfReader
        from zipfile import ZIP_DEFLATED, ZipFile

        pdf = ROOT / "informe" / "report.pdf"
        assert pdf.read_bytes().startswith(b"%PDF-")
        pages = len(PdfReader(pdf).pages)
        assert 1 <= pages <= 5, f"Report has {pages} pages, maximum is five."
        checks["pdf_pages"] = pages
        archive_path = ARTIFACTS / "submission.zip"
        files = [
            ROOT / "README.md", ROOT / "requirements.txt",
            ROOT / "notebooks" / "nlp_embeddings.ipynb",
            ROOT / "informe" / "report.md", pdf, ROOT / "informe" / "research.md",
            ARTIFACTS / "best_sgns.kv",
            *sorted((ROOT / "src").glob("*.py")), ROOT / "src" / "finish.ps1",
            *sorted(RESULTS.glob("*.json")), *sorted(RESULTS.glob("*.csv")),
            *sorted(FIGURES.glob("*.png")),
        ]
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for file in files:
                archive.write(file, file.relative_to(ROOT).as_posix())
        with ZipFile(archive_path) as archive:
            assert archive.testzip() is None
            assert "informe/artifacts/best_sgns.kv" in archive.namelist()
        checks["submission_archive"] = str(archive_path.relative_to(ROOT))
    save_json(RESULTS / "deliverable_checks.json", checks)
    print(checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args()
    if arguments.validate:
        validate_deliverables()
        return
    if not (RESULTS / "final_evaluation.json").exists():
        raise RuntimeError("Final experiments are incomplete; a results report cannot be generated yet.")
    from gensim.models import KeyedVectors

    vectors = KeyedVectors.load(str(ARTIFACTS / "best_sgns.kv"))
    vectors.save(str(ARTIFACTS / "best_sgns.kv"), separately=[])
    export_tables()
    (ROOT / "informe" / "report.md").write_text(report_text(), encoding="utf-8")
    execute_notebook(notebook_document())
    validate_deliverables(require_pdf=False)


if __name__ == "__main__":
    main()
