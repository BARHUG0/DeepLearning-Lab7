# NLP end-to-end and embeddings

Implementation of `npl_embeddings.md`. Reusable code lives in `src/`, the executed
presentation in `notebooks/`, and the report and recorded results in `informe/`.
Large datasets and checkpoints are cached locally and excluded from Git.

## Reproduction on Windows

The existing parent `.venv` is used. No new environment is created.

```powershell
uv pip install --python ..\.venv\Scripts\python.exe -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
..\.venv\Scripts\python.exe -m src.checks
..\.venv\Scripts\python.exe -m src.run prepare
..\.venv\Scripts\python.exe -m src.run pilot
..\.venv\Scripts\python.exe -m src.run embeddings
..\.venv\Scripts\python.exe -m src.run arithmetic
..\.venv\Scripts\python.exe -m src.run classification
..\.venv\Scripts\python.exe -m src.run final
..\.venv\Scripts\python.exe -m src.deliverables
quarto render informe/report.md --to pdf
..\.venv\Scripts\python.exe -m src.deliverables --validate
```

Commands must run from the repository root. The pilot is a numerical and
throughput check, not an assignment experiment. Completed embedding epochs are
checkpointed. Finished experiments are reused, while incomplete epochs restart
from the preceding checkpoint.

`.\src\finish.ps1` runs the complete sequence, executes the notebook, renders
the PDF through Quarto, and verifies all experiments and the five-page limit.
It never creates commits or pushes. If any stage fails, it stops with its error.

The official AG News test set and SimLex-999 are read only by the final stage.
Final-test output is cached and never used for configuration selection.

## Protocol

- WikiText-103 training data supplies nested subsets of at least 20, 30, and
  40 million normalized tokens. The smallest subset extends to at least
  20 million retained vocabulary tokens, so filtering cannot violate the minimum.
- The embedding vocabulary is selected from the smallest subset and held fixed
  across ablations. Only observed vocabulary items contribute training pairs.
- Seven controlled SGNS configurations vary dimensions, corpus size, window,
  and negatives. Each runs three epochs.
- Epoch selection uses accuracy on a fixed shared set of analogy questions,
  then WordSim-353 correlation. SimLex is held out.
- Gensim uses the selected SGNS configuration, the same encoded sentences,
  vocabulary, and equivalent settings. Its online optimization differs from
  PyTorch's batched summed-gradient updates.
- Random, SGNS, gensim, and GloVe classifiers use the same MLP search budget.
  Frozen and trainable tables are compared where applicable.
- Training fractions are 1%, 10%, 50%, and 100%. Stratified nested samples and
  a fixed validation set prevent data-size comparisons from changing the split.
- The missing section 5.3 is interpreted as cosine parallelism between
  independent word-pair difference vectors of the same relationship.
- The gensim classifier is included to supply the test F1 required in section 7.

The best original SGNS vectors are exported as
`informe/artifacts/best_sgns.kv`. They are not replaced by classification-tuned
vectors. Final validation creates `informe/artifacts/submission.zip` with the
executed notebook, PDF, standalone vectors, source code and recorded results.
The archive supplies the large vectors without adding training caches to Git.
