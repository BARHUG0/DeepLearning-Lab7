import numpy as np

from .preprocessing import keep_probabilities, pair_batches, pair_count, sentences, tokenize


def check_preprocessing():
    assert tokenize("King’s well-known car costs 1 @,@ 000.") == [
        "king's", "well", "known", "car", "costs", "1,000",
    ]
    assert sentences(" = Article title = ") == []
    assert sentences("King runs. Queen walks.") == [["king", "runs"], ["queen", "walks"]]
    ids = np.asarray([0, 1, 2, 3, 4])
    sentence_ids = np.asarray([0, 0, 0, 1, 1])
    actual = []
    for centers, contexts in pair_batches(ids, sentence_ids, 2, batch_size=2, chunk_size=2):
        actual.extend(zip(centers.tolist(), contexts.tolist()))
    expected = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (3, 4), (4, 3)}
    assert set(actual) == expected and len(actual) == len(expected)
    assert pair_count([3, 2], 2) == len(expected)
    probabilities = keep_probabilities([100000, 10], 1e-4)
    assert 0 < probabilities[0] < probabilities[1] <= 1
    assert np.array_equal(keep_probabilities([5, 5], 0), [1, 1])


def main():
    check_preprocessing()
    check_sgns()
    check_analogies()
    check_embedding_bag()
    check_classification()
    print("Preprocessing, SGNS, analogy, EmbeddingBag, and split checks passed.")


def check_sgns():
    import torch
    from .sgns import SGNS, negative_distribution

    torch.manual_seed(1)
    model = SGNS(5, 2)
    with torch.no_grad():
        model.input.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.], [0., -1.]]))
        model.output.weight.copy_(model.input.weight)
    centers = torch.tensor([0, 1])
    contexts = torch.tensor([2, 2])
    negatives = torch.tensor([[3, 4], [3, 4]])
    losses = model(centers, contexts, negatives)
    expected = np.logaddexp(0, -1) + np.logaddexp(0, -1) + np.logaddexp(0, 0)
    assert np.allclose(losses.detach().numpy(), expected)
    losses.sum().backward()
    assert model.input.weight.grad.is_sparse and model.output.weight.grad.is_sparse
    assert model.input.weight.grad.coalesce().values().abs().sum() > 0
    assert model.output.weight.grad.coalesce().values().abs().sum() > 0
    distribution = negative_distribution([1, 16])
    assert np.allclose(distribution, [1 / 9, 8 / 9])
    rng = np.random.default_rng(1)
    samples = rng.choice(2, 100000, p=distribution)
    assert abs((samples == 0).mean() - distribution[0]) < 0.005


def check_analogies():
    from gensim.models import KeyedVectors
    from .evaluation import analogia, analogy_scores
    import torch

    vectors = KeyedVectors(3)
    vectors.add_vectors(
        ["man", "woman", "king", "queen", "other"],
        np.asarray([[1., 0., 0.], [1., 1., 0.], [1., 0., 1.], [1., 1., 1.], [-1., 0., 0.]]),
    )
    actual = analogia(vectors, "man", "woman", "king", 2)
    expected = vectors.most_similar(positive=["woman", "king"], negative=["man"], topn=2)
    assert [word for word, _ in actual] == [word for word, _ in expected]
    assert np.allclose([score for _, score in actual], [score for _, score in expected], atol=1e-6)
    matrix = torch.nn.functional.normalize(torch.tensor(vectors.vectors), dim=1)
    assert torch.isfinite(analogy_scores(matrix, 0, 1, 2, "3CosMul")).all()
    assert len(analogia(vectors, "man", "woman", "king", 5, exclude=False)) == 5


def check_embedding_bag():
    import torch
    from torch import nn

    weights = torch.tensor([[0., 0.], [1., 2.], [3., 4.]])
    indices = torch.tensor([1, 2, 2])
    offsets = torch.tensor([0, 2, 3])
    for mode, expected in (
        ("mean", [[2., 3.], [3., 4.]]),
        ("sum", [[4., 6.], [3., 4.]]),
        ("max", [[3., 4.], [3., 4.]]),
    ):
        bag = nn.EmbeddingBag.from_pretrained(weights, mode=mode, include_last_offset=True)
        assert torch.allclose(bag(indices, offsets), torch.tensor(expected))


def check_classification():
    from .classification import BagClassifier, collate_bags, fraction_indices
    import torch

    labels = np.repeat(np.arange(4), 100)
    train = np.arange(len(labels))
    small = fraction_indices(train, labels, 0.01)
    medium = fraction_indices(train, labels, 0.1)
    assert len(small) == 4 and len(medium) == 40
    assert set(small).issubset(set(medium))
    assert np.array_equal(np.bincount(labels[small]), [1, 1, 1, 1])
    ids, offsets, target = collate_bags([([1, 2], 0), ([], 1), ([0], 2)])
    assert offsets.tolist() == [0, 2, 2, 3]
    weights = np.asarray([[0., 0.], [1., 2.], [3., 4.]], dtype=np.float32)
    frozen = BagClassifier(weights, freeze=True)
    trainable = BagClassifier(weights, freeze=False)
    assert not frozen.embedding.weight.requires_grad
    assert trainable.embedding.weight.requires_grad
    logits = trainable(ids, offsets)
    assert logits.shape == (3, 4) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, target).backward()
    assert torch.equal(trainable.embedding.weight.grad[0], torch.zeros(2))
    optimizer = torch.optim.Adam(trainable.parameters(), lr=0.01)
    trainable.eval()
    initial = float(torch.nn.functional.cross_entropy(trainable(ids, offsets), target).detach())
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(trainable(ids, offsets), target)
        loss.backward()
        optimizer.step()
    final = float(torch.nn.functional.cross_entropy(trainable(ids, offsets), target).detach())
    assert final < initial
    assert torch.equal(trainable.embedding.weight[0], torch.zeros(2))


if __name__ == "__main__":
    main()
