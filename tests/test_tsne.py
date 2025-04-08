import numpy as np
from sklearn.manifold import TSNE
import pytest

def validate_tsne_input(word_vectors, perplexity):
    if perplexity >= len(word_vectors):
        raise ValueError("perplexity must be less than n_samples")

def test_tsne_perplexity():
    word_vectors = np.random.rand(10, 5)  # 10 samples, 5 features
    perplexity = 5
    validate_tsne_input(word_vectors, perplexity)

    perplexity = 10
    with pytest.raises(ValueError, match="perplexity must be less than n_samples"):
        validate_tsne_input(word_vectors, perplexity)