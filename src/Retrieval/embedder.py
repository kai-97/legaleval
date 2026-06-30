import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps a sentence-transformers model so it loads exactly once."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Returns a float32 array of shape (len(texts), self.dim),
        each row scaled to unit length.
        """

        return self.model.encode(texts, normalize_embeddings=True,
                                 convert_to_numpy=True,
                                 batch_size=64,
                                 show_progress_bar=True)


if __name__ == "__main__":
    # Sanity check: meaning-clustering should show up in raw scores.
    embedder = Embedder()

    samples = [
        "pregnant employee fired",        # 0
        "overtime pay owed",              # 1
        "termination during pregnancy",   # 2
    ]

    vecs = embedder.embed(samples)
    print("shape:", vecs.shape)

    print("0-2 (should be higher):", vecs[0].dot(vecs[2]))
    print("0-1 (should be lower): ", vecs[0].dot(vecs[1]))

    # This proves normalization works and that "fired while pregnant"
    # sits nearer "termination during pregnancy" than "overtime pay".