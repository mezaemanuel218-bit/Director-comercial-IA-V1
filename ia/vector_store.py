import numpy as np

class VectorStore:

    def __init__(self):

        self.texts = []
        self.vectors = []

    def add(self, text, vector):

        self.texts.append(text)
        self.vectors.append(vector)

    def search(self, query_vector, top=5):

        sims = []

        for v in self.vectors:

            sim = np.dot(query_vector, v) / (
                np.linalg.norm(query_vector) * np.linalg.norm(v)
            )

            sims.append(sim)

        idx = np.argsort(sims)[-top:]

        return [self.texts[i] for i in idx]