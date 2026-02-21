from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RAGConfig:
    top_k: int


class HierarchicalRetriever:
    def __init__(self, cfg: RAGConfig, knowledge: dict[str, list[str]]):
        self.cfg = cfg
        self.knowledge = knowledge
        self._index = {}
        self._fit()

    def _fit(self) -> None:
        for key, docs in self.knowledge.items():
            if not docs:
                continue
            vec = TfidfVectorizer()
            mat = vec.fit_transform(docs)
            self._index[key] = (docs, mat, vec)

    def retrieve(self, query: str) -> dict[str, list[str]]:
        out = {}
        for key, payload in self._index.items():
            docs, mat, vec = payload
            q = vec.transform([query])
            sims = cosine_similarity(q, mat).reshape(-1)
            top = sims.argsort()[::-1][: self.cfg.top_k]
            out[key] = [docs[i] for i in top]
        return out
