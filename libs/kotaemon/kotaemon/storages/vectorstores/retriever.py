
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List, Optional
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore


class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 5,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()


    def query(
            self,
            embedding: list[float],
            top_k: int = 1,
            ids: Optional[list[str]] = None,
            **kwargs,
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """This one fits well with the kotaemon format."""
        vector_store_query = VectorStoreQuery(
            query_embedding=embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)


        nodes, scores, ids = [], [], []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes.append(node)
            scores.append(score)
            ids.append(query_result.ids[index])
        return nodes, scores, ids

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve.
        score is the cosine similarity between the query and the nodes
        """
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores