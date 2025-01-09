import os
from typing import Any, Optional, cast

from kotaemon.base import DocumentWithEmbedding
from llama_index.vector_stores.postgres import PGVectorStore as LIPGVectorStore

from .base import LlamaIndexVectorStore

import traceback
import inspect
import logging
import sys

import psycopg2
from kotaemon.base import DocumentWithEmbedding
from sqlalchemy import make_url
from llama_index.core.schema import TextNode  # Use TextNode instead of Node

from llama_index.core.query_engine import RetrieverQueryEngine

# OpenAI dependencies
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from .retriever import VectorDBRetriever

# model constants
embed_model_name = "text-embedding-3-small"
generating_model_name = "gpt-4o-mini"

#class PostgresPGVectorStore(LlamaIndexVectorStore):
class PostgresPGVectorStore:
    # _li_class = None

    # def _get_li_class(self):
    #     try:
    #         from llama_index.vector_stores.postgres import (
    #             PGVectorStore as LIPGVectorStore,
    #         )
    #     except ImportError:
    #         raise ImportError(
    #             "Please install missing package: "
    #             "'pip install llama-index-vector-stores-postgres'"
    #         )

        # return LIPGVectorStore

    def __init__(
        self,
        *args, 
        **kwargs
    ):
        
        print("\n=== New PostgresPGVectorStore Instance ===")
        print("Called from:")
        for frame in traceback.extract_stack()[:-1]:  # [:-1] excludes this line
            print(f"  File {frame.filename}, line {frame.lineno}, in {frame.name}")

        # Uncomment to see debug logs
        print("A. PostgresPGVectorStore: Initializing")
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

        print(f'***** PostgresPGVectorStore: Initializing *****')


        connection_string = "postgresql://postgres:password@localhost:5432"
        db_name = "vector_db"
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True

        with conn.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {db_name}")
            c.execute(f"CREATE DATABASE {db_name}")
        
        from sqlalchemy import make_url
        self.url = make_url(connection_string)

        print(f"url -- host: {self.url.host}, port: {self.url.port}, user: {self.url.username}, password: {self.url.password} ")

        # create a pgvector store client
        self.vector_store = LIPGVectorStore.from_params(
            database=db_name,
            host=self.url.host,
            password=self.url.password,
            port=self.url.port,
            user=self.url.username,
            table_name="generic_table",
            embed_dim=1536,  # openai embedding dimension
        )
        
        # Debug: Print available methods
        # print("\nAvailable PGVectorStore methods:")
        # for name, method in inspect.getmembers(self.vector_store, predicate=inspect.ismethod):
        #     print(f"- {name}: {inspect.signature(method)}")
       
        # self.url = make_url(connection_string)
        # print(f"url -- host: {self.url.host}, port: {self.url.port}, user: {self.url.username}, password: {self.url.password} ")

        self.embed_model = OpenAIEmbedding(model=embed_model_name) 
        self.llm = OpenAI(model=generating_model_name) # api_key="some key",  # uses OPENAI_API_KEY env var by default
        self.retriever = VectorDBRetriever(vector_store=self.vector_store, embed_model=self.embed_model)


    def add(
        self,
        embeddings: list[list[float]] | list[DocumentWithEmbedding],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None
    ) -> list[str]:
        """Add embeddings to vector store"""
        nodes = []
        for i, emb in enumerate(embeddings):
            # Extract embedding correctly based on type
            if isinstance(emb, DocumentWithEmbedding):
                embedding_value = emb.embedding  # Get embedding from DocumentWithEmbedding
            else:
                embedding_value = emb if isinstance(emb, list) else emb.tolist()
            
            node = TextNode(
                text="",
                embedding=embedding_value,
                id_=ids[i] if ids else None,
                metadata=metadatas[i] if metadatas else {}
            )
            nodes.append(node)
        # print(f'***** nodes {nodes}*****')
        return self.vector_store.add(nodes=nodes)
    
   
    def query(
            self,
            embedding: list[float],
            top_k: int = 1,
            ids: Optional[list[str]] = None,
            **kwargs,
    ) -> tuple[list[list[float]], list[float], list[str]]:
        print(f'***** PostgresPGVectorStore query method *****')
        query_engine = RetrieverQueryEngine.from_args(self.retriever, llm=self.llm)
        return query_engine.retriever.query(embedding, top_k, ids)
        

    def delete(self, ids: list[str], **kwargs):
        print(f'***** PostgresPGVectorStore delete method *****')
        print(f'***** ids {ids} *****')
        self.vector_store.delete_nodes(ids)
