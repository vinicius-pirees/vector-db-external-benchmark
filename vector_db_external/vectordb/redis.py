import logging
import os
from typing import Any, Optional, List

import numpy as np
from pydantic import SecretStr

import redis
from redis.commands.search.field import (
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from .vectordb_api import DBConfig, VectorDB
from .search_result import EmbeddingSearchResult


log = logging.getLogger(__name__)


class RedisConfig(DBConfig):
    password: SecretStr
    host: SecretStr
    port: SecretStr


def default_config():
    return RedisConfig(
        password=os.environ["REDIS_PASSWORD"],
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
    )


class Redis(VectorDB):
    def __init__(
        self,
        database_name: str = "vector_store_benchmark",
        vector_dimension: int = 1536,
        db_config: DBConfig | None = None,
        drop_old: bool = False,
        **kwargs,
    ):
    
        self.db_config = db_config if db_config is not None else default_config()
        self.index_name = database_name
        self.doc_prefix = "doc:"

        conn = redis.Redis(
            host=self.db_config.host.get_secret_value(),
            port=self.db_config.port.get_secret_value(),
            password=self.db_config.password.get_secret_value(),
            db=0,
        )
        self.__make_index(vector_dimension, conn)
        conn.close()
        conn = None


    def remove_index(self):
        try:
            self.conn.ft(self.index_name).dropindex()
        except ResponseError:
            print(f"index {self.index_name} does not exist")


    def __make_index(self, vector_dimensions: int, conn):
        try:
            conn.ft(self.index_name).info()
        except Exception:
            schema = (
                TextField("text_id"),
                TagField("metadata", separator=","),
                TextField("document"),
                VectorField(
                    "vector",
                    "HNSW", # Vector Index Type: FLAT or HNSW
                    {  
                        "TYPE": "FLOAT32",  # FLOAT32 or FLOAT64
                        "DIM": vector_dimensions,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            definition = IndexDefinition(prefix=[self.doc_prefix], index_type=IndexType.HASH)
            rs = conn.ft(self.index_name)
            rs.create_index(schema, definition=definition)

        self.conn = redis.Redis(
            host=self.db_config.host.get_secret_value(),
            port=self.db_config.port.get_secret_value(),
            password=self.db_config.password.get_secret_value(),
            db=0,
        )

    def insert_embeddings(
        self,
        ids: list[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadata: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> None:
        """Insert embeddings into the database.

        Args:
            embeddings(list[list[float]]): list of documents' embeddings
            documents(list[str]): list of textual documents
            ids(list[str]): list of ids for each given document
            metadata(dict[str, str]): dict of key and value for metadata
        """
        batch_size = 1000
        with self.conn.pipeline(transaction=False) as pipe:
            for i, embedding in enumerate(embeddings):
                embedding = np.array(embedding).astype(np.float32)
                mapping = {
                    "text_id": ids[i],
                    "vector": embedding.tobytes(),
                    
                }

                if documents:
                    mapping.update({"document": documents[i]})

                if metadata:
                    mapping.update(
                        {
                            "metadata": ",".join(
                                [f"{k}:{v}" for k, v in metadata[i].items()]
                            )
                        }
                    )
                pipe.hset(
                    f"doc:{ids[i]}",
                    mapping=mapping,
                )
                if i % batch_size == 0:
                    pipe.execute()

            pipe.execute()

    # #TODO: implement filters as expected
    def search_embedding(
        self,
        query: list[float],
        k: int = 10,
        filters: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Search embeddings from the database.

        Args:
            query(list[float]): embedding to use as a search query
            k(int): number of results to return
            filters(dict[str, str]): dict of key and value for filtering on metadata
            kwargs: other arguments



            Filter example: @brand:{nike} @country:{brazil}


        """
        query_vector = np.array(query).astype(np.float32).tobytes()

        query_prefix = "*"

        if filters:
            query_prefix = ""
            for meta_key, meta_value in filters.items():
                query_prefix += "@metadata:{" + str(meta_key) + "\:" + str(meta_value) + "} "

            query_prefix = query_prefix.strip()
    
        



        query_obj = (
            Query(f"({query_prefix})=>[KNN {k} @vector $vec as distance]")
            .sort_by("distance")
            .return_fields("id", "text_id", "distance","document","metadata")
            .paging(0, k)
            .dialect(2)
        )
        query_params = {"vec": query_vector}
        results = self.conn.ft(self.index_name).search(query_obj, query_params).docs

        ids = []
        documents = []
        metadatas = []

        for doc in results:
            ids.append(doc.text_id)
            
            if hasattr(doc, 'document'):
                documents.append(doc.document)
            else:
                documents.append(None)

            if hasattr(doc, 'metadata'):
                meta_components = {}
                for meta in doc.metadata.split(","):
                    key, value = meta.split(":")
                    meta_components[key] = value
                metadatas.append(meta_components)
            else:
                metadatas.append({})





        parsed_result = {
            "ids": ids,
            "embeddings": None,
            "documents": documents,
            "metadatas": metadatas
        }



        embedding_search_result = EmbeddingSearchResult(**parsed_result)

        return embedding_search_result
