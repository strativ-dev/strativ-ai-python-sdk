import os
from collections import namedtuple
from enum import Enum

from langchain_openai import OpenAIEmbeddings

import tiktoken


class EmbeddingModel(
    namedtuple("Embedding", ["model_name", "embedding_dimension"]),
    Enum,
):
    """Maintains a list of available embedding models"""

    TEXT_EMBEDDING_ADA_002 = ("text-embedding-ada-002", 1536)
    TEXT_EMBEDDING_3_SMALL = ("text-embedding-3-small", 512)


DEFAULT_EMBEDDING_MODEL = EmbeddingModel.TEXT_EMBEDDING_ADA_002


class EmbeddingService:
    """
    Service to generate embeddings for any given document using ANY
    embedding model available to us. For now, defaults to using OpenAI's Text
    Embedding Ada 002 model
    """

    def __init__(
        self, embedding_model: EmbeddingModel = DEFAULT_EMBEDDING_MODEL
    ) -> None:
        # TODO: Need to do something about this garbage code later on.
        if embedding_model == EmbeddingModel.TEXT_EMBEDDING_ADA_002:
            self.embedding_model = OpenAIEmbeddings(
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                model=EmbeddingModel.TEXT_EMBEDDING_ADA_002.model_name,
                openai_organization=os.environ.get("OPENAI_ORGANIZATION_ID"),
            )
        elif embedding_model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL:
            self.embedding_model = OpenAIEmbeddings(
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                model=EmbeddingModel.TEXT_EMBEDDING_3_SMALL.model_name,
                openai_organization=os.environ.get("OPENAI_ORGANIZATION_ID"),
            )

    @staticmethod
    def tokenize_document(document: str) -> list[int]:
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode(document)

    @staticmethod
    def is_valid_document(document: str) -> bool:
        if not document.strip():
            raise ValueError("Document cannot be empty")

        if len(EmbeddingService.tokenize_document(document)) > 8000:
            raise ValueError("Document cannot exceed 8000 tokens")

        return True

    def generate_embedding(self, document: str) -> list[float]:
        if EmbeddingService.is_valid_document(document):
            return self.embedding_model.embed_documents([document])
