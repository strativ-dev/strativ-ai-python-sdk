# Standard library imports

# Django imports

# Django Rest Framework imports

# Third party imports

# Local imports
from apps.knowledge.constants import DocumentType
from apps.knowledge.models import Document
from common.services.ai.embedding import EmbeddingService
from pgvector.django import CosineDistance


class KnowledgeService:
    @staticmethod
    def add_document_to_knowledge_base(text: str, document_type: str):
        """
        Add a document to the knowledge base
        """
        Document.objects.create(
            embedding=EmbeddingService().generate_embedding(text),
            tokens=len(EmbeddingService().tokenize_document(text)),
            text=text,
            document_type=document_type,
        )

    def get_document_from_knowledge_base(
        self,
        content_to_match: str,
        document_type: DocumentType = DocumentType.DESCRIPTION,
        custom_threshold: float = 0.3,
        language: str = "sv",
    ):
        content_to_match_embedding = EmbeddingService().generate_embedding(
            content_to_match
        )[0]

        best_document = (
            Document.objects.filter(
                document_type=document_type, language=language
            )
            .annotate(
                distance=CosineDistance("embedding", content_to_match_embedding)
            )
            .order_by("distance")
            .first()
        )

        if best_document and best_document.distance < custom_threshold:
            return best_document
        else:
            # TODO: This should instead raise an exception which needs to be handled.
            return None

    def get_documents_from_knowledge_base(
        self,
        content_to_match: str,
        document_type: DocumentType = DocumentType.DESCRIPTION,
        custom_threshold: float = 0.3,
        number_of_best_matches: int = 5,
        language: str = "sv",
    ):
        content_to_match_embedding = EmbeddingService().generate_embedding(
            content_to_match
        )[0]

        best_documents = (
            Document.objects.filter(
                document_type=document_type, language=language
            )
            .annotate(
                distance=CosineDistance("embedding", content_to_match_embedding)
            )
            .order_by("distance")[:number_of_best_matches]
        )

        best_documents = [
            document
            for document in best_documents
            if document.distance < custom_threshold
        ]

        return best_documents
