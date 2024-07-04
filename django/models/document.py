from django.db import models
from django.utils.text import Truncator
from django.utils.translation import gettext as _
from pgvector.django import VectorField
from common.services.ai.embedding import EmbeddingModel, EmbeddingService
from apps.knowledge.constants import DocumentType


class Document(models.Model):
    embedding = VectorField(
        dimensions=EmbeddingModel.TEXT_EMBEDDING_ADA_002.embedding_dimension,
    )

    tokens = models.PositiveIntegerField(
        verbose_name=_("Tokens"),
    )

    document_type = models.CharField(
        max_length=25,
        choices=DocumentType.choices,
        verbose_name=_("Document Type"),
        default=DocumentType.DESCRIPTION,
    )

    text = models.CharField(
        max_length=1024,
        verbose_name=_("Document"),
    )

    language = models.CharField(
        max_length=5,
        verbose_name=_("Language"),
        default="sv",
    )

    def save(self, *args, **kwargs):
        embedding_service = EmbeddingService()
        self.embedding = embedding_service.generate_embedding(self.text)[0]
        self.tokens = len(embedding_service.tokenize_document(self.text))
        super(Document, self).save(*args, **kwargs)

    def __str__(self):
        return Truncator(self.text).chars(50)

    class Meta:
        verbose_name = _("Document")
        verbose_name_plural = _("Documents")
        unique_together = ("text", "document_type", "language")
