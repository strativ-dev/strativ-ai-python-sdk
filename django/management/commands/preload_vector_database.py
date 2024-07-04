# Standard library imports

# Django imports
from django.core.management.base import BaseCommand

# Django Rest Framework imports

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from apps.knowledge.models import Document


class Command(BaseCommand):
    help = "Load vector database using pgvector into PostgreSQL"

    def handle(self, *args, **options):
        answer_knowledge_base = pd.read_csv(
            "apps/knowledge/fixtures/answer_embeddings_fixture.csv", index_col=0
        )
        answer_knowledge_base["embeddings"] = (
            answer_knowledge_base["embeddings"].apply(eval).apply(np.array)
        )

        topics_knowledge_base = pd.read_csv(
            "apps/knowledge/fixtures/topic_embeddings_fixture.csv", index_col=0
        )
        topics_knowledge_base["embeddings"] = (
            topics_knowledge_base["embeddings"].apply(eval).apply(np.array)
        )

        self.answer_knowledge_base = answer_knowledge_base
        self.topics_knowledge_base = topics_knowledge_base

        documents = []

        for answer in answer_knowledge_base.iloc[1:].iterrows():
            documents.append(
                Document(
                    embedding=answer[1]["embeddings"],
                    tokens=answer[1]["n_tokens"],
                    text=answer[1]["text"],
                    document_type="description",
                )
            )

        for topic in topics_knowledge_base.iloc[1:].iterrows():
            documents.append(
                Document(
                    embedding=topic[1]["embeddings"],
                    tokens=topic[1]["n_tokens"],
                    text=topic[1]["text"],
                    document_type="topic",
                )
            )

        documents = Document.objects.bulk_create(
            documents, unique_fields=["text"], ignore_conflicts=True
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Vector database upserted successfully with {Document.objects.count()} documents"
            )
        )
