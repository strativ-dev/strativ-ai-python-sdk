from embedding import EmbeddingService, EmbeddingModel


class TestEmbeddingService:
    def test_get_embedding(self):
        """Generate an embedding for any document passed to the service"""
        embedding_service = EmbeddingService(
            embedding_model=EmbeddingModel.TEXT_EMBEDDING_ADA_002,
        )

        embedding = embedding_service.generate_embedding("Hello, how are you?")
        assert embedding is not None

    def test_ada_embedding(self):
        """Generate an embedding using the ada 002 model for any document passed to the service"""
        ada_002_model = EmbeddingModel.TEXT_EMBEDDING_ADA_002

        embedding_service = EmbeddingService(
            embedding_model=ada_002_model,
        )

        embedding = embedding_service.generate_embedding("Hello, how are you?")
        assert embedding is not None
        assert len(embedding[0]) == ada_002_model.embedding_dimension

    def test_default_embedding(self):
        """Generate an embedding using the default model for any document passed to the service"""
        embedding_service = EmbeddingService()

        embedding = embedding_service.generate_embedding("Hello, how are you?")
        assert embedding is not None
