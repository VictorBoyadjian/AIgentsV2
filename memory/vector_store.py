"""
Weaviate vector store for semantic search over project artifacts.

Stores and retrieves architecture documents, PRDs, code files, and
research results for RAG-powered context injection.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from core.config import get_settings

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "ProjectArtifacts"


@dataclass
class VectorDocument:
    """A document stored in the vector database."""

    id: str
    content: str
    metadata: dict[str, Any]
    project_id: str
    doc_type: str  # "architecture", "prd", "code", "research", "test"
    score: float = 0.0


class VectorStore:
    """
    Weaviate-backed vector store for project artifacts.

    Used for:
    - Storing architecture docs, PRDs, code files, research results
    - Semantic search for relevant context injection
    - RAG pipeline support for agent memory
    """

    def __init__(self, weaviate_url: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self._url = weaviate_url or settings.weaviate.weaviate_url
        self._api_key = api_key or settings.weaviate.weaviate_api_key
        self._client: Any = None

    async def initialize(self) -> None:
        """Connect to Weaviate and ensure collection exists."""
        try:
            import weaviate

            self._client = weaviate.connect_to_local(
                host=self._url.replace("http://", "").split(":")[0],
                port=int(self._url.split(":")[-1]) if ":" in self._url.split("//")[-1] else 8080,
            )

            # Create collection if not exists
            if not self._client.collections.exists(COLLECTION_NAME):
                from weaviate.classes.config import Property, DataType, Configure

                self._client.collections.create(
                    name=COLLECTION_NAME,
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="project_id", data_type=DataType.TEXT),
                        Property(name="doc_type", data_type=DataType.TEXT),
                        Property(name="metadata_json", data_type=DataType.TEXT),
                        Property(name="created_at", data_type=DataType.DATE),
                    ],
                    vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                )

            logger.info("vector_store.initialized", collection=COLLECTION_NAME)

        except Exception as exc:
            logger.warning("vector_store.init_failed", error=str(exc))

    async def close(self) -> None:
        """Close Weaviate connection."""
        if self._client:
            self._client.close()

    async def add_document(
        self,
        content: str,
        project_id: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a document to the vector store."""
        import json

        doc_id = str(uuid.uuid4())

        if not self._client:
            logger.warning("vector_store.not_initialized")
            return doc_id

        try:
            collection = self._client.collections.get(COLLECTION_NAME)
            collection.data.insert(
                properties={
                    "content": content,
                    "project_id": project_id,
                    "doc_type": doc_type,
                    "metadata_json": json.dumps(metadata or {}),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                uuid=doc_id,
            )

            logger.info(
                "vector_store.document_added",
                doc_id=doc_id,
                doc_type=doc_type,
                content_len=len(content),
            )
        except Exception as exc:
            logger.error("vector_store.add_failed", error=str(exc))

        return doc_id

    async def search(
        self,
        query: str,
        project_id: str | None = None,
        doc_type: str | None = None,
        limit: int = 5,
    ) -> list[VectorDocument]:
        """Semantic search over stored documents."""
        import json

        if not self._client:
            logger.warning("vector_store.not_initialized")
            return []

        try:
            from weaviate.classes.query import Filter

            collection = self._client.collections.get(COLLECTION_NAME)

            filters = None
            if project_id and doc_type:
                filters = Filter.by_property("project_id").equal(project_id) & Filter.by_property("doc_type").equal(doc_type)
            elif project_id:
                filters = Filter.by_property("project_id").equal(project_id)
            elif doc_type:
                filters = Filter.by_property("doc_type").equal(doc_type)

            response = collection.query.near_text(
                query=query,
                limit=limit,
                filters=filters,
            )

            results: list[VectorDocument] = []
            for obj in response.objects:
                props = obj.properties
                results.append(VectorDocument(
                    id=str(obj.uuid),
                    content=props.get("content", ""),
                    metadata=json.loads(props.get("metadata_json", "{}")),
                    project_id=props.get("project_id", ""),
                    doc_type=props.get("doc_type", ""),
                    score=obj.metadata.certainty if obj.metadata and hasattr(obj.metadata, "certainty") else 0.0,
                ))

            logger.info("vector_store.search_completed", query=query[:50], results=len(results))
            return results

        except Exception as exc:
            logger.error("vector_store.search_failed", error=str(exc))
            return []

    async def delete_project_documents(self, project_id: str) -> int:
        """Delete all documents for a project."""
        if not self._client:
            return 0

        try:
            from weaviate.classes.query import Filter

            collection = self._client.collections.get(COLLECTION_NAME)
            result = collection.data.delete_many(
                where=Filter.by_property("project_id").equal(project_id)
            )
            count = result.successful if hasattr(result, "successful") else 0
            logger.info("vector_store.project_deleted", project_id=project_id, deleted=count)
            return count
        except Exception as exc:
            logger.error("vector_store.delete_failed", error=str(exc))
            return 0
