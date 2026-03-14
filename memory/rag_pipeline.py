"""
RAG pipeline combining vector store search with LLM generation.

Provides context-aware responses by retrieving relevant documents
from the vector store and injecting them into the LLM prompt.
"""

from __future__ import annotations

from typing import Any

import structlog

from core.config import TaskComplexity
from core.llm_router import LLMRouter
from memory.vector_store import VectorStore

logger = structlog.get_logger(__name__)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Combines semantic search over project artifacts with LLM generation
    to produce context-aware responses. Used by agents to access
    relevant historical information.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        llm_router: LLMRouter | None = None,
    ) -> None:
        self._vector_store = vector_store or VectorStore()
        self._llm_router = llm_router or LLMRouter()

    async def query(
        self,
        question: str,
        project_id: str,
        doc_type: str | None = None,
        top_k: int = 5,
        agent_role: str = "fallback",
    ) -> str:
        """
        Answer a question using RAG over project artifacts.

        1. Search vector store for relevant documents
        2. Build context from retrieved documents
        3. Generate answer with LLM
        """
        # Retrieve relevant documents
        documents = await self._vector_store.search(
            query=question,
            project_id=project_id,
            doc_type=doc_type,
            limit=top_k,
        )

        if not documents:
            return "No relevant documents found in the knowledge base."

        # Build context
        context_parts: list[str] = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"### Document {i} (type: {doc.doc_type}, score: {doc.score:.2f})\n{doc.content[:2000]}"
            )
        context = "\n\n".join(context_parts)

        # Generate answer
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based on "
                    "the provided context documents. Cite relevant documents when possible. "
                    "If the context doesn't contain enough information, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        response, _ = await self._llm_router.call_llm(
            agent_role=agent_role,
            messages=messages,
            task_complexity=TaskComplexity.SIMPLE,
            max_tokens=2048,
        )

        logger.info(
            "rag_pipeline.query_completed",
            question=question[:50],
            documents_found=len(documents),
        )

        return response

    async def index_document(
        self,
        content: str,
        project_id: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[str]:
        """
        Index a document by chunking and storing in vector store.

        Args:
            content: Full document content.
            project_id: Project identifier.
            doc_type: Document type for filtering.
            metadata: Additional metadata.
            chunk_size: Characters per chunk.
            chunk_overlap: Overlap between chunks.

        Returns:
            List of document IDs for the stored chunks.
        """
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)
        doc_ids: list[str] = []

        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            doc_id = await self._vector_store.add_document(
                content=chunk,
                project_id=project_id,
                doc_type=doc_type,
                metadata=chunk_metadata,
            )
            doc_ids.append(doc_id)

        logger.info(
            "rag_pipeline.document_indexed",
            project_id=project_id,
            doc_type=doc_type,
            chunks=len(chunks),
        )

        return doc_ids

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at paragraph or sentence boundary
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", " "]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > chunk_size * 0.5:
                        chunk = chunk[: last_sep + len(sep)]
                        end = start + len(chunk)
                        break

            chunks.append(chunk.strip())
            start = end - overlap

        return [c for c in chunks if c]
