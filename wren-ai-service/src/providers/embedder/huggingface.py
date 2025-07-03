import asyncio
from typing import Any, Dict, List, Tuple

from haystack import Document, component
from sentence_transformers import SentenceTransformer

from src.core.provider import EmbedderProvider
from src.providers.loader import provider
import logging

logger = logging.getLogger("wren-ai-service")


@component
class HuggingfaceTextEmbedder:
    def __init__(self, model: SentenceTransformer):
        self._model = model

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    async def run(self, text: str):
        if not isinstance(text, str):
            raise TypeError(
                "HuggingfaceTextEmbedder expects a string as input."
            )
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, lambda: self._model.encode(text))
        meta = {"model": type(self._model).__name__}
        return {"embedding": embedding.tolist(), "meta": meta}


@component
class HuggingfaceDocumentEmbedder:
    def __init__(self, model: SentenceTransformer):
        self._model = model

    async def _embed_batch(
        self, texts: List[str], batch_size: int, progress_bar: bool
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=progress_bar
            ).tolist()
        )
        meta = {"model": type(self._model).__name__}
        return embeddings, meta

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    async def run(
        self, documents: List[Document], batch_size: int = 32, progress_bar: bool = False
    ):
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError(
                "HuggingfaceDocumentEmbedder expects a list of Documents as input."
            )
        texts = [doc.content or "" for doc in documents]
        embeddings, meta = await self._embed_batch(
            texts=texts,
            batch_size=batch_size,
            progress_bar=progress_bar
        )
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        return {"documents": documents, "meta": meta}


@provider("huggingface_embedder")
class HuggingfaceEmbedderProvider(EmbedderProvider):
    def __init__(self, model: str, **kwargs):
        self._model_name = model
        self._hf_model = SentenceTransformer(model)
        logger.info(f"Initializing HuggingfaceEmbedder with model: {model}")

    def get_text_embedder(self):
        return HuggingfaceTextEmbedder(model=self._hf_model)

    def get_document_embedder(self):
        return HuggingfaceDocumentEmbedder(model=self._hf_model)
