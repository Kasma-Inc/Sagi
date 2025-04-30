#! /usr/bin/env python3

from abc import ABC
from typing import List, Optional, Type

from langchain_core.document_loaders import BaseLoader as LangchainBaseLoader
from langchain_core.documents import Document


class BaseLoader(ABC):
    """Base class for all loaders"""

    loader_type: Type[LangchainBaseLoader]

    # additional metadata to add to the loaded raw documents
    page_number_key: str = "page_number"

    def _load(self, document_path: str, **loader_args) -> List[Document]:
        raw_docs = self.loader_type(document_path, **loader_args).load()
        return raw_docs

    def load(
        self, document_path: str, document_meta: Optional[dict] = None, **loader_args
    ) -> list[Document]:
        """Load document and set the metadata of the output

        Args:
            document_path (str): The document path for langchain loader to use.
            document_meta (Optional[dict]): The document metadata to set to the output.
            loader_args (dict): The arguments for the langchain loader.

        Returns:
            list[Document]: Raw documents.
        """
        if document_meta is None:
            document_meta = {}
        raw_docs = self._load(document_path, **loader_args)
        self._set_doc_metadata(raw_docs, document_meta)
        return raw_docs

    def _set_doc_metadata(
        self, docs: List[Document], document_meta: dict
    ) -> List[Document]:
        for doc in docs:
            document_meta[self.page_number_key] = None
            doc.metadata = document_meta
