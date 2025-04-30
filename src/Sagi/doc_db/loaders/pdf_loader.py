import warnings
from typing import List

from langchain_community import document_loaders
from langchain_core.documents import Document

from Sagi.doc_db.loaders.base_loader import BaseLoader

# Suppress PyPDF warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")


class PDFLoader(BaseLoader):
    """Loads PDF documents"""

    # FIXME(tatiana): for speed, we brutaly prune the number of docs for now
    def __init__(self, max_output_docs: int = 5):
        self.loader_type = document_loaders.PyPDFLoader
        self.max_output_docs = max_output_docs

    def _load(self, document_path: str, **loader_args) -> List[Document]:
        raw_docs = self.loader_type(document_path, **loader_args).load()
        if len(raw_docs) > self.max_output_docs:
            sorted_docs = sorted(
                raw_docs, key=lambda x: len(x.page_content), reverse=True
            )
            raw_docs = sorted_docs[:10]

        return raw_docs

    def _set_doc_metadata(
        self, docs: List[Document], document_meta: dict
    ) -> List[Document]:
        for doc in docs:
            document_meta[self.page_number_key] = doc.metadata["page"]
            doc.metadata = document_meta
