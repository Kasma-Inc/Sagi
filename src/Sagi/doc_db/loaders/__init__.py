import time
from typing import List, Optional, Union

import requests
from langchain_core.documents import Document
from requests_toolbelt.multipart.encoder import MultipartEncoder

from Sagi.doc_db.loaders.csv_loader import CSVLoader
from Sagi.doc_db.loaders.excel_loader import ExcelLoader
from Sagi.doc_db.loaders.html_loader import HTMLLoader
from Sagi.doc_db.loaders.pdf_loader import PDFLoader
from Sagi.doc_db.loaders.ppt_loader import PowerPointLoader
from Sagi.doc_db.loaders.word_loader import WordLoader


class MarkifyClient:
    def __init__(
        self,
        base_url: str = "http://markify:20926",
        timeout: int = 30,
        default_mode: str = "simple",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_mode = default_mode
        self.session = requests.Session()

        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def create_job(
        self,
        file_path: str,
        mode: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        endpoint = f"{self.base_url}/api/jobs"
        # content type detection
        if not content_type:
            content_type = self._detect_mime_type(file_path)
        # request body
        multipart_data = MultipartEncoder(
            fields={
                "file": (file_path, open(file_path, "rb"), content_type),
                "mode": mode or self.default_mode,
            }
        )
        response = self.session.post(
            url=endpoint,
            data=multipart_data,
            headers={
                "Accept": "application/json",
                "Content-Type": multipart_data.content_type,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["job_id"]

    def get_job_status(self, job_id: str) -> dict:
        endpoint = f"{self.base_url}/api/jobs/{job_id}"
        try:
            response = self.session.get(
                url=endpoint,
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError("Job ID not found") from e
            raise

    def get_result(
        self, job_id: str, output_file: Optional[str] = None
    ) -> Union[str, None]:
        endpoint = f"{self.base_url}/api/jobs/{job_id}/result"
        try:
            response = self.session.get(
                url=endpoint, headers={"Accept": "text/markdown"}, timeout=self.timeout
            )
            response.raise_for_status()
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(response.text)
                return None
            return response.text
        except requests.HTTPError as e:
            if e.response.status_code == 202:
                raise RuntimeError("Job still processing") from e
            raise

    def convert_pdf(
        self,
        file_path: str,
        mode: Optional[str] = None,
        poll_interval: int = 2,
        max_wait: int = 3600,
    ) -> str:
        start_time = time.time()
        job_id = self.create_job(file_path, mode)
        while True:
            if time.time() - start_time > max_wait:
                raise TimeoutError("Max wait time exceeded")
            status = self.get_job_status(job_id)
            if status["status"] == "completed":
                return self.get_result(job_id)
            elif status["status"] == "failed":
                raise RuntimeError(f"Job failed: {status.get('error')}")
            time.sleep(poll_interval)

    @staticmethod
    def _detect_mime_type(file_path: str) -> str:
        ext = file_path.split(".")[-1].lower()
        mime_map = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "jpg": "image/jpeg",
            "png": "image/png",
            "zip": "application/zip",
            "txt": "text/plain",
            "csv": "text/csv",
            "text": "text/plain",
            "tsv": "text/tab-separated-values",
        }
        return mime_map.get(ext, "application/octet-stream")


DEFAULT_LOADER_CONFIGS = {
    "application/pdf": {
        "loader": PDFLoader,
        "args": {
            # TODO(tatiana): tune the args?
        },
        "init_args": {
            "max_output_docs": 5,
        },
    },
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
        "loader": WordLoader,
        "args": {
            # TODO(tatiana): tune the args?
        },
    },
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": {
        "loader": PowerPointLoader,
        "args": {"mode": "single", "strategy": "fast"},
    },
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {
        "loader": ExcelLoader,
        "args": {
            # TODO(tatiana): tune the args?
        },
    },
    "text/html": {
        "loader": HTMLLoader,
        "args": {},
    },
    "text/csv": {
        "loader": CSVLoader,
        "args": {},
    },
}


def load_document(
    document_path: str,
    content_type: str,
    document_meta: Optional[dict] = None,
    loader_configs: Optional[dict] = None,
) -> List[Document]:
    """Load a document from the given path and content type

    Args:
        document_path (str): The path to the document.
        content_type (str): The content type of the document.
        document_meta (Optional[dict]): The metadata of the document.
        loader_configs (Optional[dict]): If unspecified, use DEFAULT_LOADER_CONFIGS.

    Raises:
        ValueError: If the content type is not supported.

    Returns:
        List[Document]: The loaded documents.
    """
    if loader_configs is None:
        loader_configs = DEFAULT_LOADER_CONFIGS

    if content_type not in loader_configs:
        raise ValueError(f"Unsupported document type: {content_type}")
    loader_conf = loader_configs[content_type]
    if "init_args" in loader_conf:
        loader = loader_conf["loader"](**loader_conf["init_args"])
    else:
        loader = loader_conf["loader"]()
    if "args" in loader_conf:
        raw_docs = loader.load(document_path, document_meta, **loader_conf["args"])
    else:
        raw_docs = loader.load(document_path, document_meta)
    return raw_docs


__all__ = [
    "PowerPointLoader",
    "PDFLoader",
    "WordLoader",
    "ExcelLoader",
    "load_document",
    "HTMLLoader",
    "CSVLoader",
]
