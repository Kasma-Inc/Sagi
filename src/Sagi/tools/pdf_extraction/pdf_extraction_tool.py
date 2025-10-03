import asyncio
import logging
import os
import re
import tempfile
import uuid
from urllib.parse import urlparse

import httpx
from autogen_core import CancellationToken
from autogen_core.tools import BaseTool
from hirag_prod.loader import load_document
from pydantic import BaseModel
from resources.functions import get_envs
from utils.file_utils import StorageServiceType, delete_file, upload_file_to_storage


class PDFExtractionTool(BaseTool):
    """PDF document extraction tool - Download and extract PDF content from URL"""

    MAX_CONTENT_LENGTH = 120000
    TIMEOUT_SECONDS = 30

    def __init__(
        self, workspace_id: str | None = None, knowledge_base_id: str | None = None
    ):
        class PDFArgs(BaseModel):
            pdf_url: str

        super().__init__(
            args_type=PDFArgs,
            return_type=str,
            name="pdf_extractor",
            description="Extract text content from PDF documents via URL",
        )
        self.workspace_id = workspace_id
        self.knowledge_base_id = knowledge_base_id

    async def run(self, args: BaseModel, cancellation_token: CancellationToken) -> str:
        pdf_url = None
        try:
            pdf_url = getattr(args, "pdf_url", None)
        except Exception:
            pdf_url = None
        if not isinstance(pdf_url, str):
            return "Error: Missing or invalid 'pdf_url' argument"
        try:
            if not pdf_url.startswith(("http://", "https://")):
                return "Error: Please provide a valid HTTP/HTTPS URL"

            if re.search(r"\\u[0-9a-fA-F]{4}", pdf_url):
                try:
                    pdf_url = pdf_url.encode("utf-8", "backslashreplace").decode(
                        "unicode_escape"
                    )
                except UnicodeDecodeError:
                    logging.warning(
                        "Failed to decode unicode escapes in PDF URL %s; using original",
                        pdf_url,
                    )

            async with httpx.AsyncClient(timeout=self.TIMEOUT_SECONDS) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
                    return "Error: URL is not a PDF file"

                if not response.content.startswith(b"%PDF"):
                    logging.info(
                        "Rejected non-PDF content for URL %s despite content-type %s",
                        pdf_url,
                        content_type,
                    )
                    return "Error: Retrieved content is not a valid PDF document"

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(response.content)
                temp_path = f.name

            staged_cleanup = None
            processed_path = temp_path
            MAX_STAGED_PAGES = 20
            try:
                try:
                    from pypdf import PdfReader, PdfWriter

                    with open(temp_path, "rb") as input_fp:
                        reader = PdfReader(input_fp, strict=False)
                        total_pages = len(reader.pages)
                        logging.info(
                            "Downloaded PDF %s reports %s pages before extraction",
                            pdf_url,
                            total_pages,
                        )
                        if total_pages > MAX_STAGED_PAGES:
                            writer = PdfWriter()
                            for page_index in range(MAX_STAGED_PAGES):
                                writer.add_page(reader.pages[page_index])
                            with tempfile.NamedTemporaryFile(
                                suffix=".pdf", delete=False
                            ) as truncated_fp:
                                writer.write(truncated_fp)
                                processed_path = truncated_fp.name
                            logging.info(
                                "Truncated PDF %s from %s to %s pages before staging",
                                pdf_url,
                                total_pages,
                                MAX_STAGED_PAGES,
                            )
                        else:
                            logging.info(
                                "PDF %s page count %s within limit; skipping truncation",
                                pdf_url,
                                total_pages,
                            )
                except ImportError:
                    logging.info("pypdf not installed; skipping PDF page limit")
                except Exception:
                    logging.warning(
                        "Failed to limit PDF %s to first %s pages",
                        pdf_url,
                        MAX_STAGED_PAGES,
                        exc_info=True,
                    )
                    processed_path = temp_path

                content_type = "application/pdf"
                document_path, staged_cleanup = await self._stage_pdf(
                    processed_path, pdf_url
                )
                metadata = {
                    "fileName": pdf_url.split("/")[-1],
                    "private": True,
                    "uri": document_path,
                }
                if self.workspace_id is not None and self.workspace_id != "":
                    metadata["workspaceId"] = self.workspace_id
                if self.knowledge_base_id is not None and self.knowledge_base_id != "":
                    metadata["knowledgeBaseId"] = self.knowledge_base_id
                loader_type = getattr(get_envs(), "LOAD_TYPE", None) or "docling"
                _, doc_md = load_document(
                    document_path,
                    content_type,
                    loader_type,
                    document_meta=metadata,
                )
                if doc_md is None:
                    return "PDF content extraction failed: document parser returned no markdown"

                if not doc_md or not doc_md.text:
                    return "PDF content extraction failed: Unable to parse PDF content"

                text_content = self._sanitize_pdf_text(doc_md.text)
                logging.info(
                    "Parsed PDF %s (staged at %s) extracted %s characters of text",
                    pdf_url,
                    document_path,
                    len(text_content),
                )

                if text_content:
                    if len(text_content) > self.MAX_CONTENT_LENGTH:
                        truncated_content = text_content[: self.MAX_CONTENT_LENGTH]
                        return f"PDF document content ({len(text_content):,} characters, showing first {self.MAX_CONTENT_LENGTH:,}):\n\n{truncated_content}...\n\n[Content truncated - full document is longer]"
                    else:
                        return f"PDF document content ({len(text_content):,} characters):\n\n{text_content}"
                else:
                    return "PDF document parsed successfully, but no text content extracted (possibly scanned image PDF or encrypted document)"

            finally:
                if staged_cleanup is not None:
                    try:
                        await staged_cleanup()
                    except Exception:
                        logging.warning(
                            "Failed to delete staged PDF from remote storage",
                            exc_info=True,
                        )
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                if processed_path != temp_path:
                    try:
                        os.unlink(processed_path)
                    except Exception:
                        pass

        except httpx.TimeoutException:
            return "PDF download timeout: Please check network connection or try another PDF link"
        except httpx.ConnectError as e:
            detail = str(e) or e.__class__.__name__
            if "CERTIFICATE_VERIFY_FAILED" in detail:
                return (
                    "PDF download failed: SSL certificate validation error. Try a different URL "
                    "or access the site manually to verify the certificate."
                )
            return f"PDF download failed: {detail}"
        except httpx.HTTPStatusError as e:
            return f"PDF download failed: HTTP {e.response.status_code} - {e.response.reason_phrase}"
        except Exception as e:
            logging.exception("PDF processing failed when handling PDF URL %s", pdf_url)
            detail = str(e) or e.__class__.__name__
            return f"PDF processing failed: {detail}"

    async def _stage_pdf(self, local_path, source_url):
        envs = get_envs()

        prefix = getattr(envs, "OFNIL_STAGE_TEMP_PATH", "") or ""
        prefix = prefix.strip("/")
        parsed_url = urlparse(source_url)
        filename = (
            os.path.basename(parsed_url.path) if parsed_url and parsed_url.path else ""
        )
        if not filename:
            filename = os.path.basename(local_path)
        object_key = f"{uuid.uuid4().hex}-{filename}"
        if prefix:
            object_key = f"{prefix}/{object_key}"

        aws_bucket = envs.AWS_BUCKET_NAME
        aws_access_key = envs.AWS_ACCESS_KEY_ID
        aws_secret = envs.AWS_SECRET_ACCESS_KEY

        if not all([aws_bucket, aws_access_key, aws_secret]):
            raise ValueError("AWS S3 credentials are required for PDF staging")

        success = await asyncio.to_thread(
            upload_file_to_storage,
            StorageServiceType.S3,
            local_path,
            object_key,
        )
        if not success:
            raise RuntimeError("Failed to upload staged PDF to S3")

        async def cleanup():
            try:
                await asyncio.to_thread(delete_file, StorageServiceType.S3, object_key)
            except Exception:
                logging.warning(
                    "Failed to remove staged PDF %s/%s",
                    aws_bucket,
                    object_key,
                    exc_info=True,
                )

        staged_uri = f"s3://{aws_bucket}/{object_key}"
        logging.info("Staged PDF %s to %s", source_url, staged_uri)
        return staged_uri, cleanup

    def _sanitize_pdf_text(self, raw_text: str) -> str:
        if not isinstance(raw_text, str):
            return ""

        cleaned = re.sub(r"!\[[^\]]*\]\(data:image/[^)]+\)", " ", raw_text)
        cleaned = re.sub(r"!\[[^\]]*\]\(attachment:[^)]+\)", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned and isinstance(raw_text, str):
            return re.sub(r"\s+", " ", raw_text).strip()
        return cleaned
