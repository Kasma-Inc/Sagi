"""PDF processing utilities for web search results."""

import asyncio
import logging
import re
from typing import Optional
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)


def is_pdf_url(url: str) -> bool:
    """Check if a URL points to a PDF file.

    Args:
        url: The URL to check

    Returns:
        True if the URL appears to be a PDF file
    """
    if not url:
        return False

    # Parse URL to get path and query parameters
    parsed = urlparse(url.lower())

    # Check file extension
    if parsed.path.endswith(".pdf"):
        return True

    pdf_patterns = ["pdf", "doc_type=pdf", "doc_type=q", "format=pdf", "filetype=pdf"]

    query_lower = parsed.query.lower()
    for pattern in pdf_patterns:
        if pattern in query_lower:
            return True

    pdf_hosts_patterns = [
        "arxiv.org",
        "researchgate.net",
        "academia.edu",
        "semanticscholar.org",
    ]

    hostname_lower = parsed.hostname.lower() if parsed.hostname else ""
    for pattern in pdf_hosts_patterns:
        if pattern in hostname_lower and (
            "pdf" in url.lower() or "paper" in url.lower()
        ):
            return True

    return False


async def extract_pdf_content(url: str, max_pages: int = 200) -> Optional[str]:
    """Extract text content from a PDF URL.

    Args:
        url: URL of the PDF file
        max_pages: Maximum number of pages to extract (to prevent huge downloads)

    Returns:
        Extracted text content or None if extraction fails
    """
    try:
        import io

        import fitz

        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.head(url) as response:
                    content_type = response.headers.get("Content-Type", "").lower()
                    content_length = response.headers.get("Content-Length")

                    if (
                        content_type
                        and "application/pdf" not in content_type
                        and "pdf" not in content_type
                    ):
                        logger.warning(
                            f"URL {url} may not be a PDF (Content-Type: {content_type}), but trying anyway"
                        )

                    if content_length and int(content_length) > 100 * 1024 * 1024:
                        logger.warning(
                            f"PDF at {url} is too large ({content_length} bytes), skipping"
                        )
                        return None
            except Exception as e:
                logger.warning(
                    f"HEAD request failed for {url}: {e}, proceeding with GET"
                )

            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(
                        f"Failed to download PDF from {url}: HTTP {response.status}"
                    )
                    return None

                pdf_content = await response.read()

                if len(pdf_content) > 100 * 1024 * 1024:
                    logger.warning(
                        f"Downloaded PDF is too large ({len(pdf_content)} bytes), skipping extraction"
                    )
                    return None

        try:
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")

            extracted_text = []
            pages_processed = 0

            for page_num in range(len(pdf_document)):
                if pages_processed >= max_pages:
                    logger.warning(
                        f"PDF {url} has {len(pdf_document)} pages but only processed {max_pages} pages due to limit"
                    )
                    break

                page = pdf_document.load_page(page_num)
                page_text = page.get_text()

                if page_text.strip():
                    extracted_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    pages_processed += 1

            pdf_document.close()

            if not extracted_text:
                return f"""
[PDF DOCUMENT PROCESSED]
URL: {url}
Size: {len(pdf_content)} bytes
Status: PDF appears to be image-based or encrypted - no text could be extracted
Pages: {len(pdf_document)}

Note: This may be a scanned document that requires OCR processing.
"""

            full_text = "\n\n".join(extracted_text)

            truncation_warning = ""
            if pages_processed >= max_pages and len(pdf_document) > max_pages:
                truncation_warning = f"\n⚠️  WARNING: Document truncated! Only processed {max_pages} pages out of {len(pdf_document)} total pages.\n"

            result = f"""
[PDF DOCUMENT EXTRACTED]
URL: {url}
Size: {len(pdf_content)} bytes
Pages extracted: {pages_processed}/{len(pdf_document)}
Text length: {len(full_text)} characters{truncation_warning}

--- EXTRACTED CONTENT ---
{full_text}
--- END OF EXTRACTED CONTENT ---
"""

            logger.info(
                f"Successfully extracted text from PDF {url} ({pages_processed} pages, {len(full_text)} chars)"
            )
            return result

        except Exception as pdf_error:
            logger.warning(f"PyMuPDF extraction failed: {pdf_error}, trying pypdf")

            try:
                from pypdf import PdfReader

                pdf_reader = PdfReader(io.BytesIO(pdf_content))
                extracted_text = []
                pages_processed = 0

                for page_num, page in enumerate(pdf_reader.pages):
                    if pages_processed >= max_pages:
                        logger.warning(
                            f"PDF {url} has {len(pdf_reader.pages)} pages but only processed {max_pages} pages due to limit (pypdf fallback)"
                        )
                        break

                    page_text = page.extract_text()
                    if page_text.strip():
                        extracted_text.append(
                            f"--- Page {page_num + 1} ---\n{page_text}"
                        )
                        pages_processed += 1

                if not extracted_text:
                    return f"""
[PDF DOCUMENT PROCESSED]
URL: {url}
Size: {len(pdf_content)} bytes
Status: Could not extract text - PDF may be image-based or encrypted
Pages: {len(pdf_reader.pages)}
"""

                full_text = "\n\n".join(extracted_text)

                # Check if document was truncated (pypdf version)
                truncation_warning = ""
                if pages_processed >= max_pages and len(pdf_reader.pages) > max_pages:
                    truncation_warning = f"\n⚠️  WARNING: Document truncated! Only processed {max_pages} pages out of {len(pdf_reader.pages)} total pages.\n"

                result = f"""
[PDF DOCUMENT EXTRACTED]
URL: {url}
Size: {len(pdf_content)} bytes
Pages extracted: {pages_processed}/{len(pdf_reader.pages)}
Text length: {len(full_text)} characters
Method: pypdf (fallback){truncation_warning}

--- EXTRACTED CONTENT ---
{full_text}
--- END OF EXTRACTED CONTENT ---
"""

                logger.info(
                    f"Successfully extracted text from PDF {url} using pypdf fallback ({pages_processed} pages)"
                )
                return result

            except Exception as fallback_error:
                logger.error(
                    f"Both PyMuPDF and pypdf extraction failed: {fallback_error}"
                )
                return f"""
[PDF DOCUMENT PROCESSED]
URL: {url}
Size: {len(pdf_content)} bytes
Status: Text extraction failed - PDF may be corrupted, encrypted, or image-only
Error: {fallback_error}

Note: PDF was successfully downloaded but text could not be extracted.
"""

    except asyncio.TimeoutError:
        logger.error(f"Timeout while processing PDF from {url}")
        return None
    except Exception as e:
        logger.error(f"Error extracting content from PDF {url}: {e}")
        return None


def enhance_search_results_with_pdf_info(search_results: str) -> str:
    """Enhance search results by detecting and marking PDF URLs.

    Args:
        search_results: Original search results text

    Returns:
        Enhanced search results with PDF URLs marked
    """
    if not search_results:
        return search_results

    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, search_results)

    enhanced_results = search_results
    pdf_urls_found = []

    for url in urls:
        if is_pdf_url(url):
            pdf_urls_found.append(url)
            enhanced_results = enhanced_results.replace(url, f"{url} [PDF DOCUMENT]")

    if pdf_urls_found:
        pdf_summary = "\n\n=== PDF DOCUMENTS DETECTED ===\n"
        pdf_summary += (
            f"Found {len(pdf_urls_found)} PDF document(s) in search results:\n"
        )
        for i, pdf_url in enumerate(pdf_urls_found, 1):
            pdf_summary += f"{i}. {pdf_url}\n"
        pdf_summary += "\nNote: These are PDF documents that may contain detailed technical information.\n"
        pdf_summary += (
            "Consider extracting their content for more comprehensive analysis.\n"
        )

        enhanced_results = enhanced_results + pdf_summary

    return enhanced_results


async def process_pdf_urls_in_search_results(search_results: str) -> str:
    """Process search results and extract content from any PDF URLs found.

    Args:
        search_results: Original search results text

    Returns:
        Enhanced search results with PDF content included
    """
    if not search_results:
        return search_results

    # Find URLs in the search results
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, search_results)

    pdf_contents = []
    for url in urls:
        if is_pdf_url(url):
            logger.info(f"Processing PDF URL: {url}")
            content = await extract_pdf_content(url)
            if content:
                pdf_contents.append({"url": url, "content": content})

    enhanced_results = search_results

    if pdf_contents:
        pdf_section = "\n\n=== EXTRACTED PDF CONTENT ===\n"
        for i, pdf_info in enumerate(pdf_contents, 1):
            pdf_section += f"\n--- PDF {i}: {pdf_info['url']} ---\n"
            pdf_section += pdf_info["content"]
            pdf_section += "\n" + "=" * 50 + "\n"

        enhanced_results = enhanced_results + pdf_section

    return enhanced_results
