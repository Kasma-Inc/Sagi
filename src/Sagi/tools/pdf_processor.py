import asyncio
import logging
import re
from typing import Optional
from urllib.parse import urlparse

import aiohttp
import fitz
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MAX_PDF_SIZE = 100 * 1024 * 1024
MAX_PAGES = 200
DEFAULT_TIMEOUT = 30


def is_pdf_url(url: str) -> bool:
    if not url:
        return False

    url_lower = url.lower()
    parsed = urlparse(url_lower)

    if parsed.path.endswith(".pdf"):
        return True
    
    pdf_indicators = ["doc_type=pdf", "format=pdf", "filetype=pdf", ".pdf", "download.pdf"]
    if any(indicator in parsed.query.lower() for indicator in pdf_indicators):
        return True
    
    pdf_hosts = ["arxiv.org", "ieee.org", "acm.org", "researchgate.net"]
    hostname = parsed.hostname.lower() if parsed.hostname else ""
    if any(host in hostname for host in pdf_hosts):
        if any(term in url_lower for term in ["pdf", "paper", "download"]):
            return True

    return False


async def extract_pdf_content(url: str) -> Optional[str]:
    try:
        timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"❌ Failed to download PDF: HTTP {response.status}"
                
                pdf_data = await response.read()
                if len(pdf_data) > MAX_PDF_SIZE:
                    return f"❌ PDF too large: {len(pdf_data)} bytes > {MAX_PDF_SIZE}"

        doc = fitz.open(stream=pdf_data, filetype="pdf")
        text_parts = []
        
        pages_to_process = min(len(doc), MAX_PAGES)
        for page_num in range(pages_to_process):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        
        if not text_parts:
            return f"⚠️ PDF processed but no text extracted (may be image-based)"

        full_text = "\n\n".join(text_parts)
        
        total_pages = len(doc)
        doc.close()
        
        result = f"""
📄 PDF EXTRACTED
URL: {url}
Pages: {pages_to_process}/{total_pages}
Content length: {len(full_text)} chars

{full_text}
"""
        logger.info(f"Extracted PDF: {url} ({pages_to_process} pages)")
        return result

    except Exception as e:
        logger.error(f"PDF extraction failed for {url}: {e}")
        return f"❌ PDF extraction failed: {str(e)}"


async def extract_web_content(url: str, max_length: int = 30000) -> Optional[str]:
    try:
        timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                html = await response.text()

        soup = BeautifulSoup(html, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()

        content = None
        for selector in ['main', 'article', '.content', '.main-content']:
            element = soup.select_one(selector)
            if element:
                content = element.get_text(separator='\n', strip=True)
                break
        
        if not content:
            content = soup.get_text(separator='\n', strip=True)

        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        if len(content) > max_length:
            content = content[:max_length] + "\n[Content truncated...]"

        return f"""
🌐 WEB CONTENT
URL: {url}
Length: {len(content)} chars

{content}
""" if len(content.strip()) > 50 else None

    except Exception as e:
        logger.error(f"Web extraction failed for {url}: {e}")
        return None


def enhance_search_results_with_pdf_info(search_results: str) -> str:
    if not search_results:
        return search_results

    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, search_results)
    
    enhanced = search_results
    pdf_count = 0
    
    for url in urls:
        if is_pdf_url(url):
            enhanced = enhanced.replace(url, f"{url} [PDF📄]")
            pdf_count += 1

    if pdf_count > 0:
        enhanced += f"\n\n🔍 Detected {pdf_count} PDF documents for deep content extraction"
    
    return enhanced


async def process_search_results_with_documents(search_results: str, max_docs: int = 2) -> str:
    if not search_results:
        return search_results

    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, search_results)
    
    pdf_urls = [url for url in urls if is_pdf_url(url)]
    web_urls = [url for url in urls[:5] if not is_pdf_url(url)]
    
    enhanced_results = enhance_search_results_with_pdf_info(search_results)
    extracted_content = []

    for pdf_url in pdf_urls[:max_docs]:
        content = await extract_pdf_content(pdf_url)
        if content:
            extracted_content.append(content)

    remaining_slots = max_docs - len(extracted_content)
    for web_url in web_urls[:remaining_slots]:
        content = await extract_web_content(web_url)
        if content:
            extracted_content.append(content)

    if extracted_content:
        enhanced_results += "\n\n" + "="*50 + " DOCUMENT CONTENT EXTRACTION " + "="*50
        enhanced_results += "\n".join(extracted_content)

    return enhanced_results


async def enhance_search_results_with_pdf_info_async(search_results: str) -> str:
    return await process_search_results_with_documents(search_results)


async def process_pdf_urls_in_search_results(search_results: str) -> str:
    return await process_search_results_with_documents(search_results, max_docs=3)