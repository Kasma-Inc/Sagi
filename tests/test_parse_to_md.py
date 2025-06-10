import os

from hirag_prod.loader import load_document


def test_load_docx_mineru():
    document_path = f"{os.path.dirname(__file__)}/Guide-to-U.S.-Healthcare-System.pdf"
    content_type = "application/pdf"
    document_meta = {
        "type": "pdf",
        "filename": "Guide-to-U.S.-Healthcare-System.pdf",
        "uri": document_path,
        "private": False,
    }
    loader_configs = None
    documents = load_document(
        document_path,
        content_type,
        document_meta,
        loader_configs,
        loader_type="mineru",
    )
    markdown_content = "".join([doc.page_content for doc in documents])
    folder = f"{os.path.dirname(__file__)}/test_results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(f"{folder}/markdown_content.md", "w") as f:
        f.write(markdown_content)
    assert markdown_content is not ""
