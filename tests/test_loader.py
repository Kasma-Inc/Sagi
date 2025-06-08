from urllib import request as urlrequest

from Sagi.utils.pdf_mineru_loader import parse_markify_markdown


def test_load_pdf_mineru(tmp_path, monkeypatch):
    markdown = (
        "Routing easy/common questions to smaller models like Claude 3.5 Haiku and hard/unusual "
        "questions to more capable models like Claude 3.5 Sonnet to optimize cost and speed.\n\n"
        "Workflow: Parallelization LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. "
        "This workflow, parallelization\n\nSectioning: Breaking a task into independent subtasks run in parallel. "
        "Voting: Running the same task multiple times to get diverse\n\n"
        "![](http://localhost:20926/images/example.jpg)\n"
    )

    image_bytes = b"fake-image"

    class FakeResponse:
        def __init__(self, data: bytes):
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    def fake_urlopen(url):
        assert url == "http://markify:20926/images/example.jpg"
        return FakeResponse(image_bytes)

    monkeypatch.setattr(urlrequest, "urlopen", fake_urlopen)

    result = parse_markify_markdown(markdown, tmp_path)
    stored_image = tmp_path / "example.jpg"
    assert stored_image.exists()
    assert stored_image.read_bytes() == image_bytes
    assert f"![]({stored_image.as_posix()})" in result
