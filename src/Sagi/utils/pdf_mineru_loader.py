import os
import re
from pathlib import Path
from urllib import request as urlrequest


def _download_image(url: str, dest: Path) -> Path:
    with urlrequest.urlopen(url) as resp:
        dest.write_bytes(resp.read())
    return dest


def parse_markify_markdown(markdown: str, images_dir: str | Path) -> str:
    """Parse ``markdown`` and download images referenced from a Markify service.

    URLs starting with ``http://localhost:20926`` are replaced with
    ``http://markify:20926`` to allow access inside Docker. The images are
    downloaded into ``images_dir`` and the links in the markdown are updated to
    point to the downloaded files.
    """

    images_dir = Path(images_dir)
    pattern = re.compile(r"!\[[^\]]*\]\((http://[^)]+)\)")

    def replace(match: re.Match[str]) -> str:
        url = match.group(1)
        if url.startswith("http://localhost:20926"):
            url = url.replace("http://localhost:20926", "http://markify:20926")
        filename = os.path.basename(url)
        local_path = images_dir / filename
        _download_image(url, local_path)
        return f"![]({local_path.as_posix()})"

    return pattern.sub(replace, markdown)


__all__ = ["parse_markify_markdown"]