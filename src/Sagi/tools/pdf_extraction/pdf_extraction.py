import os
import re

from Sagi.tools.pdf_extraction.html_generation import HTMLGenerator
from Sagi.tools.pdf_extraction.html_template import (
    create_editable_container,
    create_initialize,
    editable_html_template,
)
from Sagi.tools.pdf_extraction.segmentation import Segmentation


class PDF_Extraction:

    @classmethod
    async def extract_pdf_with_model(
        cls,
        input_path: str,
        storage_dir: str,
        model_client,
        output_path: str | None = None,
        simultaneous_requests: bool = True,
        save_output_on_s3: bool = False,
    ) -> str:

        os.makedirs(storage_dir, exist_ok=True)
        Segmentation.call_segmentation(
            input_path, storage_dir + "/page_info", save_output_on_s3=save_output_on_s3
        )
        rect_data, leftmost, rightmost, page_width, page_height = (
            Segmentation.load_json(storage_dir + "/page_info")
        )
        html_generator = HTMLGenerator(
            input_pdf_path=input_path,
            storage_dir=storage_dir,
            output_path=output_path,
            rect_data=rect_data,
            leftmost_coordinates=leftmost,
            rightmost_coordinates=rightmost,
            page_width=page_width,
            page_height=page_height,
            model_client=model_client,
        )
        return await html_generator.generate_all_pages(
            simultaneous_requests=simultaneous_requests
        )

    # In this function, the chart/image generation will be skipped
    @classmethod
    async def extract_pdf_without_model(
        cls,
        input_path: str,
        storage_dir: str,
        output_path: str | None = None,
        save_output_on_s3: bool = False,
    ) -> str:

        os.makedirs(storage_dir, exist_ok=True)
        Segmentation.call_segmentation(
            input_path, storage_dir + "/page_info", save_output_on_s3=save_output_on_s3
        )
        rect_data, leftmost, rightmost, page_width, page_height = (
            Segmentation.load_json(storage_dir + "/page_info")
        )
        html_generator = HTMLGenerator(
            input_pdf_path=input_path,
            storage_dir=storage_dir,
            output_path=output_path,
            rect_data=rect_data,
            leftmost_coordinates=leftmost,
            rightmost_coordinates=rightmost,
            page_width=page_width,
            page_height=page_height,
        )
        return await html_generator.generate_all_pages()

    @classmethod
    def make_editable_document(cls, input_path: str, output_path: str, title: str):

        def extract_page_content(content: str, page_number: int) -> str | None:
            start_pattern = (
                rf"<div class=\'background-page\' id=\'page_{page_number}\'>"
            )
            start_match = re.search(start_pattern, content)
            if not start_match:
                return None

            start_pos = start_match.start()

            pos = start_pos
            div_count = 0
            in_page_div = False

            for i in range(start_pos, len(content)):
                if content[i : i + 5] == "<div ":
                    div_count += 1
                    if i == start_pos:
                        in_page_div = True
                elif content[i : i + 6] == "</div>":
                    div_count -= 1
                    if in_page_div and div_count == 0:
                        end_pos = i + 6
                        return content[start_pos:end_pos]
            else:
                return None

        with open(input_path, "r") as f:
            content = f.read()

        match = re.search(r"width:\s*([0-9.]+)pt", content)
        width = float(match.group(1)) if match else None
        match = re.search(r"height:\s*([0-9.]+)pt", content)
        height = float(match.group(1)) if match else None

        if width is None or height is None:
            raise ValueError("Width and height not found in input file")

        result = editable_html_template
        result = result.replace("ffff-width", f"{width}pt")
        result = result.replace("ffff-height", f"{height}pt")
        result = result.replace("ffff-title", title)
        result = result.replace("ffff-extra-width", f"{width + 0.1}pt")

        editable_container = ""
        initialize = ""
        content_dict = ""

        page_number = 0
        while True:
            page_content = extract_page_content(content, page_number)
            if page_content is None:
                break

            editable_container += create_editable_container(page_number)
            initialize += create_initialize(page_number)
            content_dict += f'"gjs{page_number}": `{page_content}`,'

            page_number += 1

        content_dict = "{" + content_dict + "}"

        result = result.replace("ffff-container", editable_container)
        result = result.replace("ffff-initialize", initialize)
        result = result.replace("ffff-content", content_dict)

        with open(output_path, "w") as f:
            f.write(result)
