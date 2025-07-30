from Sagi.tools.pdf_extraction.Segmentation import Segmentation
from Sagi.tools.pdf_extraction.HTML_generation import HTMLGenerator
import os

class PDF_Extraction:

    @classmethod
    async def extract_pdf_with_model(
        cls,
        input_path: str,
        storage_dir: str,
        model_client,
        output_path: str | None = None,
        simultaneous_requests: bool = True,
        save_output_on_s3: bool = False
    ) -> str:

        os.makedirs(storage_dir, exist_ok=True)
        Segmentation.call_segmentation(input_path, storage_dir + "/page_info", save_output_on_s3=save_output_on_s3)
        rect_data, leftmost, rightmost, page_width, page_height = Segmentation.load_json(storage_dir + "/page_info")
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
        return await html_generator.generate_all_pages(simultaneous_requests=simultaneous_requests)


    # In this function, the chart/image generation will be skipped
    @classmethod
    async def extract_pdf_without_model(
        cls,
        input_path: str,
        storage_dir: str,
        output_path: str | None = None,
        save_output_on_s3: bool = False
    ) -> str:

        os.makedirs(storage_dir, exist_ok=True)
        Segmentation.call_segmentation(input_path, storage_dir + "/page_info", save_output_on_s3=save_output_on_s3)
        rect_data, leftmost, rightmost, page_width, page_height = Segmentation.load_json(storage_dir + "/page_info")
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