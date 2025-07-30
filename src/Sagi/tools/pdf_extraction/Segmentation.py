import json
from dataclasses import dataclass
from Sagi.tools.pdf_extraction.Extraction_data import RectData
from typing import List, Dict, Any
import os
import fitz
import subprocess
from pathlib import Path
import shutil
import random
import string

from Sagi.tools.pdf_extraction._utils import upload_file_to_s3, download_file_from_s3, cnt_files_in_s3, delete_dir_from_s3, ocr_parse

@dataclass
class RectInfo:
    x0: float
    y0: float
    x1: float
    y1: float
    cat_id: int
    text: str = ""
    type: str = "non-text"

    def to_rect_data(self) -> RectData:
        return RectData(self.type, self.x0, self.y0, self.x1, self.y1, self.text)

class Segmentation:

    @classmethod
    def call_segmentation(cls, input_path: str, storage_dir: str, save_output_on_s3: bool = False):

        os.makedirs(storage_dir, exist_ok=True)
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        s3_path = f"pdf-extraction/{os.path.splitext(os.path.basename(input_path))[0]}_{random_string}"
        s3_input_path = os.path.join(s3_path, os.path.basename(input_path))
        s3_output_path = os.path.join(s3_path, "ocr_output")
        s3_page_info_path = os.path.join(s3_output_path, "page_info")

        upload_file_to_s3(input_path, s3_input_path)
        ocr_parse(s3_input_path, s3_output_path)

        for i in range(cnt_files_in_s3(s3_page_info_path)):
            filename = f"page_{i}.json"
            download_file_from_s3(os.path.join(s3_page_info_path, filename), os.path.join(storage_dir, filename))

        if not save_output_on_s3:
            delete_dir_from_s3(s3_path)

    @classmethod
    def load_json_per_page(cls, storage_json_path: str):
        with open(storage_json_path, "r") as file:
            json_data = json.load(file)
        
        json_data = json_data["layout_dets"]
        rect_info = []
        index_map: Dict[str, int] = {}
        for block in json_data:
            if block["category_id"] == 15:
                index = index_map[f"{block['bbox'][0]}_{block['bbox'][1]}_{block['bbox'][2]}_{block['bbox'][3]}"]
                rect_info[index].text = block["text"]
                rect_info[index].type = "text"
            else:
                index_map[f"{block['bbox'][0]}_{block['bbox'][1]}_{block['bbox'][2]}_{block['bbox'][3]}"] = len(rect_info)
                rect_info.append(RectInfo(block["bbox"][0], block["bbox"][1], block["bbox"][2], block["bbox"][3], block["category_id"]))

        rect_info = sorted(rect_info, key=lambda rect: (rect.y0, rect.x0))
        
        # Determine header and footer
        min_y = min([rect.y0 for rect in rect_info if rect.cat_id != 2])
        for rect in rect_info:
            if rect.cat_id == 2:
                if rect.y0 <= min_y:
                    rect.type = "header"
                else:
                    rect.type = "footer"

        return rect_info

    @classmethod
    def load_json(cls, storage_dir: str):
        result: List[List[RectData]] = []
        for i in range(len(os.listdir(storage_dir))):
            filename = f"page_{i}.json"
            rect_info = cls.load_json_per_page(os.path.join(storage_dir, filename))
            rect_data = [rect.to_rect_data() for rect in rect_info]
            result.append(rect_data)

        json_data = json.load(open(os.path.join(storage_dir, "page_0.json")))
        height = json_data["page_info"]["height"] * 72 / 200
        width = json_data["page_info"]["width"] * 72 / 200
        leftmost = min(rect.x0 for rect_data in result for rect in rect_data)
        rightmost = width - leftmost
        
        return result, leftmost, rightmost, width, height

    