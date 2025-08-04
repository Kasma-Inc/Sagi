from dataclasses import dataclass

import fitz


@dataclass
class RectData:
    type: str
    x0: float
    y0: float
    x1: float
    y1: float
    text: str = ""
    org_x0: float = 0
    org_y0: float = 0
    org_x1: float = 0
    org_y1: float = 0

    def __post_init__(self):
        self.org_x0 = self.x0
        self.org_y0 = self.y0
        self.org_x1 = self.x1
        self.org_y1 = self.y1

    def extend_rect_data(self, extend_by: int):
        return RectData(
            type=self.type,
            x0=self.x0 - extend_by * 0.5,
            y0=self.y0 - extend_by * 0.5,
            x1=self.x1 + extend_by,
            y1=self.y1 + extend_by,
            text=self.text,
            org_x0=self.org_x0,
            org_y0=self.org_y0,
            org_x1=self.org_x1,
            org_y1=self.org_y1,
        )

    def contain_rect(self, rect: fitz.Rect) -> bool:
        overlap_y0 = max(self.y0, rect.y0)
        overlap_y1 = min(self.y1, rect.y1)
        overlap_height = max(0, overlap_y1 - overlap_y0)
        rect_height = rect.y1 - rect.y0

        overlap_x0 = max(self.x0, rect.x0)
        overlap_x1 = min(self.x1, rect.x1)
        overlap_width = max(0, overlap_x1 - overlap_x0)
        rect_width = rect.x1 - rect.x0

        return (overlap_height + 1 >= 0.7 * rect_height) and (
            overlap_width + 1 >= 0.7 * rect_width
        )

    def get_org_rect(self, extend_by: int = 0):
        return fitz.Rect(
            self.org_x0 - extend_by * 0.5,
            self.org_y0 - extend_by * 0.5,
            self.org_x1 + extend_by,
            self.org_y1 + extend_by,
        )

    def get_rect(self, extend_by: int = 0):
        return fitz.Rect(
            self.x0 - extend_by * 0.5,
            self.y0 - extend_by * 0.5,
            self.x1 + extend_by,
            self.y1 + extend_by,
        )


@dataclass
class TextStyle:
    font: str
    color: int
    alpha: int
    size: float

    def same_style(self, other: "TextStyle") -> bool:
        return (
            self.font == other.font
            and self.color == other.color
            and self.alpha == other.alpha
            and self.size == other.size
        )
