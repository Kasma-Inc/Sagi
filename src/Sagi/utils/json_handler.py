import json


def format_file_content(file_content_path: str):
    """
    Format the file_content json into a string that's easy for LLM to understand.

    Args:
        file_content_path (str): The path to the file_content json file.

    Returns:
        str: The formatted string.
    """
    with open(file_content_path, "r", encoding="utf-8") as f:
        file_content = json.load(f)

    formatted_texts = []

    # Add metadata information
    if "metadata" in file_content:
        metadata = file_content["metadata"]
        formatted_texts.append("=== Document Metadata ===")
        for key, value in metadata.items():
            formatted_texts.append(f"{key}: {value}")
        formatted_texts.append("\n")

    # Process each section
    if "sections" in file_content:
        for section in file_content["sections"]:
            # Add section title
            formatted_texts.append(f"## Section: {section['title']}")

            # Process subsections
            if "subsections" in section:
                for subsection in section["subsections"]:
                    formatted_texts.append(f"\n### Subsection: {subsection['title']}")

                    # Add subsection content
                    if "content" in subsection:
                        formatted_texts.append(f"Content: {subsection['content']}")

                    # Add media captions
                    if "medias" in subsection and subsection["medias"]:
                        formatted_texts.append("\nMedia Captions:")
                        for media in subsection["medias"]:
                            if "caption" in media:
                                formatted_texts.append(f"- {media['caption']}")

            formatted_texts.append("\n" + "=" * 50 + "\n")

    return "\n".join(formatted_texts)


def format_templates(slide_induction_path: str):
    with open(slide_induction_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter out non-template items
    templates = {
        k: v for k, v in data.items() if isinstance(v, dict) and "template_id" in v
    }

    # Sort by template_id
    templates = dict(sorted(templates.items(), key=lambda item: item[1]["template_id"]))

    result = []
    for name, tpl in templates.items():
        template_id = tpl.get("template_id")
        slides = tpl.get("slides", [])
        content_schema = tpl.get("content_schema", {})

        result.append(f"=== template_id: {template_id} ===")
        result.append(f"Template name: {name}")
        result.append(f"slides: {slides}")
        result.append("Content Schema:")
        for field, info in content_schema.items():
            dtype = info.get("type", "")
            data = info.get("data", [])
            all_data = ", ".join(map(str, data))
            result.append(f"  - {field} ({dtype})")
            if all_data:
                result.append(f"    data: {all_data}")
        result.append("")

    return "\n".join(result)


def format_slide_info(slide: dict):
    """
    Format the high_level_plan json string into a list.

    Args:
        high_level_plan (str): The high_level_plan json string.
    """
    slide_info = f"""# Slide Info
  - **slide_category**: {slide.get('category', 'N/A')}
  - **Description**: {slide.get('description', 'N/A')}"""

    return slide_info


def get_template_num(slide_induction_path: str):
    template_num = 0
    with open(slide_induction_path, "r", encoding="utf-8") as f:
        slide_induction = json.load(f)

    for v in slide_induction.values():
        if isinstance(v, dict) and "template_id" in v:
            template_num = max(template_num, v["template_id"])

    return template_num
