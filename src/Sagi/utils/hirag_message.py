import json
import logging

from autogen_agentchat.messages import ToolCallSummaryMessage


def extract_texts(content):
    texts = []
    for line in content.strip().split("\n"):
        if line:
            array = json.loads(line)
            for item in array:
                if item.get("type") == "text" and "text" in item:
                    texts.append(item["text"])
    return texts


def unique_by_key(items, key):
    seen = set()
    result = []
    for item in items:
        identifier = item[key]
        if identifier not in seen:
            seen.add(identifier)
            result.append(item)
    return result


def unique_by_first_element(tuples):
    seen = set()
    result = []
    for tup in tuples:
        identifier = tup[0]
        if identifier not in seen:
            seen.add(identifier)
            result.append(tup)
    return result


def hirag_message_to_llm_message(
    message: ToolCallSummaryMessage,
) -> ToolCallSummaryMessage:
    try:
        texts = extract_texts(message.content)
        chunks, entities, relations, neighbors = [], [], [], []

        for text in texts:
            query_result_json = json.loads(text)
            chunks.extend(query_result_json.get("chunks", []))
            entities.extend(query_result_json.get("entities", []))
            relations.extend(query_result_json.get("relations", []))
            neighbors.extend(query_result_json.get("neighbors", []))

        chunks = unique_by_key(chunks, key="document_key")
        chunks_str = "\n".join([chunk["text"] for chunk in chunks])

        entities = [
            (
                entity["document_key"],
                entity["text"],
                entity["entity_type"],
                entity["description"],
            )
            for entity in entities
        ]
        neighbors = [
            (
                neighbor["id"],
                neighbor["page_content"],
                neighbor["metadata"]["entity_type"],
                neighbor["metadata"]["description"],
            )
            for neighbor in neighbors
        ]
        entities_with_neighbors = unique_by_first_element(entities + neighbors)
        entities_with_neighbors_str = "\n".join(
            [
                f"{entity[1]} with type {entity[2]} and description {entity[3]}"
                for entity in entities_with_neighbors
            ]
        )

        relations_str = "\n".join(
            [relation["properties"]["description"] for relation in relations]
        )

        # Prepare the information for the LLM to answer the question
        cleaned_message_content = (
            f"The following is the information you can use to answer the question:\n\n"
            f"Chunks:\n{chunks_str}\n\n"
            f"Entities:\n{entities_with_neighbors_str}\n\n"
            f"Relations:\n{relations_str}\n\n"
        )
        cleaned_message_content = [
            {"type": "text", "text": cleaned_message_content, "annotations": None}
        ]
        message.content = json.dumps(cleaned_message_content)
        return message
    except Exception as e:
        logging.error(f"Error in hirag_message_to_llm_message: {e}")
        return message
