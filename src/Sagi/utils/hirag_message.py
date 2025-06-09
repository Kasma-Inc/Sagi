import json
from autogen_agentchat.messages import ToolCallSummaryMessage

def hirag_message_to_llm_message(message: ToolCallSummaryMessage) -> ToolCallSummaryMessage:
    chunks, entities, relations, neighbors = [], [], [], []
    content = message.content
    breakpoint()
    query_results_str = json.loads(content)
    for query_result_str in query_results_str:
        breakpoint()
        query_result_json = json.loads(query_result_str["text"])
        chunks.extend(query_result_json["chunks"])
        entities.extend(query_result_json["entities"])
        relations.extend(query_result_json["relations"])
        neighbors.extend(query_result_json["neighbors"])

    seen_ids = set()
    chunks = [
        chunk
        for chunk in chunks
        if not (chunk["id"] in seen_ids or seen_ids.add(chunk["id"]))
    ]

    chunks_str = "\n".join([chunk["text"] for chunk in chunks])
    
    seen_ids = set()
    entities = [
        (entity["id"], entity["name"], entity["type"], entity["description"])
        for entity in entities
    ]
    neighbors = [
        (neighbor["id"], neighbor["name"], neighbor["type"], neighbor["metadata"]["description"])
        for neighbor in neighbors
    ]
    entities_with_neighbors = entities + neighbors
    
    seen_ids = set()
    entities_with_neighbors = [
        entity
        for entity in entities_with_neighbors
        if not (entity["id"] in seen_ids or seen_ids.add(entity["id"]))
    ]
    entities_with_neighbors_str = "\n".join([f"{entity[1]} with type {entity[2]} and description {entity[3]}" for entity in entities_with_neighbors])
    relations_str = "\n".join([relation["properties"]["description"] for relation in relations])
    hirag_message = f"The following is the information you can use to answer the question:\n\n"
    hirag_message += f"Chunks:\n{chunks_str}\n\n"
    hirag_message += f"Entities:\n{entities_with_neighbors_str}\n\n"
    hirag_message += f"Relations:\n{relations_str}\n\n"
    message.content = hirag_message
    return message

