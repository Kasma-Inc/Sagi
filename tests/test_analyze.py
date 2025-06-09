import requests
import uuid

url = "http://localhost:8001/api/chat?protocol=data"
headers = {
    "Content-Type": "application/json",
}
data = {
    "id": "chat148",
    "latestMessage": {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": "Query the first eight pieces of data in the database and analyzing it",
        "parts": [],
        "toolInvocations": [],
        "createdAt": "2025-06-08T12:00:00",
        "experimental_attachments": []
    },
    "selectedChatModel": "gpt-4",
    "search": False,
    "selectedWorkflow": "analyzing"
}

response = requests.post(url, json=data, headers=headers)
print(response.status_code, response.text)
