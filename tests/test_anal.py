import requests
import uuid

url = "http://localhost:58001/api/chat?protocol=data"
headers = {
    "Content-Type": "application/json",
}
data = {
    "id": "chat196",
    "latestMessage": {
        "id": str(uuid.uuid4()),
        "role": "user",
        # "content": "Start to research",
        # "content": "Query the first eight pieces of data in the database and Conduct a transaction risk assessment",
        "content": "Query the first ten pieces of people who want a loan in the database and Conduct a credit examine assessment",
        "parts": [],
        "toolInvocations": [],
        "createdAt": "2025-06-08T12:00:00",
        "experimental_attachments": []
    },
    "selectedChatModel": "gpt-4",
    "search": False,
    # "workflow": "Risk_Control"
    "workflow": "Credit_Examine"
}

response = requests.post(url, json=data, headers=headers)
print(response.status_code, response.text)