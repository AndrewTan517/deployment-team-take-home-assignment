from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatCompletionsRequest(BaseModel):
    id: str
    input: str


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionsRequest) -> dict[str, Any]:
    return {
        "id": request.id,
        "object": "chat.completion",
        "model": "dummy-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Dummy response for input: {request.input}",
                },
                "finish_reason": "stop",
            }
        ],
    }
