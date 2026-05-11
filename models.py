"""
models.py
---------
Pydantic schemas for the /chat API.
The response schema is NON-NEGOTIABLE — it is what SHL's evaluator expects.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def must_start_with_user(cls, messages):
        if messages[0].role != "user":
            raise ValueError("Conversation must start with a user message.")
        return messages


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # single letter code: A, B, C, D, E, K, M, P, S


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def max_ten(cls, recs):
        if len(recs) > 10:
            raise ValueError("recommendations must not exceed 10 items.")
        return recs


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — catalog entry shape
# ─────────────────────────────────────────────────────────────────────────────

class CatalogEntry(BaseModel):
    name: str
    url: str
    test_type: str
    test_type_label: Optional[str] = ""
    description: Optional[str] = ""
    remote_testing: Optional[bool] = False
    adaptive: Optional[bool] = False
    duration: Optional[str] = ""
