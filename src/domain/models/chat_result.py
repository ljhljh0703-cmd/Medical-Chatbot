"""
Chat domain result models.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    c_id: str
    source_spec: Optional[str] = None
    content_snippet: str = Field(default="", max_length=1000)
    similarity_score: float = 0.0


class ChatResult(BaseModel):
    query: str
    chatbot_answer: str
    ground_truth: Optional[str] = None
    retrieved_sources: list[RetrievedChunk] = Field(default_factory=list)
    red_flag_triggered: bool = False
    mode: str = "B"