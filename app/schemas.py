from pydantic import BaseModel, Field
from typing import List, Optional


class AskRequest(BaseModel):
    question: str = Field(..., description="User question")
    k: int = 4
    collection: Optional[str] = None


class Source(BaseModel):
    text: str
    metadata: dict
    score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


class IngestRequest(BaseModel):
    path: str
    collection: Optional[str] = None