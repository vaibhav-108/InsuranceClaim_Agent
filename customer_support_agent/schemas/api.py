from __future__ import annotations
from typing import List, Any, Literal
from pydantic import BaseModel, EmailStr, Field


class TicketCreateRequest(BaseModel):
    cuatomer_email: EmailStr
    customer_name: str | None=None
    customer_company: str|None=None
    subject: str= Field(min_length=3)
    discription: str = Field(min_length=10)
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    auto_generate: bool = False
    
#will get response from mongodb    
class TicketResponse(BaseModel):
    id: int
    customer_id: int
    customer_email: EmailStr
    customer_name: str 
    customer_company: str
    subject: str
    discription: str
    priority: str
    created_at: str
    updated_at: str

    
#just store count
class DraftSignals(BaseModel):
    memory_hit_count: int = 0
    knowledge_hit_count: int = 0
    tool_call_count: int = 0
    tool_error_count: int = 0
    knowledge_Sources: List[str] = Field(default_factory=list)
    
    
#list of string generarted by LLM while generate a response
class DraftHighLights(BaseModel):
    memory: list[str] = Field(default_factory=list)
    knowledge: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

#needed when call for draft tool
class DraftToolCall(BaseModel):
    name: str
    tool_call_id: str | None = None
    arguments: dict[str,Any] = Field(default_factory=dict)
    status: str
    summary: str | None = None
    output: dict[str,Any] | None = None
    output_text: str
    
# all will pass to LLM for generating drafts
class StructuredDraftContext(BaseModel):
    version: int = 2
    ticket: dict[str,Any] | None = None
    customer: dict[str,Any] | None = None
    signals: DraftSignals | dict[str,Any] | None = None
    highlights: DraftHighLights | dict[str,Any] | None = None
    memory_hits: List[dict[str,Any]] = Field(default_factory=list)
    knowledge_hits: List[dict[str,Any]] = Field(default_factory=list)
    tool_calls: List[DraftToolCall] | dict[str,Any] = Field(default_factory=list)
    error: List[str] = Field(default_factory=list)
    

class DraftResponse(BaseModel):
    id: int
    ticket_id: int
    content: str
    context_used: StructuredDraftContext | dict[str,Any] | None = None
    status: str
    created_at: str
    
    

class DraftUpdateRequest(BaseModel):
    content: str
    status: Literal["pending", "accepted", "discarded"] | None = None

    
class GenerateDraftResponse(BaseModel):
    ticket_id: int
    draft: DraftResponse  

class KnowledgeIngestRequest(BaseModel):
    clear_existing: bool = False

class KnowledgeIngestResponse(BaseModel):
    file_indexed: int
    chunk_indexed: int
    collection_count: int

class CustomerMemoriesResponse(BaseModel):
    customer_id: int
    customer_email: EmailStr
    memories: List[dict[str,Any]]


class customerMemorySearchResponse(BaseModel):
    customer_id: int
    customer_email: EmailStr
    query: str
    result: List[dict[str,Any]]






