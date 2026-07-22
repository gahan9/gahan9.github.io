"""
Data Models for Defect Triage System

Defines Pydantic models for JIRA defects, embeddings, and triage predictions.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DefectDomain(str, Enum):
    """Domain-based cluster classification for defects."""
    INFRASTRUCTURE = "infrastructure"
    SYSTEM_CALL = "system_call"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    FIRMWARE = "firmware"
    DRIVER = "driver"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class RejectionReason(str, Enum):
    """Standardized rejection/root cause reasons."""
    SETUP_ISSUE = "setup_issue"
    INFRA_ISSUE = "infrastructure_issue"
    INVALID_CONFIG = "invalid_configuration"
    KNOWN_LIMITATION = "known_limitation"
    DUPLICATE = "duplicate"
    NOT_REPRODUCIBLE = "not_reproducible"
    WORKING_AS_DESIGNED = "working_as_designed"
    FIXED = "fixed"
    PENDING_INFO = "pending_information"


class TriagePriority(str, Enum):
    """Triage priority levels."""
    CRITICAL = "P0"
    HIGH = "P1"
    MEDIUM = "P2"
    LOW = "P3"
    TRIVIAL = "P4"


class JiraDefect(BaseModel):
    """JIRA Defect model representing a validation issue."""
    
    id: str = Field(..., description="JIRA ticket ID (e.g., HPC-1234)")
    summary: str = Field(..., description="Defect summary/title")
    description: str = Field(..., description="Detailed defect description")
    component: str = Field(..., description="Affected component/module")
    platform: str = Field(default="MI300X", description="Target HPC platform")
    error_log: Optional[str] = Field(None, description="Error logs or stack traces")
    steps_to_reproduce: Optional[str] = Field(None, description="Steps to reproduce")
    environment: Optional[str] = Field(None, description="Test environment details")
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "HPC-5678",
                "summary": "GPU memory allocation fails on MI300X under high load",
                "description": "When running PyTorch distributed training with 8 GPUs...",
                "component": "ROCm-Memory",
                "platform": "MI300X",
                "error_log": "hip_memory_error: out of device memory",
            }
        }


class DefectEmbedding(BaseModel):
    """Embedding representation for semantic search."""
    
    defect_id: str
    embedding: list[float] = Field(..., description="PyTorch embedding vector")
    domain: DefectDomain
    metadata: dict = Field(default_factory=dict)


class TriagePrediction(BaseModel):
    """AI-generated triage prediction for a defect."""
    
    defect_id: str
    predicted_domain: DefectDomain
    predicted_priority: TriagePriority
    predicted_rejection_reason: Optional[RejectionReason] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    similar_defects: list[str] = Field(default_factory=list)
    suggested_resolution: Optional[str] = None
    context_summary: str = Field(..., description="LLM-generated context analysis")
    requires_manual_review: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "defect_id": "HPC-5678",
                "predicted_domain": "memory",
                "predicted_priority": "P1",
                "confidence_score": 0.87,
                "similar_defects": ["HPC-1234", "HPC-2345"],
                "suggested_resolution": "Increase GPU memory pool allocation...",
                "context_summary": "Memory allocation failure during distributed training...",
            }
        }


class TriageState(BaseModel):
    """LangGraph state for the triage workflow."""
    
    defect: JiraDefect
    embedding: Optional[list[float]] = None
    retrieved_context: list[dict] = Field(default_factory=list)
    domain_classification: Optional[DefectDomain] = None
    trend_analysis: dict = Field(default_factory=dict)
    prediction: Optional[TriagePrediction] = None
    messages: list[dict] = Field(default_factory=list)
    error: Optional[str] = None
