from pydantic import BaseModel
from typing import Dict, List
from enum import Enum

class MediaType(str, Enum):
    CINEMATIC = "cinematic"
    CARTOON = "cartoon"
    REALISTIC = "realistic"
    ARTISTIC = "artistic"

class GenerationStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PREVIEWING = "previewing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ScriptAnalysis(BaseModel):
    word_count: int
    recommended_scenes: int
    estimated_duration_minutes: float
    complexity_score: str

class ScenePrompt(BaseModel):
    scene_number: int
    scene_title: str
    script_excerpt: str
    image_prompt: str

class PreviewImage(BaseModel):
    scene_number: int
    scene_title: str
    prompt: str
    preview_url: str
    generation_time: float
    provider_used: str
    model_used: str
    approved: bool = False

class GenerationSession(BaseModel):
    session_id: str
    project_id: str
    status: GenerationStatus
    total_scenes: int
    completed_scenes: int
    previews: List[PreviewImage] = []
    scene_prompts: List[ScenePrompt] = []
    errors: List[str] = []

class ProjectInfo(BaseModel):
    project_id: str
    title: str
    created_at: str
    analysis: ScriptAnalysis
    script_content: str

# Request/Response models
class GenerationRequest(BaseModel):
    project_id: str
    num_scenes: int
    media_type: MediaType = MediaType.CINEMATIC
    ai_provider: str = "openrouter"
    ai_model: str = "openai/gpt-oss-20b:free"
    image_provider: str = "runware"
    image_model: str = "runware:101@1"

class RegenerationRequest(BaseModel):
    session_id: str
    scene_number: int
    image_provider: str = "runware"
    image_model: str = "runware:101@1"

class ApprovalRequest(BaseModel):
    session_id: str
    scene_approvals: Dict[int, bool]