import os
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

@dataclass
class APIConfig:
    """API configuration class"""
    api_key: str
    api_url: str
    models: List[str]

class Config:
    """Central configuration management"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.projects_dir = self.base_dir / "image_generation"
        self.projects_dir.mkdir(exist_ok=True)
        
        # API configurations
        self.openrouter = APIConfig(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            api_url="https://openrouter.ai/api/v1/chat/completions",
            models=[
                "openai/gpt-oss-20b:free",
                "meta-llama/llama-3.2-3b-instruct:free",
                "microsoft/phi-3-mini-128k-instruct:free"
            ]
        )
        
        self.runware = APIConfig(
            api_key=os.getenv("RUNWARE_API_KEY", ""),
            api_url="https://api.runware.ai/v1/imageInference",
            models=[
                "runware:101@1",
                "runware:100@1",
                "runware:102@1"
            ]
        )
        
        self.together = APIConfig(
            api_key=os.getenv("TOGETHER_API_KEY", ""),
            api_url="https://api.together.xyz/v1/images/generations",
            models=[
                "black-forest-labs/FLUX.1-schnell-Free",
                "stabilityai/stable-diffusion-xl-base-1.0"
            ]
        )
        
        # Application settings
        self.timeout = 60
        self.max_retries = 3
        self.retry_delay = 2

    def validate_config(self) -> Dict[str, bool]:
        """Validate API configurations"""
        return {
            "openrouter": bool(self.openrouter.api_key),
            "runware": bool(self.runware.api_key),
            "together": bool(self.together.api_key)
        }

# Global config instance
config = Config()