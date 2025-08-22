"""
Utilities Module
Author: Assistant
Version: 3.0.0
"""

import json
import logging
from typing import Tuple, List  # Add List import here
from pathlib import Path
from datetime import datetime

from models import ScriptAnalysis, ScenePrompt
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScriptAnalyzer:
    """Utility class for analyzing scripts"""
    
    @staticmethod
    def analyze(script: str) -> ScriptAnalysis:
        """Analyze script and return metrics"""
        try:
            words = script.strip().split()
            word_count = len(words)
            
            # Calculate recommended scenes
            if word_count < 50:
                recommended_scenes = 2
            elif word_count < 150:
                recommended_scenes = 3
            elif word_count < 300:
                recommended_scenes = 4
            elif word_count < 500:
                recommended_scenes = 6
            else:
                recommended_scenes = min(8, word_count // 80)
            
            # Estimate duration (200 words per minute)
            estimated_minutes = max(0.5, word_count / 200)
            
            # Complexity score
            if word_count > 0:
                avg_word_length = sum(len(word) for word in words) / word_count
                complexity = "Complex" if avg_word_length > 6 else "Simple"
            else:
                complexity = "Simple"
            
            return ScriptAnalysis(
                word_count=word_count,
                recommended_scenes=recommended_scenes,
                estimated_duration_minutes=estimated_minutes,
                complexity_score=complexity
            )
        except Exception as e:
            logger.error(f"Script analysis failed: {e}")
            return ScriptAnalysis(
                word_count=0,
                recommended_scenes=2,
                estimated_duration_minutes=1.0,
                complexity_score="Simple"
            )

class ProjectManager:
    """Manages project creation and file operations"""
    
    @staticmethod
    def create_project(project_id: str, title: str, script: str, analysis: ScriptAnalysis) -> Path:
        """Create project directory and save files"""
        try:
            project_path = config.projects_dir / project_id
            project_path.mkdir(exist_ok=True)
            (project_path / "images").mkdir(exist_ok=True)
            
            # Save project metadata
            project_meta = {
                "title": title,
                "script": script,
                "analysis": analysis.dict(),
                "created_at": datetime.now().isoformat()
            }
            
            (project_path / "project.json").write_text(
                json.dumps(project_meta, indent=2),
                encoding='utf-8'
            )
            
            logger.info(f"Created project: {project_id}")
            return project_path
            
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            raise

    @staticmethod
    def save_scene_prompts(project_path: Path, scenes: List[ScenePrompt]):
        """Save scene prompts to file"""
        try:
            prompts_data = {
                "generated_at": datetime.now().isoformat(),
                "total_scenes": len(scenes),
                "scenes": [scene.dict() for scene in scenes]
            }
            
            (project_path / "scene_prompts.json").write_text(
                json.dumps(prompts_data, indent=2),
                encoding='utf-8'
            )
            logger.info(f"Saved {len(scenes)} scene prompts")
        except Exception as e:
            logger.error(f"Failed to save scene prompts: {e}")

    @staticmethod
    def load_project(project_id: str) -> Tuple[str, str, ScriptAnalysis]:
        """Load project data"""
        project_path = config.projects_dir / project_id
        if not project_path.exists():
            raise FileNotFoundError(f"Project {project_id} not found")
        
        project_file = project_path / "project.json"
        if project_file.exists():
            project_data = json.loads(project_file.read_text(encoding='utf-8'))
            return (
                project_data.get("script", ""),
                project_data.get("title", "Untitled"),
                ScriptAnalysis(**project_data.get("analysis", {}))
            )
        else:
            raise FileNotFoundError("Project data not found")