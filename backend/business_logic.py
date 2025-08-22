import asyncio
import threading
import requests
import aiohttp  
from typing import Dict, List
from datetime import datetime
from pathlib import Path

from models import (
    ProjectInfo, GenerationRequest, RegenerationRequest, ApprovalRequest,
    ScenePrompt, PreviewImage, GenerationStatus
)
from utils import ScriptAnalyzer, ProjectManager, logger
from ai_services import PromptGenerator, ImageGenerator
from session_manager import session_manager
from config import config

class StoryToImageService:
    """Main business logic service with async support"""
    
    def __init__(self):
        self.prompt_generator = PromptGenerator()
        self.image_generator = ImageGenerator()
    
    def analyze_script(self, script: str, title: str) -> ProjectInfo:
        """Analyze script and create project"""
        if not script.strip():
            raise ValueError("Script cannot be empty")
        
        project_id = f"story_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        analysis = ScriptAnalyzer.analyze(script)
        ProjectManager.create_project(project_id, title, script, analysis)
        
        return ProjectInfo(
            project_id=project_id,
            title=title,
            created_at=datetime.now().isoformat(),
            analysis=analysis,
            script_content=script
        )
    
    def start_generation(self, request: GenerationRequest) -> str:
        """Start image generation process"""
        # Load project
        script, title, analysis = ProjectManager.load_project(request.project_id)
        
        # Generate scene prompts
        scenes = self.prompt_generator.generate_prompts(
            script, request.num_scenes, request.media_type, request.ai_model
        )
        
        if not scenes:
            raise ValueError("Failed to generate scene prompts")
        
        # Save scene prompts
        project_path = config.projects_dir / request.project_id
        ProjectManager.save_scene_prompts(project_path, scenes)
        
        # Create session
        session_id = session_manager.create_session(request.project_id, scenes)
        
        # Start background generation (async if possible, fallback to sync)
        try:
            self._start_async_generation(session_id, scenes, request.image_provider, request.image_model)
        except Exception as e:
            logger.warning(f"Async generation failed, falling back to sync: {e}")
            self._start_background_generation(session_id, scenes, request.image_provider, request.image_model)
        
        return session_id
    
    def _start_async_generation(self, session_id: str, scenes: List[ScenePrompt], provider: str, model: str):
        """Start asynchronous image generation"""
        async def generate_images_async():
            try:
                # Generate all images concurrently
                previews = await self.image_generator.generate_images_async(scenes, provider, model)
                
                # Process results
                for i, result in enumerate(previews):
                    session = session_manager.get_session(session_id)
                    if not session or session.status == GenerationStatus.CANCELLED:
                        break
                    
                    if isinstance(result, PreviewImage):
                        session_manager.add_preview(session_id, result)
                    elif isinstance(result, Exception):
                        logger.error(f"Scene {i+1} generation failed: {result}")
                        session_manager.update_session(
                            session_id, 
                            errors=session.errors + [f"Failed to generate scene {i+1}: {str(result)}"]
                        )
                    else:
                        logger.error(f"Unexpected result type for scene {i+1}: {type(result)}")
                
                # Update final status
                session = session_manager.get_session(session_id)
                if session and session.status != GenerationStatus.CANCELLED:
                    session_manager.update_session(session_id, status=GenerationStatus.PREVIEWING)
                    logger.info(f"Async generation completed for session {session_id}")
                    
            except Exception as e:
                logger.error(f"Async background generation failed: {e}")
                session_manager.update_session(
                    session_id, 
                    status=GenerationStatus.FAILED,
                    errors=[f"Generation failed: {str(e)}"]
                )
        
        def run_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(generate_images_async())
            except Exception as e:
                logger.error(f"Async loop failed: {e}")
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        logger.info(f"Started async generation thread for session {session_id}")
    
    def _start_background_generation(self, session_id: str, scenes: List[ScenePrompt], provider: str, model: str):
        """Start background image generation (fallback sync method)"""
        def generate_images():
            try:
                for scene in scenes:
                    session = session_manager.get_session(session_id)
                    if not session or session.status == GenerationStatus.CANCELLED:
                        break
                    
                    preview = self.image_generator.generate_image(scene, provider, model)
                    session_manager.add_preview(session_id, preview)
                    
                    if not preview.preview_url:
                        session_manager.update_session(
                            session_id, 
                            errors=session.errors + [f"Failed to generate scene {scene.scene_number}"]
                        )
                
                # Update final status
                session = session_manager.get_session(session_id)
                if session and session.status != GenerationStatus.CANCELLED:
                    session_manager.update_session(session_id, status=GenerationStatus.PREVIEWING)
                    logger.info(f"Sync generation completed for session {session_id}")
                    
            except Exception as e:
                logger.error(f"Background generation failed: {e}")
                session_manager.update_session(
                    session_id, 
                    status=GenerationStatus.FAILED,
                    errors=[f"Generation failed: {str(e)}"]
                )
        
        thread = threading.Thread(target=generate_images, daemon=True)
        thread.start()
        logger.info(f"Started sync generation thread for session {session_id}")
    
    def regenerate_scene(self, request: RegenerationRequest) -> PreviewImage:
        """Regenerate a specific scene"""
        session = session_manager.get_session(request.session_id)
        if not session:
            raise ValueError("Session not found")
        
        # Find scene prompt
        scene_prompt = None
        for prompt in session.scene_prompts:
            if prompt.scene_number == request.scene_number:
                scene_prompt = prompt
                break
        
        if not scene_prompt:
            raise ValueError("Scene not found")
        
        # Generate new image
        new_preview = self.image_generator.generate_image(
            scene_prompt, request.image_provider, request.image_model
        )
        
        # Update session
        session_manager.add_preview(request.session_id, new_preview)
        
        logger.info(f"Regenerated scene {request.scene_number} for session {request.session_id}")
        return new_preview
    
    async def approve_and_save_async(self, request: ApprovalRequest) -> Dict[str, int]:
        """Handle preview approvals and save images asynchronously"""
        session = session_manager.get_session(request.session_id)
        if not session:
            raise ValueError("Session not found")
        
        # Update approvals
        approved_count = 0
        for scene_num, approved in request.scene_approvals.items():
            for preview in session.previews:
                if preview.scene_number == scene_num:
                    preview.approved = approved
                    if approved:
                        approved_count += 1
                    break
        
        if approved_count == 0:
            raise ValueError("No scenes were approved")
        
        # Save approved images asynchronously
        saved_count = await self._save_approved_images_async(session)
        session_manager.update_session(request.session_id, status=GenerationStatus.COMPLETED)
        
        logger.info(f"Approved and saved {saved_count} images for session {request.session_id}")
        return {"saved_images": saved_count, "approved_scenes": approved_count}
    
    def approve_and_save(self, request: ApprovalRequest) -> Dict[str, int]:
        """Handle preview approvals and save images (sync version)"""
        session = session_manager.get_session(request.session_id)
        if not session:
            raise ValueError("Session not found")
        
        # Update approvals
        approved_count = 0
        for scene_num, approved in request.scene_approvals.items():
            for preview in session.previews:
                if preview.scene_number == scene_num:
                    preview.approved = approved
                    if approved:
                        approved_count += 1
                    break
        
        if approved_count == 0:
            raise ValueError("No scenes were approved")
        
        # Save approved images
        saved_count = self._save_approved_images(session)
        session_manager.update_session(request.session_id, status=GenerationStatus.COMPLETED)
        
        return {"saved_images": saved_count, "approved_scenes": approved_count}
    
    async def _save_approved_images_async(self, session) -> int:
        """Save approved images to project folder asynchronously"""
        try:
            project_path = config.projects_dir / session.project_id
            saved_count = 0
            
            # Prepare download tasks
            download_tasks = []
            
            async def download_and_save(preview):
                try:
                    async with aiohttp.ClientSession() as client_session:
                        async with client_session.get(preview.preview_url, timeout=30) as response:
                            if response.status == 200:
                                content = await response.read()
                                filename = f"scene_{preview.scene_number:03d}.jpg"
                                file_path = project_path / "images" / filename
                                
                                # Write file synchronously (file I/O is fast)
                                file_path.write_bytes(content)
                                logger.info(f"Saved image for scene {preview.scene_number}")
                                return True
                except Exception as e:
                    logger.error(f"Failed to save scene {preview.scene_number}: {e}")
                    return False
                return False
            
            # Create download tasks for approved images
            for preview in session.previews:
                if preview.approved and preview.preview_url:
                    download_tasks.append(download_and_save(preview))
            
            # Execute downloads concurrently
            if download_tasks:
                results = await asyncio.gather(*download_tasks, return_exceptions=True)
                saved_count = sum(1 for result in results if result is True)
            
            return saved_count
        except Exception as e:
            logger.error(f"Error saving approved images: {e}")
            return 0
    
    def _save_approved_images(self, session) -> int:
        """Save approved images to project folder (sync version)"""
        try:
            project_path = config.projects_dir / session.project_id
            saved_count = 0
            
            for preview in session.previews:
                if preview.approved and preview.preview_url:
                    try:
                        response = requests.get(preview.preview_url, timeout=30)
                        if response.status_code == 200:
                            filename = f"scene_{preview.scene_number:03d}.jpg"
                            file_path = project_path / "images" / filename
                            
                            file_path.write_bytes(response.content)
                            saved_count += 1
                            logger.info(f"Saved image for scene {preview.scene_number}")
                            
                    except Exception as e:
                        logger.error(f"Failed to save scene {preview.scene_number}: {e}")
            
            return saved_count
        except Exception as e:
            logger.error(f"Error saving approved images: {e}")
            return 0
    
    def list_projects(self) -> List[Dict]:
        """List all projects"""
        projects = []
        
        try:
            for folder in config.projects_dir.iterdir():
                if folder.is_dir() and (folder / "project.json").exists():
                    try:
                        import json
                        project_data = json.loads(
                            (folder / "project.json").read_text(encoding='utf-8')
                        )
                        
                        # Count images
                        images_count = 0
                        images_dir = folder / "images"
                        if images_dir.exists():
                            images_count = len(list(images_dir.glob("*.jpg")))
                        
                        projects.append({
                            "project_id": folder.name,
                            "title": project_data.get("title", "Untitled"),
                            "created_at": project_data.get("created_at", ""),
                            "analysis": project_data.get("analysis", {}),
                            "images_count": images_count
                        })
                    except Exception as e:
                        logger.error(f"Error loading project {folder.name}: {e}")
            
            # Sort by creation date (newest first)
            projects.sort(key=lambda x: x["created_at"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
        
        return projects
    
    def get_project_details(self, project_id: str) -> Dict:
        """Get detailed project information"""
        try:
            script, title, analysis = ProjectManager.load_project(project_id)
            
            # List images
            images_dir = config.projects_dir / project_id / "images"
            images = []
            if images_dir.exists():
                for img_file in sorted(images_dir.glob("*.jpg")):
                    images.append(f"/projects/{project_id}/images/{img_file.name}")
            
            return {
                "project_id": project_id,
                "title": title,
                "script": script,
                "analysis": analysis.dict(),
                "images": images,
                "total_images": len(images)
            }
            
        except Exception as e:
            logger.error(f"Error loading project {project_id}: {e}")
            raise
    
    def get_health_status(self) -> Dict:
        """Get service health status"""
        config_status = config.validate_config()
        active_sessions = session_manager.get_session_count()
        
        return {
            "status": "healthy" if any(config_status.values()) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "active_sessions": active_sessions,
            "total_projects": len(list(config.projects_dir.iterdir())),
            "services": config_status,
            "version": "3.0.0",
            "session_breakdown": {
                status.value: len(session_manager.get_sessions_by_status(status))
                for status in GenerationStatus
            }
        }

# Global service instance
story_service = StoryToImageService()