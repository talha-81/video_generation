import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import tempfile
import json

# Import modules to test
from models import ScriptAnalysis, ScenePrompt, MediaType, GenerationStatus, PreviewImage
from utils import ScriptAnalyzer, ProjectManager
from session_manager import SessionManager
from ai_services import PromptGenerator
from business_logic import StoryToImageService

class TestScriptAnalyzer:
    """Test script analysis functionality"""
    
    def test_analyze_simple_script(self):
        """Test analysis of simple script"""
        script = "A brave knight saves a princess from a dragon."
        analysis = ScriptAnalyzer.analyze(script)
        
        assert analysis.word_count == 9
        assert analysis.recommended_scenes == 2
        assert analysis.complexity_score in ["Simple", "Complex"]
        assert analysis.estimated_duration_minutes > 0
    
    def test_analyze_empty_script(self):
        """Test analysis of empty script"""
        analysis = ScriptAnalyzer.analyze("")
        
        assert analysis.word_count == 0
        assert analysis.recommended_scenes == 2
        assert analysis.complexity_score == "Simple"
    
    def test_analyze_long_script(self):
        """Test analysis of long script"""
        script = " ".join(["word"] * 500)  # 500 words
        analysis = ScriptAnalyzer.analyze(script)
        
        assert analysis.word_count == 500
        assert analysis.recommended_scenes == 6
        assert analysis.estimated_duration_minutes > 2
    
    def test_complex_script_detection(self):
        """Test detection of complex scripts"""
        complex_script = "Extraordinary magnificent adventurous protagonist encounters supernatural phenomena"
        analysis = ScriptAnalyzer.analyze(complex_script)
        
        assert analysis.complexity_score == "Complex"

class TestProjectManager:
    """Test project management functionality"""
    
    def test_create_project(self):
        """Test project creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock config to use temp directory
            with patch('utils.config.projects_dir', Path(temp_dir)):
                analysis = ScriptAnalysis(
                    word_count=10,
                    recommended_scenes=3,
                    estimated_duration_minutes=1.5,
                    complexity_score="Simple"
                )
                
                project_path = ProjectManager.create_project(
                    "test_project", 
                    "Test Title", 
                    "Test script content",
                    analysis
                )
                
                assert project_path.exists()
                assert (project_path / "project.json").exists()
                assert (project_path / "images").exists()
                
                # Verify project data
                project_data = json.loads(
                    (project_path / "project.json").read_text()
                )
                assert project_data["title"] == "Test Title"
                assert project_data["script"] == "Test script content"
    
    def test_save_scene_prompts(self):
        """Test saving scene prompts"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            scenes = [
                ScenePrompt(
                    scene_number=1,
                    scene_title="Test Scene",
                    script_excerpt="Test excerpt",
                    image_prompt="Test prompt"
                )
            ]
            
            ProjectManager.save_scene_prompts(project_path, scenes)
            
            assert (project_path / "scene_prompts.json").exists()
            
            # Verify saved data
            saved_data = json.loads(
                (project_path / "scene_prompts.json").read_text()
            )
            assert saved_data["total_scenes"] == 1
            assert len(saved_data["scenes"]) == 1

class TestSessionManager:
    """Test session management"""
    
    def setup_method(self):
        """Setup fresh session manager for each test"""
        self.session_manager = SessionManager(
            cleanup_interval_minutes=1,  # Short interval for testing
            max_session_age_hours=1
        )
    
    def test_create_session(self):
        """Test session creation"""
        scenes = [
            ScenePrompt(
                scene_number=1,
                scene_title="Test",
                script_excerpt="Test",
                image_prompt="Test"
            )
        ]
        
        session_id = self.session_manager.create_session("test_project", scenes)
        
        assert session_id.startswith("session_")
        session = self.session_manager.get_session(session_id)
        assert session is not None
        assert session.project_id == "test_project"
        assert session.total_scenes == 1
        assert session.status == GenerationStatus.GENERATING
    
    def test_update_session(self):
        """Test session updates"""
        scenes = [ScenePrompt(scene_number=1, scene_title="Test", script_excerpt="Test", image_prompt="Test")]
        session_id = self.session_manager.create_session("test_project", scenes)
        
        self.session_manager.update_session(session_id, status=GenerationStatus.COMPLETED)
        
        session = self.session_manager.get_session(session_id)
        assert session.status == GenerationStatus.COMPLETED
    
    def test_add_preview(self):
        """Test adding preview to session"""
        scenes = [ScenePrompt(scene_number=1, scene_title="Test", script_excerpt="Test", image_prompt="Test")]
        session_id = self.session_manager.create_session("test_project", scenes)
        
        preview = PreviewImage(
            scene_number=1,
            scene_title="Test",
            prompt="Test prompt",
            preview_url="http://example.com/image.jpg",
            generation_time=1.5,
            provider_used="test",
            model_used="test_model"
        )
        
        self.session_manager.add_preview(session_id, preview)
        
        session = self.session_manager.get_session(session_id)
        assert len(session.previews) == 1
        assert session.completed_scenes == 1
    
    def test_cleanup_session(self):
        """Test session cleanup"""
        scenes = [ScenePrompt(scene_number=1, scene_title="Test", script_excerpt="Test", image_prompt="Test")]
        session_id = self.session_manager.create_session("test_project", scenes)
        
        success = self.session_manager.cleanup_session(session_id)
        
        assert success
        assert self.session_manager.get_session(session_id) is None

class TestPromptGenerator:
    """Test AI prompt generation"""
    
    def setup_method(self):
        """Setup prompt generator"""
        self.prompt_generator = PromptGenerator()
    
    def test_fallback_scene_generation(self):
        """Test fallback scene generation when AI is unavailable"""
        script = "A brave knight saves a princess from a dragon in a magical castle."
        scenes = self.prompt_generator._generate_fallback_scenes(
            script, 3, MediaType.CINEMATIC
        )
        
        assert len(scenes) == 3
        assert all(isinstance(scene, ScenePrompt) for scene in scenes)
        assert all(scene.scene_number > 0 for scene in scenes)
        assert all("cinematic" in scene.image_prompt.lower() for scene in scenes)
    
    def test_style_mapping(self):
        """Test different style mappings"""
        script = "Test script"
        
        # Test different media types
        for media_type in MediaType:
            scenes = self.prompt_generator._generate_fallback_scenes(script, 1, media_type)
            assert len(scenes) == 1
            
            # Check that style is applied
            style_keywords = {
                MediaType.CINEMATIC: "cinematic",
                MediaType.CARTOON: "cartoon",
                MediaType.REALISTIC: "realistic",
                MediaType.ARTISTIC: "artistic"
            }
            
            expected_keyword = style_keywords[media_type]
            assert expected_keyword in scenes[0].image_prompt.lower()

class TestStoryToImageService:
    """Test main business logic service"""
    
    def setup_method(self):
        """Setup service for testing"""
        self.service = StoryToImageService()
    
    def test_analyze_script_integration(self):
        """Test script analysis integration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('business_logic.config.projects_dir', Path(temp_dir)):
                project_info = self.service.analyze_script(
                    "A test story about adventure", 
                    "Test Project"
                )
                
                assert project_info.title == "Test Project"
                assert project_info.script_content == "A test story about adventure"
                assert project_info.analysis.word_count == 6
                assert project_info.project_id.startswith("story_")
    
    def test_empty_script_validation(self):
        """Test validation of empty scripts"""
        with pytest.raises(ValueError, match="Script cannot be empty"):
            self.service.analyze_script("", "Test")
        
        with pytest.raises(ValueError, match="Script cannot be empty"):
            self.service.analyze_script("   ", "Test")
    
    def test_get_health_status(self):
        """Test health status endpoint"""
        health = self.service.get_health_status()
        
        assert "status" in health
        assert "timestamp" in health
        assert "active_sessions" in health
        assert "services" in health
        assert "version" in health
        assert health["version"] == "3.0.0"

@pytest.mark.asyncio
class TestAsyncFunctionality:
    """Test async functionality"""
    
    async def test_async_image_generation_mock(self):
        """Test async image generation with mocks"""
        # Mock the image generator
        with patch('ai_services.aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "data": [{"imageURL": "http://example.com/test.jpg"}]
            }
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            from ai_services import RunwareGenerator
            generator = RunwareGenerator()
            
            scene = ScenePrompt(
                scene_number=1,
                scene_title="Test",
                script_excerpt="Test",
                image_prompt="Test prompt"
            )
            
            # This would normally require API keys, so we'll just test the structure
            # In a real test environment, you'd mock the entire flow
            assert hasattr(generator, 'generate_async')

# Integration Tests
class TestIntegration:
    """Integration tests for the full workflow"""
    
    def test_full_workflow_simulation(self):
        """Test a complete workflow simulation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('business_logic.config.projects_dir', Path(temp_dir)):
                service = StoryToImageService()
                
                # Step 1: Analyze script
                project_info = service.analyze_script(
                    "A brave knight rescues a princess from an evil dragon.",
                    "Knight Story"
                )
                
                assert project_info.project_id.startswith("story_")
                
                # Step 2: Verify project was created
                project_path = Path(temp_dir) / project_info.project_id
                assert project_path.exists()
                assert (project_path / "project.json").exists()
                
                # Step 3: List projects
                projects = service.list_projects()
                assert len(projects) == 1
                assert projects[0]["title"] == "Knight Story"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])