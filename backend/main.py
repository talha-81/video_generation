from fastapi import FastAPI, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import aiohttp  # Add this import
from datetime import datetime

# Import our modules with proper path handling
import sys
from pathlib import Path

# Add the backend directory to Python path if not already there
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Now import our modules
from config import config
from models import (
    ProjectInfo, GenerationRequest, RegenerationRequest, ApprovalRequest
)
from business_logic import story_service
from session_manager import session_manager
from utils import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Story to Image Generator API v3.0 (Modular) starting...")
    print(f"📁 Projects directory: {config.projects_dir}")
    
    # Validate configuration
    config_status = config.validate_config()
    print("\n🔧 API Configuration Status:")
    for service, status in config_status.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {service.title()}: {'Configured' if status else 'Missing API key'}")
    
    # Print available models
    print("\n🤖 Available Models:")
    print("   AI Models (OpenRouter):")
    for model in config.openrouter.models:
        print(f"   • {model}")
    print("   Image Models:")
    print("   Runware:")
    for model in config.runware.models:
        print(f"   • {model}")
    print("   Together AI:")
    for model in config.together.models:
        print(f"   • {model}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    session_manager.force_cleanup_all()
    print("✅ Cleanup completed")

# Create FastAPI app
app = FastAPI(
    title="Story to Image Generator",
    description="Transform stories into images with AI - Modular Architecture v3.0",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files
app.mount("/projects", StaticFiles(directory=config.projects_dir), name="projects")

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Story to Image Generator API v3.0 - Modular Architecture", 
        "status": "ready",
        "active_sessions": session_manager.get_session_count(),
        "projects_count": len(list(config.projects_dir.iterdir())),
        "architecture": "modular",
        "features": [
            "Async image generation",
            "Thread-safe session management", 
            "Automatic session cleanup",
            "Comprehensive error handling"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with detailed status"""
    return story_service.get_health_status()

@app.get("/models")
async def get_available_models():
    """Get all available models"""
    return {
        "ai_models": {
            "openrouter": config.openrouter.models
        },
        "image_models": {
            "runware": config.runware.models,
            "together": config.together.models
        }
    }

@app.post("/analyze-script", response_model=ProjectInfo)
async def analyze_script_endpoint(
    script: str = Form(...), 
    title: str = Form("Untitled Story")
):
    """Analyze script and create project"""
    try:
        return story_service.analyze_script(script, title)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Script analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Script analysis failed")

@app.post("/generate-previews")
async def generate_previews(request: GenerationRequest):
    """Generate preview images with async support"""
    try:
        session_id = story_service.start_generation(request)
        
        return {
            "session_id": session_id,
            "status": "generating",
            "total_scenes": request.num_scenes,
            "project_id": request.project_id,
            "generation_type": "async_enhanced"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")

@app.post("/regenerate-scene")
async def regenerate_scene(request: RegenerationRequest):
    """Regenerate a specific scene image"""
    try:
        new_preview = story_service.regenerate_scene(request)
        
        return {
            "status": "success" if new_preview.preview_url else "failed",
            "scene_number": request.scene_number,
            "new_preview": new_preview
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Regeneration failed: {e}")
        raise HTTPException(status_code=500, detail="Regeneration failed")

@app.get("/generation-status/{session_id}")
async def get_generation_status(session_id: str):
    """Get generation status with enhanced information"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Add progress percentage
    progress_percentage = (session.completed_scenes / session.total_scenes * 100) if session.total_scenes > 0 else 0
    
    return {
        **session.dict(),
        "progress_percentage": round(progress_percentage, 1),
        "has_errors": len(session.errors) > 0
    }

@app.post("/approve-previews")
async def approve_previews(request: ApprovalRequest, background_tasks: BackgroundTasks):
    """Handle preview approvals and save images with async download"""
    try:
        # Try async version first, fallback to sync if needed
        try:
            result = await story_service.approve_and_save_async(request)
            save_method = "async"
        except Exception as async_error:
            logger.warning(f"Async save failed, falling back to sync: {async_error}")
            result = story_service.approve_and_save(request)
            save_method = "sync_fallback"
        
        session = session_manager.get_session(request.session_id)
        
        return {
            "status": "completed",
            **result,
            "total_scenes": len(session.previews) if session else 0,
            "save_method": save_method
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Approval failed: {e}")
        raise HTTPException(status_code=500, detail="Approval failed")

@app.get("/projects")
async def list_projects():
    """List all projects"""
    try:
        projects = story_service.list_projects()
        return {"projects": projects, "total_count": len(projects)}
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail="Failed to list projects")

@app.get("/projects/{project_id}")
async def get_project_details(project_id: str):
    """Get detailed project information"""
    try:
        return story_service.get_project_details(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Project not found")
    except Exception as e:
        logger.error(f"Failed to load project: {e}")
        raise HTTPException(status_code=500, detail="Failed to load project")

@app.delete("/sessions/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up session"""
    try:
        success = session_manager.cleanup_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session cleaned up successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Session cleanup failed")

# ==============================================================================
# DEBUG & MONITORING ENDPOINTS
# ==============================================================================

@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint for session information"""
    sessions_info = session_manager.get_all_sessions()
    return {
        "active_sessions": len(sessions_info),
        "sessions": sessions_info,
        "session_manager_stats": {
            "cleanup_interval_minutes": session_manager.cleanup_interval // 60,
            "max_session_age_hours": session_manager.max_session_age // 3600
        }
    }

@app.get("/debug/config")
async def debug_config():
    """Debug endpoint for configuration"""
    return {
        "config_status": config.validate_config(),
        "projects_dir": str(config.projects_dir),
        "timeout": config.timeout,
        "max_retries": config.max_retries,
        "retry_delay": config.retry_delay,
        "api_endpoints": {
            "openrouter": config.openrouter.api_url,
            "runware": config.runware.api_url,
            "together": config.together.api_url
        }
    }

@app.get("/stats")
async def get_stats():
    """General statistics endpoint"""
    try:
        projects_count = len(list(config.projects_dir.iterdir()))
        from models import GenerationStatus
        sessions_by_status = {
            status.value: len(session_manager.get_sessions_by_status(status))
            for status in GenerationStatus
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "projects": {
                "total": projects_count
            },
            "sessions": {
                "active_total": session_manager.get_session_count(),
                "by_status": sessions_by_status
            },
            "system": {
                "version": "3.0.0",
                "architecture": "modular"
            }
        }
    except Exception as e:
        logger.error(f"Stats endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

# ==============================================================================
# APPLICATION FACTORY & MAIN
# ==============================================================================

def create_app() -> FastAPI:
    """Factory function to create FastAPI app"""
    return app

def main():
    """Main entry point"""
    print("🎬 Story to Image Generator API v3.0 - Modular Architecture")
    print("📦 Enhanced with Async Support & Improved Performance")
    print("🔑 Environment Variables Required:")
    print("   - OPENROUTER_API_KEY")
    print("   - RUNWARE_API_KEY") 
    print("   - TOGETHER_API_KEY")
    print("\n🌟 New Features in v3.0:")
    print("   ✅ Modular architecture (separate files)")
    print("   ✅ Async image generation & downloads")
    print("   ✅ Enhanced session management with auto-cleanup")
    print("   ✅ Improved error handling & logging")
    print("   ✅ Performance monitoring endpoints")
    print("   ✅ Thread-safe operations")
    
    # Check for API keys
    config_status = config.validate_config()
    if not any(config_status.values()):
        print("\n⚠️  WARNING: No API keys configured!")
        print("   The application will run in fallback mode with limited functionality.")
    else:
        active_services = [service for service, status in config_status.items() if status]
        print(f"\n✅ Active Services: {', '.join(active_services)}")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()