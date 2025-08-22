import threading
import uuid
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from models import GenerationSession, GenerationStatus, ScenePrompt, PreviewImage
from utils import logger

class SessionManager:
    """Manages generation sessions with thread safety and cleanup"""
    
    def __init__(self, cleanup_interval_minutes: int = 60, max_session_age_hours: int = 24):
        self.sessions: Dict[str, GenerationSession] = {}
        self.lock = threading.RLock()  # Use RLock for nested locking
        self.cleanup_interval = cleanup_interval_minutes * 60
        self.max_session_age = max_session_age_hours * 3600
        self.session_timestamps: Dict[str, datetime] = {}
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for session cleanup"""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_old_sessions()
                    threading.Event().wait(self.cleanup_interval)
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("Session cleanup thread started")
    
    def _cleanup_old_sessions(self):
        """Remove old sessions that exceed max age"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        with self.lock:
            for session_id, timestamp in self.session_timestamps.items():
                age = (current_time - timestamp).total_seconds()
                if age > self.max_session_age:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                if session_id in self.sessions:
                    self.sessions[session_id].status = GenerationStatus.CANCELLED
                    del self.sessions[session_id]
                    del self.session_timestamps[session_id]
                    logger.info(f"Cleaned up old session: {session_id}")
    
    def create_session(self, project_id: str, scenes: List[ScenePrompt]) -> str:
        """Create a new generation session"""
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        session = GenerationSession(
            session_id=session_id,
            project_id=project_id,
            status=GenerationStatus.GENERATING,
            total_scenes=len(scenes),
            completed_scenes=0,
            scene_prompts=scenes
        )
        
        with self.lock:
            self.sessions[session_id] = session
            self.session_timestamps[session_id] = datetime.now()
        
        logger.info(f"Created session {session_id} with {len(scenes)} scenes")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[GenerationSession]:
        """Get session by ID"""
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                # Update last accessed time
                self.session_timestamps[session_id] = datetime.now()
            return session
    
    def update_session(self, session_id: str, **kwargs):
        """Update session properties"""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                # Update timestamp
                self.session_timestamps[session_id] = datetime.now()
                logger.debug(f"Updated session {session_id}: {kwargs}")
    
    def add_preview(self, session_id: str, preview: PreviewImage):
        """Add preview to session"""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                
                # Replace existing preview for the same scene or add new one
                for i, existing_preview in enumerate(session.previews):
                    if existing_preview.scene_number == preview.scene_number:
                        session.previews[i] = preview
                        break
                else:
                    session.previews.append(preview)
                
                session.completed_scenes = len(session.previews)
                self.session_timestamps[session_id] = datetime.now()
                
                logger.info(f"Added preview for scene {preview.scene_number} to session {session_id}")
    
    def cleanup_session(self, session_id: str) -> bool:
        """Remove session"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].status = GenerationStatus.CANCELLED
                del self.sessions[session_id]
                if session_id in self.session_timestamps:
                    del self.session_timestamps[session_id]
                logger.info(f"Cleaned up session: {session_id}")
                return True
        return False
    
    def get_all_sessions(self) -> Dict[str, Dict]:
        """Get all sessions for debugging"""
        with self.lock:
            current_time = datetime.now()
            return {
                sid: {
                    "status": s.status,
                    "completed": s.completed_scenes,
                    "total": s.total_scenes,
                    "project_id": s.project_id,
                    "age_minutes": int((current_time - self.session_timestamps.get(sid, current_time)).total_seconds() / 60),
                    "errors_count": len(s.errors)
                }
                for sid, s in self.sessions.items()
            }
    
    def get_session_count(self) -> int:
        """Get total number of active sessions"""
        with self.lock:
            return len(self.sessions)
    
    def get_sessions_by_status(self, status: GenerationStatus) -> List[GenerationSession]:
        """Get all sessions with a specific status"""
        with self.lock:
            return [session for session in self.sessions.values() if session.status == status]
    
    def force_cleanup_all(self):
        """Force cleanup of all sessions (for shutdown)"""
        with self.lock:
            for session in self.sessions.values():
                session.status = GenerationStatus.CANCELLED
            self.sessions.clear()
            self.session_timestamps.clear()
            logger.info("Force cleaned up all sessions")

# Global session manager instance
session_manager = SessionManager()