"""
Story to Image Generator - Streamlit Frontend (Minimal Version)
Author: Assistant
Version: 3.0.0
"""

import streamlit as st
import requests
import time
from typing import Dict, List, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Story to Image Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Direct configuration (instead of config.py)
API_BASE_URL = "http://127.0.0.1:8000"
MEDIA_TYPES = {
    "cinematic": "🎬 Cinematic",
    "cartoon": "🎨 Cartoon", 
    "realistic": "📸 Realistic",
    "artistic": "🖼️ Artistic"
}

class APIClient:
    """Client for communicating with the FastAPI backend"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
    
    def check_health(self) -> Dict:
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.json() if response.status_code == 200 else {"status": "error"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_models(self) -> Dict:
        """Get available AI and image models"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            return response.json() if response.status_code == 200 else {}
        except Exception:
            return {}
    
    def analyze_script(self, script: str, title: str) -> Optional[Dict]:
        """Analyze script and create project"""
        try:
            data = {"script": script, "title": title}
            response = requests.post(f"{self.base_url}/analyze-script", data=data, timeout=30)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Failed to analyze script: {str(e)}")
            return None
    
    def generate_previews(self, request_data: Dict) -> Optional[Dict]:
        """Start preview generation"""
        try:
            response = requests.post(f"{self.base_url}/generate-previews", json=request_data, timeout=30)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Failed to start generation: {str(e)}")
            return None
    
    def get_generation_status(self, session_id: str) -> Optional[Dict]:
        """Get generation status"""
        try:
            response = requests.get(f"{self.base_url}/generation-status/{session_id}", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None
    
    def regenerate_scene(self, request_data: Dict) -> Optional[Dict]:
        """Regenerate a specific scene"""
        try:
            response = requests.post(f"{self.base_url}/regenerate-scene", json=request_data, timeout=60)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Failed to regenerate scene: {str(e)}")
            return None
    
    def approve_previews(self, request_data: Dict) -> Optional[Dict]:
        """Approve previews and save images"""
        try:
            response = requests.post(f"{self.base_url}/approve-previews", json=request_data, timeout=120)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            st.error(f"Failed to approve previews: {str(e)}")
            return None
    
    def list_projects(self) -> List[Dict]:
        """List all projects"""
        try:
            response = requests.get(f"{self.base_url}/projects", timeout=10)
            if response.status_code == 200:
                return response.json().get("projects", [])
        except Exception:
            pass
        return []
    
    def get_project_details(self, project_id: str) -> Optional[Dict]:
        """Get project details"""
        try:
            response = requests.get(f"{self.base_url}/projects/{project_id}", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

# Initialize API client
api_client = APIClient()

def display_health_status():
    """Display API health status in sidebar"""
    with st.sidebar:
        st.subheader("🏥 System Status")
        health = api_client.check_health()
        
        if health.get("status") == "healthy":
            st.success("✅ API Connected")
        elif health.get("status") == "degraded":
            st.warning("⚠️ API Degraded")
        else:
            st.error("❌ API Offline")
            st.error(health.get("message", "Cannot connect to backend"))
            return False
        
        # Show service status
        if "services" in health:
            st.write("**Services:**")
            for service, status in health["services"].items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {service.title()}")
        
        return True

def script_analysis_page():
    """Script analysis and project creation page"""
    st.title("🎬 Story to Image Generator")
    st.subheader("Transform your stories into stunning visual narratives")
    
    # Check if API is healthy
    if not display_health_status():
        st.stop()
    
    # Get available models
    models = api_client.get_models()
    ai_models = models.get("ai_models", {}).get("openrouter", [])
    runware_models = models.get("image_models", {}).get("runware", [])
    together_models = models.get("image_models", {}).get("together", [])
    
    with st.form("script_form"):
        st.markdown("### 📝 Enter Your Story")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            title = st.text_input("Story Title", placeholder="Enter a creative title for your story")
            script = st.text_area(
                "Story Content", 
                height=200,
                placeholder="Write your story here... Be descriptive and creative!"
            )
        
        with col2:
            st.markdown("### ⚙️ Generation Settings")
            num_scenes = st.slider("Number of Scenes", 2, 8, 4)
            media_type = st.selectbox("Visual Style", list(MEDIA_TYPES.keys()),
                                    format_func=lambda x: MEDIA_TYPES[x])
            
            # Model selection
            if ai_models:
                ai_model = st.selectbox("AI Model", ai_models)
            else:
                ai_model = "openai/gpt-oss-20b:free"
                st.warning("AI models not available")
            
            image_provider = st.selectbox("Image Provider", ["runware", "together"])
            
            if image_provider == "runware" and runware_models:
                image_model = st.selectbox("Image Model", runware_models)
            elif image_provider == "together" and together_models:
                image_model = st.selectbox("Image Model", together_models)
            else:
                image_model = "runware:101@1" if image_provider == "runware" else "black-forest-labs/FLUX.1-schnell-Free"
                st.warning(f"{image_provider} models not available")
        
        submitted = st.form_submit_button("🚀 Analyze & Generate Images", use_container_width=True)
    
    if submitted and script and title:
        with st.spinner("Analyzing your story..."):
            # Analyze script
            project_info = api_client.analyze_script(script, title)
            
            if project_info:
                st.session_state.project_info = project_info
                st.session_state.generation_settings = {
                    "project_id": project_info["project_id"],
                    "num_scenes": num_scenes,
                    "media_type": media_type,
                    "ai_model": ai_model,
                    "image_provider": image_provider,
                    "image_model": image_model
                }
                
                # Display analysis results
                st.success("✅ Story analyzed successfully!")
                
                col1, col2, col3 = st.columns(3)
                analysis = project_info["analysis"]
                
                with col1:
                    st.metric("Word Count", analysis["word_count"])
                with col2:
                    st.metric("Recommended Scenes", analysis["recommended_scenes"])
                with col3:
                    st.metric("Complexity", analysis["complexity_score"])
                
                st.info(f"Estimated Duration: {analysis['estimated_duration_minutes']:.1f} minutes")
                
                # Start generation
                if st.button("🎨 Start Image Generation", use_container_width=True):
                    generation_request = st.session_state.generation_settings
                    
                    with st.spinner("Starting image generation..."):
                        result = api_client.generate_previews(generation_request)
                        
                        if result:
                            st.session_state.session_id = result["session_id"]
                            st.session_state.page = "generation"
                            st.rerun()
                        else:
                            st.error("Failed to start generation")
            else:
                st.error("Failed to analyze script")
    elif submitted:
        st.error("Please fill in both title and story content")

def generation_page():
    """Image generation monitoring page"""
    st.title("🎨 Generating Your Images")
    
    if "session_id" not in st.session_state:
        st.error("No active generation session")
        if st.button("← Back to Story Input"):
            st.session_state.page = "script"
            st.rerun()
        return
    
    session_id = st.session_state.session_id
    
    # Auto-refresh every 3 seconds
    placeholder = st.empty()
    progress_placeholder = st.empty()
    
    while True:
        status = api_client.get_generation_status(session_id)
        
        if not status:
            st.error("Failed to get generation status")
            break
        
        with progress_placeholder.container():
            # Progress bar
            progress = status.get("progress_percentage", 0)
            st.progress(progress / 100)
            st.write(f"Progress: {progress:.1f}% ({status['completed_scenes']}/{status['total_scenes']} scenes)")
        
        with placeholder.container():
            # Status display
            status_text = status["status"].title()
            if status["status"] == "generating":
                st.info(f"🔄 {status_text} - Please wait...")
            elif status["status"] == "previewing":
                st.success(f"✅ {status_text} - Images ready!")
            elif status["status"] == "failed":
                st.error(f"❌ {status_text}")
                if status.get("errors"):
                    for error in status["errors"]:
                        st.error(error)
                break
            elif status["status"] == "completed":
                st.success("🎉 Generation completed!")
                break
            
            # Show preview images if available
            if status.get("previews"):
                st.subheader("🖼️ Preview Images")
                
                cols = st.columns(min(3, len(status["previews"])))
                for i, preview in enumerate(status["previews"]):
                    with cols[i % 3]:
                        if preview["preview_url"]:
                            st.image(preview["preview_url"], 
                                   caption=f"Scene {preview['scene_number']}: {preview['scene_title']}")
                            st.write(f"⏱️ Generated in {preview['generation_time']:.1f}s")
                            st.write(f"🤖 {preview['provider_used']} - {preview['model_used']}")
                        else:
                            st.error(f"Scene {preview['scene_number']}: Generation failed")
        
        # Check if generation is complete
        if status["status"] in ["previewing", "completed"]:
            st.session_state.generation_status = status
            st.session_state.page = "preview"
            st.rerun()
        elif status["status"] == "failed":
            break
        
        # Wait before next update
        time.sleep(3)
        st.rerun()

def preview_page():
    """Preview and approval page"""
    st.title("🖼️ Preview & Approve Images")
    
    if "generation_status" not in st.session_state:
        st.error("No preview data available")
        return
    
    status = st.session_state.generation_status
    session_id = st.session_state.session_id
    previews = status.get("previews", [])
    
    if not previews:
        st.warning("No preview images available")
        return
    
    st.write(f"**Project:** {st.session_state.project_info.get('title', 'Untitled')}")
    st.write(f"**Session ID:** {session_id}")
    
    # Approval form
    with st.form("approval_form"):
        st.subheader("Select images to save:")
        
        scene_approvals = {}
        
        # Display previews in a grid
        cols = st.columns(2)
        for i, preview in enumerate(previews):
            with cols[i % 2]:
                if preview["preview_url"]:
                    st.image(preview["preview_url"], 
                           caption=f"Scene {preview['scene_number']}: {preview['scene_title']}")
                    
                    # Approval checkbox
                    approved = st.checkbox(
                        f"✅ Approve Scene {preview['scene_number']}", 
                        value=True,
                        key=f"approve_{preview['scene_number']}"
                    )
                    scene_approvals[preview["scene_number"]] = approved
                    
                    # Scene details
                    with st.expander(f"Scene {preview['scene_number']} Details"):
                        st.write(f"**Title:** {preview['scene_title']}")
                        st.write(f"**Prompt:** {preview['prompt']}")
                        st.write(f"**Generation Time:** {preview['generation_time']:.1f}s")
                        st.write(f"**Provider:** {preview['provider_used']}")
                        st.write(f"**Model:** {preview['model_used']}")
                    
                    # Regeneration option
                    if st.button(f"🔄 Regenerate Scene {preview['scene_number']}", 
                               key=f"regen_{preview['scene_number']}"):
                        with st.spinner(f"Regenerating scene {preview['scene_number']}..."):
                            regen_request = {
                                "session_id": session_id,
                                "scene_number": preview["scene_number"],
                                "image_provider": st.session_state.generation_settings["image_provider"],
                                "image_model": st.session_state.generation_settings["image_model"]
                            }
                            
                            result = api_client.regenerate_scene(regen_request)
                            if result and result["status"] == "success":
                                st.success(f"Scene {preview['scene_number']} regenerated!")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(f"Failed to regenerate scene {preview['scene_number']}")
                else:
                    st.error(f"Scene {preview['scene_number']}: Image generation failed")
                    scene_approvals[preview["scene_number"]] = False
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            save_approved = st.form_submit_button("💾 Save Approved Images", use_container_width=True)
        with col2:
            select_all = st.form_submit_button("✅ Select All", use_container_width=True)
        with col3:
            deselect_all = st.form_submit_button("❌ Deselect All", use_container_width=True)
        
        if select_all:
            for key in st.session_state:
                if key.startswith("approve_"):
                    st.session_state[key] = True
            st.rerun()
        
        if deselect_all:
            for key in st.session_state:
                if key.startswith("approve_"):
                    st.session_state[key] = False
            st.rerun()
        
        if save_approved:
            approved_scenes = [num for num, approved in scene_approvals.items() if approved]
            
            if approved_scenes:
                with st.spinner("Saving approved images..."):
                    approval_request = {
                        "session_id": session_id,
                        "scene_approvals": scene_approvals
                    }
                    
                    result = api_client.approve_previews(approval_request)
                    
                    if result:
                        st.success(f"✅ Successfully saved {result['saved_images']} images!")
                        st.balloons()
                        
                        # Show completion summary
                        st.subheader("🎉 Generation Complete!")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Images Saved", result["saved_images"])
                        with col2:
                            st.metric("Scenes Approved", result["approved_scenes"])
                        
                        # Navigation options
                        if st.button("📁 View All Projects"):
                            st.session_state.page = "projects"
                            st.rerun()
                        
                        if st.button("🆕 Create New Story"):
                            # Clear session state
                            for key in list(st.session_state.keys()):
                                if key not in ["page"]:
                                    del st.session_state[key]
                            st.session_state.page = "script"
                            st.rerun()
                    else:
                        st.error("Failed to save images")
            else:
                st.warning("Please select at least one image to save")

def projects_page():
    """Projects management page"""
    st.title("📁 Your Projects")
    
    projects = api_client.list_projects()
    
    if not projects:
        st.info("No projects found. Create your first story!")
        if st.button("🆕 Create New Story"):
            st.session_state.page = "script"
            st.rerun()
        return
    
    st.write(f"Found {len(projects)} project(s)")
    
    # Project grid
    cols = st.columns(2)
    for i, project in enumerate(projects):
        with cols[i % 2]:
            with st.container():
                st.subheader(f"🎬 {project['title']}")
                st.write(f"**Created:** {project['created_at'][:10]}")
                st.write(f"**Images:** {project['images_count']}")
                
                analysis = project.get('analysis', {})
                if analysis:
                    st.write(f"**Words:** {analysis.get('word_count', 0)}")
                    st.write(f"**Complexity:** {analysis.get('complexity_score', 'Unknown')}")
                
                if st.button(f"👁️ View Project", key=f"view_{project['project_id']}"):
                    st.session_state.selected_project = project['project_id']
                    st.session_state.page = "project_detail"
                    st.rerun()
                
                st.markdown("---")

def project_detail_page():
    """Project detail page"""
    if "selected_project" not in st.session_state:
        st.error("No project selected")
        return
    
    project_id = st.session_state.selected_project
    project_details = api_client.get_project_details(project_id)
    
    if not project_details:
        st.error("Failed to load project details")
        return
    
    st.title(f"📖 {project_details['title']}")
    
    # Project info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Story Content")
        st.text_area("Script", project_details['script'], height=200, disabled=True)
    
    with col2:
        st.subheader("Analysis")
        analysis = project_details['analysis']
        st.metric("Word Count", analysis['word_count'])
        st.metric("Recommended Scenes", analysis['recommended_scenes'])
        st.metric("Complexity", analysis['complexity_score'])
        st.metric("Duration (min)", f"{analysis['estimated_duration_minutes']:.1f}")
    
    # Images gallery
    if project_details['images']:
        st.subheader(f"🖼️ Generated Images ({project_details['total_images']})")
        
        # Display images in a grid
        cols = st.columns(3)
        for i, image_url in enumerate(project_details['images']):
            with cols[i % 3]:
                # Convert relative URL to absolute
                full_url = f"{API_BASE_URL}{image_url}"
                st.image(full_url, caption=f"Scene {i + 1}")
    else:
        st.info("No images generated for this project yet")
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Projects"):
            st.session_state.page = "projects"
            st.rerun()
    with col2:
        if st.button("🆕 Create New Story"):
            for key in list(st.session_state.keys()):
                if key not in ["page"]:
                    del st.session_state[key]
            st.session_state.page = "script"
            st.rerun()

def main():
    """Main application function"""
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "script"
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🎬 Navigation")
        
        pages = {
            "script": "📝 Create Story",
            "projects": "📁 View Projects"
        }
        
        for page_key, page_name in pages.items():
            if st.button(page_name, use_container_width=True):
                st.session_state.page = page_key
                st.rerun()
        
        # Additional info
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Write detailed, descriptive stories
        - Include character descriptions and settings
        - 3-6 scenes work best for most stories
        - Experiment with different visual styles
        """)
        
        st.markdown("---")
        st.markdown("**Version:** 3.0.0")
        st.markdown("**Backend:** FastAPI + AI Services")
    
    # Page routing
    if st.session_state.page == "script":
        script_analysis_page()
    elif st.session_state.page == "generation":
        generation_page()
    elif st.session_state.page == "preview":
        preview_page()
    elif st.session_state.page == "projects":
        projects_page()
    elif st.session_state.page == "project_detail":
        project_detail_page()
    else:
        script_analysis_page()

if __name__ == "__main__":
    main()