# Story to Image Generator - Streamlit Frontend

A beautiful, user-friendly web interface for the Story to Image Generator API built with Streamlit.

## 🌟 Features

- **📝 Story Input**: Clean, intuitive story writing interface
- **⚙️ Smart Settings**: Easy model and parameter selection
- **🎨 Real-time Generation**: Live progress monitoring with auto-refresh
- **🖼️ Image Preview**: Beautiful gallery view of generated images
- **✅ Approval System**: Select which images to save
- **🔄 Scene Regeneration**: Regenerate individual scenes with different models
- **📁 Project Management**: Browse and view all your created projects
- **💡 Smart Analysis**: Automatic script analysis with recommendations
- **🏥 Health Monitoring**: Real-time API status display
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

1. **Backend API Running**: Make sure your FastAPI backend is running first:
   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Python Environment**: Python 3.8+ with required packages

### Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Frontend**:
   ```bash
   # Option 1: Direct Streamlit
   streamlit run app.py
   
   # Option 2: Use launcher script
   python run_frontend.py
   ```

3. **Access the Interface**:
   - Open your browser to `http://localhost:8501`
   - The backend should be running on `http://localhost:8000`

## 📖 How to Use

### 1. Create a Story
- Enter a creative title for your story
- Write your story in the text area (be descriptive!)
- Configure generation settings:
  - Number of scenes (2-8)
  - Visual style (cinematic, cartoon, realistic, artistic)
  - AI model for scene generation
  - Image provider and model

### 2. Monitor Generation
- Watch real-time progress with a progress bar
- See images as they're generated
- View detailed generation statistics

### 3. Review & Approve
- Preview all generated images in a gallery
- Select which images to save to your project
- Regenerate individual scenes if needed
- Save approved images with one click

### 4. Manage Projects
- Browse all your created projects
- View project details and generated images
- Access saved images anytime

## ⚙️ Configuration

### Environment Variables

```bash
# API Configuration
API_BASE_URL=http://127.0.0.1:8000  # Backend API URL
API_TIMEOUT=30                       # Request timeout in seconds
```

### UI Customization

Edit `config.py` to customize:
- Maximum number of scenes
- Status messages
- UI colors and themes
- Default settings

## 📁 File Structure

```
frontend/
├── app.py              # Main Streamlit application
├── config.py           # Frontend configuration
├── requirements.txt    # Python dependencies
├── run_frontend.py     # Launcher script
└── README.md          # This file
```

## 🎨 Features in Detail

### Story Analysis
- **Word Count**: Tracks story length
- **Scene Recommendations**: Suggests optimal scene count
- **Complexity Detection**: Analyzes story complexity
- **Duration Estimation**: Estimates reading time

### Image Generation
- **Multiple Providers**: Support for Runware and Together AI
- **Model Selection**: Choose from various AI models
- **Style Options**: Different visual styles available
- **Concurrent Generation**: Parallel image processing

### Preview System
- **Gallery View**: Clean, organized image display
- **Scene Details**: View prompts and generation stats
- **Approval Workflow**: Select images to keep
- **Regeneration**: Retry failed or unsatisfactory scenes

### Project Management
- **Project History**: View all created projects
- **Image Archives**: Access saved images
- **Project Details**: Full project information
- **Easy Navigation**: Smooth page transitions

## 🛠️ Troubleshooting

### Common Issues

**Frontend won't start:**
```bash
# Install Streamlit
pip install streamlit

# Check if port 8501 is available
lsof -i :8501  # On Unix/Mac
netstat -ano | findstr :8501  # On Windows
```

**Can't connect to backend:**
- Verify backend is running on port 8000
- Check `API_BASE_URL` in config
- Ensure no firewall blocking

**Images not loading:**
- Check backend logs for errors
- Verify API keys are configured
- Try regenerating failed scenes

**Slow performance:**
- Close other browser tabs
- Check internet connection
- Monitor backend resource usage

### Debug Mode

Enable debug logging:
```bash
streamlit run app.py --logger.level=debug
```

## 🔧 Advanced Configuration

### Custom Themes

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#ff6b6b"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### Performance Tuning

For better performance:
- Increase `maxUploadSize` in Streamlit config
- Adjust `API_TIMEOUT` based on your models
- Use caching for repeated API calls

## 🤝 Integration

### Custom Models

To add new models, update the backend configuration and the frontend will automatically detect them through the `/models` endpoint.

### API Extensions

The frontend uses these API endpoints:
- `GET /health` - System status
- `GET /models` - Available models
- `POST /analyze-script` - Script analysis
- `POST /generate-previews` - Start generation
- `GET /generation-status/{session_id}` - Monitor progress
- `POST /regenerate-scene` - Regenerate scenes
- `POST /approve-previews` - Save images
- `GET /projects` - List projects
- `GET /projects/{project_id}` - Project details

## 📊 Monitoring

### Health Dashboard

The sidebar shows:
- ✅ API connection status
- 🔧 Service availability (OpenRouter, Runware, Together)
- 📈 Active sessions
- 🏗️ System version

### Generation Statistics

During generation, view:
- Progress percentage
- Completed vs. total scenes
- Generation times per image
- Error tracking

## 🎯 Tips for Best Results

### Writing Stories
- Be descriptive about characters and settings
- Include visual details and emotions
- Aim for 100-500 words for best results
- Use clear scene transitions

### Model Selection
- **Cinematic**: Best for movie-like scenes
- **Cartoon**: Great for animated styles
- **Realistic**: Photographic quality
- **Artistic**: Creative interpretations

### Scene Count
- 2-4 scenes: Short stories, single concepts
- 4-6 scenes: Medium stories, character arcs
- 6-8 scenes: Complex narratives, multiple locations

## 🔄 Updates

### Version 3.0 Features
- ✅ Real-time generation monitoring
- ✅ Async image processing
- ✅ Enhanced error handling
- ✅ Project management system
- ✅ Scene regeneration
- ✅ Mobile-responsive design

## 🐛 Known Issues

- Large images may load slowly on slower connections
- Some browsers may cache old images (refresh to clear)
- Very long stories (>1000 words) may need more scenes

## 📞 Support

If you encounter issues:

1. Check the backend logs first
2. Verify all API keys are configured
3. Ensure stable internet connection
4. Try regenerating failed scenes
5. Restart both backend and frontend if needed

## 🚀 Production Deployment

For production use:

1. **Configure proper URLs**:
   ```bash
   export API_BASE_URL=https://your-api-domain.com
   ```

2. **Use production WSGI server**:
   ```bash
   # For the backend
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app

   # For the frontend
   streamlit run app.py --server.port=80 --server.address=0.0.0.0
   ```

3. **Set up reverse proxy** (nginx recommended)

4. **Configure HTTPS** for secure connections

---

**Built with ❤️ using Streamlit and FastAPI**