# config.py - Configuration Management

## 📖 Overview

Centralized configuration management module that handles environment variables, API configurations, and application settings. This module ensures secure and flexible configuration across different deployment environments.

## 🎯 Purpose

- **Environment Management**: Load and validate environment variables
- **API Configuration**: Manage API keys and service endpoints
- **Path Management**: Configure file and directory paths
- **Settings Validation**: Ensure all required configurations are present

## 🚀 Key Features

### ✅ **Secure Configuration**
- Environment variable based configuration
- No hardcoded secrets in source code
- Automatic validation of required settings
- Development vs production configurations

### ✅ **Type Safety**
- Dataclass-based configuration objects
- Type hints for all configuration values
- Runtime validation of configuration types
- IDE support with autocompletion

### ✅ **Flexible Setup**
- Support for .env files
- Environment-specific overrides
- Default values for optional settings
- Easy configuration for different deployment environments

## 📊 Configuration Structure

### Core Configuration Classes

#### `APIConfig`
```python
@dataclass
class APIConfig:
    """API configuration for external services"""
    api_key: str
    base_url: str
    timeout: int = 30
    retry_attempts: int = 3
    
    def is_configured(self) -> bool:
        """Check if API is properly configured"""
        return bool(self.api_key and self.api_key != "your_key_here")
```

#### `AppConfig`
```python
@dataclass
class AppConfig:
    """Main application configuration"""
    # AI Service APIs
    openrouter: APIConfig
    runware: APIConfig
    
    # Application Settings
    debug_mode: bool
    log_level: str
    max_sessions: int
    session_cleanup_hours: int
    
    # Directory Paths
    image_generation_dir: Path
    static_files_dir: Path
    
    # CORS Settings
    cors_origins: List[str]
```

### Configuration Loading

#### Environment Variable Mapping
```python
# API Keys
OPENROUTER_API_KEY=sk-or-v1-...
RUNWARE_API_KEY=your-runware-key

# Application Settings
DEBUG_MODE=true
LOG_LEVEL=INFO
MAX_SESSIONS=100
SESSION_CLEANUP_HOURS=24

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,https://yourapp.com

# Directory Paths
IMAGE_GENERATION_DIR=./image_generation
STATIC_FILES_DIR=./static
```

#### Automatic Path Creation
```python
def ensure_directories(config: AppConfig) -> None:
    """Ensure all required directories exist"""
    config.image_generation_dir.mkdir(parents=True, exist_ok=True)
    config.static_files_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Ensured directory: {config.image_generation_dir}")
    logger.info(f"Ensured directory: {config.static_files_dir}")
```

## 🔧 Usage Examples

### Basic Configuration Loading
```python
from config import config

# Access API configurations
if config.openrouter.is_configured():
    # Use OpenRouter API
    api_key = config.openrouter.api_key
    timeout = config.openrouter.timeout

# Access application settings
if config.debug_mode:
    logger.setLevel(logging.DEBUG)

# Access directory paths
project_dir = config.image_generation_dir / "project_123"
```

### Configuration Validation
```python
def validate_configuration() -> Dict[str, str]:
    """Validate all configuration settings"""
    status = {}
    
    # Check API configurations
    if config.openrouter.is_configured():
        status["openrouter"] = "configured"
    else:
        status["openrouter"] = "missing_key"
    
    if config.runware.is_configured():
        status["runware"] = "configured"
    else:
        status["runware"] = "missing_key"
    
    # Check directories
    if config.image_generation_dir.exists():
        status["image_dir"] = "exists"
    else:
        status["image_dir"] = "missing"
    
    return status
```

## 🔒 Security Features

### Environment Variable Security
```python
# Secure loading with validation
def load_api_key(env_var: str, service_name: str) -> str:
    """Securely load and validate API key"""
    key = os.getenv(env_var)
    
    if not key:
        raise ConfigurationError(f"Missing required API key: {env_var}")
    
    if key.startswith("your_") or key == "":
        raise ConfigurationError(f"Invalid API key for {service_name}")
    
    logger.info(f"✅ {service_name} API key loaded successfully")
    return key
```

### Configuration Sanitization
```python
def get_safe_config() -> Dict[str, Any]:
    """Get configuration without sensitive data"""
    return {
        "debug_mode": config.debug_mode,
        "log_level": config.log_level,
        "max_sessions": config.max_sessions,
        "apis": {
            "openrouter": "configured" if config.openrouter.is_configured() else "not_configured",
            "runware": "configured" if config.runware.is_configured() else "not_configured"
        },
        "directories": {
            "image_generation": str(config.image_generation_dir),
            "static_files": str(config.static_files_dir)
        }
    }
```

## 🌍 Environment-Specific Configurations

### Development Environment
```python
# .env.development
DEBUG_MODE=true
LOG_LEVEL=DEBUG
CORS_ORIGINS=*
MAX_SESSIONS=10
SESSION_CLEANUP_HOURS=1

# Development API endpoints (if different)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
RUNWARE_BASE_URL=https://api.runware.ai/v1
```

### Production Environment
```python
# .env.production
DEBUG_MODE=false
LOG_LEVEL=INFO
CORS_ORIGINS=https://yourapp.com,https://www.yourapp.com
MAX_SESSIONS=1000
SESSION_CLEANUP_HOURS=24

# Production-specific timeouts
API_TIMEOUT=60
RETRY_ATTEMPTS=5
```

### Staging Environment
```python
# .env.staging
DEBUG_MODE=false
LOG_LEVEL=WARNING
CORS_ORIGINS=https://staging.yourapp.com
MAX_SESSIONS=100
SESSION_CLEANUP_HOURS=12
```

## 📊 Configuration Validation

### Startup Validation
```python
def validate_startup_config() -> None:
    """Validate configuration at application startup"""
    errors = []
    
    # Validate required API keys
    if not config.openrouter.is_configured():
        errors.append("OpenRouter API key not configured")
    
    if not config.runware.is_configured():
        errors.append("Runware API key not configured")
    
    # Validate directories
    try:
        ensure_directories(config)
    except Exception as e:
        errors.append(f"Cannot create directories: {e}")
    
    # Check numeric ranges
    if config.max_sessions < 1:
        errors.append("max_sessions must be >= 1")
    
    if config.session_cleanup_hours < 1:
        errors.append("session_cleanup_hours must be >= 1")
    
    if errors:
        raise ConfigurationError(f"Configuration errors: {'; '.join(errors)}")
    
    logger.info("✅ Configuration validation passed")
```

### Runtime Health Check
```python
def get_config_health() -> Dict[str, str]:
    """Get current configuration health status"""
    health = {}
    
    # API service health
    try:
        # Test API connectivity (optional)
        health["openrouter"] = "available" if test_openrouter_connection() else "unavailable"
        health["runware"] = "available" if test_runware_connection() else "unavailable"
    except Exception:
        health["openrouter"] = "unknown"
        health["runware"] = "unknown"
    
    # Directory health
    health["directories"] = "ok" if all([
        config.image_generation_dir.exists(),
        config.static_files_dir.exists()
    ]) else "missing"
    
    return health
```

## 🔧 Configuration Templates

### .env.example Template
```bash
# Story to Image Generator - Environment Configuration
# Copy this file to .env and fill in your API keys

# OpenRouter API (for AI text generation)
# Get your key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Runware API (for image generation)  
# Get your key from: https://runware.ai/api-keys
RUNWARE_API_KEY=your_runware_api_key_here

# Application Settings
DEBUG_MODE=false
LOG_LEVEL=INFO
MAX_SESSIONS=100
SESSION_CLEANUP_HOURS=24

# CORS Settings (comma-separated)
CORS_ORIGINS=http://localhost:3000

# Directory Configuration
IMAGE_GENERATION_DIR=./image_generation
STATIC_FILES_DIR=./static

# API Timeout Settings
API_TIMEOUT=30
RETRY_ATTEMPTS=3
```

### Docker Environment Template
```bash
# Docker-specific environment variables
# These override .env file settings when using Docker

# API Keys (from Docker secrets or environment)
OPENROUTER_API_KEY_FILE=/run/secrets/openrouter_key
RUNWARE_API_KEY_FILE=/run/secrets/runware_key

# Container-specific paths
IMAGE_GENERATION_DIR=/app/data/images
STATIC_FILES_DIR=/app/data/static

# Container networking
CORS_ORIGINS=https://yourdomain.com
API_TIMEOUT=60
```

## 🧪 Testing Configuration

### Test Configuration
```python
@pytest.fixture
def test_config():
    """Configuration for testing"""
    return AppConfig(
        openrouter=APIConfig(
            api_key="test-key-openrouter",
            base_url="https://api.test.com",
            timeout=10
        ),
        runware=APIConfig(
            api_key="test-key-runware", 
            base_url="https://api.test.com",
            timeout=10
        ),
        debug_mode=True,
        log_level="DEBUG",
        max_sessions=5,
        session_cleanup_hours=1,
        image_generation_dir=Path("/tmp/test_images"),
        static_files_dir=Path("/tmp/test_static"),
        cors_origins=["*"]
    )
```

### Mock Configuration
```python
def mock_configuration():
    """Mock configuration for unit tests"""
    with patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "mock-openrouter-key",
        "RUNWARE_API_KEY": "mock-runware-key",
        "DEBUG_MODE": "true",
        "MAX_SESSIONS": "10"
    }):
        yield reload_config()
```

## 📈 Performance Considerations

### Configuration Caching
- Configuration loaded once at startup
- No file I/O during request processing
- Immutable configuration objects
- Fast attribute access with dataclasses

### Memory Usage
- Minimal memory footprint
- No configuration file watching in production
- Efficient string storage
- Path objects cached

## 🚀 Deployment Best Practices

### Container Deployment
```dockerfile
# Use build args for non-sensitive config
ARG DEBUG_MODE=false
ARG LOG_LEVEL=INFO

# Use secrets for sensitive data
RUN --mount=type=secret,id=openrouter_key \
    --mount=type=secret,id=runware_key \
    cat /run/secrets/openrouter_key > /app/.openrouter_key && \
    cat /run/secrets/runware_key > /app/.runware_key
```

### Kubernetes Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  DEBUG_MODE: "false"
  LOG_LEVEL: "INFO"
  MAX_SESSIONS: "1000"
  CORS_ORIGINS: "https://yourapp.com"
---
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
type: Opaque
data:
  openrouter-key: <base64-encoded-key>
  runware-key: <base64-encoded-key>
```

---

**config.py provides secure, flexible, and type-safe configuration management for all deployment environments! 🔧**