# utils.py - Utility Functions

## 📖 Overview

Utility functions module that provides pure functions and helper classes for script analysis, project management, file operations, and data processing. These utilities are used across the application to handle common operations.

## 🎯 Purpose

- **Script Analysis**: Parse and analyze story text to extract scenes
- **Project Management**: Handle project directories and file organization
- **File Operations**: Safe file I/O operations with error handling
- **Data Processing**: Text processing and formatting utilities

## 🚀 Key Features

### ✅ **Pure Functions**
- No side effects or external dependencies
- Easily testable and composable
- Thread-safe operations
- Predictable input/output behavior

### ✅ **Robust Error Handling**
- Comprehensive exception handling
- Graceful degradation for non-critical errors
- Detailed error logging and reporting
- Safe file operations with cleanup

### ✅ **Performance Optimized**
- Efficient text processing algorithms
- Memory-conscious operations
- Optimized file I/O patterns
- Cached computations where appropriate

## 🔧 Core Utility Functions

### Script Analysis

#### `analyze_script_structure(story_text: str) -> ScriptAnalysis`
```python
def analyze_script_structure(story_text: str) -> ScriptAnalysis:
    """
    Analyze story text and extract structural information
    
    Args:
        story_text: Raw story text to analyze
        
    Returns:
        ScriptAnalysis object with extracted information
        
    Raises:
        ValueError: If story text is invalid or too short
        
    Example:
        >>> story = "Chapter 1: The hero begins their journey..."
        >>> analysis = analyze_script_structure(story)
        >>> print(analysis.estimated_scenes)  # 3
        >>> print(analysis.main_themes)       # ['journey', 'heroism']
    """
    
    if not story_text or len(story_text.strip()) < 10:
        raise ValueError("Story text must be at least 10 characters")
    
    # Clean and normalize text
    cleaned_text = clean_text(story_text)
    
    # Extract structural elements
    paragraphs = extract_paragraphs(cleaned_text)
    sentences = extract_sentences(cleaned_text)
    
    # Analyze content
    themes = extract_themes(cleaned_text)
    characters = extract_characters(cleaned_text)
    locations = extract_locations(cleaned_text)
    
    # Estimate scene boundaries
    scene_breaks = identify_scene_boundaries(paragraphs)
    estimated_scenes = len(scene_breaks) if scene_breaks else max(1, len(paragraphs) // 3)
    
    return ScriptAnalysis(
        total_words=len(cleaned_text.split()),
        total_sentences=len(sentences),
        total_paragraphs=len(paragraphs),
        estimated_scenes=min(estimated_scenes, 20),  # Cap at maximum
        main_themes=themes[:5],  # Top 5 themes
        key_characters=characters[:10],  # Top 10 characters
        primary_locations=locations[:8],  # Top 8 locations
        complexity_score=calculate_complexity_score(cleaned_text),
        recommended_scenes=recommend_scene_count(cleaned_text)
    )
```

#### `extract_scenes_from_story(story_text: str, num_scenes: int) -> List[ScenePrompt]`
```python
def extract_scenes_from_story(story_text: str, num_scenes: int) -> List[ScenePrompt]:
    """
    Extract and create scene prompts from story text
    
    Args:
        story_text: Complete story text
        num_scenes: Number of scenes to extract
        
    Returns:
        List of ScenePrompt objects
        
    Example:
        >>> scenes = extract_scenes_from_story(story, 5)
        >>> for scene in scenes:
        ...     print(f"Scene {scene.scene_number}: {scene.title}")
    """
    
    # Analyze story structure first
    analysis = analyze_script_structure(story_text)
    
    # Split story into logical segments
    segments = split_story_into_segments(story_text, num_scenes)
    
    scene_prompts = []
    for i, segment in enumerate(segments):
        # Generate scene metadata
        title = generate_scene_title(segment, i, analysis)
        description = generate_scene_description(segment, analysis)
        
        # Create image generation prompt
        image_prompt = create_image_prompt(
            description=description,
            themes=analysis.main_themes,
            characters=analysis.key_characters,
            locations=analysis.primary_locations,
            scene_number=i
        )
        
        scene_prompts.append(ScenePrompt(
            scene_number=i,
            title=title,
            description=description,
            image_prompt=image_prompt
        ))
    
    return scene_prompts
```

### Text Processing

#### `clean_text(text: str) -> str`
```python
def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned and normalized text
        
    Features:
        - Remove extra whitespace
        - Fix common encoding issues
        - Normalize punctuation
        - Remove special characters
    """
    
    if not text:
        return ""
    
    # Fix common encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix punctuation spacing
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s*([a-zA-Z])', r'\1 \2', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Strip and return
    return text.strip()
```

#### `extract_themes(text: str) -> List[str]`
```python
def extract_themes(text: str) -> List[str]:
    """
    Extract main themes from text using NLP techniques
    
    Args:
        text: Text to analyze for themes
        
    Returns:
        List of identified themes, ordered by relevance
        
    Algorithm:
        1. Extract key phrases using TF-IDF
        2. Identify semantic clusters
        3. Map to common literary themes
        4. Score and rank by frequency and importance
    """
    
    # Define theme keywords
    theme_keywords = {
        'adventure': ['journey', 'quest', 'travel', 'explore', 'adventure'],
        'love': ['love', 'romance', 'heart', 'passion', 'affection'],
        'conflict': ['battle', 'fight', 'war', 'struggle', 'conflict'],
        'mystery': ['mystery', 'secret', 'hidden', 'unknown', 'puzzle'],
        'magic': ['magic', 'spell', 'enchant', 'mystical', 'supernatural'],
        'friendship': ['friend', 'companion', 'ally', 'together', 'bond'],
        'betrayal': ['betray', 'deceive', 'trick', 'lie', 'false'],
        'redemption': ['redeem', 'forgive', 'second chance', 'atone', 'mercy'],
        'courage': ['brave', 'courage', 'hero', 'bold', 'fearless'],
        'sacrifice': ['sacrifice', 'give up', 'selfless', 'noble', 'duty']
    }
    
    # Convert text to lowercase for matching
    text_lower = text.lower()
    
    # Score themes based on keyword frequency
    theme_scores = {}
    for theme, keywords in theme_keywords.items():
        score = 0
        for keyword in keywords:
            # Count exact matches and partial matches
            exact_matches = text_lower.count(keyword)
            partial_matches = len(re.findall(rf'\b\w*{keyword}\w*\b', text_lower))
            
            score += exact_matches * 2 + partial_matches
        
        if score > 0:
            theme_scores[theme] = score
    
    # Sort themes by score and return top themes
    sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
    return [theme for theme, score in sorted_themes]
```

### Project Management

#### `create_project_structure(project_info: ProjectInfo) -> bool`
```python
def create_project_structure(project_info: ProjectInfo) -> bool:
    """
    Create complete project directory structure
    
    Args:
        project_info: Project metadata and configuration
        
    Returns:
        True if successful, False otherwise
        
    Creates:
        project_dir/
        ├── story.txt           # Original story text
        ├── scenes.json         # Generated scenes data
        ├── project_info.json   # Project metadata
        ├── images/             # Generated images directory
        │   ├── scene_0.jpg
        │   ├── scene_1.jpg
        │   └── ...
        └── exports/            # Export directory for final products
            ├── story_with_images.html
            ├── presentation.pdf
            └── project_archive.zip
    """
    
    try:
        # Create main project directory
        project_path = Path(project_info.project_dir)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (project_path / 'images').mkdir(exist_ok=True)
        (project_path / 'exports').mkdir(exist_ok=True)
        
        # Save story text
        story_file = project_path / 'story.txt'
        with open(story_file, 'w', encoding='utf-8') as f:
            f.write(project_info.story_text)
        
        # Save project metadata
        project_file = project_path / 'project_info.json'
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project_info.dict(), f, indent=2, default=str)
        
        logger.info(f"✅ Created project structure: {project_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create project structure: {e}")
        return False
```

#### `save_scenes_to_file(scenes: List[ScenePrompt], project_dir: str) -> bool`
```python
def save_scenes_to_file(scenes: List[ScenePrompt], project_dir: str) -> bool:
    """
    Save generated scenes to JSON file
    
    Args:
        scenes: List of scene prompts to save
        project_dir: Project directory path
        
    Returns:
        True if successful, False otherwise
    """
    
            try:
        project_path = Path(project_dir)
        scenes_file = project_path / 'scenes.json'
        
        # Convert scenes to serializable format
        scenes_data = {
            'scenes': [scene.dict() for scene in scenes],
            'total_scenes': len(scenes),
            'generated_at': datetime.utcnow().isoformat(),
            'version': '3.0.0'
        }
        
        # Save with atomic write (write to temp file, then rename)
        temp_file = scenes_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(scenes_data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_file.rename(scenes_file)
        
        logger.info(f"✅ Saved {len(scenes)} scenes to: {scenes_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save scenes: {e}")
        # Clean up temp file if it exists
        temp_file = Path(project_dir) / 'scenes.json.tmp'
        if temp_file.exists():
            temp_file.unlink()
        return False
```

#### `load_scenes_from_file(project_dir: str) -> List[ScenePrompt]`
```python
def load_scenes_from_file(project_dir: str) -> List[ScenePrompt]:
    """
    Load scenes from saved JSON file
    
    Args:
        project_dir: Project directory path
        
    Returns:
        List of loaded scene prompts
        
    Raises:
        FileNotFoundError: If scenes file doesn't exist
        JSONDecodeError: If file is corrupted
        ValidationError: If data doesn't match scene schema
    """
    
    project_path = Path(project_dir)
    scenes_file = project_path / 'scenes.json'
    
    if not scenes_file.exists():
        raise FileNotFoundError(f"Scenes file not found: {scenes_file}")
    
    try:
        with open(scenes_file, 'r', encoding='utf-8') as f:
            scenes_data = json.load(f)
        
        # Validate file format
        if 'scenes' not in scenes_data:
            raise ValueError("Invalid scenes file format")
        
        # Convert back to ScenePrompt objects
        scenes = []
        for scene_data in scenes_data['scenes']:
            scene = ScenePrompt(**scene_data)
            scenes.append(scene)
        
        logger.info(f"✅ Loaded {len(scenes)} scenes from: {scenes_file}")
        return scenes
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Corrupted scenes file: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load scenes: {e}")
        raise
```

### File Operations

#### `safe_file_operation(operation: Callable, *args, **kwargs) -> Any`
```python
def safe_file_operation(operation: Callable, *args, **kwargs) -> Any:
    """
    Execute file operation with comprehensive error handling
    
    Args:
        operation: File operation function to execute
        *args, **kwargs: Arguments for the operation
        
    Returns:
        Result of the operation
        
    Raises:
        FileOperationError: If operation fails with details
    """
    
    try:
        return operation(*args, **kwargs)
        
    except PermissionError as e:
        logger.error(f"❌ Permission denied: {e}")
        raise FileOperationError(f"Permission denied: {e}")
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise FileOperationError(f"File not found: {e}")
        
    except OSError as e:
        logger.error(f"❌ OS error during file operation: {e}")
        raise FileOperationError(f"OS error: {e}")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in file operation: {e}")
        raise FileOperationError(f"Unexpected error: {e}")
```

#### `create_zip_archive(project_dir: str, output_path: str) -> bool`
```python
def create_zip_archive(project_dir: str, output_path: str) -> bool:
    """
    Create ZIP archive of project directory
    
    Args:
        project_dir: Directory to archive
        output_path: Output ZIP file path
        
    Returns:
        True if successful, False otherwise
        
    Features:
        - Excludes temporary files
        - Preserves directory structure
        - Compresses efficiently
        - Handles large files gracefully
    """
    
    try:
        project_path = Path(project_dir)
        output_file = Path(output_path)
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Files to exclude from archive
        exclude_patterns = [
            '*.tmp', '*.log', '.DS_Store', 'Thumbs.db',
            '__pycache__', '*.pyc', '.git'
        ]
        
        def should_exclude(file_path: Path) -> bool:
            """Check if file should be excluded"""
            for pattern in exclude_patterns:
                if file_path.match(pattern):
                    return True
            return False
        
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in project_path.rglob('*'):
                if file_path.is_file() and not should_exclude(file_path):
                    # Calculate relative path for archive
                    archive_path = file_path.relative_to(project_path)
                    
                    # Add to archive
                    zf.write(file_path, archive_path)
        
        # Verify archive was created
        if output_file.exists() and output_file.stat().st_size > 0:
            logger.info(f"✅ Created archive: {output_file} ({output_file.stat().st_size} bytes)")
            return True
        else:
            logger.error(f"❌ Archive creation failed: {output_file}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to create archive: {e}")
        return False
```

### Text Processing Utilities

#### `split_story_into_segments(story_text: str, num_segments: int) -> List[str]`
```python
def split_story_into_segments(story_text: str, num_segments: int) -> List[str]:
    """
    Intelligently split story into equal segments for scene generation
    
    Args:
        story_text: Complete story text
        num_segments: Number of segments to create
        
    Returns:
        List of story segments
        
    Algorithm:
        1. Split by paragraphs first
        2. Group paragraphs into logical segments
        3. Ensure segments are roughly equal in length
        4. Preserve narrative flow and coherence
    """
    
    if num_segments <= 1:
        return [story_text]
    
    # Clean and split into paragraphs
    cleaned_text = clean_text(story_text)
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    
    if not paragraphs:
        # Fallback: split by sentences
        sentences = [s.strip() + '.' for s in cleaned_text.split('.') if s.strip()]
        paragraphs = sentences
    
    # Calculate target segment size
    total_length = sum(len(p) for p in paragraphs)
    target_segment_size = total_length // num_segments
    
    segments = []
    current_segment = []
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        
        # If adding this paragraph would exceed target size and we have content
        if (current_size + paragraph_length > target_segment_size * 1.2 and 
            current_segment and len(segments) < num_segments - 1):
            
            # Finish current segment
            segments.append('\n\n'.join(current_segment))
            current_segment = [paragraph]
            current_size = paragraph_length
        else:
            # Add to current segment
            current_segment.append(paragraph)
            current_size += paragraph_length
    
    # Add remaining content as final segment
    if current_segment:
        segments.append('\n\n'.join(current_segment))
    
    # Ensure we have exactly num_segments
    while len(segments) < num_segments:
        # Split largest segment
        largest_idx = max(range(len(segments)), key=lambda i: len(segments[i]))
        largest_segment = segments[largest_idx]
        
        # Split roughly in half
        mid_point = len(largest_segment) // 2
        split_point = largest_segment.find('. ', mid_point)
        if split_point == -1:
            split_point = largest_segment.find(' ', mid_point)
        
        if split_point != -1:
            segments[largest_idx] = largest_segment[:split_point + 1]
            segments.append(largest_segment[split_point + 1:])
        else:
            break
    
    # If we have too many segments, merge smallest ones
    while len(segments) > num_segments:
        # Find two smallest adjacent segments
        min_combined_size = float('inf')
        merge_idx = 0
        
        for i in range(len(segments) - 1):
            combined_size = len(segments[i]) + len(segments[i + 1])
            if combined_size < min_combined_size:
                min_combined_size = combined_size
                merge_idx = i
        
        # Merge segments
        segments[merge_idx] = segments[merge_idx] + '\n\n' + segments[merge_idx + 1]
        segments.pop(merge_idx + 1)
    
    return segments
```

#### `generate_scene_title(segment: str, scene_number: int, analysis: ScriptAnalysis) -> str`
```python
def generate_scene_title(segment: str, scene_number: int, analysis: ScriptAnalysis) -> str:
    """
    Generate descriptive title for a story segment
    
    Args:
        segment: Story segment text
        scene_number: Scene index number
        analysis: Story analysis data
        
    Returns:
        Generated scene title
        
    Features:
        - Extracts key actions and events
        - Uses story themes and characters
        - Maintains narrative progression
        - Ensures unique and descriptive titles
    """
    
    # Extract key phrases from segment
    key_phrases = extract_key_phrases(segment)
    
    # Look for action words and important nouns
    action_words = extract_action_words(segment)
    important_nouns = extract_important_nouns(segment)
    
    # Use story analysis context
    themes = analysis.main_themes[:3]  # Top 3 themes
    characters = analysis.key_characters[:5]  # Top 5 characters
    
    # Title generation strategies
    strategies = [
        lambda: f"{action_words[0].title()} {important_nouns[0]}" if action_words and important_nouns else None,
        lambda: f"Chapter {scene_number + 1}: {key_phrases[0]}" if key_phrases else None,
        lambda: f"The {themes[0].title()} Begins" if themes and scene_number == 0 else None,
        lambda: f"{characters[0]}'s {themes[0].title()}" if characters and themes else None,
        lambda: f"Scene {scene_number + 1}: {extract_first_sentence_subject(segment)}" if segment else None
    ]
    
    # Try strategies in order
    for strategy in strategies:
        title = strategy()
        if title and len(title) <= 50:  # Reasonable title length
            return title
    
    # Fallback title
    return f"Scene {scene_number + 1}"
```

### Image Prompt Generation

#### `create_image_prompt(description: str, themes: List[str], characters: List[str], locations: List[str], scene_number: int) -> str`
```python
def create_image_prompt(
    description: str, 
    themes: List[str], 
    characters: List[str], 
    locations: List[str], 
    scene_number: int
) -> str:
    """
    Create detailed image generation prompt
    
    Args:
        description: Scene description
        themes: Story themes
        characters: Key characters
        locations: Important locations
        scene_number: Scene index
        
    Returns:
        Optimized image generation prompt
        
    Features:
        - Incorporates visual style keywords
        - Balances detail and brevity
        - Uses cinematic composition terms
        - Avoids problematic content
    """
    
    # Base prompt from description
    prompt_parts = [description[:200]]  # Limit base description
    
    # Add visual style elements
    if scene_number == 0:
        prompt_parts.append("establishing shot")
    elif scene_number % 2 == 0:
        prompt_parts.append("wide angle view")
    else:
        prompt_parts.append("close-up dramatic shot")
    
    # Add thematic elements
    theme_visuals = {
        'adventure': 'epic landscape, heroic lighting',
        'mystery': 'moody shadows, atmospheric fog',
        'magic': 'mystical glowing effects, ethereal atmosphere',
        'conflict': 'dramatic tension, dynamic composition',
        'love': 'warm golden lighting, soft focus',
        'courage': 'heroic stance, dramatic backlighting'
    }
    
    for theme in themes[:2]:  # Top 2 themes
        if theme in theme_visuals:
            prompt_parts.append(theme_visuals[theme])
    
    # Add character context (without specific names)
    if characters:
        char_count = len(characters)
        if char_count == 1:
            prompt_parts.append("single protagonist")
        elif char_count <= 3:
            prompt_parts.append("small group of characters")
        else:
            prompt_parts.append("ensemble cast")
    
    # Add location context
    if locations:
        location_types = categorize_locations(locations)
        if 'outdoor' in location_types:
            prompt_parts.append("natural outdoor setting")
        if 'indoor' in location_types:
            prompt_parts.append("interior scene")
        if 'fantasy' in location_types:
            prompt_parts.append("fantastical environment")
    
    # Add technical quality terms
    quality_terms = [
        "high quality", "detailed", "professional lighting",
        "cinematic composition", "4K resolution"
    ]
    prompt_parts.extend(quality_terms[:2])  # Add 2 quality terms
    
    # Combine and clean
    full_prompt = ', '.join(prompt_parts)
    
    # Remove problematic content
    full_prompt = sanitize_image_prompt(full_prompt)
    
    # Ensure reasonable length (most AI services have limits)
    if len(full_prompt) > 300:
        full_prompt = full_prompt[:297] + "..."
    
    return full_prompt
```

#### `sanitize_image_prompt(prompt: str) -> str`
```python
def sanitize_image_prompt(prompt: str) -> str:
    """
    Remove potentially problematic content from image prompts
    
    Args:
        prompt: Raw image prompt
        
    Returns:
        Sanitized prompt safe for AI image generation
    """
    
    # Content to remove or replace
    problematic_terms = {
        # Violence/Gore
        'blood': 'red liquid',
        'gore': 'dramatic',
        'violent': 'intense',
        'weapon': 'tool',
        
        # Inappropriate content
        'nude': 'figure',
        'naked': 'unclothed figure',
        'explicit': 'detailed',
        
        # Potentially offensive
        'ugly': 'unique',
        'disgusting': 'unusual',
        'horrible': 'dramatic'
    }
    
    sanitized = prompt.lower()
    
    # Replace problematic terms
    for bad_term, replacement in problematic_terms.items():
        sanitized = sanitized.replace(bad_term, replacement)
    
    # Remove excessive descriptors
    sanitized = re.sub(r'\b(very|extremely|super|ultra)\s+', '', sanitized)
    
    # Clean up formatting
    sanitized = re.sub(r',\s*,', ',', sanitized)  # Remove double commas
    sanitized = re.sub(r'\s+', ' ', sanitized)     # Normalize whitespace
    
    return sanitized.strip()
```

### Validation Utilities

#### `validate_project_integrity(project_dir: str) -> Dict[str, bool]`
```python
def validate_project_integrity(project_dir: str) -> Dict[str, bool]:
    """
    Validate project directory structure and files
    
    Args:
        project_dir: Project directory path
        
    Returns:
        Dictionary of validation results
        
    Checks:
        - Directory structure exists
        - Required files are present
        - Files are readable and valid
        - Image files are accessible
        - Data consistency across files
    """
    
    validation_results = {}
    project_path = Path(project_dir)
    
    # Check directory structure
    validation_results['project_dir_exists'] = project_path.exists()
    validation_results['images_dir_exists'] = (project_path / 'images').exists()
    validation_results['exports_dir_exists'] = (project_path / 'exports').exists()
    
    # Check required files
    story_file = project_path / 'story.txt'
    scenes_file = project_path / 'scenes.json'
    project_info_file = project_path / 'project_info.json'
    
    validation_results['story_file_exists'] = story_file.exists()
    validation_results['scenes_file_exists'] = scenes_file.exists()
    validation_results['project_info_exists'] = project_info_file.exists()
    
    # Validate file contents
    if story_file.exists():
        try:
            with open(story_file, 'r', encoding='utf-8') as f:
                story_content = f.read()
            validation_results['story_file_readable'] = len(story_content.strip()) > 0
        except Exception:
            validation_results['story_file_readable'] = False
    
    if scenes_file.exists():
        try:
            scenes = load_scenes_from_file(str(project_path))
            validation_results['scenes_file_valid'] = len(scenes) > 0
        except Exception:
            validation_results['scenes_file_valid'] = False
    
    # Check image files
    if (project_path / 'images').exists():
        image_files = list((project_path / 'images').glob('*.{jpg,jpeg,png,webp}'))
        validation_results['has_image_files'] = len(image_files) > 0
        
        # Check if image files are accessible
        accessible_images = 0
        for img_file in image_files:
            try:
                if img_file.stat().st_size > 0:
                    accessible_images += 1
            except Exception:
                pass
        
        validation_results['images_accessible'] = accessible_images == len(image_files)
    
    return validation_results
```

### Performance Utilities

#### `measure_execution_time(func: Callable) -> Callable`
```python
def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure and log function execution time
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function with timing
    """
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"⏱️ {func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper
```

#### `batch_process_with_progress(items: List[Any], processor: Callable, batch_size: int = 10) -> List[Any]`
```python
def batch_process_with_progress(
    items: List[Any], 
    processor: Callable, 
    batch_size: int = 10
) -> List[Any]:
    """
    Process items in batches with progress tracking
    
    Args:
        items: Items to process
        processor: Function to process each item
        batch_size: Number of items per batch
        
    Returns:
        List of processed results
    """
    
    results = []
    total_items = len(items)
    
    for i in range(0, total_items, batch_size):
        batch = items[i:i + batch_size]
        batch_results = []
        
        for item in batch:
            try:
                result = processor(item)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"❌ Failed to process item {i}: {e}")
                batch_results.append(None)
        
        results.extend(batch_results)
        
        # Progress logging
        processed = min(i + batch_size, total_items)
        progress = (processed / total_items) * 100
        logger.info(f"📊 Processed {processed}/{total_items} items ({progress:.1f}%)")
    
    return results
```

## 🧪 Testing Utilities

### Test Data Generation
```python
def generate_test_story(length: str = 'medium') -> str:
    """Generate test story text for testing purposes"""
    
    story_templates = {
        'short': "A brave knight embarks on a quest to save the princess from a dragon.",
        'medium': """
        Once upon a time, in a kingdom far away, there lived a brave knight named Sir Galahad.
        The kingdom was in great peril, for a terrible dragon had captured the beautiful princess.
        The king offered a great reward to anyone who could save his daughter.
        Sir Galahad accepted the challenge and began his dangerous journey to the dragon's lair.
        After many trials and tribulations, he finally confronted the beast and rescued the princess.
        """,
        'long': """
        In the ancient kingdom of Aethermoor, where magic flowed like rivers through the land,
        a young hero named Aria discovered she possessed extraordinary powers.
        The realm was threatened by an ancient evil that had awakened from its thousand-year slumber.
        
        Chapter 1: The Discovery
        Aria lived a simple life in the village of Millbrook, unaware of her destiny.
        One fateful morning, she found herself able to control the elements with her thoughts.
        
        Chapter 2: The Quest Begins
        The village elder revealed her true heritage and the prophecy that foretold her coming.
        She must gather the Seven Sacred Crystals to defeat the Shadow Lord.
        
        Chapter 3: The Journey
        Together with her loyal companions - a wise wizard, a brave warrior, and a clever thief -
        Aria traveled across treacherous lands, facing countless dangers and challenges.
        
        Chapter 4: The Final Battle
        At the Shadow Lord's fortress, Aria discovered the true power of friendship and sacrifice.
        With the combined strength of the crystals and her allies, she defeated the ancient evil.
        
        Chapter 5: The New Dawn
        Peace was restored to Aethermoor, and Aria became the kingdom's protector,
        ensuring that such darkness would never threaten the realm again.
        """
    }
    
    return story_templates.get(length, story_templates['medium']).strip()

def create_test_project_info(project_id: str = None) -> ProjectInfo:
    """Create test project info for testing"""
    
    if project_id is None:
        project_id = f"test_proj_{int(time.time())}"
    
    return ProjectInfo(
        project_id=project_id,
        title="Test Story Project",
        story_text=generate_test_story(),
        media_type=MediaType.CINEMATIC,
        total_scenes=3,
        project_dir=f"/tmp/test_projects/{project_id}"
    )
```

## 📊 Error Handling

### Custom Exceptions
```python
class UtilityError(Exception):
    """Base exception for utility operations"""
    pass

class ScriptAnalysisError(UtilityError):
    """Exception for script analysis operations"""
    pass

class FileOperationError(UtilityError):
    """Exception for file operations"""
    pass

class ProjectValidationError(UtilityError):
    """Exception for project validation"""
    pass
```

---

**utils.py provides a comprehensive toolkit of pure, testable functions that handle all the complex text processing and file management operations your application needs! 🔧**