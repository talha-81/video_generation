import requests
import uuid
import json
import time
import asyncio
import aiohttp
from typing import Optional, List
from abc import ABC, abstractmethod

from models import ScenePrompt, PreviewImage, MediaType
from config import config, APIConfig
from utils import logger

class AIService(ABC):
    """Base class for AI services"""
    
    def __init__(self, api_config: APIConfig):
        self.config = api_config
    
    async def _make_async_request(self, session: aiohttp.ClientSession, payload: dict, timeout: int = 30) -> Optional[dict]:
        """Make async API request with error handling"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            async with session.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API error: {response.status} - {await response.text()}")
                    return None
                    
        except Exception as e:
            logger.error(f"Async request failed: {e}")
            return None
        
    def _make_request(self, payload: dict, timeout: int = 30) -> Optional[dict]:
        """Make synchronous API request with error handling (fallback)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

class PromptGenerator(AIService):
    """Generates scene prompts using AI"""
    
    def __init__(self):
        super().__init__(config.openrouter)
        self.style_map = {
            MediaType.CINEMATIC: "cinematic movie scene with dramatic lighting and professional composition",
            MediaType.CARTOON: "vibrant cartoon illustration with bold colors and expressive characters",
            MediaType.REALISTIC: "photorealistic scene with natural lighting and detailed textures",
            MediaType.ARTISTIC: "artistic illustration with creative interpretation and painterly style"
        }
    
    def generate_prompts(self, script: str, num_scenes: int, media_type: MediaType, model: str) -> List[ScenePrompt]:
        """Generate scene prompts using OpenRouter"""
        try:
            if not self.config.api_key:
                logger.warning("OpenRouter API key not configured, using fallback")
                return self._generate_fallback_scenes(script, num_scenes, media_type)
            
            style = self.style_map.get(media_type, self.style_map[MediaType.CINEMATIC])
            
            prompt = f"""Create {num_scenes} detailed visual scene descriptions from this story.

Style: {style}

Story: "{script}"

Return ONLY valid JSON in this exact format:
{{
    "scenes": [
        {{
            "scene_number": 1,
            "scene_title": "Brief descriptive title",
            "script_excerpt": "relevant text from story",
            "image_prompt": "detailed visual description for {style}"
        }}
    ]
}}"""

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            response = self._make_request(payload)
            if response:
                content = response["choices"][0]["message"]["content"]
                return self._parse_ai_response(content, num_scenes)
            else:
                return self._generate_fallback_scenes(script, num_scenes, media_type)
                
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}")
            return self._generate_fallback_scenes(script, num_scenes, media_type)
    
    def _parse_ai_response(self, content: str, num_scenes: int) -> List[ScenePrompt]:
        """Parse AI response and extract scenes"""
        try:
            # Clean JSON response
            content = content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            
            # Find JSON start
            if content.find("{") > 0:
                content = content[content.find("{"):]
            
            result = json.loads(content)
            scenes = [ScenePrompt(**scene) for scene in result["scenes"][:num_scenes]]
            logger.info(f"Generated {len(scenes)} AI scene prompts")
            return scenes
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return []
    
    def _generate_fallback_scenes(self, script: str, num_scenes: int, media_type: MediaType) -> List[ScenePrompt]:
        """Generate basic scene prompts without AI"""
        try:
            words = script.strip().split()
            if not words:
                words = ["A", "simple", "scene"]
            
            words_per_scene = max(1, len(words) // num_scenes)
            style = self.style_map.get(media_type, self.style_map[MediaType.CINEMATIC])
            scenes = []
            
            for i in range(num_scenes):
                start = i * words_per_scene
                end = min((i + 1) * words_per_scene, len(words))
                excerpt = " ".join(words[start:end])
                
                if len(excerpt) > 100:
                    excerpt = excerpt[:97] + "..."
                
                prompt = f"{style} showing {excerpt}. High quality, detailed rendering."
                
                scenes.append(ScenePrompt(
                    scene_number=i + 1,
                    scene_title=f"Scene {i + 1}",
                    script_excerpt=excerpt,
                    image_prompt=prompt
                ))
            
            logger.info(f"Generated {len(scenes)} fallback scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return []

class ImageGenerator:
    """Handles image generation from different providers"""
    
    def __init__(self):
        self.providers = {
            "runware": RunwareGenerator(),
            "together": TogetherGenerator()
        }
    
    def generate_image(self, scene: ScenePrompt, provider: str, model: str) -> PreviewImage:
        """Generate image with retry logic"""
        if provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        generator = self.providers[provider]
        return generator.generate_with_retry(scene, model)
    
    async def generate_images_async(self, scenes: List[ScenePrompt], provider: str, model: str) -> List[PreviewImage]:
        """Generate multiple images asynchronously"""
        if provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}")
        
        generator = self.providers[provider]
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(3)
        
        async def generate_with_semaphore(scene):
            async with semaphore:
                return await generator.generate_async(scene, model)
        
        tasks = [generate_with_semaphore(scene) for scene in scenes]
        return await asyncio.gather(*tasks, return_exceptions=True)

class BaseImageGenerator(AIService, ABC):
    """Base class for image generators"""
    
    @abstractmethod
    async def generate_async(self, scene: ScenePrompt, model: str) -> PreviewImage:
        """Generate image asynchronously"""
        pass
    
    def generate_with_retry(self, scene: ScenePrompt, model: str) -> PreviewImage:
        """Generate image with retry logic (synchronous fallback)"""
        start_time = time.time()
        
        for attempt in range(config.max_retries):
            try:
                if not self.config.api_key:
                    logger.error(f"{self.__class__.__name__} API key not configured")
                    break
                
                result = self._generate_image(scene, model)
                if result:
                    return PreviewImage(
                        scene_number=scene.scene_number,
                        scene_title=scene.scene_title,
                        prompt=scene.image_prompt,
                        preview_url=result,
                        generation_time=time.time() - start_time,
                        provider_used=self.__class__.__name__.lower().replace('generator', ''),
                        model_used=model
                    )
                    
            except Exception as e:
                logger.error(f"{self.__class__.__name__} attempt {attempt + 1} failed: {e}")
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_delay)
        
        # Return failed result
        return PreviewImage(
            scene_number=scene.scene_number,
            scene_title=scene.scene_title,
            prompt=scene.image_prompt,
            preview_url="",
            generation_time=time.time() - start_time,
            provider_used=self.__class__.__name__.lower().replace('generator', ''),
            model_used=model
        )
    
    @abstractmethod
    def _generate_image(self, scene: ScenePrompt, model: str) -> Optional[str]:
        """Generate single image (to be implemented by subclasses)"""
        pass

class RunwareGenerator(BaseImageGenerator):
    """Runware AI image generator"""
    
    def __init__(self):
        super().__init__(config.runware)
    
    async def generate_async(self, scene: ScenePrompt, model: str) -> PreviewImage:
        """Generate image asynchronously"""
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(config.max_retries):
                try:
                    if not self.config.api_key:
                        logger.error("Runware API key not configured")
                        break
                    
                    result = await self._generate_image_async(session, scene, model)
                    if result:
                        return PreviewImage(
                            scene_number=scene.scene_number,
                            scene_title=scene.scene_title,
                            prompt=scene.image_prompt,
                            preview_url=result,
                            generation_time=time.time() - start_time,
                            provider_used="runware",
                            model_used=model
                        )
                        
                except Exception as e:
                    logger.error(f"Runware async attempt {attempt + 1} failed: {e}")
                    if attempt < config.max_retries - 1:
                        await asyncio.sleep(config.retry_delay)
            
            # Return failed result
            return PreviewImage(
                scene_number=scene.scene_number,
                scene_title=scene.scene_title,
                prompt=scene.image_prompt,
                preview_url="",
                generation_time=time.time() - start_time,
                provider_used="runware",
                model_used=model
            )
    
    async def _generate_image_async(self, session: aiohttp.ClientSession, scene: ScenePrompt, model: str) -> Optional[str]:
        """Generate single image using Runware (async)"""
        payload = [{
            "taskType": "imageInference",
            "taskUUID": str(uuid.uuid4()),
            "outputType": "URL",
            "outputFormat": "JPG",
            "positivePrompt": scene.image_prompt,
            "height": 1024,
            "width": 1024,
            "model": model,
            "steps": 20,
            "CFGScale": 7.0,
            "numberResults": 1
        }]
        
        response = await self._make_async_request(session, payload, config.timeout)
        if response and "data" in response and response["data"]:
            return response["data"][0].get("imageURL", "")
        return None
    
    def _generate_image(self, scene: ScenePrompt, model: str) -> Optional[str]:
        """Generate single image using Runware (sync fallback)"""
        payload = [{
            "taskType": "imageInference",
            "taskUUID": str(uuid.uuid4()),
            "outputType": "URL",
            "outputFormat": "JPG",
            "positivePrompt": scene.image_prompt,
            "height": 1024,
            "width": 1024,
            "model": model,
            "steps": 20,
            "CFGScale": 7.0,
            "numberResults": 1
        }]
        
        response = self._make_request(payload, config.timeout)
        if response and "data" in response and response["data"]:
            return response["data"][0].get("imageURL", "")
        return None

class TogetherGenerator(BaseImageGenerator):
    """Together AI image generator"""
    
    def __init__(self):
        super().__init__(config.together)
    
    async def generate_async(self, scene: ScenePrompt, model: str) -> PreviewImage:
        """Generate image asynchronously"""
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(config.max_retries):
                try:
                    if not self.config.api_key:
                        logger.error("Together API key not configured")
                        break
                    
                    result = await self._generate_image_async(session, scene, model)
                    if result:
                        return PreviewImage(
                            scene_number=scene.scene_number,
                            scene_title=scene.scene_title,
                            prompt=scene.image_prompt,
                            preview_url=result,
                            generation_time=time.time() - start_time,
                            provider_used="together",
                            model_used=model
                        )
                        
                except Exception as e:
                    logger.error(f"Together async attempt {attempt + 1} failed: {e}")
                    if attempt < config.max_retries - 1:
                        await asyncio.sleep(config.retry_delay)
            
            # Return failed result
            return PreviewImage(
                scene_number=scene.scene_number,
                scene_title=scene.scene_title,
                prompt=scene.image_prompt,
                preview_url="",
                generation_time=time.time() - start_time,
                provider_used="together",
                model_used=model
            )
    
    async def _generate_image_async(self, session: aiohttp.ClientSession, scene: ScenePrompt, model: str) -> Optional[str]:
        """Generate single image using Together AI (async)"""
        payload = {
            "model": model,
            "prompt": scene.image_prompt,
            "width": 1024,
            "height": 1024,
            "steps": 4 if "schnell" in model.lower() else 20,
            "n": 1,
            "response_format": "url"
        }
        
        response = await self._make_async_request(session, payload, config.timeout)
        if response and "data" in response and response["data"]:
            return response["data"][0].get("url", "")
        return None
    
    def _generate_image(self, scene: ScenePrompt, model: str) -> Optional[str]:
        """Generate single image using Together AI (sync fallback)"""
        payload = {
            "model": model,
            "prompt": scene.image_prompt,
            "width": 1024,
            "height": 1024,
            "steps": 4 if "schnell" in model.lower() else 20,
            "n": 1,
            "response_format": "url"
        }
        
        response = self._make_request(payload, config.timeout)
        if response and "data" in response and response["data"]:
            return response["data"][0].get("url", "")
        return None