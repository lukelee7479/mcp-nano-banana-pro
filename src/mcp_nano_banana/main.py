import asyncio
import logging
import os
import base64
import uuid
import json
import httpx
from typing import Literal

from io import BytesIO
from urllib.parse import urlparse
from typing import Dict, Any, Optional

from PIL import Image
from mcp.server.fastmcp import FastMCP
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

image_tasks = {}
edit_image_tasks = {}

AspectRatioType = Literal["1:1", "3:4", "4:3", "9:16", "16:9"]
ThinkingLevelType = Literal["MINIMAL", "HIGH"]
ResolutionType = Literal["0.5K", "1K", "2K", "4K"]

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_THINKING_LEVEL: ThinkingLevelType = "MINIMAL"
DEFAULT_ENABLE_GROUNDING = False
DEFAULT_RESOLUTION: ResolutionType = "1K"
DEFAULT_ASPECT_RATIO: AspectRatioType = "16:9"

GENAI_CLIENT = None
ENV_VARS = None
HTTPX_CLIENT = None

def get_httpx_client():
    global HTTPX_CLIENT
    if HTTPX_CLIENT is None:
        HTTPX_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    return HTTPX_CLIENT

def get_env_vars() -> Dict[str, str]:
    global ENV_VARS
    if ENV_VARS is None:
        ENV_VARS = validate_environment_variables()
    return ENV_VARS

def get_genai_client():
    global GENAI_CLIENT
    if GENAI_CLIENT is None:
        env_vars = get_env_vars()
        GENAI_CLIENT = genai.Client(api_key=env_vars["GEMINI_API_KEY"])
    return GENAI_CLIENT




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Error Handling Classes ---
class ImageGenerationError(Exception):
    """Custom exception for image generation errors"""
    pass

class ImageUploadError(Exception):
    """Custom exception for image upload errors"""
    pass

class ValidationError(Exception):
    """Custom exception for input validation errors"""
    pass

class APIError(Exception):
    """Custom exception for API-related errors"""
    pass

# --- Utility Functions ---
def validate_prompt(prompt: str) -> None:
    """Validate image generation prompt"""
    if not prompt or not isinstance(prompt, str):
        raise ValidationError("Prompt must be a non-empty string")
    
    if len(prompt.strip()) == 0:
        raise ValidationError("Prompt cannot be empty or only whitespace")
    
    #if len(prompt) > 1000:
    #    raise ValidationError("Prompt is too long (maximum 1000 characters)")
    
    # Check for potentially problematic content
    if any(char in prompt for char in ['<', '>', '&', '"', "'"]):
        logger.warning("Prompt contains potentially problematic characters")

def validate_image_url(url: str) -> None:
    """Validate image URL"""
    if not url or not isinstance(url, str):
        raise ValidationError("Image URL must be a non-empty string")
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValidationError("Invalid URL format")
        
        if parsed.scheme not in ['http', 'https']:
            raise ValidationError("URL must use HTTP or HTTPS protocol")
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {str(e)}")

def validate_environment_variables() -> Dict[str, str]:
    """Validate required environment variables"""
    errors = []
    env_vars = {}
    
    # Check GEMINI_API_KEY
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        errors.append("GEMINI_API_KEY environment variable not set")
    elif not gemini_key.strip():
        errors.append("GEMINI_API_KEY environment variable is empty")
    else:
        env_vars['GEMINI_API_KEY'] = gemini_key
    
    # Check IMGBB_API_KEY
    imgbb_key = os.getenv("IMGBB_API_KEY")
    if not imgbb_key:
        errors.append("IMGBB_API_KEY environment variable not set")
    elif not imgbb_key.strip():
        errors.append("IMGBB_API_KEY environment variable is empty")
    else:
        env_vars['IMGBB_API_KEY'] = imgbb_key
    
    if errors:
        raise ValidationError(f"Environment validation failed: {'; '.join(errors)}")
    
    return env_vars

def create_error_response(error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> str:
    """Create a standardized error response"""
    error_response = {
        "error": True,
        "error_type": error_type,
        "message": message,
        "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None
    }
    
    if details:
        error_response["details"] = details
    
    return json.dumps(error_response)

def create_success_response(data: Any) -> str:
    """Create a standardized success response"""
    success_response = {
        "success": True,
        "data": data,
        "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None
    }
    return json.dumps(success_response)

# --- MCP Server Setup ---
# Create a FastMCP server instance
mcp = FastMCP(
    name="image_generator_mcp_server",
)
logger.info(f"MCP server '{mcp.name}' created.")


# --- Tool Definition ---
@mcp.tool(
    name="generate_image",
    description=(
        "Call this tool IMMEDIATELY when the user requests to create, draw, or generate a new image. "
        "Input: 'prompt' "
        "Output: Returns the generated image URL. "
        "CONSTRAINT: You MUST call this tool EXACTLY ONCE per user request. Do NOT call in parallel."
    ),
)
async def generate_image(
    prompt: str,
    thinking_level: ThinkingLevelType = DEFAULT_THINKING_LEVEL,
    enable_grounding: bool = DEFAULT_ENABLE_GROUNDING,
    resolution: ResolutionType = DEFAULT_RESOLUTION,
    aspect_ratio: AspectRatioType = DEFAULT_ASPECT_RATIO
) -> str:
    """
    Generates an image from a text prompt and returns the url of the image.
    """
    cache_key = prompt.strip().lower()

    # 1. same job running
    if cache_key in image_tasks:
        logger.info(f"Duplicate request detected. Waiting for the existing task for prompt: {prompt}")
        try:
            # 먼저 시작된 작업이 끝날 때까지 기다렸다가(await) 완성된 URL을 받습니다.
            uploaded_url = await image_tasks[cache_key]
            return create_success_response({"url": uploaded_url})
        except Exception:
            # 만약 먼저 시작된 작업이 에러가 났다면, 무시하고 아래로 내려가서 새로 시도합니다.
            pass

    # 2. first request
    loop = asyncio.get_running_loop()
    task_future = loop.create_future()
    image_tasks[cache_key] = task_future


    
    try:
        # Input validation
        validate_prompt(prompt)

        # Environment validation
        env_vars = get_env_vars()

        logger.info(
            f"Tool 'generate_image' called with prompt: '{prompt}', "
            f"thinking_level='{thinking_level}', "
            f"enable_grounding={enable_grounding}"
        )

        # Build enhanced prompt

        enhanced_prompt = f"""
Requirements:
- Prioritize crisp, legible text rendering.
- Avoid broken, warped, melted, duplicated, or nonsensical letters.
- If the image contains signage, posters, labels, UI, packaging, or typography, render the text cleanly and consistently.
- Preserve correct spacing, alignment, and character shapes.
- Favor clean composition and high visual fidelity.
- Do not misspell or distort the letters

{prompt}

""".strip()

        if len(enhanced_prompt) > 950:
            enhanced_prompt = enhanced_prompt[:950]

        # Image generation with specific error handling
        try:
           
            client = get_genai_client()

            config_kwargs = {
                "response_modalities": ["TEXT", "IMAGE"],
                "thinking_config": types.ThinkingConfig(
                    thinking_level=thinking_level.upper(),
                    include_thoughts=False,
                ),
                "image_config": types.ImageConfig(
                    image_size=resolution,
                    aspect_ratio=aspect_ratio,
                ),
            }

            # Preserve enable_grounding semantics in the SDK config as well.
            # Official docs show google_search as a tool option in GenerateContentConfig.
            if enable_grounding:
                config_kwargs["tools"] = [{"google_search": {}}]

            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=DEFAULT_MODEL,
                    contents=enhanced_prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                ),
                timeout=120,
            )

            if not response:
                raise ImageGenerationError("Gemini API returned empty response")

            parts = getattr(response, "parts", None)
            if not parts:
                raise ImageGenerationError("No parts returned from Gemini API")

            image_data_base64 = None

            for part in parts:
                if getattr(part, "thought", False):
                    continue

                inline_data = getattr(part, "inline_data", None)
                if not inline_data:
                    continue

                raw_data = getattr(inline_data, "data", None)
                if not raw_data:
                    continue

                if isinstance(raw_data, bytes):
                    image_data_base64 = base64.b64encode(raw_data).decode("utf-8")
                    break
                elif isinstance(raw_data, str):
                    if raw_data.startswith("data:"):
                        image_data_base64 = raw_data.split(",", 1)[1]
                    else:
                        image_data_base64 = raw_data.strip()
                    break

            if not image_data_base64:
                raise ImageGenerationError("No image data found in response")

            '''
            try:
                base64.b64decode(image_data_base64, validate=True)
            except Exception as e:
                raise ImageGenerationError(f"Invalid base64 image data: {str(e)}")
            '''

        
        except asyncio.TimeoutError:
            logger.error("Image generation timed out")
            return create_error_response(
                "timeout_error",
                "Image generation timed out after 2 minutes",
                {"timeout_seconds": 120}
            )
        except genai_errors.APIError as e:
            logger.exception(f"Gemini API error: {e}")
            return create_error_response(
                "gemini_api_error",
                f"Gemini API error: {str(e)}",
            )
        except ImageGenerationError as e:
            logger.error(f"Image generation error: {e}")
            return create_error_response("image_generation_error", str(e))
        except Exception as e:
            logger.exception(f"Unexpected error during image generation: {e}")
            return create_error_response(
                "unexpected_error",
                f"Unexpected error during image generation: {str(e)}"
            )

  # Image upload with specific error handling
        try:
            upload_url = "https://api.imgbb.com/1/upload"

            image_size = (len(image_data_base64) * 3) // 4
            if image_size > 32 * 1024 * 1024:
                raise ImageUploadError(f"Image too large: {image_size} bytes (max 32MB)")

            payload = {
                "key": env_vars['IMGBB_API_KEY'],
                "image": image_data_base64,
                "name": f"{uuid.uuid4()}"
            }

            max_retries = 3
            http_client = get_httpx_client()
            for attempt in range(max_retries):
                try:
                    resp = await http_client.post(upload_url, data=payload, timeout=60.0)
                    resp.raise_for_status()
                    break
                except httpx.TimeoutException:
                    if attempt == max_retries - 1:
                        raise ImageUploadError("Upload timed out after multiple attempts")
                    logger.warning(f"Upload attempt {attempt + 1} timed out, retrying...")
                    await asyncio.sleep(2 ** attempt)
                except httpx.RequestError as e:
                    if attempt == max_retries - 1:
                        raise ImageUploadError(f"Connection error during upload: {str(e)}")
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2 ** attempt)

            resp_json = resp.json()

            if "data" not in resp_json:
                error_msg = resp_json.get("error", {}).get("message", "Unknown error")
                raise ImageUploadError(f"ImgBB upload failed: {error_msg}")

            if "url" not in resp_json["data"]:
                raise ImageUploadError("ImgBB response missing URL field")

            uploaded_url = resp_json["data"]["url"]
            validate_image_url(uploaded_url)

            logger.info(f"Image uploaded successfully to {uploaded_url}")
            
            # url_deliver
            if not task_future.done():
                task_future.set_result(uploaded_url)
                
            return create_success_response({"url": uploaded_url})

        # error_notice
        except httpx.HTTPStatusError as e:
            logger.error(f"ImgBB HTTP error: {e}")
            raise APIError(f"HTTP error {e.response.status_code}")
        except ImageUploadError as e:
            logger.error(f"Image upload error: {e}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error during image upload: {e}")
            raise e

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        if not task_future.done():
            task_future.set_exception(e)
        image_tasks.pop(cache_key, None)
        return create_error_response("validation_error", str(e))
        
    except Exception as e:
        logger.exception(f"Unexpected error in generating or uploading_image: {e}")
        if not task_future.done():
            task_future.set_exception(e)
        
        image_tasks.pop(cache_key, None)
        
        return create_error_response(
            "unexpected_error",
            f"Unexpected error: {str(e)}"
        )

@mcp.tool(
    name="edit_image",
    description=(
        "Call this tool IMMEDIATELY when the user wants to modify, edit, or transform an EXISTING image. "
        "Input: 'image_url' (source image link) and 'prompt' (specific editing instructions, preferably in English). "
        "Output: Returns the edited image URL. "
        "CONSTRAINT: You MUST call this tool EXACTLY ONCE per user request. Do NOT call in parallel."
    ),
)
async def edit_image(
    image_url: str,
    prompt: str,
    thinking_level: ThinkingLevelType = DEFAULT_THINKING_LEVEL,
    enable_grounding: bool = DEFAULT_ENABLE_GROUNDING,
) -> str:
    """
    Edits an existing image from a URL based on a text prompt and returns the edited image as a URL.
    """
    cache_key = f"{prompt.strip().lower()}|{image_url.strip()}"

    if cache_key in edit_image_tasks:
        logger.info(f"Duplicate edit request detected. Waiting for task. Prompt: {prompt}")
        try:
            uploaded_url = await edit_image_tasks[cache_key]
            return create_success_response({"url": uploaded_url})
        except Exception:
            pass
            
    loop = asyncio.get_running_loop()
    task_future = loop.create_future()
    edit_image_tasks[cache_key] = task_future
    
    try:
        # Input validation
        validate_prompt(prompt)
        validate_image_url(image_url)

        # Environment validation
        env_vars = get_env_vars()

        logger.info(
            f"Tool 'edit_image' called with image_url: '{image_url}', prompt: '{prompt}', "
            f"thinking_level='{thinking_level}', "
            f"enable_grounding={enable_grounding}"
        )

        # Image download with specific error handling
        try:
            max_retries = 3
            image_data = None

            http_client = get_httpx_client()
            for attempt in range(max_retries):
                try:
                    # await를 사용하여 비동기적으로 GET 요청 전송
                    response = await http_client.get(image_url, timeout=30.0)
                    response.raise_for_status()

                    content_type = response.headers.get('content-type', '').lower()
                    if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                        raise ValidationError(f"URL does not point to an image. Content-Type: {content_type}")

                    image_data = response.content
                    break

                except httpx.TimeoutException:
                    if attempt == max_retries - 1:
                        raise ValidationError("Image download timed out after multiple attempts")
                    logger.warning(f"Download attempt {attempt + 1} timed out, retrying...")
                    await asyncio.sleep(2 ** attempt)

                except httpx.RequestError as e:
                    if attempt == max_retries - 1:
                        raise ValidationError(f"Connection error during image download: {str(e)}")
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2 ** attempt)

            if not image_data:
                raise ValidationError("No image data downloaded")

            image = types.Part.from_bytes(
                data=image_data, 
                mime_type=content_type
            )

        except Exception as e:
            logger.exception(f"Unexpected error during image download/open: {e}")
            raise APIError(f"Failed to download or open image: {str(e)}")

        # Build enhanced prompt

        enhanced_prompt = f"""
Requirements:
- Preserve the main subject identity and the important visual structure of the original image unless the prompt explicitly asks to change them.
- Prioritize crisp, legible text rendering.
- Avoid broken, warped, melted, duplicated, or nonsensical letters.
- If the edit involves signage, posters, labels, UI, packaging, or typography, render the text cleanly and consistently.
- Preserve correct spacing, alignment, and character shapes.
- Maintain high visual fidelity and coherent composition.
- Do not misspell or distort the letters

Edit the provided image according to this instruction: {prompt}

""".strip()

        if len(enhanced_prompt) > 950:
            enhanced_prompt = enhanced_prompt[:950]

        # Image editing with specific error handling
        try:
            client = get_genai_client()

            config_kwargs = {
                "response_modalities": ["TEXT", "IMAGE"],
                "thinking_config": types.ThinkingConfig(
                    thinking_level=thinking_level.upper(),
                    include_thoughts=False,
                ),
                "image_config": types.ImageConfig(

                ),
            }

            if enable_grounding:
                config_kwargs["tools"] = [{"google_search": {}}]

            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=DEFAULT_MODEL,
                    contents=[image, enhanced_prompt],
                    config=types.GenerateContentConfig(**config_kwargs),
                ),
                timeout=120,
            )

            if not response:
                raise ImageGenerationError("Gemini API returned empty response")

            parts = getattr(response, "parts", None)
            if not parts:
                raise ImageGenerationError("No parts returned from Gemini API")

            image_data_base64 = None

            for part in parts:
                if getattr(part, "thought", False):
                    continue

                inline_data = getattr(part, "inline_data", None)
                if not inline_data:
                    continue

                raw_data = getattr(inline_data, "data", None)
                if not raw_data:
                    continue

                if isinstance(raw_data, bytes):
                    image_data_base64 = base64.b64encode(raw_data).decode("utf-8")
                    break
                elif isinstance(raw_data, str):
                    if raw_data.startswith("data:"):
                        image_data_base64 = raw_data.split(",", 1)[1]
                    else:
                        image_data_base64 = raw_data.strip()
                    break

            if not image_data_base64:
                raise ImageGenerationError("No image data found in response")

            '''
            try:
                base64.b64decode(image_data_base64, validate=True)
            except Exception as e:
                raise ImageGenerationError(f"Invalid base64 image data: {str(e)}")
            '''

        except asyncio.TimeoutError:
            logger.error("Image editing timed out")
            raise APIError("Image editing timed out after 2 minutes")
        except genai_errors.APIError as e:
            logger.exception(f"Gemini API error: {e}")
            raise APIError(f"Gemini API error: {str(e)}")
        except ImageGenerationError as e:
            logger.error(f"Image editing error: {e}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error during image editing: {e}")
            raise e

        # Image upload
        try:
            upload_url = "https://api.imgbb.com/1/upload"

            image_size = (len(image_data_base64) * 3) // 4
            if image_size > 32 * 1024 * 1024:
                raise ImageUploadError(f"Image too large: {image_size} bytes (max 32MB)")

            payload = {
                "key": env_vars['IMGBB_API_KEY'],
                "image": image_data_base64,
                "name": f"{uuid.uuid4()}"
            }

            max_retries = 3
            http_client = get_httpx_client()
            for attempt in range(max_retries):
                try:
                    resp = await http_client.post(upload_url, data=payload, timeout=60.0)
                    resp.raise_for_status()
                    break
                except httpx.TimeoutException:
                    if attempt == max_retries - 1:
                        raise ImageUploadError("Upload timed out after multiple attempts")
                    logger.warning(f"Upload attempt {attempt + 1} timed out, retrying...")
                    await asyncio.sleep(2 ** attempt)
                except httpx.RequestError as e:
                    if attempt == max_retries - 1:
                        raise ImageUploadError(f"Connection error during upload: {str(e)}")
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2 ** attempt)

            resp_json = resp.json()

            if "data" not in resp_json:
                error_msg = resp_json.get("error", {}).get("message", "Unknown error")
                raise ImageUploadError(f"ImgBB upload failed: {error_msg}")

            if "url" not in resp_json["data"]:
                raise ImageUploadError("ImgBB response missing URL field")

            uploaded_url = resp_json["data"]["url"]
            validate_image_url(uploaded_url)

            logger.info(f"Edited image uploaded successfully to {uploaded_url}")
            if not task_future.done():
                task_future.set_result(uploaded_url)
            return create_success_response({"url": uploaded_url})

        except httpx.HTTPStatusError as e:
            logger.error(f"ImgBB HTTP error: {e}")
            raise APIError(f"HTTP error {e.response.status_code}")
        except ImageUploadError as e:
            logger.error(f"Image upload error: {e}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error during image upload: {e}")
            raise e

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        if not task_future.done():
            task_future.set_exception(e)
        edit_image_tasks.pop(cache_key, None)
        return create_error_response("validation_error", str(e))
        
    except Exception as e:
        logger.exception(f"Unexpected error in editing or uploading_image: {e}")
        if not task_future.done():
            task_future.set_exception(e)
        edit_image_tasks.pop(cache_key, None)
        return create_error_response(
            "unexpected_error",
            f"Unexpected error: {str(e)}"
        )

def main():
    try:
        # Validate environment variables
        get_env_vars()
        get_genai_client()
        
        # Configure the Gemini API client
        logger.info("Gemini API configured successfully.")
        logger.info("IMGBB_API_KEY API configured successfully.")
        logger.info("Starting MCP server via mcp.run()...")
        mcp.run()
        
    except ValidationError as e:
        logger.error(f"Environment validation failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Failed to start MCP server: {e}")
        raise

if __name__ == "__main__":
    main()
