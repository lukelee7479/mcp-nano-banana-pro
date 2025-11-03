import asyncio
import logging
import os
import base64
import uuid
import json
from mcp.server.fastmcp import FastMCP
import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
from typing import Dict, Any, Optional

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
    
    if len(prompt) > 1000:
        raise ValidationError("Prompt is too long (maximum 1000 characters)")
    
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
    description="Generates an image based on a text prompt using the Gemini API and returns the image as a url.",
)
async def generate_image(prompt: str) -> str:
    """
    Generates an image from a text prompt and returns the url of the image.
    """
    try:
        # Input validation
        validate_prompt(prompt)
        
        # Environment validation
        env_vars = validate_environment_variables()
        
        logger.info(f"Tool 'generate_image' called with prompt: '{prompt}'")

        # Image generation with specific error handling
        try:
            model = genai.GenerativeModel('gemini-2.5-flash-image-preview')
            
            # Generate content with timeout handling
            response = await asyncio.wait_for(
                model.generate_content_async([f"Generate a high-quality, detailed image of: {prompt}"]),
                timeout=120  # 2 minute timeout for generation
            )
            
            if not response:
                raise ImageGenerationError("Gemini API returned empty response")
            
            # Access response attributes directly (AsyncGenerateContentResponse doesn't have to_dict())
            if not hasattr(response, 'candidates') or not response.candidates:
                raise ImageGenerationError("No candidates returned from Gemini API")
            
            candidate = response.candidates[0]
            if not hasattr(candidate, 'content') or not candidate.content:
                raise ImageGenerationError("Invalid candidate structure: missing 'content' field")
            
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                raise ImageGenerationError("Invalid content structure: missing 'parts' field")
            
            parts = candidate.content.parts
            if not parts:
                raise ImageGenerationError("No parts returned in content")
            
            last_part = parts[-1]
            if not hasattr(last_part, 'inline_data') or not last_part.inline_data:
                raise ImageGenerationError("Last part does not contain image data")
            
            if not hasattr(last_part.inline_data, 'data') or not last_part.inline_data.data:
                raise ImageGenerationError("Image data field is missing")
            
            raw_data = last_part.inline_data.data
            
            # Handle different data formats
            if isinstance(raw_data, bytes):
                # If data is already bytes, encode to base64 string
                image_data_base64 = base64.b64encode(raw_data).decode('utf-8')
            elif isinstance(raw_data, str):
                # If data is a string, check if it has a data URI prefix
                if raw_data.startswith('data:'):
                    # Extract base64 part after comma
                    image_data_base64 = raw_data.split(',', 1)[1]
                else:
                    # Assume it's already base64 string, strip whitespace
                    image_data_base64 = raw_data.strip()
            else:
                raise ImageGenerationError(f"Unexpected data type: {type(raw_data)}")
            
            # Validate base64 data
            if not image_data_base64:
                raise ImageGenerationError("Empty image data received")
            
            # Test if base64 is valid
            try:
                base64.b64decode(image_data_base64, validate=True)
            except Exception as e:
                raise ImageGenerationError(f"Invalid base64 image data: {str(e)}")
                
        except asyncio.TimeoutError:
            logger.error("Image generation timed out")
            return create_error_response(
                "timeout_error",
                "Image generation timed out after 2 minutes",
                {"timeout_seconds": 120}
            )
        except genai.types.BlockedPromptException as e:
            logger.error(f"Prompt blocked by Gemini API: {e}")
            return create_error_response(
                "content_policy_error",
                "Prompt was blocked by content policy",
                {"blocked_reason": str(e)}
            )
        except genai.types.StopCandidateException as e:
            logger.error(f"Generation stopped by Gemini API: {e}")
            return create_error_response(
                "generation_stopped_error",
                "Image generation was stopped by the API",
                {"stop_reason": str(e)}
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
            
            # Validate image size (ImgBB has a 32MB limit)
            image_size = len(base64.b64decode(image_data_base64))
            if image_size > 32 * 1024 * 1024:  # 32MB
                raise ImageUploadError(f"Image too large: {image_size} bytes (max 32MB)")
            
            payload = {
                "key": env_vars['IMGBB_API_KEY'],
                "image": image_data_base64,
                "name": f"{uuid.uuid4()}"
            }
            
            # Upload with timeout and retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = requests.post(upload_url, data=payload, timeout=60)
                    resp.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        raise ImageUploadError("Upload timed out after multiple attempts")
                    logger.warning(f"Upload attempt {attempt + 1} timed out, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                except requests.exceptions.ConnectionError as e:
                    if attempt == max_retries - 1:
                        raise ImageUploadError(f"Connection error during upload: {str(e)}")
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2 ** attempt)
            
            resp_json = resp.json()
            
            # Validate ImgBB response
            if "data" not in resp_json:
                error_msg = resp_json.get("error", {}).get("message", "Unknown error")
                raise ImageUploadError(f"ImgBB upload failed: {error_msg}")
            
            if "url" not in resp_json["data"]:
                raise ImageUploadError("ImgBB response missing URL field")
            
            uploaded_url = resp_json["data"]["url"]
            
            # Validate the returned URL
            validate_image_url(uploaded_url)
            
            logger.info(f"Image uploaded successfully to {uploaded_url}")
            return create_success_response({"url": uploaded_url})
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 400:
                error_msg = "Bad request to ImgBB API"
            elif status_code == 401:
                error_msg = "Invalid ImgBB API key"
            elif status_code == 403:
                error_msg = "ImgBB API access forbidden"
            elif status_code == 413:
                error_msg = "Image file too large for ImgBB"
            elif status_code == 429:
                error_msg = "ImgBB API rate limit exceeded"
            elif status_code >= 500:
                error_msg = "ImgBB server error"
            else:
                error_msg = f"HTTP error {status_code}"
            
            logger.error(f"ImgBB HTTP error: {e}")
            return create_error_response(
                "upload_http_error",
                error_msg,
                {"status_code": status_code, "response_text": e.response.text}
            )
        except ImageUploadError as e:
            logger.error(f"Image upload error: {e}")
            return create_error_response("image_upload_error", str(e))
        except Exception as e:
            logger.exception(f"Unexpected error during image upload: {e}")
            return create_error_response(
                "unexpected_error",
                f"Unexpected error during image upload: {str(e)}"
            )

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return create_error_response("validation_error", str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in generate_image: {e}")
        return create_error_response(
            "unexpected_error",
            f"Unexpected error: {str(e)}"
        )
    

@mcp.tool(
    name="edit_image",
    description="Edits an existing image based on a text prompt using the Gemini API. Takes an image URL and a prompt, then returns the edited image as a URL.",
)
async def edit_image(image_url: str, prompt: str) -> str:
    """
    Edits an existing image from a URL based on a text prompt and returns the edited image as a URL.
    """
    try:
        # Input validation
        validate_prompt(prompt)
        validate_image_url(image_url)
        
        # Environment validation
        env_vars = validate_environment_variables()
        
        logger.info(f"Tool 'edit_image' called with image_url: '{image_url}' and prompt: '{prompt}'")

        # Image download with specific error handling
        try:
            # Download the image from the URL with timeout and retry logic
            max_retries = 3
            image_data = None
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(image_url, timeout=30)
                    response.raise_for_status()
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                        raise ValidationError(f"URL does not point to an image. Content-Type: {content_type}")
                    
                    # Check file size (10MB limit for download)
                    if len(response.content) > 10 * 1024 * 1024:
                        raise ValidationError("Image file too large (max 10MB)")
                    
                    image_data = response.content
                    break
                    
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        raise ImageGenerationError("Image download timed out after multiple attempts")
                    logger.warning(f"Download attempt {attempt + 1} timed out, retrying...")
                    await asyncio.sleep(2 ** attempt)
                except requests.exceptions.ConnectionError as e:
                    if attempt == max_retries - 1:
                        raise ImageGenerationError(f"Connection error during image download: {str(e)}")
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2 ** attempt)
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code
                    if status_code == 404:
                        raise ValidationError("Image not found at the provided URL")
                    elif status_code == 403:
                        raise ValidationError("Access forbidden to the image URL")
                    elif status_code == 410:
                        raise ValidationError("Image is no longer available at the provided URL")
                    elif status_code >= 500:
                        if attempt == max_retries - 1:
                            raise ImageGenerationError(f"Server error downloading image: {status_code}")
                        logger.warning(f"Server error {status_code} on attempt {attempt + 1}, retrying...")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise ImageGenerationError(f"HTTP error downloading image: {status_code}")
            
            if not image_data:
                raise ImageGenerationError("Failed to download image after all retry attempts")
            
            # Validate and process image
            try:
                image = Image.open(BytesIO(image_data))
                
                # Validate image format
                if image.format not in ['JPEG', 'PNG', 'WEBP', 'BMP', 'GIF']:
                    raise ValidationError(f"Unsupported image format: {image.format}")
                
                # Check image dimensions
                width, height = image.size
                if width > 4096 or height > 4096:
                    raise ValidationError(f"Image too large: {width}x{height} (max 4096x4096)")
                
                if width < 1 or height < 1:
                    raise ValidationError("Invalid image dimensions")
                
                # Convert to RGB if necessary (for compatibility)
                if image.mode not in ['RGB', 'RGBA']:
                    image = image.convert('RGB')
                
            except Exception as e:
                if "cannot identify image file" in str(e).lower():
                    raise ValidationError("Invalid image file format or corrupted image")
                else:
                    raise ImageGenerationError(f"Error processing image: {str(e)}")
            
        except ValidationError as e:
            logger.error(f"Image validation error: {e}")
            return create_error_response("validation_error", str(e))
        except ImageGenerationError as e:
            logger.error(f"Image download error: {e}")
            return create_error_response("image_download_error", str(e))
        except Exception as e:
            logger.exception(f"Unexpected error during image download: {e}")
            return create_error_response(
                "unexpected_error",
                f"Unexpected error during image download: {str(e)}"
            )

        # Image editing with specific error handling
        try:
            model = genai.GenerativeModel('gemini-2.5-flash-image-preview')

            # Generate content with timeout handling
            response = await asyncio.wait_for(
                model.generate_content_async([prompt, image]),
                timeout=120  # 2 minute timeout for editing
            )
            
            if not response:
                raise ImageGenerationError("Gemini API returned empty response")
            
            # Access response attributes directly (AsyncGenerateContentResponse doesn't have to_dict())
            if not hasattr(response, 'candidates') or not response.candidates:
                raise ImageGenerationError("No candidates returned from Gemini API")
            
            candidate = response.candidates[0]
            if not hasattr(candidate, 'content') or not candidate.content:
                raise ImageGenerationError("Invalid candidate structure: missing 'content' field")
            
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                raise ImageGenerationError("Invalid content structure: missing 'parts' field")
            
            parts = candidate.content.parts
            if not parts:
                raise ImageGenerationError("No parts returned in content")
            
            last_part = parts[-1]
            if not hasattr(last_part, 'inline_data') or not last_part.inline_data:
                raise ImageGenerationError("Last part does not contain image data")
            
            if not hasattr(last_part.inline_data, 'data') or not last_part.inline_data.data:
                raise ImageGenerationError("Image data field is missing")
            
            raw_data = last_part.inline_data.data
            
            # Handle different data formats
            if isinstance(raw_data, bytes):
                # If data is already bytes, encode to base64 string
                image_data_base64 = base64.b64encode(raw_data).decode('utf-8')
            elif isinstance(raw_data, str):
                # If data is a string, check if it has a data URI prefix
                if raw_data.startswith('data:'):
                    # Extract base64 part after comma
                    image_data_base64 = raw_data.split(',', 1)[1]
                else:
                    # Assume it's already base64 string, strip whitespace
                    image_data_base64 = raw_data.strip()
            else:
                raise ImageGenerationError(f"Unexpected data type: {type(raw_data)}")
            
            # Validate base64 data
            if not image_data_base64:
                raise ImageGenerationError("Empty image data received")
            
            # Test if base64 is valid
            try:
                base64.b64decode(image_data_base64, validate=True)
            except Exception as e:
                raise ImageGenerationError(f"Invalid base64 image data: {str(e)}")
                
        except asyncio.TimeoutError:
            logger.error("Image editing timed out")
            return create_error_response(
                "timeout_error",
                "Image editing timed out after 2 minutes",
                {"timeout_seconds": 120}
            )
        except genai.types.BlockedPromptException as e:
            logger.error(f"Prompt blocked by Gemini API: {e}")
            return create_error_response(
                "content_policy_error",
                "Prompt was blocked by content policy",
                {"blocked_reason": str(e)}
            )
        except genai.types.StopCandidateException as e:
            logger.error(f"Editing stopped by Gemini API: {e}")
            return create_error_response(
                "generation_stopped_error",
                "Image editing was stopped by the API",
                {"stop_reason": str(e)}
            )
        except ImageGenerationError as e:
            logger.error(f"Image editing error: {e}")
            return create_error_response("image_editing_error", str(e))
        except Exception as e:
            logger.exception(f"Unexpected error during image editing: {e}")
            return create_error_response(
                "unexpected_error",
                f"Unexpected error during image editing: {str(e)}"
            )

        # Image upload with specific error handling (same as generate_image)
        try:
            upload_url = "https://api.imgbb.com/1/upload"
            
            # Validate image size (ImgBB has a 32MB limit)
            image_size = len(base64.b64decode(image_data_base64))
            if image_size > 32 * 1024 * 1024:  # 32MB
                raise ImageUploadError(f"Image too large: {image_size} bytes (max 32MB)")
            
            payload = {
                "key": env_vars['IMGBB_API_KEY'],
                "image": image_data_base64,
                "name": f"{uuid.uuid4()}"
            }
            
            # Upload with timeout and retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = requests.post(upload_url, data=payload, timeout=60)
                    resp.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        raise ImageUploadError("Upload timed out after multiple attempts")
                    logger.warning(f"Upload attempt {attempt + 1} timed out, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                except requests.exceptions.ConnectionError as e:
                    if attempt == max_retries - 1:
                        raise ImageUploadError(f"Connection error during upload: {str(e)}")
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2 ** attempt)
            
            resp_json = resp.json()
            
            # Validate ImgBB response
            if "data" not in resp_json:
                error_msg = resp_json.get("error", {}).get("message", "Unknown error")
                raise ImageUploadError(f"ImgBB upload failed: {error_msg}")
            
            if "url" not in resp_json["data"]:
                raise ImageUploadError("ImgBB response missing URL field")
            
            uploaded_url = resp_json["data"]["url"]
            
            # Validate the returned URL
            validate_image_url(uploaded_url)
            
            logger.info(f"Edited image uploaded successfully to {uploaded_url}")
            return create_success_response({"url": uploaded_url})
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 400:
                error_msg = "Bad request to ImgBB API"
            elif status_code == 401:
                error_msg = "Invalid ImgBB API key"
            elif status_code == 403:
                error_msg = "ImgBB API access forbidden"
            elif status_code == 413:
                error_msg = "Image file too large for ImgBB"
            elif status_code == 429:
                error_msg = "ImgBB API rate limit exceeded"
            elif status_code >= 500:
                error_msg = "ImgBB server error"
            else:
                error_msg = f"HTTP error {status_code}"
            
            logger.error(f"ImgBB HTTP error: {e}")
            return create_error_response(
                "upload_http_error",
                error_msg,
                {"status_code": status_code, "response_text": e.response.text}
            )
        except ImageUploadError as e:
            logger.error(f"Image upload error: {e}")
            return create_error_response("image_upload_error", str(e))
        except Exception as e:
            logger.exception(f"Unexpected error during image upload: {e}")
            return create_error_response(
                "unexpected_error",
                f"Unexpected error during image upload: {str(e)}"
            )

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return create_error_response("validation_error", str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in edit_image: {e}")
        return create_error_response(
            "unexpected_error",
            f"Unexpected error: {str(e)}"
        )


def main():
    try:
        # Validate environment variables
        env_vars = validate_environment_variables()
        
        # Configure the Gemini API client
        genai.configure(api_key=env_vars['GEMINI_API_KEY'])
        logger.info("Gemini API configured successfully.")
        logger.info("IMGBB_API_KEY API configured successfully.")

        logger.info("Starting MCP server via mcp.run()...")
        asyncio.run(mcp.run())
        
    except ValidationError as e:
        logger.error(f"Environment validation failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Failed to start MCP server: {e}")
        raise

if __name__ == "__main__":
    main()
