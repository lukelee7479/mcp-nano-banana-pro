import asyncio
import logging
import os
import base64
import uuid
import json
import httpx
#import time
#import hashlib
#from botocore.exceptions import ClientError
from typing import Literal

#from io import BytesIO
from urllib.parse import urlparse, unquote
from typing import Dict, Any, Optional

#from PIL import Image
from mcp.server.fastmcp import FastMCP, Context
from mcp_nano_banana.mcp_s3 import upload_file, ROOT, BUCKET, s3
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

image_tasks = {}
edit_image_tasks = {}
_task_lock = None


#CACHE_TTL_SECONDS = 3600

def get_task_lock():
    global _task_lock
    if _task_lock is None:
        _task_lock = asyncio.Lock()
    return _task_lock

DEFAULT_MODEL = ["gemini-nano-banana-2.1", "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image" ]


DEFAULT_ENABLE_GROUNDING = False

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
    env_vars = get_env_vars()
    return genai.Client(api_key=env_vars["GEMINI_API_KEY"])




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
    concat_envs = os.getenv("NANOBANANA_API_KEY")
    try:
        banana_envs = [value.strip() for value in concat_envs.split("++", 2)]
        gemini_key = banana_envs[0]
    except:
        gemini_key = concat_envs
        
    if not gemini_key:
        errors.append("GEMINI_API_KEY environment variable not set")
    elif not gemini_key.strip():
        errors.append("GEMINI_API_KEY environment variable is empty")
    else:
        env_vars['GEMINI_API_KEY'] = gemini_key
    
    # Check IMGBB_API_KEY
    imgbb_key = os.getenv("IMGBB_API_KEY")
    if imgbb_key and imgbb_key.strip():
        env_vars["IMGBB_API_KEY"] = imgbb_key.strip()

    
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
"""
def get_cache_s3_key(prompt: str) -> str:
    prompt_hash = hashlib.sha256(prompt.strip().lower().encode('utf-8')).hexdigest()
    return f"cache/{prompt_hash}.json"
    
async def get_s3_cache(prompt: str) -> Optional[str]:
    if not BUCKET or not s3:
        return None
        
    cache_key = get_cache_s3_key(prompt)
    try:
        def _fetch():
            resp = s3.get_object(Bucket=BUCKET, Key=cache_key)
            return resp['Body'].read().decode('utf-8')

        data_str = await asyncio.to_thread(_fetch)
        cache_data = json.loads(data_str)

        if time.time() - cache_data["timestamp"] <= CACHE_TTL_SECONDS:
            return cache_data["url"]
            
    except ClientError as e:
        if e.response['Error']['Code'] != 'NoSuchKey':
            logger.warning(f"S3 Cache read error: {e}")
    except Exception as e:
        logger.warning(f"S3 Cache parsing error: {e}")
        
    return None

async def set_s3_cache(prompt: str, url: str):
    if not BUCKET or not s3:
        return
        
    cache_key = get_cache_s3_key(prompt)
    cache_data = {
        "url": url,
        "timestamp": time.time()
    }
    try:
        def _upload():
            s3.put_object(
                Bucket=BUCKET,
                Key=cache_key,
                Body=json.dumps(cache_data).encode('utf-8'),
                ContentType='application/json'
            )
        await asyncio.to_thread(_upload)
    except Exception as e:
        logger.warning(f"S3 Cache write error: {e}")
"""

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
    enable_grounding: bool = DEFAULT_ENABLE_GROUNDING,
    ctx: Context = None
) -> str:
    """
    Generates an image from a text prompt and returns the url of the image.
    """
    """
    cached_url = await get_s3_cache(prompt)
    if cached_url:
        logger.info(f"same request, return generated image: {prompt}")
        return create_success_response({"url": cached_url})
    """
                
    lock = get_task_lock()
    is_new_task = False
    task_future = None

    cache_key = prompt.strip().lower()

    # 1. same job running
    async with lock:
        
        if cache_key in image_tasks:
            logger.info(f"Duplicate request detected. Waiting for the existing task for: {prompt}")
            task_future = image_tasks[cache_key]
        else:
            loop = asyncio.get_running_loop()
            task_future = loop.create_future()
            image_tasks[cache_key] = task_future
            is_new_task = True

    if not is_new_task:     
        try:
            uploaded_url = await asyncio.wait_for(asyncio.shield(task_future), timeout=150)
            return create_success_response({"url": uploaded_url})
        except asyncio.TimeoutError:
            return create_error_response("timeout_error", "현재 요청의 대기시간이 초과되었습니다. 최초 요청은 처리중일 수 있습니다.")
        except asyncio.CancelledError:
            if task_future.cancelled():
                return create_error_response("task_cancelled", "기존 이미지 처리 작업이 취소되었습니다.")
            logger.info("Duplicate request was cancelled")
            raise
        except Exception as e:
            return create_error_response("task_failed", f"기다리던 기존 요청이 실패했습니다: {str(e)}")

    error_type = None
    error_msg = None
    exception_to_set = None

    try:
        validate_prompt(prompt)
        env_vars = get_env_vars()

        logger.info(
            f"Tool 'generate_image' called with prompt: '{prompt}', "
            f"enable_grounding={enable_grounding}"
        )

        enhanced_prompt = f"""
Requirements:

- 생성되는 이미지는 전체적으로 자연스럽고 완성도 높은 시각적 결과물을 목표로 해야 한다.
- 이미지의 주요 피사체와 배경 요소는 사용자의 요청 내용을 충실히 반영해야 하며, 사용자가 명시하지 않은 불필요한 사물이나 인물을 임의로 추가하지 않는다.
- 이미지 안에 등장하는 모든 요소는 장면의 맥락과 자연스럽게 연결되어야 하며, 서로 겹치거나 비정상적으로 잘려 보이거나 구조적으로 불가능한 형태가 나타나지 않도록 한다.
- 인물의 얼굴, 손, 손가락, 팔, 다리, 신체 관절은 해부학적으로 자연스럽게 표현해야 하며, 손가락 수가 잘못되거나 팔다리가 추가되거나 일부가 사라지는 오류가 발생하지 않도록 한다.
- 사람이 등장하는 경우 표정과 시선, 자세와 동작이 장면의 상황에 맞도록 자연스럽게 표현해야 한다.
- 인물의 얼굴은 좌우 비대칭이 지나치게 심하지 않아야 하며, 눈, 코, 입, 귀의 위치가 정상적으로 유지되어야 한다.
- 동일한 인물이 여러 번 중복 생성되거나 한 인물의 일부가 복제되어 나타나지 않도록 한다.
- 배경에 존재하는 사람들도 얼굴과 신체가 지나치게 왜곡되거나 불완전한 형태로 생성되지 않도록 한다.
- 건축물과 구조물은 실제 물리적 구조와 원근법에 맞게 표현하며, 문, 창문, 기둥, 계단, 난간 등이 비정상적으로 끊기거나 연결되지 않도록 한다.
- 건물의 수직선과 수평선은 장면의 원근감에 맞게 일관되게 유지한다.
- 도로, 보도, 계단, 난간, 펜스 등 반복적인 구조가 나타날 경우 패턴이 중간에 갑자기 변형되거나 사라지지 않도록 한다.
- 나무, 식물, 꽃, 풀, 숲과 같은 자연 요소는 실제 생태적 형태와 질감을 참고하여 자연스럽게 표현한다.
- 나뭇잎이나 꽃잎이 비정상적으로 반복되거나 동일한 패턴이 기계적으로 복제된 것처럼 보이지 않도록 한다.
- 산, 강, 호수, 바다, 하늘, 구름과 같은 자연환경은 장면 전체의 조명과 날씨 조건에 맞게 일관성 있게 표현한다.
- 하늘의 밝기와 색온도는 장면의 시간대와 기상 조건에 적합해야 한다.
- 그림자 방향은 주요 광원의 위치와 일치해야 하며, 서로 다른 피사체의 그림자가 각기 다른 방향으로 발생하지 않도록 한다.
- 반사광이 발생하는 유리, 금속, 물, 거울 등은 주변 환경과 조명을 자연스럽게 반영하도록 한다.
- 반사되는 장면에 원본 장면과 무관한 인물이나 물체가 추가되지 않도록 한다.
- 이미지 전체의 원근법과 깊이감은 일관되게 유지하며, 앞쪽의 사물이 지나치게 작거나 뒤쪽의 사물이 비정상적으로 크게 표현되지 않도록 한다.
- 사용자가 특정 카메라 시점이나 구도를 요청한 경우 해당 시점을 우선적으로 준수한다.
- 카메라 렌즈 특성을 표현할 경우 왜곡이 지나치게 과장되어 주요 피사체의 형태가 무너지지 않도록 한다.
- 화면의 중심 피사체는 명확하게 인식될 수 있어야 하며, 배경 요소가 중심 피사체보다 과도하게 시선을 끌지 않도록 한다.
- 이미지 내부의 텍스트는 별도의 중요한 시각 요소로 취급한다.
- 사용자가 직접 제공한 한국어 문구는 원문 그대로 정확히 유지해야 한다.
- 제공된 한국어 문구를 임의로 번역하거나 영어로 바꾸거나 의미가 비슷한 다른 표현으로 바꾸지 않는다.
- 한국어 음절 블록을 정확히 유지하고 자음과 모음이 잘못 결합되거나 분리되지 않도록 한다.
- 한국어 문장 내에서 동일한 글자가 중복되거나 일부 글자가 빠지거나 다른 글자로 치환되지 않도록 한다.
- 한국어의 받침, 겹받침, 된소리, 이중모음 등 복합적인 음절 구조도 원문과 동일하게 유지해야 한다.
- 한글이 포함된 제목, 부제목, 설명문, 라벨, 간판, 표지판, 메뉴, 버튼, 배너, 포스터 문구를 가능한 한 정확하게 렌더링한다.
- 사용자가 별도의 언어를 지정하지 않은 경우 이미지 내부에서 새롭게 생성되는 텍스트는 기본적으로 한국어를 사용한다.
- 영어 텍스트는 사용자가 명시적으로 영어 사용을 요청하거나 고유명사, 브랜드명, 국제적으로 통용되는 표현처럼 영어 표기가 자연스러운 경우에만 제한적으로 사용한다.
- 사용자가 제공한 문장이 큰따옴표나 작은따옴표 안에 포함되어 있을 경우 해당 문자열을 수정 없이 그대로 렌더링한다.
- 사용자가 제공한 제목을 요약하거나 짧게 바꾸지 않는다.
- 사용자가 제공한 텍스트의 띄어쓰기와 문장부호도 가능한 한 그대로 유지한다.
- 쉼표, 마침표, 콜론, 괄호, 물음표, 느낌표, 따옴표 등 문장부호가 불필요하게 누락되거나 중복되지 않도록 한다.
- 숫자가 포함된 문구는 숫자의 순서와 자릿수를 정확하게 유지한다.
- 날짜, 시간, 전화번호, 가격, 수량, 퍼센트, 단위와 같이 숫자 정확성이 중요한 요소는 원문과 동일하게 표현한다.
- 숫자 0과 영문 대문자 O, 숫자 1과 영문 소문자 l, 숫자 5와 영문 S처럼 모양이 비슷한 문자들이 서로 혼동되지 않도록 한다.
- 이미지 내 텍스트는 가능한 한 선명하고 읽기 쉬운 형태로 생성한다.
- 텍스트 획이 녹거나 뭉개지거나 끊어지거나 서로 붙지 않도록 한다.
- 한글 글자의 초성, 중성, 종성이 잘못 분리되어 독립적인 기호처럼 보이지 않도록 한다.
- 텍스트의 글꼴 스타일은 전체 디자인과 어울리도록 선택하되, 가독성을 저해할 정도로 장식적인 형태는 피한다.
- 굵은 글씨가 필요한 제목은 획이 지나치게 두꺼워 내부 공간이 막히지 않도록 한다.
- 작은 크기의 글자는 가능하면 단순하고 명확한 형태로 표현한다.
- 중요한 제목은 배경과 충분한 명도 대비를 가지도록 배치한다.
- 밝은 배경 위에는 너무 밝은 글자를 사용하지 않으며, 어두운 배경 위에는 너무 어두운 글자를 사용하지 않는다.
- 텍스트 뒤에 복잡한 배경 무늬가 있을 경우 필요하면 자연스러운 배경 정리나 여백을 활용하여 가독성을 높인다.
- 사용자가 특정 색상의 글자를 요청한 경우 해당 색상을 우선 적용한다.
- 사용자가 특정 위치에 문구를 배치하도록 요청한 경우 위치를 임의로 변경하지 않는다.
- 제목과 본문, 설명문, 라벨 사이의 시각적 위계를 명확히 한다.
- 동일한 문구가 불필요하게 여러 번 반복되지 않도록 한다.
- 이미지 내에 사용자가 요청하지 않은 가상의 로고나 브랜드명을 임의로 추가하지 않는다.
- 실존 브랜드의 로고가 필요하지 않은 경우 비슷하게 보이는 가상의 로고를 임의로 삽입하지 않는다.
- 워터마크, 서명, 제작자 표시, 임의의 작은 글씨를 자동으로 추가하지 않는다.
- 사용자가 포스터를 요청한 경우 제목, 주요 이미지, 보조 정보가 명확한 구조를 갖도록 구성한다.
- 사용자가 인포그래픽을 요청한 경우 정보의 순서와 시각적 계층을 명확히 표현한다.
- 인포그래픽의 각 항목은 서로 충분한 간격을 두어 혼동되지 않도록 한다.
- 아이콘과 설명 텍스트가 잘못 연결되거나 다른 항목과 섞이지 않도록 한다.
- 표나 도표가 등장하는 경우 행과 열이 비정상적으로 끊기거나 중간에서 합쳐지지 않도록 한다.
- 표 안의 텍스트는 각 셀에 맞게 정렬되고, 다른 셀의 내용과 겹치지 않도록 한다.
- 화살표가 등장할 경우 시작점과 도착점이 명확해야 하며 엉뚱한 요소를 가리키지 않도록 한다.
- 순서도가 등장할 경우 진행 방향이 일관되게 유지되어야 한다.
- 번호가 있는 목록은 1, 2, 3과 같은 순서가 중복되거나 누락되지 않도록 한다.
- 지도나 안내도가 등장하는 경우 실제 지리적 정확성이 요구되지 않는 한 시각적으로 명확한 안내 구조를 우선한다.
- 사용자가 실제 지명이나 위치를 명시한 경우 지명을 임의로 변형하지 않는다.
- 방향 표시가 있는 경우 동서남북의 방향이 서로 모순되지 않도록 한다.
- 교통수단이 등장하는 경우 자동차, 버스, 기차, 비행기, 자전거 등 각각의 구조가 현실적으로 표현되도록 한다.
- 자동차의 바퀴 수가 잘못되거나 차체 구조가 비정상적으로 이어지지 않도록 한다.
- 기차나 모노레일이 등장할 경우 동일한 차량이 중복 생성되거나 선로와 분리되어 떠 있는 형태로 표현되지 않도록 한다.
- 동물이 등장하는 경우 종의 특징을 자연스럽게 반영하며 다리, 꼬리, 귀, 날개 등이 비정상적으로 추가되거나 사라지지 않도록 한다.
- 새가 등장하는 경우 부리, 날개, 꼬리, 발의 형태가 자연스럽게 표현되도록 한다.
- 물고기가 등장하는 경우 지느러미와 꼬리의 구조가 물속 환경과 자연스럽게 어울리도록 한다.
- 계절이 명시된 경우 해당 계절의 식생, 날씨, 색감, 의상, 조명을 일관성 있게 반영한다.
- 봄 장면에서는 새싹, 꽃, 부드러운 햇빛과 같은 요소를 자연스럽게 표현한다.
- 여름 장면에서는 풍부한 녹음과 비교적 강한 햇빛을 자연스럽게 표현한다.
- 가을 장면에서는 단풍과 낙엽, 따뜻한 색감을 자연스럽게 표현한다.
- 겨울 장면에서는 눈, 서리, 차가운 공기감과 계절에 맞는 식생을 자연스럽게 표현한다.
- 비가 오는 장면에서는 젖은 바닥, 반사광, 빗방울, 흐린 하늘이 서로 일관되게 나타나도록 한다.
- 눈이 오는 장면에서는 눈의 적설 정도가 장면 전체에서 균형 있게 유지되도록 한다.
- 특정 물체 위에만 비정상적으로 눈이 쌓이거나 주변에는 전혀 눈이 없는 모순이 발생하지 않도록 한다.
- 안개가 있는 장면에서는 거리에 따른 대비 감소와 색감 변화를 자연스럽게 표현한다.
- 야간 장면에서는 광원의 종류와 위치에 따라 주변 밝기와 그림자를 자연스럽게 표현한다.
- 낮 장면에서는 사용자가 요청하지 않는 한 달이나 별이 과도하게 보이지 않도록 한다.
- 실내 장면에서는 창문, 조명, 가구, 벽, 바닥의 원근과 조명이 일관되게 표현되도록 한다.
- 가구는 물리적으로 사용 가능한 형태와 크기로 표현한다.
- 책상이나 의자의 다리가 비정상적인 위치에 추가되지 않도록 한다.
- 전자기기가 등장하는 경우 화면, 버튼, 케이블, 본체의 위치 관계가 자연스럽도록 한다.
- 노트북 화면 안의 내용은 지나치게 복잡한 가짜 텍스트로 채우지 않는다.
- 화면 안에 텍스트가 필요한 경우 핵심 문구만 선명하게 표현하는 것을 우선한다.
- 로봇이나 기계 장치가 등장하는 경우 각 관절과 구조물의 연결 관계가 논리적으로 이어지도록 한다.
- 로봇팔의 관절 수와 방향이 비정상적으로 변형되지 않도록 한다.
- 케이블이나 배선이 필요할 경우 시작점과 연결점이 자연스럽게 이어지도록 한다.
- 과학 실험 장면에서는 실험기구의 위치와 기능적 관계가 최소한 시각적으로 모순되지 않도록 한다.
- 교육용 이미지에서는 지나치게 장식적인 요소보다 정보 전달의 명확성을 우선한다.
- 학생이 등장하는 경우 학습 활동과 관련된 도구와 환경이 자연스럽게 배치되도록 한다.
- 여러 사람이 함께 작업하는 장면에서는 각 인물의 역할이 구도상 구분될 수 있도록 한다.
- 발표 장면에서는 발표자, 화면, 청중의 시선 방향이 자연스럽게 연결되도록 한다.
- 회의 장면에서는 인물들이 서로를 바라보거나 자료를 확인하는 등 상황에 맞는 행동을 하도록 한다.
- 사용자가 실사 스타일을 요청하면 사진 촬영 결과처럼 자연스러운 질감과 조명을 유지한다.
- 일러스트 스타일을 요청하면 스타일의 일관성을 유지하면서도 주요 사물의 형태를 명확히 표현한다.
- 유화 스타일을 요청하면 붓터치와 물감 질감을 표현하되 중요한 텍스트와 피사체의 형태가 지나치게 무너지지 않도록 한다.
- 수채화 스타일을 요청하면 번짐과 투명감을 표현하되 주요 윤곽이 완전히 사라지지 않도록 한다.
- 만화 스타일을 요청하면 선과 색상의 규칙성을 유지하고 장면 내 모든 요소가 동일한 화풍을 공유하도록 한다.
- 한 이미지 안에서 서로 다른 미술 스타일이 의도 없이 섞이지 않도록 한다.
- 사용자가 특정 작가풍이나 시대적 분위기를 일반적인 미술 특성으로 요청한 경우 해당 시각적 특징을 자연스럽게 반영한다.
- 이미지 전체의 색조는 사용자의 요청과 주제에 맞게 통일한다.
- 특정 색상을 강조하더라도 모든 요소가 동일한 색으로 물들어 형태 구분이 어려워지지 않도록 한다.
- 채도가 지나치게 높아 세부 구조가 사라지지 않도록 한다.
- 대비가 지나치게 강해 밝은 부분이나 어두운 부분의 디테일이 완전히 소실되지 않도록 한다.
- 중요한 피사체의 윤곽은 배경에서 명확하게 구분되도록 한다.
- 장면에 여러 개의 광원이 있는 경우 각 광원의 색온도와 밝기를 자연스럽게 조화시킨다.
- 불필요한 렌즈 플레어나 과도한 빛 번짐을 자동으로 추가하지 않는다.
- 사용자가 요청하지 않은 영화적 효과를 임의로 지나치게 적용하지 않는다.
- 이미지 가장자리에서 사람이나 물체가 어색하게 절단되지 않도록 구도를 조정한다.
- 중요한 텍스트가 이미지 가장자리에 너무 가까이 배치되어 잘릴 위험이 없도록 한다.
- 이미지 생성 시 안전한 여백을 확보하여 주요 제목이나 핵심 요소가 화면 밖으로 벗어나지 않도록 한다.
- 16:9, 4:3, 1:1 등 지정된 화면비가 있는 경우 주요 구성 요소가 해당 화면비 안에서 자연스럽게 배치되도록 한다.
- 가로형 이미지에서는 수평적 구성을 활용하되 중요한 요소가 지나치게 좌우 끝에 몰리지 않도록 한다.
- 세로형 이미지에서는 위아래의 시각적 균형을 유지한다.
- 정사각형 이미지에서는 중앙 집중형 또는 균형 잡힌 배치를 사용한다.
- 사용자가 특정 객체의 위치를 지정한 경우 해당 객체의 위치를 가능한 한 정확하게 유지한다.
- 편집 작업에서는 사용자가 변경을 요청한 부분 외에는 원본 이미지의 주요 구도와 객체 위치를 최대한 보존한다.
- 편집 과정에서 원본에 없던 새로운 인물이나 사물이 임의로 등장하지 않도록 한다.
- 특정 객체를 제거하는 경우 제거된 공간을 주변 배경과 자연스럽게 연결한다.
- 특정 객체를 추가하는 경우 주변 조명, 원근법, 그림자와 일치하도록 한다.
- 객체의 색상만 변경하는 요청에서는 객체의 형태와 위치까지 불필요하게 바꾸지 않는다.
- 날씨만 변경하는 요청에서는 주요 건물, 인물, 차량, 식생의 위치를 최대한 유지한다.
- 시간대만 변경하는 요청에서는 장면의 구조를 유지하면서 조명과 하늘을 자연스럽게 변경한다.
- 계절 변경 요청에서는 기존 구도를 유지하며 계절적 요소만 자연스럽게 변화시킨다.
- 원본 이미지의 주요 피사체 정체성을 유지해야 하는 경우 얼굴, 의상, 자세, 위치의 급격한 변경을 피한다.
- 사용자가 특정 사물이나 요소를 유지하라고 요청한 경우 편집 과정에서 해당 요소를 삭제하거나 가리지 않는다.
- 사용자가 특정 사물을 숨기라고 요청한 경우 다른 요소를 이용해 자연스럽게 가리되 원본 장면과 어울리도록 한다.
- 연속된 이미지나 장면의 일부로 사용되는 경우 이전 프레임과 객체의 위치, 크기, 방향이 자연스럽게 이어질 수 있도록 한다.
- 장면 간 연속성이 중요할 경우 동일한 인물이나 물체의 외형을 일관되게 유지한다.
- 동일한 차량이나 기계가 장면 중간에 복제되어 두 개 이상 나타나지 않도록 한다.
- 사용자가 명시적으로 요청하지 않는 한 이미지 생성 중간에 새로운 주요 객체를 추가하지 않는다.
- 시각적으로 중요한 표지판이나 비석의 글자는 가능한 한 정확히 유지하고 주변 질감과 자연스럽게 조화시킨다.
- 글자가 지나치게 또렷해 주변의 회화적 질감과 분리되어 보이지 않도록 하되 읽을 수 있는 수준의 정확성은 유지한다.
- 회화 스타일에서도 텍스트는 무작위 획이나 의미 없는 문양으로 대체하지 않는다.
- 정확히 표현하기 어려운 긴 문장이 필요한 경우 최소한 주요 제목과 핵심 단어의 정확성을 우선한다.
- 사용자가 여러 문구를 제공한 경우 각 문구를 서로 섞지 않고 지정된 위치와 용도에 맞게 구분한다.
- 제목 문구가 본문 문구와 합쳐지거나 서로 겹치지 않도록 한다.
- 한국어 문장과 영어 문장이 함께 필요한 경우 언어별 문자열을 정확히 구분한다.
- 괄호 안에 있는 영문 표기나 약어를 임의로 삭제하지 않는다.
- 고유명사는 사용자가 제공한 표기를 그대로 유지한다.
- 사람 이름, 기관명, 장소명, 제품명은 철자를 임의로 수정하거나 유사한 다른 단어로 변경하지 않는다.
- 사용자가 특정 스타일의 서체를 요청한 경우 가능한 범위에서 해당 특징을 반영하되 텍스트 정확성과 가독성을 우선한다.
- 손글씨 스타일을 요청해도 글자가 읽을 수 없는 수준으로 심하게 왜곡되지 않도록 한다.
- 캘리그래피 스타일을 요청해도 각 음절의 자모 구조가 유지되도록 한다.
- 포스터 제목은 시각적으로 강조하되 이미지의 주요 피사체를 과도하게 가리지 않도록 한다.
- 설명문은 제목보다 작은 크기로 표현하되 충분히 읽을 수 있도록 한다.
- 이미지 내부에 불필요한 긴 문단을 자동으로 생성하지 않는다.
- 사용자가 제공하지 않은 임의의 날짜, 전화번호, 주소, URL, 가격을 만들어 넣지 않는다.
- 사용자가 요구한 핵심 정보가 이미지에서 누락되지 않도록 한다.
- 사용자가 강조한 우선순위가 서로 충돌하는 경우 핵심 피사체, 정확한 텍스트, 전체 구도, 세부 장식 순으로 우선순위를 적용한다.
- 시각적으로 아름다운 결과를 만들기 위해 사용자의 명시적인 요구사항을 희생하지 않는다.
- 장식적 요소를 추가하더라도 주요 메시지 전달을 방해하지 않도록 한다.
- 전체적으로 이미지가 복잡해질 경우 불필요한 보조 요소를 줄이고 핵심 정보의 명확성을 우선한다.
- 사용자가 단순한 디자인을 요청한 경우 불필요한 텍스처, 패턴, 장식, 아이콘을 추가하지 않는다.
- 사용자가 풍부한 디자인을 요청한 경우에도 시각적 위계가 무너지지 않도록 한다.
- 사실적인 장면에서는 물체의 재질 차이를 자연스럽게 표현한다.
- 금속은 금속 특유의 반사와 광택을, 나무는 결을, 유리는 투명성과 반사를, 천은 주름과 섬유감을 자연스럽게 표현한다.
- 피부 질감은 지나치게 플라스틱처럼 매끈하거나 과도하게 거칠지 않도록 한다.
- 음식이 등장하는 경우 형태와 질감이 실제 음식처럼 자연스럽게 표현되도록 한다.
- 음료나 액체가 담긴 용기는 액체 표면과 용기의 형태가 물리적으로 자연스럽도록 한다.
- 그림자와 접촉면을 활용하여 물체가 공중에 떠 있는 것처럼 보이지 않도록 한다.
- 바닥과 접촉하는 사람이나 사물에는 자연스러운 접촉 그림자가 나타나도록 한다.
- 반복되는 요소는 일정한 형태를 유지하되 완전히 동일한 복제 패턴처럼 보이지 않도록 자연스러운 변화를 준다.
- 이미지 생성 결과가 전체적으로 하나의 완성된 작품처럼 보이도록 스타일, 조명, 색상, 질감, 원근을 통일한다.
- 사용자의 요청을 최우선으로 따르며, 별도의 요구사항이 없는 부분에서만 모델의 창의성을 활용한다.
- 사용자가 명시적으로 금지한 요소는 어떤 형태로도 추가하지 않는다.
- 사용자가 “새로운 객체를 추가하지 말라”고 요청한 경우 기존 객체와 배경 요소만 활용한다.
- 사용자가 “정확히 유지하라”고 요청한 요소는 위치, 크기, 방향, 형태를 최대한 보존한다.
- 사용자가 특정 텍스트의 정확성을 강조한 경우 미적 효과보다 철자 정확성을 우선한다.
- 최종 이미지를 생성하기 전에 장면 내 주요 요소, 텍스트, 인물, 구조물의 일관성을 종합적으로 고려한다.
- 최종 출력에서 의미 없는 문자열, 임의의 문자, 가짜 단어, 읽을 수 없는 텍스트가 나타나지 않도록 한다.
- 최종 출력에서 사용자가 요청한 한국어 문구가 존재하는 경우 해당 문구가 원문과 일치하는지 최대한 정확하게 반영한다.
- 최종 결과는 사용자의 요구사항을 충실히 반영하면서도 자연스럽고 전문적이며 시각적으로 완성도 높은 이미지가 되도록 한다.

{prompt}
""".strip()

#        if len(enhanced_prompt) > 950:
#            enhanced_prompt = enhanced_prompt[:950]

        client = get_genai_client()
        config_kwargs = {
            "response_modalities": ["IMAGE"],
            "image_config": types.ImageConfig(
                image_size='1K',
                aspect_ratio='16:9',
            ),
            "thinking_config": types.ThinkingConfig(
                thinking_level="high"
            ),
        }

        enable_grounding = False

        max_gemini_retries = len(DEFAULT_MODEL)
        response = None

        for attempt in range(max_gemini_retries):
            try:
                if attempt==max_gemini_retries-1:
                    enhanced_prompt = "Render every text in English.\n" + prompt[:700]
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=DEFAULT_MODEL[attempt],
                        contents=enhanced_prompt,
                        config=types.GenerateContentConfig(**config_kwargs),
                    ),
                    timeout=120,
                )
                break 
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt == max_gemini_retries - 1:
                    raise e
                await asyncio.sleep(2) 


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

        uploaded_url = None

        try:
            image_bytes = base64.b64decode(image_data_base64, validate=True)
        except Exception as e:
            raise ImageGenerationError(f"Invalid image data returned by nanobanana:{e}")


        try:
            logger.info("Trying S3 upload first")

            ext = "jpg"
            if image_bytes.startswith(b"\x89PNG"):
                ext = "png"
            elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
                ext = "webp"
       

            
            temp_filename = f"{uuid.uuid4()}.{ext}"
            temp_path = os.path.join(ROOT, temp_filename)
            log_filename = f"prompt{uuid.uuid4()}.txt"
            log_path = os.path.join(ROOT, log_filename)
            
            try:
                with open(temp_path, "wb") as f:
                    f.write(image_bytes)
                s3_response = await upload_file(local_path=temp_filename, ctx=ctx)
                uploaded_url = s3_response.url

                with open(log_path, "w", encoding="utf-8") as g:
                    g.write(enhanced_prompt)
                await upload_file(local_path=log_filename, ctx=ctx)
                
                logger.info("s3 upload done")
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if os.path.exists(log_path):
                    os.remove(log_path)
            

        except Exception as s3_error:
            logger.warning(f"s3 failed, try imgbb upload. reason: {s3_error}")
            imgbb_key = env_vars.get("IMGBB_API_KEY")
            if not imgbb_key:
                raise ImageUploadError("IMGBB_API_KEY is not configured")

            upload_url = "https://api.imgbb.com/1/upload"
        
            payload = {
                "key": env_vars['IMGBB_API_KEY'],
                "image": image_data_base64,
                "name": f"{uuid.uuid4()}"
            }
            try:
                image_size = (len(image_data_base64) * 3) // 4
                if image_size > 32 * 1024 * 1024:
                    raise ImageUploadError(f"Image too large: {image_size} bytes (max 32MB)")

                max_retries = 2
                http_client = get_httpx_client()
                resp = None
        
                for attempt in range(max_retries):
                    try:
                        resp = await http_client.post(upload_url, data=payload, timeout=30.0)
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
                    error_msg_upload = resp_json.get("error", {}).get("message", "Unknown error")
                    raise ImageUploadError(f"ImgBB upload failed: {error_msg_upload}")

                if "url" not in resp_json["data"]:
                    raise ImageUploadError("ImgBB response missing URL field")
                

                uploaded_url = resp_json["data"]["url"]
                validate_image_url(uploaded_url)
            
            except Exception as imgbb_error:
                raise ImageUploadError(f"upload both failed"
                                       f"S3: {s3_error};"
                                       f"ImgBB: {imgbb_error}")          
            

        
        logger.info(f"Image uploaded successfully to {uploaded_url}")

        if not task_future.done():
            task_future.set_result(uploaded_url)

        """
        await set_s3_cache(prompt, uploaded_url)
        logger.info("s3 cache saved")
        """
        
        return create_success_response({"url": uploaded_url})

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        error_type, error_msg, exception_to_set = "validation_error", str(e), e
    except asyncio.TimeoutError as e:
        logger.error("Image generation timed out")
        error_type, error_msg, exception_to_set = "timeout_error", "Image generation timed out after 2 minutes", e
    except genai_errors.APIError as e:
        logger.exception(f"Gemini API error: {e}")
        error_type, error_msg, exception_to_set = "gemini_api_error", f"Gemini API error: {str(e)}", e
    except ImageGenerationError as e:
        logger.error(f"Image generation error: {e}")
        error_type, error_msg, exception_to_set = "image_generation_error", str(e), e
    except httpx.HTTPStatusError as e:
        logger.error(f"ImgBB HTTP error: {e}")
        error_type, error_msg, exception_to_set = "http_error", f"HTTP error {e.response.status_code}", e
    except ImageUploadError as e:
        logger.error(f"Image upload error: {e}")
        error_type, error_msg, exception_to_set = "image_upload_error", str(e), e
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        error_type, error_msg, exception_to_set = "unexpected_error", f"Unexpected error: {str(e)}", e

    finally:
        image_tasks.pop(cache_key, None)
        if task_future and not task_future.done():
            if exception_to_set:
                task_future.set_exception(exception_to_set)
            else:
                task_future.cancel("원본 태스크가 예기치 않게 취소되었습니다.")

    if error_type and error_msg:
        return create_error_response(error_type, error_msg)


@mcp.tool(
    name="edit_image",
    description=(
        "Call this tool IMMEDIATELY when the user wants to modify, edit, or transform an EXISTING image. "
        "Input: 'image_url' (source image link) and 'prompt'. "
        "Output: Returns the edited image URL. "
        "CONSTRAINT: You MUST call this tool EXACTLY ONCE per user request. Do NOT call in parallel."
    ),
)
async def edit_image(
    image_url: str,
    prompt: str,
    enable_grounding: bool = DEFAULT_ENABLE_GROUNDING,
    ctx: Context = None
) -> str:
    """
    Edits an existing image from a URL based on a text prompt and returns the edited image as a URL.
    """
    cache_key = f"{prompt.strip().lower()}|{image_url.strip()}"

    lock = get_task_lock()  
    is_new_task = False
    task_future = None

    async with lock:
        if cache_key in edit_image_tasks:
            logger.info(f"Duplicate edit request detected. Waiting for {prompt}")
            task_future = edit_image_tasks[cache_key]
        else:
            loop = asyncio.get_running_loop()
            task_future = loop.create_future()
            edit_image_tasks[cache_key] = task_future
            is_new_task = True

    if not is_new_task:       
        try:
            uploaded_url = await asyncio.wait_for(asyncio.shield(task_future), timeout=150)
            return create_success_response({"url": uploaded_url})
        except asyncio.TimeoutError:
            return create_error_response("timeout_error", "현재 요청의 대기시간이 초과되었습니다. 기존 요청은 작업 중일 수 있습니다.")
        except asyncio.CancelledError:
            if task_future.cancelled():
                return create_error_response("task_cancelled", "기존 이미지 처리 작업이 취소되었습니다.")
            logger.info("Duplicate request was cancelled")
            raise
        except Exception as e:
            return create_error_response("task_failed", f"기다리던 기존 요청이 실패하였습니다: {str(e)}")
            
    
    try:
        # Input validation
        validate_prompt(prompt)
        validate_image_url(image_url)

        # Environment validation
        env_vars = get_env_vars()

        logger.info(
            f"Tool 'edit_image' called with image_url: '{image_url}', prompt: '{prompt}', "
            f"enable_grounding={enable_grounding}"
        )

        # Image download with specific error handling
        try:
            
            image_data = None
            content_type = "image/jpg"

            clean_url = image_url.strip().replace("&amp;", "&")

            if BUCKET and BUCKET in clean_url and "amazonaws.com" in clean_url:
                logger.info("s3 url detected. trying boto3 download")
                try:
                    parsed_url = urlparse(clean_url)
                    s3_key = unquote(parsed_url.path.lstrip('/'))
                    def fetch_from_s3():
                        resp = s3.get_object(Bucket=BUCKET, Key=s3_key)
                        return resp['Body'].read(), resp.get('ContentType', 'image/jpeg')
                        
                    image_data, content_type = await asyncio.to_thread(fetch_from_s3)
                    logger.info("image downloaded from s3")
                except Exception as e:
                    logger.warning(f"download from s3 failed: {e}")



            if not image_data:
                max_retries = 3

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "image/*,*/*;q=0.8"
                }

                http_client = get_httpx_client()
                for attempt in range(max_retries):
                    try:
                        # await를 사용하여 비동기적으로 GET 요청 전송
                        response = await http_client.get(
                            clean_url,
                            headers=headers,
                            timeout=30.0,
                            follow_redirects=True
                        )
                        response.raise_for_status()

                        res_content_type = response.headers.get('content-type', '').lower()
                        if not any(img_type in res_content_type for img_type in ['image/', 'application/octet-stream']):
                            raise ValidationError(f"URL does not point to an image. Content-Type: {res_content_type}")

                        image_data = response.content
                        content_type = res_content_type
                        break

                    except httpx.HTTPStatusError as e:
                        if attempt == max_retries -1:
                            raise ValidationError(f"HTTP download status error: {e.response.status_code}")
                        await asyncio.sleep(2 ** attempt)

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

            safe_mime_type = content_type if "image/" in content_type else "image/jpeg"
            image = types.Part.from_bytes(
                data=image_data, 
                mime_type=safe_mime_type
            )

        except Exception as e:
            logger.exception(f"Unexpected error during image download/open: {e}")
            raise APIError(f"Failed to download or open image: {str(e)}")

        # Build enhanced prompt

        enhanced_prompt = f"""
Requirements:
- Preserve the main subject identity and the important visual structure of the original image unless the prompt explicitly asks to change them.
- If the edit involves texts, render the text cleanly and consistently. Do not misspell or distort the letters.

Edit the provided image according to this instruction: {prompt}

""".strip()

        if len(enhanced_prompt) > 950:
            enhanced_prompt = enhanced_prompt[:950]

        # Image editing with specific error handling
        try:
            client = get_genai_client()

            config_kwargs = {
                "response_modalities": ["IMAGE"],
                "image_config": types.ImageConfig(
       
                ),
            }

            enable_grounding=False

            if enable_grounding:
                config_kwargs["tools"] = [{"google_search": {}}]

            max_gemini_retries = len(DEFAULT_MODEL)
            response = None

            for attempt in range(max_gemini_retries):
                try:
                    if attempt>0:
                        enhanced_prompt = "Render every text in English.\n" + enhanced_prompt[:500]
                    response = await asyncio.wait_for(
                        client.aio.models.generate_content(
                            model=DEFAULT_MODEL[attempt],
                            contents=[image, enhanced_prompt],
                            config=types.GenerateContentConfig(**config_kwargs),
                        ),
                        timeout=120,
                    )
                    break
                except Exception as e:
                    logger.warning(f"[edit_image] Gemini API attempt {attempt + 1} failed: {e}")
                    if attempt == max_gemini_retries - 1:
                        raise e
                    await asyncio.sleep(2)

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
            image_bytes = base64.b64decode(image_data_base64, validate=True)
        except Exception as e:
            raise ImageGenerationError(f"Invalid base64 image data returned by nanobanana: {str(e)}")
            

        try:
            logger.info("Trying S3 upload first")

            ext = "jpg"
            if image_bytes.startswith(b"\x89PNG"):
                ext = "png"
            elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
                ext = "webp"
                    
            temp_filename = f"{uuid.uuid4()}.{ext}"
            temp_path = os.path.join(ROOT, temp_filename)

            try:
                with open(temp_path, "wb") as f:
                    f.write(image_bytes)

                s3_response = await upload_file(local_path=temp_filename, ctx=ctx)
                uploaded_url = s3_response.url
                logger.info("S3 upload done")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path) 
                    
        except Exception as s3_error:
            logger.warning(f"s3 failed, try imgbb upload. reason: {s3_error}")
            imgbb_key = env_vars.get("IMGBB_API_KEY")
            if not imgbb_key:
                raise ImageUploadError("IMGBB_API_KEY is not configured")
                
            upload_url = "https://api.imgbb.com/1/upload"

            payload = {
                "key": env_vars['IMGBB_API_KEY'],
                "image": image_data_base64,
                "name": f"{uuid.uuid4()}"
            }

            try:
                image_size = (len(image_data_base64) * 3) // 4
                if image_size > 32 * 1024 * 1024:
                    raise ImageUploadError(f"Image too large: {image_size} bytes (max 32MB)")
                    
                max_retries = 2
                http_client = get_httpx_client()
                resp = None
                
                for attempt in range(max_retries):
                    try:
                        resp = await http_client.post(upload_url, data=payload, timeout=30.0)
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
                
            except httpx.HTTPStatusError as e:
                logger.error(f"ImgBB HTTP error: {e}")
                raise ImageUploadError(f"ImgBB HTTP error {e.response.status_code}")
            except ImageUploadError as e:
                logger.error(f"Image upload error: {e}")
                raise e
            except Exception as imgbb_error:
                raise ImageUploadError(f"upload both failed"
                                       f"S3: {s3_error};"
                                       f"ImgBB: {imgbb_error}")                          

        logger.info(f"Edited image uploaded successfully to {uploaded_url}")
        if not task_future.done():
            task_future.set_result(uploaded_url)

        return create_success_response({"url": uploaded_url})



    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        if not task_future.done():
            task_future.set_exception(e)
        edit_image_tasks.pop(cache_key, None)
        return create_error_response("validation_error", str(e))

    except asyncio.CancelledError:
        logger.info("Original edit request was cancelled")

        if task_future and not task_future.done():
            task_future.cancel("원본 이미지 편집 작업이 취소되었습니다.")
        raise
        
    except Exception as e:
        logger.exception(f"Unexpected error in editing or uploading_image: {e}")
        if not task_future.done():
            task_future.set_exception(e)
        edit_image_tasks.pop(cache_key, None)
        return create_error_response(
            "unexpected_error",
            f"Unexpected error: {str(e)}"
        )
    finally:
        edit_image_tasks.pop(cache_key, None)

def main():
    try:
        # Validate environment variables
        get_env_vars()
        
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
