import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone

from mcp_nano_banana.mcp_s3 import BUCKET, s3

logger = logging.getLogger(__name__)


async def save_gemini_error_log(error, *, model, attempt, request_id):
    """Save the Gemini SDK error response, not the request or prompt, to S3.

    Best effort: a logging failure must never alter model fallback behavior.
    """
    if s3 is None or not BUCKET:
        logger.warning("Gemini error log skipped: S3 client or bucket missing")
        return

    try:
        raw_details = getattr(error, "details", None)
        if isinstance(raw_details, str):
            try:
                raw_details = json.loads(raw_details)
            except ValueError:
                pass
        if isinstance(raw_details, list) and len(raw_details) == 1:
            raw_details = raw_details[0]

        api_error = raw_details.get("error", raw_details) if isinstance(raw_details, dict) else {}
        if not isinstance(api_error, dict):
            api_error = {}
        entries = api_error.get("details", [])
        if not isinstance(entries, list):
            entries = []

        quota_violations = []
        retry_delay = None
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            kind = entry.get("@type", "")
            if kind.endswith("google.rpc.QuotaFailure"):
                quota_violations.extend(entry.get("violations", []))
            elif kind.endswith("google.rpc.RetryInfo"):
                retry_delay = entry.get("retryDelay")

        http_response = getattr(error, "response", None)
        headers = getattr(http_response, "headers", None)
        message = getattr(error, "message", None) or str(error)
        if isinstance(error, asyncio.TimeoutError) and not message:
            message = "Local Gemini API call timed out after 120 seconds"

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "project_label": os.getenv("GCP_PROJECT_LABEL", "unknown"),
            "operation": "generate_image",
            "model": model,
            "attempt": attempt,
            "exception_type": type(error).__name__,
            "http_status": getattr(error, "code", None) or api_error.get("code"),
            "google_status": getattr(error, "status", None) or api_error.get("status"),
            "message": message,
            "sdk_error_details": raw_details,
            "quota_violations": quota_violations,
            "retry_delay": retry_delay,
            "retry_after_header": headers.get("retry-after") if headers else None,
        }

        # Never put the prompt, API keys, or credentials in the diagnostic log.
        body = json.dumps(record, ensure_ascii=False, default=str, indent=2)
        for name in (
            "NANOBANANA_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
            "IMGBB_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        ):
            secret = os.getenv(name, "")
            for value in secret.split("++"):
                if len(value) >= 12:
                    body = body.replace(value, "[REDACTED]")

        now = datetime.now(timezone.utc)
        key = (
            f"gemini-error-logs/{now:%Y/%m/%d}/"
            f"{now:%H%M%S_%f}_{request_id}_attempt{attempt}_{uuid.uuid4().hex[:8]}.json"
        )
        await asyncio.wait_for(
            asyncio.to_thread(
                s3.put_object,
                Bucket=BUCKET,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/json; charset=utf-8",
            ),
            timeout=10,
        )
        logger.info("Gemini error details saved to S3: %s", key)
    except Exception as log_error:
        logger.warning("Could not save Gemini error details to S3: %s", log_error)
