#!/usr/bin/env python3
"""
MCP S3 File Uploader - A Model Context Protocol server for S3 uploads
Usage:
  python mcp_s3.py --bucket mybucket --root ~/Uploads
"""
import argparse
import mimetypes
import os
import uuid
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from fastmcp import FastMCP, Context
from pydantic import BaseModel
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--bucket", help="Target S3 bucket name (overrides .env)")
parser.add_argument("--root", required=True, help="Absolute path on disk allowed for uploads")
parser.add_argument("--access-key", required=True, help="AWS Access Key ID for local use")
parser.add_argument("--secret-key", required=True, help="AWS Secret Access Key")
args, unknown = parser.parse_known_args()

ROOT = os.path.abspath(os.path.expanduser(args.root))

# Get bucket name from CLI args or environment
BUCKET = args.bucket or os.getenv("S3_BUCKET_NAME")
if not BUCKET:
    raise ValueError("S3 bucket name must be provided via --bucket argument or S3_BUCKET_NAME in .env file")

# Initialize S3 client with credentials from environment
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


s3 = boto3.client(
    "s3",
    aws_access_key_id=args.access_key,
    aws_secret_access_key=args.secret_key,
    region_name=aws_region
)


# Initialize FastMCP
mcp = FastMCP("MCP S3 File Uploader")


class UploadResponse(BaseModel):
    url: str
    size: int
    mime_type: str
    s3_key: str


class DownloadResponse(BaseModel):
    local_path: str
    size: int
    mime_type: str
    s3_key: str


class S3Object(BaseModel):
    key: str
    size: int
    last_modified: str
    mime_type: str


class ListFilesResponse(BaseModel):
    objects: list[S3Object]
    prefix: str
    total_count: int
    truncated: bool


class FileInfoResponse(BaseModel):
    s3_key: str
    size: int
    last_modified: str
    mime_type: str
    etag: str
    storage_class: str
    server_side_encryption: Optional[str] = None
    metadata: dict = {}
    expires: Optional[str] = None
    cache_control: Optional[str] = None
    content_encoding: Optional[str] = None


def safe_join(root: str, user_path: str) -> str:
    """Safely join paths and prevent directory traversal"""
    abs_path = os.path.abspath(os.path.join(root, user_path))
    if not abs_path.startswith(root):
        raise ValueError("Path escapes allowed root")
    return abs_path


def generate_presigned_url(key: str, expires_in: int) -> str:
    """Generate a presigned URL for the uploaded file"""
    try:
        return s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": BUCKET, "Key": key},
            ExpiresIn=expires_in,
        )
    except ClientError as e:
        raise ValueError(f"Failed generating presigned URL: {e}")


async def upload_with_progress(local_path: str, bucket: str, key: str, ctx: Context):
    """Upload file with progress reporting using multipart upload for better control"""
    file_size = os.path.getsize(local_path)
    
    # Report start
    await ctx.report_progress(progress=0, total=100)
    
    try:
        # For files larger than 100MB, use multipart upload with progress
        if file_size > 100 * 1024 * 1024:  # 100MB
            # Initialize multipart upload
            response = s3.create_multipart_upload(Bucket=bucket, Key=key)
            upload_id = response['UploadId']
            
            parts = []
            part_size = 50 * 1024 * 1024  # 50MB parts
            part_number = 1
            
            with open(local_path, 'rb') as f:
                while True:
                    data = f.read(part_size)
                    if not data:
                        break
                    
                    # Upload part
                    part_response = s3.upload_part(
                        Bucket=bucket,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=data
                    )
                    
                    parts.append({
                        'ETag': part_response['ETag'],
                        'PartNumber': part_number
                    })
                    
                    # Report progress
                    progress = (part_number * part_size * 100) // file_size
                    progress = min(progress, 95)  # Cap at 95% until completion
                    await ctx.report_progress(progress=progress, total=100)
                    
                    part_number += 1
            
            # Complete multipart upload
            s3.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
        else:
            # For smaller files, use simple upload with progress checkpoints
            await ctx.report_progress(progress=25, total=100)
            s3.upload_file(local_path, bucket, key)
            await ctx.report_progress(progress=90, total=100)
    
    except ClientError as e:
        await ctx.report_progress(progress=0, total=100)  # Reset on error
        raise
    
    # Report completion
    await ctx.report_progress(progress=100, total=100)


@mcp.tool()
async def upload_file(local_path: str, ctx: Context, expires_in: int = 86400, force_overwrite: bool = False) -> UploadResponse:
    """Upload a local file to S3 using original filename and return a presigned URL with progress tracking
    
    By default, preserves original filename. If file exists in S3, upload is aborted unless force_overwrite=True.
    When force_overwrite=True, existing files are replaced (creates new S3 version if versioning enabled).
    """
    
    # Log the upload start
    await ctx.info(f"Starting upload of {local_path}")
    
    # Validate and resolve the local path
    full_local_path = safe_join(ROOT, local_path)
    if not os.path.isfile(full_local_path):
        raise ValueError("File not found")
    
    # Get file info
    file_size = os.path.getsize(full_local_path)
    original_filename = os.path.basename(full_local_path)
    s3_key = original_filename

    await ctx.info(f"File size: {file_size:,} bytes, target S3 key: {s3_key}")

    # Check for naming conflicts (unless force_overwrite is True)
    if not force_overwrite:
        try:
            # Check if file already exists
            s3.head_object(Bucket=BUCKET, Key=s3_key)
            
            # If we get here, file exists - abort upload
            await ctx.error(f"File '{s3_key}' already exists in S3. Use force_overwrite=true to replace it, or rename your local file.")
            raise ValueError(f"File '{s3_key}' already exists in S3. Use force_overwrite=true to replace it, or rename your local file.")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # File doesn't exist, safe to upload
                await ctx.info(f"S3 key '{s3_key}' is available, proceeding with upload")
            else:
                # Some other error (permissions, etc.)
                await ctx.error(f"Failed to check if file exists: {e}")
                raise ValueError(f"Failed to check if file exists: {e}")
    else:
        await ctx.info(f"Force overwrite enabled, will replace existing file if present")
    
    # Upload to S3 with progress tracking
    try:
        await upload_with_progress(full_local_path, BUCKET, s3_key, ctx)
    except ClientError as e:
        await ctx.error(f"Failed to upload file: {e}")
        raise ValueError(f"Failed to upload file: {e}")
    
    # Generate presigned URL
    presigned_url = generate_presigned_url(s3_key, expires_in)
    
    # Determine MIME type
    mime_type = mimetypes.guess_type(full_local_path)[0] or "application/octet-stream"
    
    await ctx.info(f"Upload completed successfully. Presigned URL expires in {expires_in} seconds.")
    
    return UploadResponse(
        url=presigned_url,
        size=file_size,
        mime_type=mime_type,
        s3_key=s3_key
    )


async def download_with_progress(s3_key: str, local_path: str, ctx: Context):
    """Download file with progress reporting"""
    
    # Get object info first
    try:
        response = s3.head_object(Bucket=BUCKET, Key=s3_key)
        file_size = response['ContentLength']
    except ClientError as e:
        raise ValueError(f"File not found in S3: {e}")
    
    # Report start
    await ctx.report_progress(progress=0, total=100)
    
    try:
        # For large files, use streaming download with progress
        if file_size > 50 * 1024 * 1024:  # 50MB
            response = s3.get_object(Bucket=BUCKET, Key=s3_key)
            
            with open(local_path, 'wb') as f:
                downloaded = 0
                chunk_size = 8192  # 8KB chunks
                
                for chunk in response['Body'].iter_chunks(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Report progress
                    progress = min((downloaded * 100) // file_size, 95)
                    await ctx.report_progress(progress=progress, total=100)
        else:
            # For smaller files, simple download
            await ctx.report_progress(progress=25, total=100)
            s3.download_file(BUCKET, s3_key, local_path)
            await ctx.report_progress(progress=90, total=100)
            
    except ClientError as e:
        await ctx.report_progress(progress=0, total=100)  # Reset on error
        raise ValueError(f"Failed to download file: {e}")
    
    # Report completion
    await ctx.report_progress(progress=100, total=100)


@mcp.tool()
async def download_file(s3_key: str, local_path: str, ctx: Context) -> DownloadResponse:
    """Download a file from S3 to local filesystem with progress tracking"""
    
    # Log the download start
    await ctx.info(f"Starting download of S3 key: {s3_key}")
    
    # Validate and resolve the local path
    full_local_path = safe_join(ROOT, local_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(full_local_path), exist_ok=True)
    
    # Get file info from S3
    try:
        response = s3.head_object(Bucket=BUCKET, Key=s3_key)
        file_size = response['ContentLength']
        mime_type = response.get('ContentType', 'application/octet-stream')
    except ClientError as e:
        await ctx.error(f"File not found in S3: {e}")
        raise ValueError(f"File not found in S3: {e}")
    
    await ctx.info(f"File size: {file_size:,} bytes, downloading to {local_path}")
    
    # Download from S3 with progress tracking
    try:
        await download_with_progress(s3_key, full_local_path, ctx)
    except Exception as e:
        await ctx.error(f"Failed to download file: {e}")
        raise ValueError(f"Failed to download file: {e}")
    
    await ctx.info(f"Download completed successfully to {local_path}")
    
    return DownloadResponse(
        local_path=local_path,
        size=file_size,
        mime_type=mime_type,
        s3_key=s3_key
    )


@mcp.tool()
async def list_files(prefix: str = "", max_keys: int = 100, ctx: Context = None) -> ListFilesResponse:
    """List files in S3 bucket with optional prefix filtering"""
    
    # Log the list operation
    if ctx:
        if prefix:
            await ctx.info(f"Listing files with prefix: '{prefix}' (max {max_keys})")
        else:
            await ctx.info(f"Listing all files in bucket (max {max_keys})")
    
    try:
        # Prepare S3 list objects parameters
        list_params = {
            'Bucket': BUCKET,
            'MaxKeys': max_keys
        }
        
        # Add prefix if provided
        if prefix:
            list_params['Prefix'] = prefix
        
        # List objects from S3
        response = s3.list_objects_v2(**list_params)
        
        # Process the response
        objects = []
        if 'Contents' in response:
            for obj in response['Contents']:
                # Get object metadata for MIME type
                try:
                    head_response = s3.head_object(Bucket=BUCKET, Key=obj['Key'])
                    mime_type = head_response.get('ContentType', 'application/octet-stream')
                except ClientError:
                    mime_type = 'application/octet-stream'
                
                s3_object = S3Object(
                    key=obj['Key'],
                    size=obj['Size'],
                    last_modified=obj['LastModified'].isoformat(),
                    mime_type=mime_type
                )
                objects.append(s3_object)
        
        total_count = len(objects)
        truncated = response.get('IsTruncated', False)
        
        if ctx:
            await ctx.info(f"Found {total_count} objects" + (" (truncated)" if truncated else ""))
        
        return ListFilesResponse(
            objects=objects,
            prefix=prefix,
            total_count=total_count,
            truncated=truncated
        )
        
    except ClientError as e:
        if ctx:
            await ctx.error(f"Failed to list files: {e}")
        raise ValueError(f"Failed to list files: {e}")


@mcp.tool()
async def get_file_info(s3_key: str, ctx: Context) -> FileInfoResponse:
    """Get detailed information about a file in S3"""
    
    # Log the info operation
    await ctx.info(f"Getting info for S3 key: {s3_key}")
    
    try:
        # Get object metadata using head_object
        response = s3.head_object(Bucket=BUCKET, Key=s3_key)
        
        # Extract basic information
        size = response['ContentLength']
        last_modified = response['LastModified'].isoformat()
        mime_type = response.get('ContentType', 'application/octet-stream')
        etag = response['ETag'].strip('"')  # Remove quotes from ETag
        storage_class = response.get('StorageClass', 'STANDARD')
        
        # Extract optional metadata
        server_side_encryption = response.get('ServerSideEncryption')
        expires = response.get('Expires')
        expires_str = expires.isoformat() if expires else None
        cache_control = response.get('CacheControl')
        content_encoding = response.get('ContentEncoding')
        
        # Extract custom metadata (user-defined metadata has 'x-amz-meta-' prefix)
        metadata = {}
        if 'Metadata' in response:
            metadata = response['Metadata']
        
        await ctx.info(f"File info retrieved: {size:,} bytes, {storage_class} storage")
        
        return FileInfoResponse(
            s3_key=s3_key,
            size=size,
            last_modified=last_modified,
            mime_type=mime_type,
            etag=etag,
            storage_class=storage_class,
            server_side_encryption=server_side_encryption,
            metadata=metadata,
            expires=expires_str,
            cache_control=cache_control,
            content_encoding=content_encoding
        )
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            await ctx.error(f"File not found: {s3_key}")
            raise ValueError(f"File not found: {s3_key}")
        elif error_code == 'Forbidden':
            await ctx.error(f"Access denied to file: {s3_key}")
            raise ValueError(f"Access denied to file: {s3_key}")
        else:
            await ctx.error(f"Failed to get file info: {e}")
            raise ValueError(f"Failed to get file info: {e}")


def main():
    """Main entry point for the MCP S3 server"""
    mcp.run()


if __name__ == "__main__":
    main()
