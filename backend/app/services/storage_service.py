import os
import hashlib
import logging
from typing import Optional, Dict, Any, BinaryIO
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config
import minio
from minio.error import S3Error
from app.core.config import settings

logger = logging.getLogger(__name__)

class StorageService:
    """
    Secure storage service supporting both AWS S3 and MinIO S3-compatible storage.
    Handles audio file storage with encryption, checksums, and metadata.
    """
    
    def __init__(self):
        self.storage_type = self._detect_storage_type()
        self.client = self._initialize_client()
        self.bucket_name = settings.S3_BUCKET_NAME
        self._ensure_bucket_exists()
    
    def _detect_storage_type(self) -> str:
        """Detect whether to use AWS S3 or MinIO based on endpoint URL."""
        if settings.S3_ENDPOINT_URL and 'localhost' in settings.S3_ENDPOINT_URL:
            return 'minio'
        return 'aws'
    
    def _initialize_client(self):
        """Initialize the appropriate storage client."""
        try:
            if self.storage_type == 'minio':
                return self._initialize_minio_client()
            else:
                return self._initialize_s3_client()
        except Exception as e:
            logger.error(f"Failed to initialize storage client: {e}")
            raise
    
    def _initialize_s3_client(self):
        """Initialize AWS S3 client with encryption and security settings."""
        config = Config(
            region_name=settings.S3_REGION,
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            max_pool_connections=50
        )
        
        if settings.S3_ENDPOINT_URL:
            return boto3.client(
                's3',
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY_ID,
                aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
                config=config
            )
        else:
            return boto3.client('s3', config=config)
    
    def _initialize_minio_client(self):
        """Initialize MinIO client for local development."""
        endpoint = settings.S3_ENDPOINT_URL.replace('http://', '').replace('https://', '')
        secure = settings.S3_ENDPOINT_URL.startswith('https://')
        
        return minio.Minio(
            endpoint,
            access_key=settings.S3_ACCESS_KEY_ID,
            secret_key=settings.S3_SECRET_ACCESS_KEY,
            secure=secure
        )
    
    def _ensure_bucket_exists(self):
        """Ensure the storage bucket exists with proper configuration."""
        try:
            if self.storage_type == 'minio':
                if not self.client.bucket_exists(self.bucket_name):
                    self.client.make_bucket(self.bucket_name)
                    logger.info(f"Created MinIO bucket: {self.bucket_name}")
            else:
                try:
                    self.client.head_bucket(Bucket=self.bucket_name)
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        self.client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={
                                'LocationConstraint': settings.S3_REGION
                            } if settings.S3_REGION != 'us-east-1' else {}
                        )
                        
                        # Enable versioning and encryption
                        self.client.put_bucket_versioning(
                            Bucket=self.bucket_name,
                            VersioningConfiguration={'Status': 'Enabled'}
                        )
                        
                        self.client.put_bucket_encryption(
                            Bucket=self.bucket_name,
                            ServerSideEncryptionConfiguration={
                                'Rules': [{
                                    'ApplyServerSideEncryptionByDefault': {
                                        'SSEAlgorithm': 'AES256'
                                    }
                                }]
                            }
                        )
                        logger.info(f"Created S3 bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to ensure bucket exists: {e}")
            raise
    
    def upload_audio_file(
        self, 
        file_data: BinaryIO, 
        session_id: str, 
        task_name: str,
        participant_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload audio file with encryption, checksum validation, and metadata.
        
        Args:
            file_data: Binary audio file data
            session_id: Session identifier
            task_name: Recording task name
            participant_id: Participant identifier
            metadata: Additional metadata to store
            
        Returns:
            Dict containing file info, checksum, and storage location
        """
        try:
            # Generate secure file path
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            file_extension = self._detect_audio_format(file_data)
            object_key = f"audio/{participant_id}/{session_id}/{task_name}_{timestamp}.{file_extension}"
            
            # Calculate checksum
            file_data.seek(0)
            file_content = file_data.read()
            checksum = hashlib.sha256(file_content).hexdigest()
            file_size = len(file_content)
            
            # Prepare metadata
            upload_metadata = {
                'session-id': session_id,
                'task-name': task_name,
                'participant-id': participant_id,
                'upload-timestamp': datetime.utcnow().isoformat(),
                'file-size': str(file_size),
                'checksum-sha256': checksum,
                'content-type': f'audio/{file_extension}'
            }
            
            if metadata:
                for key, value in metadata.items():
                    upload_metadata[f'custom-{key}'] = str(value)
            
            # Upload file
            file_data.seek(0)
            if self.storage_type == 'minio':
                self.client.put_object(
                    self.bucket_name,
                    object_key,
                    file_data,
                    length=file_size,
                    content_type=f'audio/{file_extension}',
                    metadata=upload_metadata
                )
            else:
                self.client.upload_fileobj(
                    file_data,
                    self.bucket_name,
                    object_key,
                    ExtraArgs={
                        'Metadata': upload_metadata,
                        'ContentType': f'audio/{file_extension}',
                        'ServerSideEncryption': 'AES256'
                    }
                )
            
            logger.info(f"Successfully uploaded audio file: {object_key}")
            
            return {
                'object_key': object_key,
                'checksum': checksum,
                'file_size': file_size,
                'upload_timestamp': datetime.utcnow(),
                'storage_url': self._generate_storage_url(object_key),
                'metadata': upload_metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to upload audio file: {e}")
            raise
    
    def download_audio_file(self, object_key: str) -> bytes:
        """Download and verify audio file integrity."""
        try:
            if self.storage_type == 'minio':
                response = self.client.get_object(self.bucket_name, object_key)
                file_data = response.read()
            else:
                response = self.client.get_object(Bucket=self.bucket_name, Key=object_key)
                file_data = response['Body'].read()
            
            # Verify checksum if available
            metadata = self.get_file_metadata(object_key)
            if 'checksum-sha256' in metadata:
                calculated_checksum = hashlib.sha256(file_data).hexdigest()
                stored_checksum = metadata['checksum-sha256']
                if calculated_checksum != stored_checksum:
                    raise ValueError(f"Checksum mismatch for {object_key}")
            
            logger.info(f"Successfully downloaded and verified: {object_key}")
            return file_data
            
        except Exception as e:
            logger.error(f"Failed to download audio file {object_key}: {e}")
            raise
    
    def get_file_metadata(self, object_key: str) -> Dict[str, str]:
        """Retrieve file metadata."""
        try:
            if self.storage_type == 'minio':
                stat = self.client.stat_object(self.bucket_name, object_key)
                return stat.metadata or {}
            else:
                response = self.client.head_object(Bucket=self.bucket_name, Key=object_key)
                return response.get('Metadata', {})
                
        except Exception as e:
            logger.error(f"Failed to get metadata for {object_key}: {e}")
            return {}
    
    def generate_presigned_url(
        self, 
        object_key: str, 
        expiration: int = 3600,
        method: str = 'GET'
    ) -> str:
        """Generate presigned URL for secure file access."""
        try:
            if self.storage_type == 'minio':
                from datetime import timedelta
                return self.client.presigned_get_object(
                    self.bucket_name, 
                    object_key, 
                    expires=timedelta(seconds=expiration)
                )
            else:
                return self.client.generate_presigned_url(
                    method.lower() + '_object',
                    Params={'Bucket': self.bucket_name, 'Key': object_key},
                    ExpiresIn=expiration
                )
                
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {object_key}: {e}")
            raise
    
    def delete_file(self, object_key: str) -> bool:
        """Securely delete file from storage."""
        try:
            if self.storage_type == 'minio':
                self.client.remove_object(self.bucket_name, object_key)
            else:
                self.client.delete_object(Bucket=self.bucket_name, Key=object_key)
            
            logger.info(f"Successfully deleted file: {object_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {object_key}: {e}")
            return False
    
    def list_session_files(self, session_id: str) -> list:
        """List all files for a specific session."""
        try:
            prefix = f"audio/{session_id}/"
            files = []
            
            if self.storage_type == 'minio':
                objects = self.client.list_objects(self.bucket_name, prefix=prefix)
                for obj in objects:
                    files.append({
                        'key': obj.object_name,
                        'size': obj.size,
                        'last_modified': obj.last_modified,
                        'etag': obj.etag
                    })
            else:
                response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
                for obj in response.get('Contents', []):
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'],
                        'etag': obj['ETag']
                    })
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files for session {session_id}: {e}")
            return []
    
    def _detect_audio_format(self, file_data: BinaryIO) -> str:
        """Detect audio file format from file header."""
        file_data.seek(0)
        header = file_data.read(12)
        file_data.seek(0)
        
        if header.startswith(b'RIFF') and b'WAVE' in header:
            return 'wav'
        elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
            return 'mp3'
        elif header.startswith(b'OggS'):
            return 'ogg'
        elif header.startswith(b'fLaC'):
            return 'flac'
        else:
            return 'wav'  # Default fallback
    
    def _generate_storage_url(self, object_key: str) -> str:
        """Generate internal storage URL reference."""
        return f"storage://{self.bucket_name}/{object_key}"
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            total_size = 0
            file_count = 0
            
            if self.storage_type == 'minio':
                objects = self.client.list_objects(self.bucket_name, recursive=True)
                for obj in objects:
                    total_size += obj.size
                    file_count += 1
            else:
                paginator = self.client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=self.bucket_name):
                    for obj in page.get('Contents', []):
                        total_size += obj['Size']
                        file_count += 1
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'file_count': file_count,
                'bucket_name': self.bucket_name,
                'storage_type': self.storage_type
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {
                'total_size_bytes': 0,
                'total_size_mb': 0,
                'file_count': 0,
                'bucket_name': self.bucket_name,
                'storage_type': self.storage_type,
                'error': str(e)
            }

# Global storage service instance
storage_service = StorageService()
