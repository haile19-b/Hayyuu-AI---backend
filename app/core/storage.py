import anyio
import boto3
from botocore.config import Config
from app.core.env import settings

class StorageUtility:
    def __init__(self):
        # Configure client with endpoint, key, secret.
        # Cloudflare R2 requires standard signature version v4.
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=settings.STORAGE_ENDPOINT_URL,
            aws_access_key_id=settings.STORAGE_ACCESS_KEY_ID,
            aws_secret_access_key=settings.STORAGE_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )
        self.bucket_name = settings.STORAGE_BUCKET_NAME

    def _upload_file_sync(self, file_content: bytes, file_path: str, content_type: str) -> None:
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=file_path,
            Body=file_content,
            ContentType=content_type,
        )

    def _delete_file_sync(self, file_path: str) -> None:
        self.s3_client.delete_object(
            Bucket=self.bucket_name,
            Key=file_path,
        )

    def _generate_download_url_sync(self, file_path: str, expiration: int) -> str:
        return self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": file_path},
            ExpiresIn=expiration,
        )

    async def upload_file(self, file_content: bytes, file_path: str, content_type: str) -> None:
        await anyio.to_thread.run_sync(
            self._upload_file_sync, file_content, file_path, content_type
        )

    async def delete_file(self, file_path: str) -> None:
        await anyio.to_thread.run_sync(
            self._delete_file_sync, file_path
        )

    async def generate_download_url(self, file_path: str, expiration: int = 3600) -> str:
        return await anyio.to_thread.run_sync(
            self._generate_download_url_sync, file_path, expiration
        )

storage_utility = StorageUtility()
