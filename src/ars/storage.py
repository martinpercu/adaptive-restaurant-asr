"""Storage abstraction (plan/phases/phase-0 §0.3, plan/02-architecture.md §5).

`Storage` is the interface; `LocalStorage` is the phase-0 implementation.
`S3Storage` (MinIO, boto3) sits behind the same interface and is used only when
`ARS_S3_ENDPOINT` is set. All paths are storage-relative (POSIX, forward slashes).
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path


class Storage(ABC):
    """Minimal object-store interface. Keys are relative POSIX paths."""

    @abstractmethod
    def put(self, path: str, data: bytes) -> str: ...

    @abstractmethod
    def get(self, path: str) -> bytes: ...

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def url(self, path: str) -> str: ...


class LocalStorage(Storage):
    """Filesystem-backed storage rooted at a base directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _abs(self, path: str) -> Path:
        # Prevent escaping the root via absolute paths or `..`.
        rel = Path(path)
        if rel.is_absolute():
            raise ValueError(f"storage paths must be relative, got: {path!r}")
        target = (self.root / rel).resolve()
        if self.root not in target.parents and target != self.root:
            raise ValueError(f"path escapes storage root: {path!r}")
        return target

    def put(self, path: str, data: bytes) -> str:
        target = self._abs(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return path

    def get(self, path: str) -> bytes:
        return self._abs(path).read_bytes()

    def exists(self, path: str) -> bool:
        return self._abs(path).exists()

    def list(self, prefix: str = "") -> list[str]:
        base = self._abs(prefix) if prefix else self.root
        if not base.exists():
            return []
        roots = [base] if base.is_dir() else [base.parent]
        out: list[str] = []
        for r in roots:
            for p in r.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(self.root).as_posix()
                    if rel.startswith(prefix):
                        out.append(rel)
        return sorted(out)

    def url(self, path: str) -> str:
        return self._abs(path).as_uri()


class S3Storage(Storage):
    """S3-compatible storage (MinIO). Credentials/endpoint from env (ARS_S3_*).

    Imported lazily so boto3 is not a hard phase-0 dependency; only constructed
    when ARS_S3_ENDPOINT is configured (smoke-tested, not part of CI).
    """

    def __init__(
        self,
        bucket: str | None = None,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        import boto3  # noqa: PLC0415  (lazy: optional dependency)

        self.bucket = bucket or os.environ["ARS_S3_BUCKET"]
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or os.environ.get("ARS_S3_ENDPOINT"),
            aws_access_key_id=access_key or os.environ.get("ARS_S3_ACCESS_KEY"),
            aws_secret_access_key=secret_key or os.environ.get("ARS_S3_SECRET_KEY"),
        )

    def put(self, path: str, data: bytes) -> str:
        self._client.put_object(Bucket=self.bucket, Key=path, Body=data)
        return path

    def get(self, path: str) -> bytes:
        return self._client.get_object(Bucket=self.bucket, Key=path)["Body"].read()

    def exists(self, path: str) -> bool:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        try:
            self._client.head_object(Bucket=self.bucket, Key=path)
            return True
        except ClientError:
            return False

    def list(self, prefix: str = "") -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            out.extend(obj["Key"] for obj in page.get("Contents", []))
        return sorted(out)

    def url(self, path: str) -> str:
        return self._client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": path}, ExpiresIn=3600
        )


def get_storage(root: str | Path, backend: str = "local") -> Storage:
    """Factory: LocalStorage by default; S3Storage when backend=='s3'."""
    if backend == "s3" or os.environ.get("ARS_S3_ENDPOINT"):
        return S3Storage()
    return LocalStorage(root)


def copy_into(storage: Storage, src: str | Path, dest_key: str) -> str:
    """Convenience: copy a local file into storage under dest_key."""
    if isinstance(storage, LocalStorage):
        target = storage._abs(dest_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        return dest_key
    return storage.put(dest_key, Path(src).read_bytes())
