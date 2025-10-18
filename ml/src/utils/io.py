import json
import os
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml
from PIL import Image


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_image(image: Image.Image, path: Path) -> None:
    ensure_dir(path.parent)
    image.save(path, format="PNG")


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def default_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def pack_outputs_to_zip(output_dir: Path, zip_path: Optional[Path] = None) -> Path:
    zip_path = zip_path or output_dir.with_suffix(".zip")
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(output_dir.parent))
    return zip_path


@contextmanager
def optional_import(module_name: str):
    try:
        module = __import__(module_name)
        yield module
    except ImportError:
        yield None


def upload_to_s3(directory: Path, bucket: str, prefix: Optional[str] = None) -> Optional[str]:
    with optional_import("boto3") as boto3:
        if boto3 is None:
            return None
        s3 = boto3.client("s3")
        prefix = prefix.rstrip("/") + "/" if prefix else ""
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                key = f"{prefix}{file_path.relative_to(directory)}"
                s3.upload_file(str(file_path), bucket, key)
        return f"s3://{bucket}/{prefix}"


def upload_to_drive(directory: Path, drive_folder_id: str) -> Optional[str]:
    with optional_import("googleapiclient.discovery") as discovery:
        if discovery is None:
            return None
        drive_service = discovery.build("drive", "v3")

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                media_body = __import__("googleapiclient.http").http.MediaFileUpload(
                    str(file_path), mimetype="image/png" if file_path.suffix == ".png" else "application/json"
                )
                drive_service.files().create(  # type: ignore[attr-defined]
                    body={
                        "name": file_path.name,
                        "parents": [drive_folder_id],
                    },
                    media_body=media_body,
                    fields="id",
                ).execute()
        return f"https://drive.google.com/drive/folders/{drive_folder_id}"
