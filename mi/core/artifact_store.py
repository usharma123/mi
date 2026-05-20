from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from mi.core.schema import RunManifest

T = TypeVar("T", bound=BaseModel)


def slugify(value: str, fallback: str = "trace") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:48] or fallback


def default_run_dir(prompt: str, base_dir: Path = Path("runs")) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base_dir / f"{timestamp}-{slugify(prompt)}"


class ArtifactStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    @classmethod
    def create(cls, root: Path | str) -> "ArtifactStore":
        store = cls(root)
        store.root.mkdir(parents=True, exist_ok=True)
        return store

    @property
    def run_id(self) -> str:
        return self.root.name

    def path(self, name: str) -> Path:
        return self.root / name

    def relative_ref(self, path: Path | str) -> str:
        return Path(path).relative_to(self.root).as_posix()

    def write_json(self, name: str, payload: BaseModel | dict[str, Any] | list[Any]) -> Path:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, BaseModel):
            text = payload.model_dump_json(indent=2)
        else:
            text = json.dumps(payload, indent=2, sort_keys=True)
        path.write_text(text + "\n", encoding="utf-8")
        return path

    def read_json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def read_model(self, name: str, model_type: type[T]) -> T:
        return model_type.model_validate(self.read_json(name))

    def write_text(self, name: str, text: str) -> Path:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_manifest(self, manifest: RunManifest) -> Path:
        return self.write_json("manifest.json", manifest)

    def read_manifest(self) -> RunManifest:
        return self.read_model("manifest.json", RunManifest)
