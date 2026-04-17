import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

from prediction.interfaces import ModelRegistry, ResidualModel


class FileModelRegistry(ModelRegistry):
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def _artifact_dir(self, road_id: str, granularity: str) -> Path:
        safe_road = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(road_id))
        return self.base_dir / safe_road / granularity

    def save(self, road_id: str, granularity: str, model: ResidualModel, metadata: Dict) -> str:
        artifact_dir = self._artifact_dir(road_id, granularity)
        os.makedirs(artifact_dir, exist_ok=True)

        model_path = artifact_dir / "model.pkl"
        metadata_path = artifact_dir / "metadata.json"

        model.save(str(model_path))

        version = datetime.now().strftime("%Y%m%d%H%M%S")
        metadata_to_save = {
            "version": version,
            "road_id": road_id,
            "granularity": granularity,
            **metadata,
        }
        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata_to_save, file_obj, ensure_ascii=False, indent=2)

        return version
