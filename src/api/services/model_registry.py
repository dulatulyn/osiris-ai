from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.config import ARTIFACTS_DIR, REGISTRY_PATH


class ModelRegistry:
    """Manages named model profiles (name → artifact directory path).

    Profiles are stored in artifacts/registry.json. Any subdirectory of
    artifacts/ that contains pipeline.joblib is auto-discovered even if not
    registered explicitly.

    Usage:
        registry.list_profiles()           → dict with 'active' + 'profiles'
        registry.set_active("v2")          → hot-swap active profile
        registry.get_profile_path("v1")    → Path to the artifact dir
        registry.register("v3", "/path")   → add a new profile
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not REGISTRY_PATH.exists():
            default = {"active": "v1", "profiles": {}}
            with open(REGISTRY_PATH, "w") as f:
                json.dump(default, f, indent=2)

    def _load(self) -> dict:
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        with open(REGISTRY_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def list_profiles(self) -> dict:
        """Return all profiles, including auto-discovered ones."""
        with self._lock:
            data = self._load()
            profiles: dict = data.get("profiles", {})

            # Auto-discover artifact subdirs with pipeline.joblib
            if ARTIFACTS_DIR.exists():
                for subdir in sorted(ARTIFACTS_DIR.iterdir()):
                    if (
                        subdir.is_dir()
                        and (subdir / "pipeline.joblib").exists()
                        and subdir.name not in profiles
                    ):
                        profiles[subdir.name] = {
                            "path": str(subdir),
                            "description": f"Auto-discovered profile '{subdir.name}'",
                            "created_at": datetime.fromtimestamp(
                                subdir.stat().st_mtime, tz=timezone.utc
                            ).isoformat(),
                        }

            return {"active": data.get("active", "v1"), "profiles": profiles}

    def get_active(self) -> str:
        with self._lock:
            return self._load().get("active", "v1")

    def set_active(self, profile_name: str) -> None:
        """Set the active profile (does NOT load the model, call prediction_service.load separately)."""
        with self._lock:
            profiles = self.list_profiles()["profiles"]
            if profile_name not in profiles:
                available = list(profiles.keys())
                raise ValueError(
                    f"Profile '{profile_name}' not found. Available: {available}"
                )
            data = self._load()
            data["active"] = profile_name
            self._save(data)

    def get_profile_path(self, profile_name: str | None = None) -> Path:
        """Resolve the artifact directory Path for a profile."""
        if profile_name is None:
            profile_name = self.get_active()

        profiles = self.list_profiles()["profiles"]
        if profile_name not in profiles:
            # Try treating it as a direct subdirectory name
            fallback = ARTIFACTS_DIR / profile_name
            if fallback.exists():
                return fallback
            raise ValueError(
                f"Profile '{profile_name}' not found. Available: {list(profiles.keys())}"
            )

        path_str = profiles[profile_name]["path"]
        path = Path(path_str)
        if not path.is_absolute():
            path = ARTIFACTS_DIR.parent / path
        return path

    def register(self, name: str, path: str, description: str = "") -> None:
        """Register a new profile or update an existing one."""
        with self._lock:
            data = self._load()
            if "profiles" not in data:
                data["profiles"] = {}
            data["profiles"][name] = {
                "path": path,
                "description": description,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
            }
            self._save(data)


model_registry = ModelRegistry()
