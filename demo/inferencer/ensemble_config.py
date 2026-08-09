from pathlib import Path

from omegaconf import OmegaConf


REQUIRED_STREAM_FIELDS = ("config", "checkpoint", "weight", "temperature")


def _resolve_path(value, project_root):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _as_float(value, field, stream_name):
    if isinstance(value, bool):
        raise ValueError(f"Stream '{stream_name}' {field} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Stream '{stream_name}' {field} must be numeric, got {value!r}"
        ) from exc


def load_ensemble_streams(config_path, project_root, expected_num_classes=22):
    """Load and validate demo ensemble streams with project-relative paths."""
    config_path = Path(config_path)
    project_root = Path(project_root).resolve()
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"{config_path} must define a non-empty 'streams' list")

    unset = []
    for index, stream in enumerate(streams):
        if not isinstance(stream, dict):
            raise ValueError(f"streams[{index}] must be a mapping")
        for field in REQUIRED_STREAM_FIELDS:
            if stream.get(field) in (None, ""):
                unset.append(f"streams[{index}].{field}")
    if unset:
        raise ValueError(
            "Ensemble config contains unset required values: " + ", ".join(unset)
        )

    resolved_streams = []
    for index, stream in enumerate(streams):
        name = str(stream.get("name") or f"stream_{index}")
        model_config = _resolve_path(stream["config"], project_root)
        checkpoint = _resolve_path(stream["checkpoint"], project_root)
        if not model_config.is_file():
            raise FileNotFoundError(f"Stream '{name}' config not found: {model_config}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Stream '{name}' checkpoint not found: {checkpoint}")

        model_cfg = OmegaConf.load(model_config)
        num_classes = int(model_cfg.model.cls_head.num_classes)
        if num_classes != expected_num_classes:
            raise ValueError(
                f"Stream '{name}' config has num_classes={num_classes}; "
                f"expected {expected_num_classes}"
            )

        weight = _as_float(stream["weight"], "weight", name)
        temperature = _as_float(stream["temperature"], "temperature", name)
        if weight < 0:
            raise ValueError(f"Stream '{name}' weight must be non-negative")
        if temperature <= 0:
            raise ValueError(f"Stream '{name}' temperature must be positive")

        resolved_streams.append(
            {
                "name": name,
                "config": str(model_config),
                "checkpoint": str(checkpoint),
                "weight": weight,
                "temperature": temperature,
            }
        )

    if sum(stream["weight"] for stream in resolved_streams) <= 0:
        raise ValueError("At least one ensemble stream weight must be positive")
    return resolved_streams
