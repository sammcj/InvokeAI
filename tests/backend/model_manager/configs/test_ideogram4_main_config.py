import json
from pathlib import Path
from tempfile import TemporaryDirectory

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


def _write_pipeline(path: Path, class_name: str) -> None:
    """Write a minimal diffusers model_index.json declaring the pipeline class."""
    (path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": class_name,
                "transformer": ["diffusers", "Ideogram4Transformer2DModel"],
                "unconditional_transformer": ["diffusers", "Ideogram4Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
                "text_encoder": ["transformers", "Qwen3VLModel"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
            }
        )
    )


def test_ideogram4_pipeline_is_classified_as_ideogram4_main_diffusers() -> None:
    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir)
        _write_pipeline(model_path, "Ideogram4Pipeline")

        result = ModelConfigFactory.from_model_on_disk(model_path, allow_unknown=True)
        config = result.config

        assert config is not None
        assert config.base == BaseModelType.Ideogram4
        assert config.type == ModelType.Main
        assert config.format == ModelFormat.Diffusers
        assert config.variant == "v4"


def test_non_ideogram4_pipeline_is_not_classified_as_ideogram4() -> None:
    with TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir)
        _write_pipeline(model_path, "Flux2Pipeline")

        result = ModelConfigFactory.from_model_on_disk(model_path, allow_unknown=True)
        config = result.config

        # The Ideogram 4 config must not claim a non-Ideogram pipeline.
        assert config is None or config.base != BaseModelType.Ideogram4
