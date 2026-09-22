"""Generate a fixed random-direction control for the refusal pilot."""

import hashlib
import json
from pathlib import Path

import torch

from activation_steering_study.steering.pilot import generate_answers, save_results
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


def sample_random_direction(hidden_size: int, norm: float, seed: int) -> torch.Tensor:
    """Return a reproducible CPU float32 direction with the requested norm."""
    # A local generator keeps this control draw from changing global torch RNG state.
    # https://docs.pytorch.org/docs/stable/generated/torch.Generator.html
    generator = torch.Generator().manual_seed(seed)
    # CPU float32 matches the extracted refusal/DiM direction.
    # https://docs.pytorch.org/docs/stable/generated/torch.randn.html
    direction = torch.randn(hidden_size, generator=generator, dtype=torch.float32)
    # Scale to the saved direction magnitude using the Euclidean vector norm.
    # https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
    return direction * (norm / torch.linalg.vector_norm(direction))


def main() -> None:
    random_seed = 42
    source_artifact = Path("artifacts/refusal-pilot.json")
    source_bytes = source_artifact.read_bytes()
    reference = json.loads(source_bytes)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    assert reference["model_id"] == MODEL_ID, "Pilot and loaded model IDs differ"
    assert reference["model_revision"] == MODEL_REVISION, "Pilot and loaded revisions differ"

    layer_index = reference["layer_index"]
    alpha = reference["alpha"]
    generation_kwargs = reference["generation_kwargs"]
    reference_norm = reference["direction_norm"]
    tokenizer, model = load_qwen()
    model_dtype = str(model.dtype)
    assert model_dtype == reference["model_dtype"], "Pilot and loaded model dtypes differ"
    direction = sample_random_direction(model.config.hidden_size, reference_norm, random_seed)

    results = generate_answers(
        model,
        tokenizer,
        reference["results"],
        {"random": direction},
        layer_index,
        alpha,
        generation_kwargs,
    )
    save_results(
        Path("artifacts/refusal-random.json"),
        {
            "source_artifact": str(source_artifact),
            "source_artifact_sha256": source_sha256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_dtype": model_dtype,
            "layer_index": layer_index,
            "alpha": alpha,
            "generation_kwargs": generation_kwargs,
            "random_seed": random_seed,
            "reference_direction_norm": reference_norm,
            "random_direction_norm": torch.linalg.vector_norm(direction).item(),
        },
        results,
    )


if __name__ == "__main__":
    main()
