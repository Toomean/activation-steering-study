"""Frozen exploratory refusal-domain extraction and condition runner."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import cast

import torch

from activation_steering_study.analysis.geometry import analyze_geometry
from activation_steering_study.data.mmlu import LABELS, SOURCE_PATH, sample_mmlu
from activation_steering_study.data.refusal import SampleRefusalItem, load_reviewed_refusal
from activation_steering_study.data.refusal_domains import load_domain_extraction, source_hashes
from activation_steering_study.evaluation.coherence import prompt_distribution_kl, response_likelihood
from activation_steering_study.evaluation.mmlu import render_question, score_question
from activation_steering_study.evaluation.refusal import has_refusal_substring
from activation_steering_study.extraction.mean import mean_activation
from activation_steering_study.steering.choice_sweep import (
    ALPHAS as CHOICE_ALPHAS, LAYERS as CHOICE_LAYERS, SOURCE_REVISIONS, TARGET_NORM,
    load_sweep_sources, score_choice_condition, sweep_code_hashes,
)
from activation_steering_study.steering.pilot import GenerationKwargs, generate_completion
from activation_steering_study.steering.random_control import sample_random_direction
from activation_steering_study.utils.choice_prompt import ANSWER_SUFFIX
from activation_steering_study.utils.json_io import save_json
from activation_steering_study.utils.qwen import MODEL_ID, MODEL_REVISION, load_qwen


LAYER = 14
ALPHAS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
DOMAINS = ("cyber_intrusion", "dangerous_substances", "disinformation")
TARGETS = ("pooled", *DOMAINS)
CONTROLS = ("baseline", "direction", "random", "candidate-sycophancy", "candidate-honesty")
DIRECTION_PATH = Path("artifacts/refusal-domains.pt")
EXTRACTION_PATH = Path("artifacts/refusal-domains.json")
CONDITIONS_DIR = Path("artifacts/refusal-conditions")
GENERATION: GenerationKwargs = {"max_new_tokens": 256, "do_sample": False}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    modules = (
        "analysis/geometry.py", "data/mmlu.py", "data/refusal.py", "data/refusal_domains.py",
        "evaluation/choices.py", "evaluation/coherence.py",
        "evaluation/mmlu.py", "evaluation/refusal.py", "evaluation/scoring.py",
        "evaluation/sycophancy.py", "extraction/mean.py", "steering/choice_sweep.py",
        "steering/generation.py", "steering/intervention.py", "steering/random_control.py",
        "steering/pilot.py", "steering/refusal_domain.py", "utils/qwen.py",
        "utils/choice_prompt.py", "utils/json_io.py",
    )
    paths = [Path("src/activation_steering_study") / module for module in modules]
    return {str(path): sha256(path) for path in paths}


def direction_from_rows(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Keep the raw harmful-minus-shared-harmless DiM in FP32."""
    direction = harmful.float().mean(0) - harmless.float().mean(0)
    if not bool(torch.isfinite(direction).all()) or direction.norm().item() <= 0:
        raise ValueError("Zero or nonfinite refusal direction")
    return direction


def extract() -> None:
    domains, harmless_items = load_domain_extraction()
    reviewed = load_reviewed_refusal()
    pooled_items = reviewed["train"]["harmful"]
    if len(pooled_items) != 28:
        raise ValueError("Expected 28 reviewed pooled harmful rows")
    tokenizer, model = load_qwen()
    harmless, harmless_mean = mean_activation(
        model, tokenizer, [row["instruction"] for row in harmless_items], LAYER
    )
    rows: dict[str, torch.Tensor] = {"harmless": harmless.cpu()}
    means: dict[str, torch.Tensor] = {"harmless": harmless_mean.cpu()}
    directions: dict[str, torch.Tensor] = {}
    ids: dict[str, list[SampleRefusalItem]] = {"harmless": list(harmless_items)}
    for name, items in (("pooled", pooled_items), *domains.items()):
        captured, harmful_mean = mean_activation(
            model, tokenizer, [row["instruction"] for row in items], LAYER
        )
        rows[name] = captured.cpu()
        means[name] = harmful_mean.cpu()
        directions[name] = direction_from_rows(captured, harmless).cpu()
        ids[name] = list(items)
        print(f"{name}: {len(items)} rows, raw norm {directions[name].norm().item():.6f}", flush=True)
    geometry = analyze_geometry({name: rows[name] for name in DOMAINS}, harmless)
    DIRECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "means": means, "directions": directions}, DIRECTION_PATH)
    EXTRACTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(EXTRACTION_PATH, {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_dtype": str(model.dtype), "layer_index": LAYER,
        "capture": "FP32 block output at final rendered prompt token",
        "formula": "mean harmful - same 32-row reviewed harmless mean",
        "source_ids": ids, "source_sha256": source_hashes(),
        "code_sha256": code_hashes(), "tensor_path": str(DIRECTION_PATH),
        "tensor_sha256": sha256(DIRECTION_PATH),
        "direction_norms": {name: direction.norm().item() for name, direction in directions.items()},
        "geometry": geometry, "status": "exploratory; provisional row labels and semantic overlap",
    })


def condition_key(target: str, alpha: float, control: str) -> str:
    if control == "baseline":
        return "baseline"
    sign = "positive" if alpha > 0 else "negative"
    magnitude = str(abs(alpha)).replace(".", "p")
    return f"{target}-{control}-{sign}-{magnitude}"


def paired_accuracy(baseline: list[dict[str, object]], current: list[dict[str, object]]) -> dict[str, object]:
    if len(baseline) != len(current) or [r["id"] for r in baseline] != [r["id"] for r in current]:
        raise ValueError("MMLU source IDs differ from baseline")
    before = [bool(row["correct"]) for row in baseline]
    after = [bool(row["correct"]) for row in current]
    gains = sum(not b and a for b, a in zip(before, after, strict=True))
    losses = sum(b and not a for b, a in zip(before, after, strict=True))
    return {"correct": sum(after), "total": len(after), "baseline_correct": sum(before),
            "wrong_to_correct": gains, "correct_to_wrong": losses,
            "paired_accuracy_change": (gains - losses) / len(after),
            "mean_option_mass": sum([cast(float, row["option_mass"]) for row in current], 0.0) / len(current)}


def _load_extraction() -> tuple[dict[str, object], dict[str, object]]:
    metadata = json.loads(EXTRACTION_PATH.read_text(encoding="utf-8"))
    if (metadata["model_id"] != MODEL_ID or metadata["model_revision"] != MODEL_REVISION
            or metadata["layer_index"] != LAYER or metadata["source_sha256"] != source_hashes()
            or metadata["code_sha256"] != code_hashes()
            or metadata["tensor_sha256"] != sha256(DIRECTION_PATH)):
        raise ValueError("Extraction provenance changed; rerun extract in a fresh snapshot")
    tensor = torch.load(DIRECTION_PATH, map_location="cpu", weights_only=True)
    return metadata, tensor


def _validate_candidate_sweep(
    sweep: dict[str, object], behaviour: str, tensor_path: Path, model_dtype: str
) -> None:
    """Reject stale or mismatched block-14 A/B evidence before refusal generation."""
    prompts_path, source_path, samples = load_sweep_sources(behaviour)
    development_ids = [sample["source_index"] for sample in samples
                       if sample["split"] == "development"]
    baseline = cast(dict[str, object], cast(dict[str, object], sweep["conditions"])["baseline"])
    baseline_ids = [row["source_index"] for row in cast(list[dict[str, object]], baseline["results"])]
    expected = {
        "behaviour": behaviour, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_dtype": model_dtype, "source_revision": SOURCE_REVISIONS[behaviour],
        "source_path": str(source_path), "source_sha256": sha256(source_path),
        "selected_path": str(prompts_path), "selected_sha256": sha256(prompts_path),
        "code_sha256": sweep_code_hashes(),
        "answer_suffix": ANSWER_SUFFIX,
        "hook_policy": "final prompt token",
        "capture_policy": "completed bare answer letter final token; float32 block outputs",
        "pair_formula": "source-labelled matching answer activation - opposite answer activation",
        "aggregation": "mean of original/swapped within source, then mean across sources",
        "layers": list(CHOICE_LAYERS), "screen_alpha": 1.0,
        "selected_alphas": list(CHOICE_ALPHAS), "target_norm": TARGET_NORM,
        "random_seed": 42, "tensor_path": str(tensor_path),
        "tensor_sha256": sha256(tensor_path),
        "extraction_count": sum(sample["split"] == "extraction" for sample in samples),
        "development_count": len(development_ids),
    }
    for field, current in expected.items():
        if sweep.get(field) != current:
            raise ValueError(f"Candidate {behaviour} sweep has stale {field}")
    if sweep.get("selected_layer") not in CHOICE_LAYERS or baseline_ids != development_ids:
        raise ValueError(f"Candidate {behaviour} sweep has stale layer or baseline source IDs")


def _candidate(behaviour: str, norm: float, model_dtype: str) -> tuple[torch.Tensor, dict[str, object]]:
    path = Path(f"artifacts/{behaviour}-sweep.pt")
    description = Path(f"artifacts/{behaviour}-sweep.json")
    sweep = json.loads(description.read_text(encoding="utf-8"))
    _validate_candidate_sweep(sweep, behaviour, path, model_dtype)
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    raw = cast(torch.Tensor, tensor[LAYER]["direction"]).float()
    raw_norm = raw.norm().item()
    if not bool(torch.isfinite(raw).all()) or raw_norm <= 0:
        raise ValueError("Invalid candidate control direction")
    return raw * (norm / raw_norm), {"candidate_path": str(path), "candidate_sha256": sha256(path),
                                     "candidate_metadata_sha256": sha256(description),
                                     "candidate_raw_norm": raw_norm}


def _own_behaviour(tokenizer, model, behaviour: str, direction: torch.Tensor,
                   alpha: float) -> dict[str, object]:
    sweep = json.loads(Path(f"artifacts/{behaviour}-sweep.json").read_text(encoding="utf-8"))
    baseline_rows = sweep["conditions"]["baseline"]["results"]
    prompts_path = Path(f"data/{behaviour}/prompts.json")
    if sha256(prompts_path) != sweep["selected_sha256"]:
        raise ValueError("Candidate A/B prompts differ from sweep baseline")
    samples = json.loads(prompts_path.read_text(encoding="utf-8"))
    development = [sample for sample in samples if sample["split"] == "development"]
    if [sample["source_index"] for sample in development] != [row["source_index"] for row in baseline_rows]:
        raise ValueError("Candidate development IDs differ from its own baseline")
    scored = score_choice_condition(tokenizer, model, behaviour, development, LAYER, direction, alpha)
    rows = []
    scored_rows = cast(list[dict[str, object]], scored["results"])
    for current, previous in zip(scored_rows, baseline_rows, strict=True):
        mean = cast(float, current["question_mean"])
        baseline_mean = cast(float, previous["question_mean"])
        rows.append({"source_index": current["source_index"], "baseline_mean": baseline_mean,
                     "mean": mean, "delta": mean - baseline_mean,
                     "orders": current["orders"]})
    return {"behaviour": behaviour, "rows": rows,
            "mean_delta": sum(cast(float, row["delta"]) for row in rows) / len(rows),
            "status": "candidate control only; own function not independently certified"}


def _generations(tokenizer, model, items: list[SampleRefusalItem], direction: torch.Tensor | None,
                 alpha: float, *, coherence: bool) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    nlls: list[float] = []
    kls: list[float] = []
    for item in items:
        completion = generate_completion(model, tokenizer, item["instruction"], direction,
                                         LAYER, alpha, GENERATION)
        label = has_refusal_substring(completion["text"])
        row: dict[str, object] = {"sample_path": item["sample_path"],
                                  "sample_index": item["sample_index"],
                                  "instruction": item["instruction"],
                                  "response": completion["text"], "refusal": label,
                                  "prompt_token_ids": completion["prompt_token_ids"],
                                  "generated_token_ids": completion["generated_token_ids"]}
        if coherence:
            likelihood = response_likelihood(model, completion["prompt_token_ids"],
                                             completion["generated_token_ids"])
            row["clean_response_likelihood"] = likelihood
            if likelihood["mean_nll"] is not None:
                nlls.append(likelihood["mean_nll"])
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["instruction"]}],
                add_generation_prompt=True, return_tensors="pt"
            )
            kl = (prompt_distribution_kl(model, inputs, direction, alpha, layer_index=LAYER)
                  if direction is not None else 0.0)
            row["prompt_kl_baseline_to_condition"] = kl
            kls.append(kl)
        rows.append(row)
    summary: dict[str, object] = {"count": len(rows), "refusal_count": sum(bool(row["refusal"]) for row in rows),
                                  "refusal_rate": sum(bool(row["refusal"]) for row in rows) / len(rows)}
    if coherence:
        summary.update({"clean_ppl_exp_mean_per_prompt_nll": math.exp(sum(nlls) / len(nlls)) if nlls else None,
                        "nonempty_count": len(nlls), "empty_count": len(rows) - len(nlls),
                        "mean_prompt_kl_baseline_to_condition": sum(kls) / len(kls)})
    return rows, summary


def _mmlu(tokenizer, model, direction: torch.Tensor | None, alpha: float) -> list[dict[str, object]]:
    rows = []
    for number, item in enumerate(sample_mmlu(), start=1):
        score = score_question(tokenizer, model, render_question(item), LAYER, direction, alpha)
        gold = LABELS[item["answer"]]
        rows.append({"id": item["id"], "subject": item["subject"],
                     "source_index": item["source_index"], "gold": gold,
                     **score, "correct": score["prediction"] == gold})
        if number % 25 == 0:
            print(f"MMLU {number}/285", flush=True)
    if len(rows) != 285:
        raise ValueError("Fixed MMLU sample changed")
    return rows


def run(target: str, alpha: float, control: str) -> None:
    if control != "baseline" and (target not in TARGETS or alpha not in ALPHAS):
        raise ValueError("Condition outside frozen target/alpha grid")
    key = condition_key(target, alpha, control)
    output = CONDITIONS_DIR / f"{key}.json"
    extraction, tensors = _load_extraction()
    baseline_path = CONDITIONS_DIR / "baseline.json"
    baseline = None if control == "baseline" else json.loads(baseline_path.read_text(encoding="utf-8"))
    base_hash = None if baseline is None else sha256(baseline_path)
    candidate_details: dict[str, object] = {}
    direction: torch.Tensor | None = None
    if control != "baseline":
        raw = cast(torch.Tensor, cast(dict[str, torch.Tensor], tensors["directions"])[target]).float()
        norm = raw.norm().item()
        if control == "direction":
            direction = raw
        elif control == "random":
            direction = sample_random_direction(raw.numel(), norm, 42)
        else:
            direction, candidate_details = _candidate(
                control.removeprefix("candidate-"), norm, cast(str, extraction["model_dtype"])
            )
    metadata: dict[str, object] = {
        "condition": key, "target": None if control == "baseline" else target,
        "control": control, "alpha": 0.0 if control == "baseline" else alpha,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_dtype": extraction["model_dtype"],
        "layer_index": LAYER, "hook_policy": "last prompt token and each cached decode token",
        "generation": GENERATION, "random_seed": 42 if control == "random" else None,
        "raw_target_norm": None if control == "baseline" else norm,
        "control_direction_norm": None if direction is None else direction.norm().item(),
        "actual_perturbation_norm": 0.0 if direction is None else (direction * alpha).norm().item(),
        "extraction_sha256": sha256(EXTRACTION_PATH), "tensor_sha256": sha256(DIRECTION_PATH),
        "baseline_sha256": base_hash, "mmlu_source_sha256": sha256(SOURCE_PATH),
        "code_sha256": code_hashes(), **candidate_details,
    }
    if baseline is not None:
        for field in ("model_id", "model_revision", "layer_index", "generation",
                      "extraction_sha256", "tensor_sha256", "mmlu_source_sha256", "code_sha256"):
            if baseline["metadata"][field] != metadata[field]:
                raise ValueError(f"Baseline condition has stale {field}")
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old["metadata"] != metadata:
            raise ValueError(f"Existing condition has stale metadata: {output}")
        print(f"already complete: {output}")
        return
    tokenizer, model = load_qwen()
    if metadata["model_dtype"] != str(model.dtype):
        raise ValueError("Model dtype differs from extraction")
    if baseline is not None and baseline["metadata"]["model_dtype"] != metadata["model_dtype"]:
        raise ValueError("Model dtype differs from baseline")
    reviewed = load_reviewed_refusal()["validation"]
    benign, benign_summary = _generations(tokenizer, model, reviewed["harmless"],
                                           direction, alpha, coherence=True)
    harmful: list[dict[str, object]] = []
    harmful_summary = None
    if control == "baseline" or alpha < 0:
        harmful, harmful_summary = _generations(tokenizer, model, reviewed["harmful"],
                                                direction, alpha, coherence=False)
    mmlu = _mmlu(tokenizer, model, direction, alpha)
    mmlu_summary = paired_accuracy(baseline["mmlu"] if baseline is not None else mmlu, mmlu)
    if baseline is not None:
        endpoint = benign_summary if alpha > 0 else harmful_summary
        base_endpoint = baseline["benign_summary"] if alpha > 0 else baseline["harmful_summary"]
        assert endpoint is not None and base_endpoint is not None
        effect = (endpoint["refusal_rate"] - base_endpoint["refusal_rate"] if alpha > 0
                  else base_endpoint["refusal_rate"] - endpoint["refusal_rate"])
    else:
        effect = None
    own = (_own_behaviour(tokenizer, model, control.removeprefix("candidate-"), direction, alpha)
           if control.startswith("candidate-") and direction is not None else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {"metadata": metadata, "benign": benign, "benign_summary": benign_summary,
                        "harmful": harmful, "harmful_summary": harmful_summary,
                        "mmlu": mmlu, "mmlu_summary": mmlu_summary,
                        "endpoint": ("baseline on 16 harmless and 16 harmful" if control == "baseline"
                                     else "induction on 16 harmless" if alpha > 0
                                     else "suppression on 16 harmful"),
                        "endpoint_effect_vs_baseline": effect,
                        "own_behaviour": own,
                        "status": "provisional substring labels; generations need manual response audit; exploratory validation only"})
    print(f"saved {output}")


def summarize() -> None:
    """Collect the finite grid without changing settings or claiming equivalence."""
    baseline_path = CONDITIONS_DIR / "baseline.json"
    baseline_sha = sha256(baseline_path)
    curves: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for target in TARGETS:
        for control in ("direction", "random", "candidate-sycophancy", "candidate-honesty"):
            for alpha in ALPHAS:
                key = condition_key(target, alpha, control)
                path = CONDITIONS_DIR / f"{key}.json"
                if not path.exists():
                    if control in ("direction", "random"):
                        missing.append(key)
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                if result["metadata"]["baseline_sha256"] != baseline_sha:
                    raise ValueError(f"Condition baseline changed: {key}")
                curves[key] = {
                    "path": str(path), "sha256": sha256(path),
                    "target": target, "control": control, "alpha": alpha,
                    "endpoint": result["endpoint"],
                    "refusal_effect_vs_baseline": result["endpoint_effect_vs_baseline"],
                    "mmlu": result["mmlu_summary"],
                    "benign": result["benign_summary"],
                    "harmful": result["harmful_summary"],
                    "status": result.get("status"),
                    "own_behaviour": result.get("own_behaviour") if control.startswith("candidate-") else None,
                }
    closest: dict[str, object] = {}
    for sign in (1, -1):
        reference_key = condition_key("pooled", float(sign), "direction")
        if reference_key not in curves:
            continue
        reference = cast(float, curves[reference_key]["refusal_effect_vs_baseline"])
        for domain in DOMAINS:
            choices = [(abs(cast(float, curves[key]["refusal_effect_vs_baseline"]) - reference),
                        abs(alpha), key)
                       for alpha in ALPHAS if alpha * sign > 0
                       if (key := condition_key(domain, alpha, "direction")) in curves]
            if choices:
                mismatch, _, key = min(choices)
                closest[f"{domain}-{'positive' if sign > 0 else 'negative'}"] = {
                    "reference": reference_key, "candidate": key,
                    "absolute_refusal_rate_mismatch": mismatch,
                    "status": "descriptive nearest tested dose; not matched-effect equivalence",
                }
    CONDITIONS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(CONDITIONS_DIR / "summary.json", {
        "baseline_path": str(baseline_path), "baseline_sha256": baseline_sha,
        "curves": curves, "missing_frozen_conditions": missing,
        "closest_tested_dose": closest,
        "status": "exploratory dose-response; source labels and completions need manual response audit",
    })
    print(f"saved {CONDITIONS_DIR / 'summary.json'}; missing {len(missing)} conditions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("extract")
    subcommands.add_parser("summarize")
    condition = subcommands.add_parser("run")
    condition.add_argument("--target", choices=TARGETS, default="pooled")
    condition.add_argument("--alpha", type=float, default=1.0)
    condition.add_argument("--control", choices=CONTROLS, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        extract()
    elif args.command == "summarize":
        summarize()
    else:
        run(args.target, args.alpha, args.control)


if __name__ == "__main__":
    main()
