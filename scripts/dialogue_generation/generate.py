"""Observation JSON -> terminal-anchored, text-only co-creation conversation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import secrets
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dialogue_generation.planner import build_plan, validate_plan, MODULES
from tools.dialogue_realizer import QwenRealizer, DEFAULT_OLLAMA_MODEL, DEFAULT_LOCAL_MODEL, MAX_REWRITE_ATTEMPTS
from tools.dialogue_personas import choose_personas, USER_PERSONAS
from tools.dialogue_models import resolve_model
from cocreation.qwen_sampling import generation_options
from scripts.dialogue_generation.stats import collect_stats, generation_snapshot


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def markdown(result):
    lines = ["# Video co-creation dialogue", "",
             f"Seed: {result['seed']} | Dialogue model: {result['realization']['model']}", "",
             f"User style: {result['personas']['user']['id']}", "",
             "The dialogue contains text and media placeholders.", ""]
    for turn in result["turns"]:
        role = turn["role"].title()
        labels = ", ".join(result["behavior_labels"][str(i)] for i in turn["intent"]["behavior_ids"])
        lines += [f"**{turn['id']}. {role}**", "", turn["content"], "",
                  f"> Intent: {turn['intent']['description']}",
                  f"> Module: {turn['intent']['module']} | Behaviors: {labels or 'General interaction'}", ""]
    lines += ["## Media placeholders", ""]
    for media in result["placeholders"]:
        lines.append(f"- {media['placeholder']}: {media['description']}")
    return "\n".join(lines) + "\n"


def run(input_path, output_dir, *, seed=0, reworks=1, p_compile=0.65,
        modules=None, extra_behaviors=True, realizer=None, backend="ollama",
        model=None, base_url="http://127.0.0.1:11434", max_tokens=256,
        timeout=180, progress=True, user_persona=None, context_tokens=32768, resume=False,
        temperature=0.9, top_p=0.95, top_k=50, rewrite_seed=None, model_mode=None):
    started = time.perf_counter()
    sampling = generation_options(dict(temperature=temperature, top_p=top_p, top_k=top_k))
    sampling.pop("do_sample")  # Ollama samples with positive temperature; local Qwen sets do_sample itself.
    if rewrite_seed is not None and (type(rewrite_seed) is not int or not 0 <= rewrite_seed < 2**31):
        raise ValueError("rewrite_seed must be an integer in [0, 2**31), or None")
    source = Path(input_path).expanduser().resolve()
    observation = json.loads(source.read_text(encoding="utf-8"))
    # Plan/validate before opening a model or touching the output directory.
    plan = build_plan(observation, seed=seed, reworks=reworks,
                      p_compile=p_compile, modules=modules, extra_behaviors=extra_behaviors)
    plan["personas"] = choose_personas(seed, user_persona)
    output = Path(output_dir).expanduser().resolve()
    completed_count = 0
    if resume:
        if (json.loads((output / "observation.json").read_text()) != observation
                or json.loads((output / "plan.json").read_text()) != plan):
            raise ValueError("Resume requires the same observation, seed, modules and persona")
        result = json.loads((output / "dialogue.json").read_text())
        validate_plan(result)
        completed_count = result["realization"]["completed_turns"]
        if len(result["turns"]) != len(plan["turns"]) or not 0 <= completed_count <= len(plan["turns"]):
            raise ValueError("Invalid completed-turn checkpoint")
        for i, turn in enumerate(result["turns"]):
            if {**turn, "content": None} != plan["turns"][i] or bool(turn["content"]) != (i < completed_count):
                raise ValueError("Checkpoint is not an intact realized prefix of this plan")
    else:
        output.mkdir(parents=True, exist_ok=False)
        save(output / "plan.json", plan)
        save(output / "observation.json", observation)
        save(output / "placeholders.json", [dict(placeholder=m["placeholder"], kind=m["kind"],
                                               description=m["description"],
                                               reference_materials=m["reference_materials"],
                                               reference_depth=m["reference_depth"],
                                               **({"source_video": m["source_video"]} if "source_video" in m else {}))
                                            for m in plan["placeholders"]])
        result = deepcopy(plan)
    # Preserve the seed schedule of older checkpoints when resuming them.
    seed_stride = result.get("realization", {}).get("rewrite_seed_stride", 3) if resume else MAX_REWRITE_ATTEMPTS
    saved_sampling = result.get("realization", {}).get("sampling") if resume else None
    saved_rewrite_seed = result.get("realization", {}).get("rewrite_seed") if resume else None
    if saved_sampling is not None and saved_sampling != sampling:
        raise ValueError("Resume requires the same temperature, top_p and top_k")
    if saved_rewrite_seed is not None:
        if rewrite_seed is not None and rewrite_seed != saved_rewrite_seed:
            raise ValueError("Resume requires the same rewrite_seed")
        rewrite_seed = saved_rewrite_seed
    elif rewrite_seed is None:
        rewrite_seed = secrets.randbits(31)
    if resume and result["realization"]["status"] == "complete":
        if realizer is None:
            requested_model, requested_mode = resolve_model(backend, model, model_mode)
            requested_backend = backend
        else:
            requested_backend = getattr(realizer, "backend", "injected")
            requested_model = getattr(realizer, "model", "injected-test-realizer")
            requested_mode = getattr(realizer, "model_mode", "injected")
        saved = result["realization"]
        if (saved["backend"] != requested_backend or saved["model"] != requested_model
                or saved.get("model_mode", requested_mode) != requested_mode):
            raise ValueError("Resume must use the same realization model/backend/mode")
        return result
    owned = realizer is None
    if owned:
        realizer = QwenRealizer(backend=backend, model=model, base_url=base_url,
                                output_dir=output / "qwen", max_tokens=max_tokens, timeout=timeout,
                                personas=plan["personas"], context_tokens=context_tokens,
                                model_mode=model_mode, **sampling,
                                asset_names={a["id"]: a.get("name", a["id"]) for a in observation["assets"]})
    if resume and (result["realization"]["backend"] != getattr(realizer, "backend", "injected")
                   or result["realization"]["model"] != getattr(realizer, "model", "injected-test-realizer")
                   or result["realization"].get("model_mode", getattr(realizer, "model_mode", "injected")) != getattr(realizer, "model_mode", "injected")):
        if owned:
            realizer.close()
        raise ValueError("Resume must use the same realization model/backend")
    prior_stats = result.get("stats", {})
    previous_elapsed = prior_stats.get("time_elapsed_seconds", 0.0)
    previous_generation = {k:v for k,v in prior_stats.get("generation", {}).items() if isinstance(v, (int, float))}
    baseline = generation_snapshot(realizer)
    injected_attempts = 0
    def refresh_stats():
        current = generation_snapshot(realizer)
        generation = {key: previous_generation.get(key, 0) + current.get(key, 0) - baseline.get(key, 0)
                      for key in current.keys() | previous_generation.keys()}
        if not hasattr(realizer, "rewrite_attempts"):
            generation["rewrite_attempts"] = previous_generation.get("rewrite_attempts", 0) + injected_attempts
        generation["token_counts_source"] = "ollama" if getattr(realizer, "backend", None) == "ollama" else "unavailable"
        # Only numeric counters take part in resume arithmetic.
        result["stats"] = collect_stats(result, observation,
            elapsed_seconds=previous_elapsed + time.perf_counter() - started, generation=generation)

    result["realization"] = dict(backend=getattr(realizer, "backend", "injected"),
                                 model=getattr(realizer, "model", "injected-test-realizer"),
                                 model_mode=getattr(realizer, "model_mode", "injected"),
                                 status="running", completed_turns=completed_count,
                                 sampling=sampling, rewrite_seed=rewrite_seed, rewrite_seed_stride=seed_stride)
    media = {m["placeholder"]: m for m in result["placeholders"]}
    completed = result["turns"][:completed_count]
    refresh_stats()
    save(output / "dialogue.json", result)
    try:
        for turn in result["turns"][completed_count:]:
            # New runs reserve a distinct seed for every bounded rewrite attempt.
            injected_attempts += 1
            text = realizer(turn, media, completed, seed=rewrite_seed + turn["id"] * seed_stride)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Realizer returned empty text")
            refs = "\nReferences: " + " ".join(turn["reference_media"]) if turn["reference_media"] else ""
            attachments = "\n" + " ".join(turn["output_media"]) if turn["output_media"] else ""
            turn["content"] = text.strip() + refs + attachments
            completed.append(turn)
            result["realization"]["completed_turns"] = len(completed)
            refresh_stats()
            save(output / "dialogue.json", result)
            if progress:
                print(f"[{turn['id']}/{len(result['turns'])}] {turn['role']}: {text.strip()}",
                      file=sys.stderr, flush=True)
        validate_plan(result)
        result["realization"]["status"] = "complete"
        refresh_stats()
        save(output / "dialogue.json", result)
        (output / "dialogue.md").write_text(markdown(result), encoding="utf-8")
    except BaseException as exc:
        result["realization"].update(status="failed", error=str(exc))
        refresh_stats()
        save(output / "dialogue.json", result)
        raise
    finally:
        if owned:
            realizer.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reworks", type=int, choices=(0, 1, 2), default=1,
                        help="Number of distinct assets reserved for post-shot rework; remaining edits are requested together")
    parser.add_argument("--p-compile", type=float, default=0.65)
    parser.add_argument("--modules", nargs="+", choices=MODULES,
                        help="Limit preparation modules; a last single asset may fall back to direct")
    parser.add_argument("--no-extra-behaviors", action="store_true",
                        help="Disable optional ambiguity, constraint and shot-requirement conflict episodes")
    parser.add_argument("--backend", choices=("ollama", "local"), default="ollama")
    parser.add_argument("--model", help=f"Default: {DEFAULT_OLLAMA_MODEL} (Ollama), {DEFAULT_LOCAL_MODEL} (local)")
    parser.add_argument("--model-mode", choices=("instruct", "thinking"),
                        help="Override mode inference for custom model names; explicit instruct/thinking aliases are recommended")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--temperature", type=float, default=0.9, help="Positive Qwen sampling temperature; default 0.9")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rewrite-seed", type=int,
                        help="Fix wording randomness independently of --seed; default fresh random seed per run, restored on resume")
    parser.add_argument("--user-persona", choices=tuple(USER_PERSONAS), help="Default: seeded random style")
    parser.add_argument("--context-tokens", type=int, default=32768,
                        help="Rewrite prompt budget; dialogue context contains the three most recent messages")
    parser.add_argument("--resume", action="store_true", help="Continue an intact prefix with identical planning options")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.observation, args.output_dir, seed=args.seed, reworks=args.reworks,
                     p_compile=args.p_compile, modules=args.modules,
                     extra_behaviors=not args.no_extra_behaviors, backend=args.backend,
                     model=args.model, model_mode=args.model_mode, base_url=args.base_url, max_tokens=args.max_tokens,
                     timeout=args.timeout, progress=not args.quiet,
                     user_persona=args.user_persona, context_tokens=args.context_tokens, resume=args.resume,
                     temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                     rewrite_seed=args.rewrite_seed)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"dialogue generation failed: {exc}\n")
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()), turns=len(result["turns"]),
                          placeholders=len(result["placeholders"]), final=result["final"]["media"],
                          behavior_coverage=result["behavior_coverage"],
                          attempts=result.get("stats", {}).get("number_of_attempts"),
                          time_elapsed_seconds=result.get("stats", {}).get("time_elapsed_seconds")), ensure_ascii=False))


if __name__ == "__main__":
    main()
