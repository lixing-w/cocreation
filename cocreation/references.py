"""Stable artifact identity and per-call model-local numbering."""
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Reference:
    artifact_id: str
    path: str | Path
    kind: str

    def __post_init__(self):
        if self.kind not in ("image", "video", "audio"):
            raise ValueError("Reference kind must be image, video or audio")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.artifact_id):
            raise ValueError("artifact_id must contain only letters, digits, _, ., :, -")


def prepare(prompt: str, references: list[Reference], *, labels=None):
    counts = {"image": 0, "video": 0, "audio": 0}
    labels = labels or {"image": "Image", "video": "Video", "audio": "Audio"}
    bindings = {}
    media = []
    for ref in references:
        if ref.artifact_id in bindings:
            raise ValueError(f"Duplicate artifact: {ref.artifact_id}")
        counts[ref.kind] += 1
        label = f"<{labels[ref.kind]} {counts[ref.kind]}>"
        row = dict(artifact_id=ref.artifact_id, kind=ref.kind,
                   path=str(Path(ref.path).expanduser().resolve()), label=label)
        bindings[ref.artifact_id] = row
        media.append(row)

    def substitute(match):
        key = match.group(1)
        if key not in bindings:
            raise ValueError(f"Prompt refers to an unattached artifact: {key}")
        return bindings[key]["label"]

    resolved = re.sub(r"\{\{artifact:([^{}]+)\}\}", substitute, prompt)
    if "{{artifact:" in resolved:
        raise ValueError("Malformed artifact placeholder")
    return dict(prompt=resolved, original_prompt=prompt, references=media,
                bindings=bindings)
