"""Ideogram 4 "magic prompt" expansion: system prompt presets + caption post-processing.

Ideogram 4 is trained on structured JSON captions; a plain prompt samples out-of-distribution and
the model often returns its baked-in "blocked by safety filter" card. The fix is to expand the plain
prompt into the caption schema with an LLM. InvokeAI already ships a local text-LLM "expand prompt"
path (``POST /api/v1/utilities/expand-prompt``); this module supplies the Ideogram-specific system
prompt and validates/normalises the LLM's output.

The system prompt (``magic_prompt_system_prompts/v1.txt``) and the post-processing helpers are ported
from the reference implementation (Apache-2.0): https://github.com/ideogram-ai/ideogram4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

from invokeai.backend.ideogram4.caption_verifier import CaptionVerifier
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)

SYSTEM_PROMPT_DIR = Path(__file__).resolve().parent / "magic_prompt_system_prompts"

# Preset name passed by the frontend / API to select the Ideogram 4 v1 magic prompt.
IDEOGRAM4_V1 = "ideogram4_v1"

# Maps a preset name to the system-prompt file that backs it.
_PRESET_FILES: dict[str, str] = {
    IDEOGRAM4_V1: "v1.txt",
}


@dataclass(frozen=True)
class PromptExpansionPreset:
    """A resolved magic-prompt preset: the system prompt plus an output post-processor."""

    name: str
    system_prompt: str
    postprocess: Callable[[str], str]


@lru_cache(maxsize=None)
def _load_sections(filename: str) -> dict[str, str]:
    """Parse a system-prompt file into its ``[SECTION]`` blocks.

    Files use ``[NAME]`` markers alone on a line (``[META]``, ``[SYSTEM]``, ``[USER]``). Returns a
    mapping of lower-cased section name to its text body, stripped. Cached so a file is read once.
    """
    raw = (SYSTEM_PROMPT_DIR / filename).read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and " " not in stripped:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = stripped[1:-1].strip().lower()
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    if "system" not in sections:
        raise ValueError(f"{filename} has no [SYSTEM] section")
    return sections


def get_preset(name: str) -> PromptExpansionPreset | None:
    """Resolve a preset name to its system prompt + post-processor, or None if unknown."""
    filename = _PRESET_FILES.get(name)
    if filename is None:
        return None
    system_prompt = _load_sections(filename)["system"]
    return PromptExpansionPreset(name=name, system_prompt=system_prompt, postprocess=postprocess_caption)


def _strip_code_fences(text: str) -> str:
    """Drop a surrounding ```json ... ``` fence if a model wraps its output in one."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _reorder_caption_keys(caption: dict[str, Any]) -> dict[str, Any]:
    """Reorder a caption's object keys to the canonical schema order.

    JSON key order is semantically irrelevant, but CaptionVerifier enforces a canonical order (e.g.
    elements as ``type`` before ``desc``). An LLM may emit keys in a different order, so we reorder
    ``style_description``, ``compositional_deconstruction``, and each element. Unknown keys are kept,
    appended after the known ones.
    """
    verifier = CaptionVerifier()

    def _ordered(d: dict[str, Any], order: Sequence[str]) -> dict[str, Any]:
        known = [k for k in order if k in d]
        extra = [k for k in d if k not in order]
        return {k: d[k] for k in (*known, *extra)}

    sd = caption.get("style_description")
    if isinstance(sd, dict):
        try:
            caption["style_description"] = _ordered(sd, verifier._style_description_key_order(sd))
        except ValueError:
            pass  # ambiguous photo/art_style; leave order for the verifier to flag

    cd = caption.get("compositional_deconstruction")
    if isinstance(cd, dict):
        cd = _ordered(cd, verifier.compositional_deconstruction_key_order)
        elements = cd.get("elements")
        if isinstance(elements, list):
            reordered = []
            for element in elements:
                if isinstance(element, dict):
                    try:
                        element = _ordered(element, verifier._element_key_order(element))
                    except ValueError:
                        pass  # missing/unknown "type"; leave order for the verifier to flag
                reordered.append(element)
            cd["elements"] = reordered
        caption["compositional_deconstruction"] = cd

    return caption


def _strip_aspect_ratio_and_bboxes(caption: dict[str, Any], *, strip_bboxes: bool) -> dict[str, Any]:
    """Drop the non-schema ``aspect_ratio`` key and, optionally, per-element ``bbox`` values.

    Matches the reference default of stripping LLM-generated bounding boxes: they are often
    unreliable and a wrong box hurts layout more than no box.
    """
    caption.pop("aspect_ratio", None)
    if strip_bboxes:
        elements = caption.get("compositional_deconstruction", {}).get("elements", [])
        if isinstance(elements, list):
            for element in elements:
                if isinstance(element, dict):
                    element.pop("bbox", None)
    return caption


def postprocess_caption(text: str, *, strip_bboxes: bool = True) -> str:
    """Normalise an LLM's expanded caption into a clean minified JSON string.

    Strips code fences, parses, reorders keys to the schema, drops ``aspect_ratio`` and (by default)
    bboxes, and logs any CaptionVerifier warnings. If the text is not valid JSON we return it as-is:
    the model accepts any string, so a plain-text expansion still generates rather than failing.
    """
    cleaned = _strip_code_fences(text)
    try:
        caption = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Ideogram 4 magic prompt: LLM output was not valid JSON; using it verbatim.")
        return cleaned

    if not isinstance(caption, dict):
        logger.warning("Ideogram 4 magic prompt: expected a JSON object; using the output verbatim.")
        return cleaned

    caption = _reorder_caption_keys(caption)
    caption = _strip_aspect_ratio_and_bboxes(caption, strip_bboxes=strip_bboxes)

    serialised = json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
    warnings = CaptionVerifier().verify(caption)
    if warnings:
        logger.warning(
            "Ideogram 4 magic prompt: expanded caption has %d schema warning(s): %s",
            len(warnings),
            "; ".join(warnings[:5]),
        )
    return serialised
