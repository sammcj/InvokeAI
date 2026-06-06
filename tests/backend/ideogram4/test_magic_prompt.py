import json

from invokeai.backend.ideogram4.caption_verifier import CaptionVerifier
from invokeai.backend.ideogram4.magic_prompt import IDEOGRAM4_V1, get_preset, postprocess_caption

# A schema-clean caption: photo style (exactly one of photo/art_style), ordered keys, valid bbox/hex.
CLEAN_CAPTION = {
    "high_level_description": "a cat in a room",
    "style_description": {
        "aesthetics": "cozy",
        "lighting": "soft window light",
        "photo": "50mm, f/2",
        "medium": "photograph",
        "color_palette": ["#1B1B2F", "#E43F5A"],
    },
    "compositional_deconstruction": {
        "background": "a sunlit room",
        "elements": [{"type": "obj", "bbox": [100, 200, 800, 900], "desc": "a ginger cat"}],
    },
}


def test_get_preset_resolves_ideogram4_v1() -> None:
    preset = get_preset(IDEOGRAM4_V1)
    assert preset is not None
    assert preset.name == IDEOGRAM4_V1
    assert len(preset.system_prompt) > 0
    assert callable(preset.postprocess)


def test_get_preset_returns_none_for_unknown() -> None:
    assert get_preset("does-not-exist") is None


def test_postprocess_strips_code_fences_and_minifies() -> None:
    fenced = "```json\n" + json.dumps(CLEAN_CAPTION) + "\n```"
    out = postprocess_caption(fenced)
    # Round-trips to the same data, and is minified (separators carry no spaces).
    assert json.loads(out)["high_level_description"] == "a cat in a room"
    assert out == json.dumps(json.loads(out), ensure_ascii=False, separators=(",", ":"))


def test_postprocess_drops_aspect_ratio_and_strips_bboxes_by_default() -> None:
    caption = {**CLEAN_CAPTION, "aspect_ratio": "1:1"}
    out = json.loads(postprocess_caption(json.dumps(caption)))
    assert "aspect_ratio" not in out
    assert "bbox" not in out["compositional_deconstruction"]["elements"][0]


def test_postprocess_keeps_bboxes_when_requested() -> None:
    out = json.loads(postprocess_caption(json.dumps(CLEAN_CAPTION), strip_bboxes=False))
    assert out["compositional_deconstruction"]["elements"][0]["bbox"] == [100, 200, 800, 900]


def test_postprocess_reorders_keys_to_schema_order() -> None:
    # Element with keys out of order: desc before type, bbox last.
    scrambled = {
        "compositional_deconstruction": {
            "elements": [{"desc": "a cat", "bbox": [1, 2, 3, 4], "type": "obj"}],
            "background": "a room",
        },
        "high_level_description": "a cat",
    }
    out = postprocess_caption(json.dumps(scrambled), strip_bboxes=False)
    # compositional_deconstruction: background before elements; element: type, bbox, desc.
    assert out.index('"background"') < out.index('"elements"')
    assert '"type":"obj","bbox":[1,2,3,4],"desc":"a cat"' in out


def test_postprocess_returns_non_json_verbatim() -> None:
    assert postprocess_caption("a plain language prompt") == "a plain language prompt"


def test_postprocess_returns_json_non_dict_verbatim() -> None:
    assert postprocess_caption("[1, 2, 3]") == "[1, 2, 3]"


def test_caption_verifier_passes_clean_caption() -> None:
    assert CaptionVerifier().verify(CLEAN_CAPTION) == []


def test_caption_verifier_warns_on_missing_compositional_deconstruction() -> None:
    warnings = CaptionVerifier().verify({"high_level_description": "a cat"})
    assert any("compositional_deconstruction" in w for w in warnings)
