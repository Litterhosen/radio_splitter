from ui_utils import build_groups


def _format_preview_title(total_clips, groups):
    title = f"🎧 Preview Selected ({total_clips} clips)"
    if groups:
        title += f" — {len(groups)} groups"
    return title


def test_group_by_phrase_preview_title_no_nameerror():
    rows = [
        {"pick": True, "clip_text_signature": "alpha beta"},
        {"pick": True, "clip_text_signature": "alpha beta"},
        {"pick": True, "clip_text_signature": "gamma"},
    ]

    selected_rows = [r for r in rows if r.get("pick")]
    total_clips = len(selected_rows)
    groups = build_groups(selected_rows, "phrase")

    title = _format_preview_title(total_clips, groups)

    assert total_clips == 3
    assert groups is not None
    assert len(groups) == 2
    assert "3 clips" in title
    assert "2 groups" in title
