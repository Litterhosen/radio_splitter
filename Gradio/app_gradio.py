from __future__ import annotations

import csv
import json
import os
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Gradio.audio_split import (
        cut_segment_to_mp3,
        cut_segment_to_wav,
        detect_non_silent_intervals,
        fixed_length_intervals,
        get_duration_seconds,
    )
except Exception:
    from audio_split import (
        cut_segment_to_mp3,
        cut_segment_to_wav,
        detect_non_silent_intervals,
        fixed_length_intervals,
        get_duration_seconds,
    )

from audio_detection import detect_audio_type
from broadcast_splitter import detect_broadcast_segments
from downloaders import DownloadError, download_audio
from hook_finder import ffmpeg_to_wav16k_mono, find_hooks, normalize_bpm_family
from transcribe import transcribe_wav

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    from tagging import auto_tags
except Exception:
    auto_tags = None

# Keep runtime artifacts on local disk by default (not cloud-synced project dir).
RUNTIME_ROOT = Path(
    os.getenv(
        "RADIO_SPLITTER_RUNTIME_ROOT",
        str(Path.home() / "AppData" / "Local" / "radio_splitter2"),
    )
).resolve()
OUTPUT_ROOT = Path(
    os.getenv("RADIO_SPLITTER_GRADIO_OUTPUT_ROOT", str(RUNTIME_ROOT / "output_gradio"))
).resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
_runtime_temp = RUNTIME_ROOT / "temp"
_runtime_temp.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(_runtime_temp))
os.environ.setdefault("TEMP", str(_runtime_temp))
os.environ.setdefault("TMPDIR", str(_runtime_temp))

_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in path.stem)[:72] or "audio"


def _fmt_clock(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    r = s % 60
    return f"{h:02d}:{m:02d}:{r:02d}"


def _update_progress(progress: gr.Progress, started_at: float, done: float, total: float, label: str) -> None:
    total_safe = max(float(total), 1.0)
    frac = min(max(float(done) / total_safe, 0.0), 1.0)
    elapsed = max(0.0, perf_counter() - started_at)
    eta = (elapsed / frac - elapsed) if frac > 1e-6 else 0.0
    progress(frac, desc=f"{int(frac * 100):3d}% | {label} | elapsed {_fmt_clock(elapsed)} | ETA {_fmt_clock(eta)}")


def _load_model(model_size: str, device: str, compute_type: str):
    key = (model_size, device, compute_type)
    if key not in _MODEL_CACHE:
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed.")
        _MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _MODEL_CACHE[key]


def _collect_inputs(files: Optional[List[str]], url: str, run_dir: Path) -> Tuple[List[Path], List[str]]:
    paths: List[Path] = []
    notes: List[str] = []

    for fp in files or []:
        p = Path(fp)
        if p.exists():
            paths.append(p)

    if url and url.strip():
        try:
            dl_path, _ = download_audio(url.strip(), run_dir / "downloads")
            paths.append(dl_path)
            notes.append(f"Downloaded: {dl_path.name}")
        except DownloadError as e:
            notes.append(f"Download failed: {e}")
        except Exception as e:
            notes.append(f"Download failed: {type(e).__name__}: {e}")

    deduped: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped, notes


def _sample_intervals(intervals: List[Tuple[float, float]], max_samples: int) -> List[Tuple[float, float]]:
    if max_samples <= 0 or not intervals:
        return []
    if len(intervals) <= max_samples:
        return intervals
    if max_samples == 1:
        return [intervals[len(intervals) // 2]]
    step = (len(intervals) - 1) / float(max_samples - 1)
    idxs = sorted({int(round(i * step)) for i in range(max_samples)})
    return [intervals[i] for i in idxs]


def _build_simple_intervals(
    src: Path,
    split_mode: str,
    fixed_len_s: float,
    noise_db: float,
    min_silence_s: float,
    pad_s: float,
    min_segment_s: float,
) -> Tuple[List[Tuple[float, float]], List[str]]:
    notes: List[str] = []
    if split_mode == "Fixed length":
        return fixed_length_intervals(src, segment_len=float(fixed_len_s)), notes

    intervals = detect_non_silent_intervals(
        src,
        noise_db=float(noise_db),
        min_silence_s=float(min_silence_s),
        pad_s=float(pad_s),
        min_segment_s=float(min_segment_s),
    )
    # Guardrail: if silence splitting returns a single near-fullfile segment, fallback.
    duration = float(get_duration_seconds(src))
    if len(intervals) <= 1 and duration > max(120.0, float(fixed_len_s) * 3.0):
        fallback_len = max(6.0, min(20.0, float(fixed_len_s)))
        fallback = fixed_length_intervals(src, segment_len=fallback_len)
        if len(fallback) > len(intervals):
            intervals = fallback
            notes.append(f"Fallback to fixed-length split ({fallback_len:.1f}s) due sparse silence boundaries.")
    return intervals, notes


def preview_clip(selected_path: Optional[str]):
    if not selected_path:
        return None, None
    p = Path(selected_path)
    if not p.exists():
        return None, None
    return str(p), str(p)


def run_pipeline(
    files: Optional[List[str]],
    url: str,
    processing_mode: str,
    export_format: str,
    max_clips_per_file: int,
    transcribe_enabled: bool,
    model_size: str,
    device: str,
    compute_type: str,
    language_ui: str,
    simple_split_mode: str,
    fixed_len_s: float,
    noise_db: float,
    min_silence_s: float,
    pad_s: float,
    min_segment_s: float,
    song_hook_min: float,
    song_hook_max: float,
    song_prefer_len: float,
    song_topn: int,
    song_gap: float,
    prefer_bars: int,
    beats_per_bar: int,
    broadcast_method: str,
    broadcast_max_segment: float,
    broadcast_merge_gap: float,
    broadcast_chunk_s: float,
    broadcast_export_without_transcript: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    state = {"done": 0.0, "total": 1.0}

    try:
        input_paths, notes = _collect_inputs(files, url, run_dir)
        if not input_paths:
            return (
                "No valid input files found.",
                [],
                None,
                gr.update(choices=[], value=None),
                None,
                None,
            )

        state["total"] = max(1.0, float(len(input_paths) * 4))
        _update_progress(progress, started, state["done"], state["total"], "Starting run")

        lang = None if language_ui == "Auto" else language_ui.lower()
        model = None
        should_transcribe = bool(transcribe_enabled)
        if should_transcribe:
            model = _load_model(model_size=model_size, device=device, compute_type=compute_type)
            state["done"] += 1.0
            _update_progress(progress, started, state["done"], state["total"], "Whisper model loaded")

        rows: List[Dict[str, Any]] = []
        clip_total = 0

        for file_idx, src in enumerate(input_paths, start=1):
            state["done"] += 1.0
            _update_progress(progress, started, state["done"], state["total"], f"File {file_idx}/{len(input_paths)}: preparing")

            src_dur = float(get_duration_seconds(src))
            src_stem = _safe_stem(src)
            src_out = run_dir / src_stem
            src_out.mkdir(parents=True, exist_ok=True)

            # Common lightweight analysis signal for classification + hook/broadcast split.
            wav16 = src_out / "_analysis_16k_mono.wav"
            ffmpeg_to_wav16k_mono(src, wav16)
            audio_info = detect_audio_type(
                wav16,
                sr=16000,
                duration=24.0,
                known_duration_sec=src_dur,
            )

            interval_specs: List[Dict[str, float]] = []
            mode_detail = processing_mode
            mode_broadcast = processing_mode.startswith("Broadcast")

            if processing_mode.startswith("Song"):
                hooks, global_bpm, global_conf = find_hooks(
                    wav16,
                    hook_len_range=(float(song_hook_min), float(song_hook_max)),
                    prefer_len=float(song_prefer_len),
                    hop_s=1.0,
                    topn=int(song_topn),
                    min_gap_s=float(song_gap),
                    prefer_bars=int(prefer_bars),
                    beats_per_bar=int(beats_per_bar),
                )
                interval_specs = [
                    {
                        "start": float(h.start),
                        "end": float(h.end),
                        "bpm": float(normalize_bpm_family(h.bpm, h.bpm_confidence)),
                        "score": float(h.score),
                    }
                    for h in hooks
                ]
                mode_detail = f"hook (global_bpm={int(global_bpm)}, conf={float(global_conf):.2f})"
            elif processing_mode.startswith("Broadcast"):
                intervals, split_method, chunking_enabled = detect_broadcast_segments(
                    wav16,
                    min_segment_sec=max(1.5, float(min_segment_s)),
                    max_segment_sec=float(broadcast_max_segment),
                    merge_gap_sec=float(broadcast_merge_gap),
                    chunk_sec=float(broadcast_chunk_s),
                    silence_noise_db=float(noise_db),
                    silence_min_s=float(min_silence_s),
                    silence_pad_s=float(pad_s),
                    prefer_method=("vad" if str(broadcast_method).startswith("VAD") else "energy"),
                )
                interval_specs = [{"start": float(a), "end": float(b), "bpm": 0.0, "score": 0.0} for a, b in intervals]
                mode_detail = f"broadcast:{split_method}, chunking={chunking_enabled}"
            else:
                simple_intervals, simple_notes = _build_simple_intervals(
                    src,
                    split_mode=simple_split_mode,
                    fixed_len_s=float(fixed_len_s),
                    noise_db=float(noise_db),
                    min_silence_s=float(min_silence_s),
                    pad_s=float(pad_s),
                    min_segment_s=float(min_segment_s),
                )
                interval_specs = [{"start": float(a), "end": float(b), "bpm": 0.0, "score": 0.0} for a, b in simple_intervals]
                notes.extend(simple_notes)
                mode_detail = f"simple:{simple_split_mode}"

            # Guardrail: avoid exporting a single full-file clip on long files.
            if (
                not processing_mode.startswith("Song")
                and len(interval_specs) == 1
                and src_dur >= 90.0
            ):
                first = interval_specs[0]
                first_dur = max(0.0, float(first.get("end", 0.0)) - float(first.get("start", 0.0)))
                if first_dur >= src_dur * 0.90:
                    fallback_len = min(
                        max(8.0, float(fixed_len_s)),
                        float(broadcast_max_segment),
                    )
                    fallback_intervals = fixed_length_intervals(src, segment_len=fallback_len)
                    if len(fallback_intervals) > 1:
                        interval_specs = [
                            {"start": float(a), "end": float(b), "bpm": 0.0, "score": 0.0}
                            for a, b in fallback_intervals
                        ]
                        notes.append(
                            f"{src.name}: fallback fixed split ({fallback_len:.1f}s) "
                            f"because primary split produced one near-full-file interval."
                        )

            if max_clips_per_file > 0:
                interval_specs = interval_specs[: int(max_clips_per_file)]

            if not interval_specs:
                notes.append(f"{src.name}: no intervals found")
                continue

            state["total"] += float(len(interval_specs))
            notes.append(f"{src.name}: {len(interval_specs)} interval(s), mode={mode_detail}")

            do_transcribe_clips = should_transcribe and not (mode_broadcast and bool(broadcast_export_without_transcript))

            for idx, spec in enumerate(interval_specs, start=1):
                a = float(spec["start"])
                b = float(spec["end"])
                dur = max(0.0, b - a)
                if dur <= 0.0:
                    continue

                state["done"] += 1.0
                if idx == 1 or idx == len(interval_specs) or idx % 8 == 0:
                    _update_progress(
                        progress,
                        started,
                        state["done"],
                        state["total"],
                        f"File {file_idx}/{len(input_paths)}: clip {idx}/{len(interval_specs)}",
                    )

                clip_total += 1
                clip_name = f"{src_stem}__{idx:04d}__{int(a*1000)}-{int(b*1000)}"

                if export_format == "wav":
                    clip_path = src_out / f"{clip_name}.wav"
                    cut_segment_to_wav(src, clip_path, a, b)
                else:
                    clip_path = src_out / f"{clip_name}.mp3"
                    cut_segment_to_mp3(src, clip_path, a, b, bitrate="192k")

                transcript = ""
                transcript_json: Dict[str, Any] = {}
                tags: List[str] = []

                if do_transcribe_clips and model is not None:
                    tmp_wav = src_out / f"_tmp_{idx:04d}.wav"
                    cut_segment_to_wav(src, tmp_wav, a, b)
                    transcript_json = transcribe_wav(model, tmp_wav, language=lang)
                    transcript = (transcript_json.get("text") or "").strip()
                    if auto_tags is not None:
                        tags = auto_tags(transcript)
                    tmp_wav.unlink(missing_ok=True)
                elif auto_tags is not None:
                    tags = ["ukendt"]

                txt_path = src_out / f"{clip_name}.txt"
                json_path = src_out / f"{clip_name}.json"
                txt_path.write_text(transcript + "\n", encoding="utf-8")
                json_path.write_text(json.dumps(transcript_json, ensure_ascii=False, indent=2), encoding="utf-8")

                bpm_val = int(spec.get("bpm", 0.0) or 0.0)
                score_val = float(spec.get("score", 0.0) or 0.0)
                rows.append(
                    {
                        "source": src.name,
                        "clip": f"{idx:04d}",
                        "start_sec": round(a, 3),
                        "end_sec": round(b, 3),
                        "dur_sec": round(dur, 3),
                        "filename": clip_path.name,
                        "mode": mode_detail,
                        "bpm": bpm_val,
                        "score": score_val,
                        "tags": ", ".join(tags),
                        "text": transcript[:200],
                        "audio_type": audio_info.get("audio_type_guess", "unknown"),
                        "audio_conf": audio_info.get("audio_type_confidence", 0.0),
                        "path": str(clip_path.resolve()),
                    }
                )

        manifest_path = run_dir / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            else:
                f.write("source,clip,start_sec,end_sec,dur_sec,filename,mode,bpm,score,tags,text,audio_type,audio_conf,path\n")

        zip_path = run_dir / "export_bundle.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, arcname="manifest.csv")
            for row in rows:
                clip_fp = Path(row["path"])
                txt_fp = clip_fp.with_suffix(".txt")
                json_fp = clip_fp.with_suffix(".json")
                if clip_fp.exists():
                    zf.write(clip_fp, arcname=f"clips/{clip_fp.name}")
                if txt_fp.exists():
                    zf.write(txt_fp, arcname=f"meta/{txt_fp.name}")
                if json_fp.exists():
                    zf.write(json_fp, arcname=f"meta/{json_fp.name}")

        _update_progress(progress, started, state["total"], state["total"], "Run finished")

        table_rows = [
            [
                r["source"],
                r["clip"],
                r["start_sec"],
                r["end_sec"],
                r["dur_sec"],
                r["filename"],
                r["mode"],
                r["bpm"],
                r["score"],
                r["tags"],
                r["text"],
            ]
            for r in rows
        ]

        preview_choices = [(f"{r['source']} | {r['filename']}", r["path"]) for r in rows]
        first_preview_path = preview_choices[0][1] if preview_choices else None

        status = (
            f"Processed {len(input_paths)} file(s), produced {clip_total} clip(s).\n"
            f"Run folder: {run_dir}"
        )
        if notes:
            status += "\n\n" + "\n".join(f"- {n}" for n in notes)

        return (
            status,
            table_rows,
            str(zip_path.resolve()),
            gr.update(choices=preview_choices, value=first_preview_path),
            first_preview_path,
            first_preview_path,
        )
    except Exception as e:
        crash_path = run_dir / "crash.log"
        crash_path.write_text(traceback.format_exc(), encoding="utf-8")
        return (
            f"Run failed: {type(e).__name__}: {e}\nCrash log: {crash_path}",
            [],
            None,
            gr.update(choices=[], value=None),
            None,
            None,
        )


with gr.Blocks(title="The Sample Machine - Gradio") as demo:
    gr.Markdown(
        "## The Sample Machine (Gradio)\n"
        "Song Hunter + Broadcast Hunter with progress/ETA, preview, and ZIP export."
    )

    with gr.Row():
        files_in = gr.Files(label="Upload audio files", file_types=["audio"], type="filepath")
        url_in = gr.Textbox(label="URL (optional, yt-dlp)", placeholder="https://...")

    with gr.Accordion("Core Settings", open=True):
        mode_in = gr.Dropdown(
            choices=["Song Hunter (Loops)", "Broadcast Hunter (Mix)", "Simple Split"],
            value="Broadcast Hunter (Mix)",
            label="Processing mode",
        )
        export_format_in = gr.Dropdown(choices=["mp3", "wav"], value="mp3", label="Export format")
        max_clips_in = gr.Slider(0, 400, value=0, step=1, label="Max clips per file (0 = no limit)")

    with gr.Accordion("Transcription", open=False):
        transcribe_in = gr.Checkbox(value=True, label="Enable transcription")
        model_size_in = gr.Dropdown(
            choices=["tiny", "base", "small", "medium"],
            value="small",
            label="Whisper model size",
        )
        device_in = gr.Dropdown(choices=["cpu"], value="cpu", label="Device")
        compute_in = gr.Dropdown(choices=["int8", "float32"], value="int8", label="Compute type")
        lang_in = gr.Dropdown(choices=["Auto", "da", "en"], value="Auto", label="Language")

    with gr.Accordion("Song Hunter Settings", open=False):
        song_min_in = gr.Slider(2, 30, value=4, step=0.5, label="Min hook length (sec)")
        song_max_in = gr.Slider(2, 30, value=15, step=0.5, label="Max hook length (sec)")
        song_pref_in = gr.Slider(2, 24, value=8, step=0.5, label="Preferred length (sec)")
        song_topn_in = gr.Slider(3, 60, value=12, step=1, label="Top N hooks")
        song_gap_in = gr.Slider(0, 10, value=2, step=0.5, label="Min gap between hooks (sec)")
        prefer_bars_in = gr.Dropdown(choices=[1, 2, 4, 8, 16], value=2, label="Preferred bars")
        beats_per_bar_in = gr.Slider(3, 7, value=4, step=1, label="Beats per bar")

    with gr.Accordion("Broadcast Hunter Settings", open=True):
        broadcast_method_in = gr.Dropdown(
            choices=["VAD-first (recommended)", "Energy-first"],
            value="VAD-first (recommended)",
            label="Split method",
        )
        broadcast_max_segment_in = gr.Slider(10, 120, value=45, step=1, label="Max segment (sec)")
        broadcast_merge_gap_in = gr.Slider(0.1, 1.0, value=0.35, step=0.05, label="Merge gap (sec)")
        broadcast_chunk_in = gr.Slider(300, 1200, value=600, step=60, label="Chunk size (sec)")
        broadcast_no_transcript_in = gr.Checkbox(value=True, label="Export without transcript (faster)")

    with gr.Accordion("Simple Split Settings", open=False):
        simple_mode_in = gr.Dropdown(
            choices=["Silence-aware", "Fixed length"],
            value="Silence-aware",
            label="Simple split mode",
        )
        fixed_len_in = gr.Slider(2, 120, value=8, step=1, label="Fixed length (sec)")
        noise_db_in = gr.Slider(-60, -10, value=-35, step=1, label="Silence threshold (dB)")
        min_silence_in = gr.Slider(0.1, 3.0, value=0.7, step=0.1, label="Min silence (sec)")
        pad_in = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Padding (sec)")
        min_segment_in = gr.Slider(0.3, 8.0, value=1.5, step=0.1, label="Min segment (sec)")

    run_btn = gr.Button("Process", variant="primary")
    status_out = gr.Markdown(label="Status")
    table_out = gr.Dataframe(
        headers=["source", "clip", "start_sec", "end_sec", "dur_sec", "filename", "mode", "bpm", "score", "tags", "text"],
        datatype=["str", "str", "number", "number", "number", "str", "str", "number", "number", "str", "str"],
        interactive=False,
        wrap=True,
        label="Manifest Preview",
    )
    zip_out = gr.File(label="Export ZIP")

    with gr.Row():
        preview_select = gr.Dropdown(label="Preview clip", choices=[], value=None)
        preview_file = gr.File(label="Download selected clip")
    preview_audio = gr.Audio(label="Clip preview", type="filepath")

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            files_in,
            url_in,
            mode_in,
            export_format_in,
            max_clips_in,
            transcribe_in,
            model_size_in,
            device_in,
            compute_in,
            lang_in,
            simple_mode_in,
            fixed_len_in,
            noise_db_in,
            min_silence_in,
            pad_in,
            min_segment_in,
            song_min_in,
            song_max_in,
            song_pref_in,
            song_topn_in,
            song_gap_in,
            prefer_bars_in,
            beats_per_bar_in,
            broadcast_method_in,
            broadcast_max_segment_in,
            broadcast_merge_gap_in,
            broadcast_chunk_in,
            broadcast_no_transcript_in,
        ],
        outputs=[status_out, table_out, zip_out, preview_select, preview_audio, preview_file],
    )

    preview_select.change(
        fn=preview_clip,
        inputs=[preview_select],
        outputs=[preview_audio, preview_file],
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        share=True,
        allowed_paths=[str(OUTPUT_ROOT)],
    )
