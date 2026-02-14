# Final Patch Deliverable

Når patchen er klar, svar kun med:

## Liste over ændrede filer

1. **downloaders.py** - YouTube download with JS runtime support and enhanced error reporting
2. **audio_split.py** - Pre-roll, fades, and zero-crossing alignment for smooth clips
3. **hook_finder.py** - Preferred bars control hook window length
4. **app.py** - UI improvements, session state fix, manifest v2 fields
5. **README.md** - Added link to smoke tests
6. **README_SMOKE_TESTS.md** (new) - Comprehensive test documentation
7. **FINAL_PATCH_SUMMARY.md** (new) - Complete patch summary

---

## Kort "what changed" pr fil

### downloaders.py
- Added `check_js_runtime()` to detect Node.js
- Enhanced error reporting with `url` and `hint` parameters
- Implemented JS runtime auto-detection and extractor fallback
- Better hint formatting using list-based approach
- Logs show JS runtime status and extractor strategy

### audio_split.py
- New `find_zero_crossing()` function for audio analysis
- New `cut_segment_with_fades()` for high-quality export
- Implements 25ms pre-roll, 15ms fade-in/out using ffmpeg
- Zero-crossing alignment (±10ms window) for smooth clip boundaries
- Returns detailed export metadata dictionary

### hook_finder.py
- Added `prefer_bars` and `beats_per_bar` parameters
- Window length calculated: `prefer_bars * bar_duration`
- Bar duration: `(60 / BPM) * beats_per_bar`
- Falls back to `prefer_len` if bars not specified

### app.py
- Version 1.1.4
- Fixed `prefer_bars_ui` session_state warning
- Updated `export_clip_with_tail()` returns tuple with metadata
- Added 15+ new manifest v2 fields
- Enhanced YouTube error display with log file links
- Passes `prefer_bars` to hook finder

### README.md
- Added reference to README_SMOKE_TESTS.md

### README_SMOKE_TESTS.md (new)
- 5 comprehensive test scenarios
- Expected results and acceptance criteria
- Troubleshooting guide
- Quick checklist

### FINAL_PATCH_SUMMARY.md (new)
- Complete implementation summary
- Technical details for each change
- Example filenames and manifest
- Security analysis
- Acceptance criteria status

---

## 3 eksempel-filenames

```
1. Daft_Punk - One_More_Time__0001__128bpm__16bar__00-15-01-30__around_the_world__a1b2c3_tail.mp3

2. Rick_Astley - Never_Gonna_Give_You_Up__0005__113bpm_est__8bar_est__01-45-02-12__never_gonna__f4e5d6_tail.mp3

3. UnknownArtist - track_20240214__0003__140bpm__4bar__00-32-00-37__noslug__9z8y7x_tail.mp3
```

**Format forklaring:**
- `{Artist} - {Title}` starter alle filnavne
- `__{clip_number:04d}` (0001, 0005, etc.)
- `__{bpm}bpm` eller `{bpm}bpm_est` (estimated)
- `__{bars}bar` eller `{bars}bar_est`
- `__{start_mmss}-{end_mmss}` (00-15-01-30 = 0:15 til 1:30)
- `__{slug}` eller `noslug`
- `__{uid6}` (6-tegn UID)
- `_tail` (hvis decay tail)

---

## 1 eksempel manifest-række (CSV)

```csv
filename,track_artist,track_title,filename_template_version,bpm_global,bpm_global_confidence,bpm_used,bpm_used_source,bars_estimated,bars_used,bars_used_source,core_start_sec,core_end_sec,core_dur_sec,export_start_sec,export_end_sec,export_dur_sec,pre_roll_ms,fade_in_ms,fade_out_ms,tail_sec,zero_cross_applied,refined,refined_reason
Daft_Punk - One_More_Time__0001__128bpm__16bar__00-15-01-30__around_the_world__a1b2c3_tail.mp3,Daft_Punk,One_More_Time,v2,128,0.87,128,refined_grid,16,16,refined_grid,15.025,45.325,30.300,15.000,46.075,31.075,25.0,15.0,15.0,0.75,true,true,
```

**Nøgle felter:**
- `filename_template_version: v2` - ny format identifier
- `track_artist: Daft_Punk` - fra ID3 eller YouTube
- `track_title: One_More_Time`
- `bpm_global: 128` - global BPM for hele track
- `bpm_global_confidence: 0.87` - høj confidence
- `bpm_used: 128` - BPM brugt til dette clip
- `bpm_used_source: refined_grid` - beat-aligned
- `bars_used: 16` - faktisk antal bars (som requested)
- `bars_used_source: refined_grid` - beat grid aligned
- `core_start_sec: 15.025` - zero-crossing adjusted start
- `core_end_sec: 45.325` - zero-crossing adjusted end
- `export_start_sec: 15.000` - start minus pre-roll (25ms)
- `export_end_sec: 46.075` - end plus tail (0.75s)
- `pre_roll_ms: 25.0` - pre-roll før start
- `fade_in_ms: 15.0` - fade-in duration
- `fade_out_ms: 15.0` - fade-out duration
- `tail_sec: 0.75` - decay tail længde
- `zero_cross_applied: true` - boundaries snappet til zero-crossings
- `refined: true` - beat refinement successful
- `refined_reason:` (tom når refined=true)

---

## Smoke test instruktioner (3 punkter)

### 1. **Local MP3 Test**
Upload en MP3, sæt **16 bars**, process.
- ✅ Clips er ~25.3s for 152 BPM track (±150ms)
- ✅ Filnavne starter: `{Artist} - {Title}__...`
- ✅ Ingen audio "pops" ved start/slut

### 2. **YouTube Test**
Indtast YouTube URL, klik Download.
- ✅ Success: MP3 downloaded + log viser "verified"
- ✅ Failure: Error message + log path + hint
- ✅ App crasher ikke

### 3. **Batch Test (2 Tracks)**
Upload 2 forskellige MP3s, process begge.
- ✅ Ingen crash, separate output folders
- ✅ Hver clip har korrekt artist/title i filnavn
- ✅ Ingen session_state warnings

---

## Status

✅ **Alle krav opfyldt**
✅ **0 security vulnerabilities** (CodeQL scan)
✅ **Code review gennemført**
✅ **Production ready**

Se `FINAL_PATCH_SUMMARY.md` for komplet dokumentation.
