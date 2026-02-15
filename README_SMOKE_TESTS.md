# Smoke Tests - The Sample Machine

This document contains smoke test instructions to verify the application works correctly after deployment or updates.

## Prerequisites

- Python 3.11+
- FFmpeg installed
- Test audio files (MP3 or WAV)
- (Optional) Node.js for YouTube download testing

## Test 1: Local MP3 Processing

**Purpose:** Verify basic audio processing, beat detection, and file naming work correctly.

### Steps:

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Configure settings in sidebar:
   - Mode: **🎵 Song Hunter (Loops)**
   - Model size: **small**
   - Format: **mp3 (192k)**
   - Preferred loop length: **16 bars**
   - Beats per bar: **4**
   - Refine to beat grid: **Checked**

3. Upload a local MP3 file (preferably 3-5 minutes, ~120-152 BPM)

4. Click **Load Whisper Model** and wait for completion

5. Click **Process** and wait for results

### Expected Results:

✅ **Processing completes without errors**

✅ **Clips found (typically 5-15 clips)**

✅ **Clip lengths match preferred bars:**
   - For track at ~152 BPM: 16 bars ≈ 25.3s (±150ms)
   - For track at ~120 BPM: 16 bars ≈ 32.0s (±150ms)

✅ **Filenames follow format:**
   ```
   {Artist} - {Title}__0001__{bpm}bpm__{bars}bar__{start}-{end}__{slug}__{uid6}_tail.mp3
   ```
   Example:
   ```
   The_Beatles - Let_It_Be__0001__152bpm__16bar__00-15-01-30__slug_text__abc123_tail.mp3
   ```

✅ **Manifest CSV contains these fields:**
   - `track_artist`, `track_title`, `track_id`
   - `filename_template_version` = "v2"
   - `bpm_global`, `bpm_global_confidence`, `bpm_used`, `bpm_used_source`
   - `bars_estimated`, `bars_used`, `bars_used_source`
   - `core_start_sec`, `core_end_sec`, `export_start_sec`, `export_end_sec`
   - `pre_roll_ms`, `fade_in_ms`, `fade_out_ms`, `tail_sec`, `zero_cross_applied`

✅ **Audio clips start smoothly (no "pop" or harsh cut at beginning)**

### Verify in manifest:
- Open the downloaded ZIP
- Check `manifest_selected.csv`
- Verify `bars_used` column shows `16` for most clips
- Verify `pre_roll_ms` = `25.0`
- Verify `fade_in_ms` = `15.0`, `fade_out_ms` = `15.0`
- Verify `tail_sec` = `0.75`

---

## Test 2: YouTube Download

**Purpose:** Verify YouTube download works with JS runtime support and proper error reporting.

### Steps:

1. Navigate to **🔗 Hent fra Link** tab

2. Enter a YouTube URL (use a known working URL, e.g., a Creative Commons track)
   ```
   https://www.youtube.com/watch?v=jNQXAC9IVRw
   ```

3. Click **Download**

### Expected Results:

#### Scenario A: Successful Download

✅ **Download completes successfully**

✅ **Success message shows:** `✅ Downloaded: {filename}.mp3`

✅ **File appears in:** `output/Downloads/`

✅ **Log file created:** `output/Downloads/download_log_{timestamp}.txt`

✅ **Log shows:**
   - "JavaScript runtime detected: {path}" (if Node.js installed)
   - OR "No JavaScript runtime found" warning (if Node.js not installed)
   - "Download successful and verified"
   - File size and duration

#### Scenario B: Download Failure (Blocked/Unavailable Video)

✅ **Error message displayed:** `❌ Download failed: ...`

✅ **Log file path shown:** `📄 Log file saved: output/Downloads/download_log_{timestamp}.txt`

✅ **Hint displayed with specific guidance:**
   - "Install Node.js ..." (if Node.js missing)
   - "Video may be geo-blocked ..." (if geo-restricted)
   - "Video may require authentication ..." (if login required)

✅ **Log file contains:**
   - All attempted format strategies
   - JavaScript runtime status
   - Extractor strategy used
   - Last error message
   - Troubleshooting tips

✅ **Application does not crash** - error is handled gracefully

✅ **Error classification code shown in UI:**
   - `ERR_JS_RUNTIME_MISSING`
   - `ERR_VIDEO_UNAVAILABLE`
   - `ERR_GEO_BLOCK`
   - `ERR_LOGIN_REQUIRED`
   - `ERR_NETWORK`

✅ **If Node.js is unavailable on Streamlit Cloud, UI explains platform limitation and next action**

### Verify:

- Check `output/Downloads/` folder for audio file (success) or log file (failure)
- If Node.js is installed, verify log mentions "JavaScript runtime detected"
- If download fails, verify error message is helpful and actionable

---

## Test 3: Batch Processing (2 Tracks)

**Purpose:** Verify application handles multiple files without crashing or mixing data.

### Steps:

1. Upload **2 different MP3 files** in the upload tab
   - File 1: Track A (e.g., "Song_A.mp3")
   - File 2: Track B (e.g., "Song_B.mp3")

2. Configure settings:
   - Mode: **🎵 Song Hunter (Loops)**
   - Preferred loop length: **4 bars**
   - Other settings: default

3. Click **Load Whisper Model** (if not already loaded)

4. Click **Process**

### Expected Results:

✅ **Both files process without errors**

✅ **No session_state warnings in UI or logs**

✅ **Output folders separated by track + run id:**
   - `output/Song_A__{run_id}/` contains clips from Track A
   - `output/Song_B__{run_id}/` contains clips from Track B

✅ **Filenames include correct artist/title for each track:**
   - Track A clips start with Track A's artist/title
   - Track B clips start with Track B's artist/title

✅ **Manifest CSV has separate rows for each clip**

✅ **No file name collisions or overwrites**

✅ **UID values are unique across all clips**

### Verify in manifest:

- Check `source` column shows correct source file for each clip
- Check `session_dir` column shows correct output folder
- Check `track_artist` and `track_title` match the correct source track
- Verify no duplicate `clip_path` values

---

## Test 4: Different Bar Lengths

**Purpose:** Verify preferred bars setting actually controls clip length.

### Steps:

1. Process the **same track 3 times** with different bar settings:
   - Run 1: **2 bars**
   - Run 2: **8 bars**
   - Run 3: **16 bars**

2. Use same track with known BPM (e.g., ~152 BPM track)

### Expected Results:

For a track at ~152 BPM (bar duration ≈ 1.58s):

| Setting | Expected Duration | Tolerance |
|---------|------------------|-----------|
| 2 bars  | ~3.2s            | ±150ms    |
| 8 bars  | ~12.6s           | ±150ms    |
| 16 bars | ~25.3s           | ±150ms    |

✅ **Clip durations match expected values (within tolerance)**

✅ **Manifest shows correct `bars_used` values**

✅ **Hook windows in UI reflect the chosen bar count**

---

## Test 5: Zero-Crossing & Fade Verification

**Purpose:** Verify audio quality improvements (no pops, smooth start/end).

### Steps:

1. Process a track with **16 bars** setting

2. Download the ZIP and extract clips

3. Listen to the **first 3 clips** in audio editor or player

### Expected Results:

✅ **No "pop" or click at clip start** - fade-in is smooth

✅ **No harsh cut at clip end** - fade-out is smooth

✅ **Clips don't start mid-word** - zero-crossing alignment helps

✅ **Manifest shows:**
   - `zero_cross_applied` = `true` for most clips
   - `pre_roll_ms` = `25.0`
   - `fade_in_ms` = `15.0`
   - `fade_out_ms` = `15.0`

### Verify in audio editor (optional):

- Load clip in Audacity or similar
- Check waveform starts near zero amplitude (not mid-peak)
- Visual fade-in/out curves present
- No DC offset or clicks

---

## Quick Checklist

Use this checklist for rapid smoke testing:

- [ ] **Local MP3:** Uploads, processes, generates clips
- [ ] **File naming:** Starts with `{Artist} - {Title}`
- [ ] **Bar length:** 16 bars → ~25s clips (152 BPM track)
- [ ] **Manifest:** Contains all required v2 fields
- [ ] **Audio quality:** No pops at start/end
- [ ] **YouTube:** Either downloads OR shows clear error with log
- [ ] **Batch:** 2 tracks process without crash
- [ ] **Session state:** No warnings about widget conflicts
- [ ] **Export:** ZIP downloads with correct structure

---

## Troubleshooting Common Issues

### "FFmpeg not found"
- **Local:** Install FFmpeg and add to PATH
- **Cloud:** Add `ffmpeg` to `packages.txt`

### "No JavaScript runtime found" warning (YouTube)
- **Local:** Install Node.js from https://nodejs.org/
- **Cloud:** Not needed on Streamlit Cloud (Python only environment)

### Clips are wrong length
- Check global BPM detection in UI output
- Verify `prefer_bars` setting is correct
- Check if beat refinement is enabled

### Session state warnings
- Restart Streamlit app
- Check for duplicate widget keys in code

### Download fails silently
- Check `output/Downloads/download_log_{timestamp}.txt`
- Update yt-dlp: `pip install --upgrade yt-dlp`
- Try a different video URL

---

## Contact

If smoke tests fail after following troubleshooting steps, please report with:
1. Which test failed
2. Error messages or screenshots
3. Log files (if available)
4. Python/OS/FFmpeg versions

---

## Test 6: Long-file Broadcast Hunter (>= 30 minutes)

**Purpose:** Validate robust VAD-first chunked segmentation and full transcript deliverables on long recordings.

### Steps:

1. Open mode: **📻 Broadcast Hunter (Mix)**.
2. Set:
   - Split method: **VAD-first (recommended)**
   - Max segment: **45 sec**
   - Merge gap: **0.35 sec**
   - Chunk size: **600 sec**
3. Upload a long file (podcast/radio mix) of **at least 30 minutes**.
4. Click **Load Whisper Model** then **Process**.
5. In results, open **🧾 Full Transcripts** and verify:
   - per-source `.txt` and `.json` download buttons
   - transcript search works
   - `▶️ Play from` seeks near the chosen timestamp
6. Export selected clips ZIP.

### Expected Results:

✅ Processing completes without crash/OOM on long file.

✅ Segmentation reports method used (`vad`, `energy`, or `silence`) and chunking status.

✅ `manifest_selected.csv` includes:
- `transcript_full_txt_path`
- `transcript_full_json_path`
- `language_detected`
- `transcribe_model`
- `split_method_used`
- `chunking_enabled`

✅ Full transcript files are present both per-source and in run ZIP exports.

### Sample output ZIP structure

```text
sample_machine_clips.zip
├── manifest_selected.csv
└── My_Source/
    ├── transcript_full.txt
    ├── transcript_full.json
    ├── Artist - Title__0001__00-00-45__slug__abc123.mp3
    ├── Artist - Title__0001__00-00-45__slug__abc123.txt
    ├── Artist - Title__0001__00-00-45__slug__abc123.json
    └── ...
```

---

## Test 7: Auto-Detection + Transcript Grouping Features

**Purpose:** Verify new auto-detection, language detection, and transcript grouping features work correctly.

### Steps:

1. Process a file with **either mode** (Song Hunter or Broadcast Hunter)

2. After processing completes, check the UI output for:
   - **Audio Type Detection** message showing detected type and confidence
   - **Recommended Mode** message

3. Scroll down to **Results Browser** section

4. Test grouping options:
   - Set **Grouping** dropdown to **"Group by Phrase"**
   - Verify clips are grouped by similar text signatures
   - Check that group count is shown
   - Expand a group to verify clips inside are displayed correctly
   
5. Try other grouping modes:
   - **Group by Tag/Theme**: Should group clips with similar tags
   - **Group by Language**: Should group clips by detected language (da/en/unknown)
   - **None**: Standard paginated view

6. Verify clip preview cards show:
   - Transcript snippet (first ~150 chars)
   - Language label with flag emoji (🇩🇰 DA, 🇬🇧 EN, or 🌐)
   - Tags and themes
   - "Copy text" button for each clip with transcript

### Expected Results:

✅ **Audio detection appears after conversion:**
```
🔍 Detecting audio type...
🎯 Audio Type: music (confidence: 0.75)
💡 Recommended Mode: Song Hunter | Broadcast Hunter
```

✅ **Grouping dropdown is visible with 4 options**

✅ **Group by Phrase works:**
- Clips with same or similar text are grouped together
- Groups show count: "📋 \"text snippet...\" (3 clips)"
- Groups are collapsible

✅ **Group by Language works:**
- Groups labeled with flags: "🇩🇰 Danish", "🇬🇧 English", etc.
- Clips correctly grouped by detected language

✅ **Clip preview cards show enhanced metadata:**
- Language label with appropriate emoji
- Transcript snippet (truncated if long)
- Copy text button appears for clips with transcripts
- Tags and themes displayed separately

✅ **Audio playback works in all grouping modes**

### Verify in manifest CSV:

After downloading and opening `manifest_selected.csv`, check for new columns:

- `audio_type_guess` (music|speech|mixed|jingle_ad|unknown)
- `audio_type_confidence` (0.0 to 1.0)
- `recommended_mode` (mode recommendation string)
- `language_guess_file` (file-level language detection)
- `language_confidence_file` (0.0 to 1.0)
- `language_guess_clip` (clip-level language, e.g., "da", "en", "unknown")
- `language_confidence_clip` (0.0 to 1.0)
- `clip_text_signature` (normalized text for grouping)

### Sample manifest row (new fields only):

```csv
audio_type_guess,audio_type_confidence,recommended_mode,language_guess_file,language_confidence_file,language_guess_clip,language_confidence_clip,clip_text_signature
music,0.82,Song Hunter | Broadcast Hunter,music,0.82,da,0.8,hold on jeg kommer
```

---

## Screenshot Checklist

To demonstrate the new features are working, capture screenshots showing:

1. **Audio detection output** - The detection message with type and confidence
2. **Grouping dropdown** - The UI showing grouping options
3. **Grouped view** - At least one grouping mode with collapsed/expanded groups
4. **Enhanced clip card** - Showing language label, snippet, and copy button
5. **Manifest with new fields** - CSV opened in Excel/editor showing new columns

---
