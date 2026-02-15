# UI Feature Guide: Auto-Detection + Grouping

This guide shows where to find the new features in the UI.

## 1. Audio Detection Display (During Processing)

After clicking "Process", you'll see:

```
### File 1/1: MyTrack.mp3
🎵 Track: **Artist - Title**
⏳ Converting audio...
✅ Conversion complete.
🔍 Detecting audio type...
🎯 **Audio Type:** music (confidence: 0.82)
💡 **Recommended Mode:** Song Hunter | Broadcast Hunter
🎵 Finding hooks...
```

**What it means:**
- `music` = Musical content detected (strong beat, tempo)
- `speech` = Voice/dialogue detected (high ZCR variation)
- `mixed` = Both music and speech
- `jingle_ad` = Short musical segment with speech
- `unknown` = Unable to classify confidently

**Confidence scores:**
- 0.0-0.5 = Low confidence
- 0.5-0.7 = Moderate confidence
- 0.7-0.9 = Good confidence
- 0.9-1.0 = Very high confidence

## 2. Results Browser Section

After processing completes, scroll down to see:

```
📊 Results Browser

┌─────────────────────────────────────┐
│ [Data Editor with all clips]        │
│ Columns: pick, filename, bpm, tags, │
│ themes, duration, text, etc.        │
└─────────────────────────────────────┘

Selected: 15 clips

┌──────────────┬──────────────┐
│ Grouping:    │ Sort by:     │
│ [Dropdown]   │ [Dropdown]   │
└──────────────┴──────────────┘
```

## 3. Grouping Options

Click the "Grouping" dropdown to see 4 options:

### Option 1: None (Default)
```
┌─────────────────────────────────────────────┐
│ 🎧 Preview Selected (15 clips)              │
│                                              │
│ Page 1 of 1                                 │
│                                              │
│ Clip 1 - Artist-Title__0001.mp3            │
│ ⏱️ 25.3s | 🎵 152 BPM | 📊 16 bars | 🇬🇧 EN│
│ 📝 Hold on I'm coming to get you...        │
│ 🏷️ musik, hook | 🎨 THEME:TIME             │
│ [Audio Player]                              │
│                                              │
│ Clip 2 - Artist-Title__0002.mp3            │
│ ...                                          │
└─────────────────────────────────────────────┘
```

### Option 2: Group by Phrase
```
┌─────────────────────────────────────────────┐
│ 🎧 Preview Selected (15 clips) - 8 groups   │
│                                              │
│ ▼ 📋 "hold on i'm coming..." (3 clips)     │
│   │ Clip 1 - [details]                      │
│   │ Clip 5 - [details]                      │
│   │ Clip 9 - [details]                      │
│                                              │
│ ▼ 📋 "breaking news just in..." (2 clips)  │
│   │ Clip 2 - [details]                      │
│   │ Clip 7 - [details]                      │
│                                              │
│ ▶ 📋 "velkommen til morgen..." (1 clip)    │
│ ▶ 📋 "[No text]" (9 clips)                 │
└─────────────────────────────────────────────┘
```

**Use case:** Find repeated content (jingles, ads, catchphrases)

### Option 3: Group by Tag/Theme
```
┌─────────────────────────────────────────────┐
│ 🎧 Preview Selected (15 clips) - 5 groups   │
│                                              │
│ ▼ 🏷️ musik, hook (10 clips)                │
│   │ Clip 1 - [details]                      │
│   │ Clip 2 - [details]                      │
│   │ ...                                      │
│                                              │
│ ▼ 🏷️ radio/nyheder, THEME:META (3 clips)  │
│   │ Clip 11 - [details]                     │
│   │ Clip 12 - [details]                     │
│   │ Clip 13 - [details]                     │
│                                              │
│ ▶ 🏷️ reklame (2 clips)                     │
└─────────────────────────────────────────────┘
```

**Use case:** Organize by content type (music, news, ads, etc.)

### Option 4: Group by Language
```
┌─────────────────────────────────────────────┐
│ 🎧 Preview Selected (15 clips) - 3 groups   │
│                                              │
│ ▼ 🇩🇰 Danish (8 clips)                      │
│   │ Clip 1 - Velkommen til...               │
│   │ Clip 3 - God morgen...                  │
│   │ ...                                      │
│                                              │
│ ▼ 🇬🇧 English (5 clips)                     │
│   │ Clip 2 - Breaking news...               │
│   │ Clip 7 - Welcome to...                  │
│   │ ...                                      │
│                                              │
│ ▶ ❓ Unknown (2 clips)                      │
│   │ [Instrumental clips with no text]       │
└─────────────────────────────────────────────┘
```

**Use case:** Separate bilingual content by language

## 4. Enhanced Clip Preview Cards

Each clip now shows:

```
┌─────────────────────────────────────────────┐
│ **Artist - Title__0001__152bpm__16bar.mp3** │
│                                              │
│ ⏱️ 25.3s | 🎵 152 BPM | 📊 16 bars | 🇬🇧 EN│
│                                              │
│ 📝 Hold on I'm coming to get you right     │
│ now don't wait up for me I'll be there     │
│ in just a minute or two hold tight...      │
│                                              │
│ 📋 Show full text ←── Click to see all     │
│                                              │
│ 🏷️ musik, hook, 16bar | 🎨 THEME:TIME     │
│                                              │
│             [🔊 Audio Player]                │
└─────────────────────────────────────────────┘
```

**Legend:**
- ⏱️ Duration in seconds
- 🎵 BPM (beats per minute)
- 📊 Bar count
- 🇩🇰 Danish | 🇬🇧 English | 🌐 Other language
- 📝 Transcript snippet (first ~150 chars)
- 📋 Button to show complete transcript
- 🏷️ Auto-detected tags
- 🎨 Auto-detected themes
- 🔊 Audio player (standard Streamlit widget)

## 5. Manifest Export

When you download the ZIP, the manifest CSV now includes:

```csv
...,audio_type_guess,audio_type_confidence,recommended_mode,language_guess_file,language_confidence_file,language_guess_clip,language_confidence_clip,clip_text_signature,...
...,music,0.82,Song Hunter | Broadcast Hunter,en,0.7,en,0.8,hold on i'm coming,...
...,speech,0.75,Broadcast Hunter,da,0.7,da,0.8,velkommen til morgen,...
...,mixed,0.68,Broadcast Hunter | Song Hunter,da,0.7,en,0.7,breaking news just in,...
```

## 6. Feature Comparison: Before vs After

### Before This PR
```
🎧 Preview Selected (15 clips)

Clip 1 - long_filename_here.mp3
Duration: 25.30s | BPM: 152 | Bars: 16
📝 Hold on I'm coming to get you right now...
[Audio Player]
```

### After This PR
```
Grouping: [Dropdown▼] Sort by: [Dropdown▼]

🎧 Preview Selected (15 clips) - 8 groups

▼ 📋 "hold on i'm coming..." (3 clips)
  
  **Artist - Title__0001__152bpm__16bar.mp3**
  ⏱️ 25.3s | 🎵 152 BPM | 📊 16 bars | 🇬🇧 EN
  
  📝 Hold on I'm coming to get you right now...
  📋 Show full text
  
  🏷️ musik, hook, 16bar | 🎨 THEME:TIME
  [Audio Player]
```

## Key Improvements

1. **Better Organization**: Group related clips together
2. **More Context**: Language labels, tags, and themes visible
3. **Cleaner Display**: Icons and emojis for quick scanning
4. **Deduplication**: Find repeated content easily
5. **Full Text Access**: View complete transcripts
6. **Richer Metadata**: 8 new fields in manifest for analysis

## Workflow Tips

### For Music Producers
1. Use **Group by Phrase** to find repeated hooks
2. Filter by high hook_score in manifest
3. Look for clips with same `clip_text_signature`

### For Radio Editors
1. Use **Group by Language** for bilingual content
2. Use **Group by Tag/Theme** to separate ads/news/music
3. Check `jingle_score` field to find commercial content

### For Researchers
1. Export manifest for analysis in Excel/Python
2. Use `audio_type_guess` to filter content types
3. Use `language_guess_clip` for language statistics
4. Check `audio_type_confidence` for quality control

## Troubleshooting

**Q: Grouping doesn't appear?**  
A: Make sure you have processed files and clips are in Results Browser

**Q: Audio type shows "unknown"?**  
A: Low confidence detection - check audio quality or length

**Q: Language shows "unknown"?**  
A: Clip has no transcribed text (instrumental or silence)

**Q: Groups have unexpected counts?**  
A: Check text signatures - short clips may group together

**Q: "Show full text" button doesn't appear?**  
A: Button only appears if text is >150 characters

## Screenshot Locations

To capture screenshots for documentation:

1. **Audio Detection**: During "Processing" status, after "Converting audio"
2. **Grouping Dropdown**: In Results Browser, above preview section
3. **Grouped View**: After selecting a grouping option, expanded group
4. **Clip Card**: Any clip in preview showing all metadata
5. **Manifest**: Downloaded CSV opened in Excel showing new columns
