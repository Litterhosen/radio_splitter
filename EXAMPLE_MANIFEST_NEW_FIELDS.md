# Example Manifest with New Auto-Detection Fields

This document shows example rows from the manifest CSV with the new auto-detection and language detection fields.

## New Fields Added

### File-Level Detection (Per Source File)
- `audio_type_guess`: Detected audio type (music | speech | mixed | jingle_ad | unknown)
- `audio_type_confidence`: Confidence score (0.0 to 1.0)
- `recommended_mode`: Recommended processing mode based on detection
- `language_guess_file`: Language detected at file level (da | en | unknown)
- `language_confidence_file`: Confidence for file-level language detection

### Clip-Level Detection (Per Clip)
- `language_guess_clip`: Language detected for this specific clip
- `language_confidence_clip`: Confidence for clip-level language detection
- `clip_text_signature`: Normalized text signature for grouping/deduplication

## Example 1: Song Hunter Mode

```csv
source,filename,clip,start_sec,end_sec,dur_sec,bpm_used,bars_used,text,audio_type_guess,audio_type_confidence,recommended_mode,language_guess_file,language_confidence_file,language_guess_clip,language_confidence_clip,clip_text_signature,tags,themes
MyTrack.mp3,Artist-Title__0001__152bpm__16bar__hold_on__abc123_tail.mp3,1,45.2,70.5,25.3,152,16,Hold on I'm coming to get you,music,0.82,Song Hunter | Broadcast Hunter,en,0.7,en,0.8,hold on i'm coming,musik|hook|16bar,THEME:TIME
MyTrack.mp3,Artist-Title__0002__152bpm__16bar__noslug__def456_tail.mp3,2,120.0,145.3,25.3,152,16,,music,0.82,Song Hunter | Broadcast Hunter,en,0.7,unknown,0.0,,musik|hook|16bar,
```

## Example 2: Broadcast Hunter Mode

```csv
source,filename,clip,start_sec,end_sec,dur_sec,text,audio_type_guess,audio_type_confidence,recommended_mode,language_guess_file,language_confidence_file,language_guess_clip,language_confidence_clip,clip_text_signature,tags,themes,jingle_score
RadioShow.mp3,Host - Morning_Show__0001__00-02-15__velkommen__xyz789.mp3,1,135.4,142.8,7.4,Velkommen til morgen programmet,mixed,0.68,Broadcast Hunter | Song Hunter,da,0.7,da,0.8,velkommen til morgen programmet,radio/nyheder,THEME:META,0.15
RadioShow.mp3,Host - Morning_Show__0002__00-02-45__breaking_news__uvw123.mp3,2,165.2,178.9,13.7,Breaking news just in from downtown,mixed,0.68,Broadcast Hunter | Song Hunter,da,0.7,en,0.8,breaking news just in from,radio/nyheder,THEME:META,0.25
```

## Example 3: Mixed Content

```csv
source,filename,clip,audio_type_guess,audio_type_confidence,recommended_mode,language_guess_clip,clip_text_signature
MixedAudio.mp3,Track__0001__jingle.mp3,1,jingle_ad,0.75,Broadcast Hunter,da,køb nu og spar
MixedAudio.mp3,Track__0002__speech.mp3,2,speech,0.85,Broadcast Hunter,en,this is the weather forecast
MixedAudio.mp3,Track__0003__music.mp3,3,music,0.90,Song Hunter | Broadcast Hunter,unknown,
```

## Field Descriptions

### audio_type_guess
Classification based on spectral features, beat strength, and energy patterns:
- **music**: Strong beat, consistent tempo, musical spectral profile
- **speech**: High zero-crossing rate variation, lower spectral centroid
- **mixed**: Contains both musical and speech characteristics
- **jingle_ad**: Short musical segment with potential speech
- **unknown**: Unable to classify with confidence

### audio_type_confidence
Confidence score from 0.0 (no confidence) to 1.0 (very confident)
- < 0.5: Low confidence, treat with caution
- 0.5 - 0.7: Moderate confidence
- 0.7 - 0.9: Good confidence
- > 0.9: Very high confidence

### recommended_mode
Suggested processing mode based on audio type:
- Music → "Song Hunter | Broadcast Hunter"
- Speech → "Broadcast Hunter"
- Mixed → "Broadcast Hunter | Song Hunter"

### language_guess_clip vs language_guess_file
- `language_guess_file`: Detected from first transcribed clip (inherits from first language detection)
- `language_guess_clip`: Detected from individual clip transcription via Whisper
- Clips can have different languages than the overall file language

### clip_text_signature
Normalized version of transcript (lowercase, first 10 words) used for:
- Grouping similar phrases together
- Detecting repeated content/jingles
- Deduplication in post-processing

Example:
- Original: "Hold on, I'm coming to get you right now!"
- Signature: "hold on i'm coming to get you right now"

## Usage in Analysis

### Filtering by Audio Type
```python
# Filter to only music clips
music_clips = df[df['audio_type_guess'] == 'music']

# Filter high-confidence speech
confident_speech = df[(df['audio_type_guess'] == 'speech') & (df['audio_type_confidence'] > 0.7)]
```

### Grouping by Language
```python
# Group clips by detected language
by_language = df.groupby('language_guess_clip').size()
```

### Finding Repeated Phrases
```python
# Find repeated phrases (jingles, ads, etc.)
repeated = df[df.duplicated(subset=['clip_text_signature'], keep=False)]
```

### Quality Control
```python
# Flag clips where file and clip language detection disagree
language_mismatch = df[df['language_guess_file'] != df['language_guess_clip']]
```
