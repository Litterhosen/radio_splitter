# Mini Splitter

Lille Streamlit-app til:

1. Transcribe (DA/EN) med TXT + SRT + JSON
2. Tale-klip ud fra segment-timestamps
3. Musik-loops (4–16 bars) med beat-baserede vinduer

## Installation (Windows)

```bat
winget install Gyan.FFmpeg
ffmpeg -version

python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## Kørsel

```bat
streamlit run app_mini.py
```

## Struktur

```text
mini_splitter/
  app.py              # compat wrapper -> app_mini.py
  app_mini.py
  ms_audio_utils.py
  ms_transcribe.py
  ms_loops.py
  ms_clipper.py
  transcribe_fw.py
  requirements.txt
  output/
```
