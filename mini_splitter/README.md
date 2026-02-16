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
streamlit run app.py
```

## Struktur

```text
mini_splitter/
  app.py
  audio_utils.py
  transcribe_fw.py
  loops.py
  clipper.py
  requirements.txt
  output/
```
