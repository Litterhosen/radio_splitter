# Radio Splitter + Whisper + Hooks (Streamlit)

Gratis lokal/Cloud app til:
- Radio/tale: split på stilhed + Whisper + navngivning
- Jingles: fixed length
- Songs: hooks/loops 4–15s + (valgfri) beat-refine til 1–2 bar loops
- Chorus-aware: find 30–45s vinduer og udtræk loops inde i dem

## Krav
- Python 3.12
- FFmpeg (lokalt) eller packages.txt på Streamlit Cloud

## Kør lokalt
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
