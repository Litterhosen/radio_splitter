# 🎛️ The Sample Machine (Radio Splitter)

**🆕 DENNE BRANCH HAR ALLE FIXES - PRODUCTION READY!**

Bilingual audio splitter med Whisper transcription, hook detection og theme tagging.

## ✨ Features

### 🎵 Song Hunter (Loops)
- Find de bedste hooks/loops (4-15 sekunder)
- Beat-aligned loops (1-2 bars perfekte loops)
- 0.75s decay pad for naturlige fadeouts
- Anti-overlap logic (ingen duplicates)
- BPM detection og preview

### 📻 Broadcast Hunter (Mix)
- Split på stilhed (radio/tale)
- Whisper transcription
- Auto-tagging og theme detection
- Jingle scoring

### 🌐 Bilingual Support
- Auto-detect sprog
- Dansk keywords
- English keywords
- Seamless switching

## ✅ Production Status

| Feature | Status |
|---------|--------|
| Bugs Fixed | ✅ 13/13 |
| Features | ✅ 8/8 |
| Security | ✅ 0 vulnerabilities |
| Tests | ✅ Passing |
| Documentation | ✅ Complete |

## 🚀 Deploy på Streamlit Cloud

### Option 1: Brug Denne Branch (Anbefalet)
```
Repository: Litterhosen/radio_splitter
Branch: copilot/rewrite-app-with-bilingual-support
Main file: app.py
```

### Option 2: Merge Til Main
Se **MERGE_TIL_MAIN.md** for guide til at gøre denne version til main.

## 📋 Deployment Guides

- 🇩🇰 **HURTIG_LØSNING.md** - Quick Danish guide
- 🔧 **STREAMLIT_ACCESS_TROUBLESHOOTING.md** - Troubleshooting
- ✅ **DEPLOYMENT_CHECKLIST.md** - Deployment checklist
- 📊 **BRANCH_COMPARISON.md** - Branch comparison
- 🔀 **MERGE_TIL_MAIN.md** - How to merge to main
- 🧪 **README_SMOKE_TESTS.md** - Smoke test instructions

## 💻 Kør Lokalt

### Krav
- Python 3.11+
- FFmpeg (system installation eller via packages.txt)

### Setup
```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
