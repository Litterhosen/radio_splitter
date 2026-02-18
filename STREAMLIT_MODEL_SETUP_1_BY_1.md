# Streamlit Model Setup (1 for 1)

Denne guide viser, hvordan du sætter de relevante modeller op én ad gangen til:
- Main app (`app_main.py`)
- Mini app (`mini_splitter/app_mini.py`)

## 1) Aktivér venv

```powershell
cd C:\Users\brian\Programmering\radio_splitter2
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Preload faster-whisper modeller (én ad gangen)

Kør disse kommandoer én efter én. Første gang downloader modellen, derefter er den cachet.

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8'); print('tiny OK')"
python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('base OK')"
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8'); print('small OK')"
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8'); print('medium OK')"
```

## 3) Start Main Streamlit og verificér model-load

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_main_streamlit.ps1 -Port 8501
```

I UI:
1. Vælg `Model size` (fx `small`)
2. Vælg `Device = cpu`
3. Vælg `Compute type = int8`
4. Klik `Load Whisper Model`
5. Bekræft at du får `Model loaded successfully`

## 4) Start Mini Streamlit og verificér backend

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_mini_streamlit.ps1 -Port 8502
```

I Mini UI:
1. `ASR backend = Fast (faster-whisper)` for stabil standardkørsel
2. `Model = tiny/base/small`
3. Kør en fil og verificér at `transcript.txt/json/srt` bliver oprettet

## 5) (Valgfrit) WhisperX i Mini

Mini understøtter `Precise (WhisperX)`. Hvis WhisperX ikke er installeret, falder appen automatisk tilbage til faster-whisper.

Installér valgfrit i aktiv venv:

```powershell
python -m pip install whisperx
```

Verificér:

```powershell
python -c "import whisperx; print('whisperx OK')"
```

Kør Mini igen og vælg `Precise (WhisperX)`.

## 6) Fejlfinding (kort)

- Hvis model-load fejler:
  - tjek aktiv venv: `where python`
  - reinstall dependencies: `python -m pip install -r requirements.txt`
- Hvis Gradio også bruges:
  - `powershell -ExecutionPolicy Bypass -File tools\setup_local_gradio_venv.ps1 -ProjectRoot . -Recreate`

