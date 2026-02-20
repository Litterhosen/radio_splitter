# Publicer Og Del Apps (Streamlit + Gradio)

Denne guide gør dine apps klar til deling med links.

## 1) Installer alt lokalt (én kommando)

```powershell
cd C:\Users\brian\Programmering\radio_splitter2
powershell -ExecutionPolicy Bypass -File tools\install_apps.ps1 -SmokeTest
```

Hvis du vil genopbygge miljøerne helt fra bunden:

```powershell
powershell -ExecutionPolicy Bypass -File tools\install_apps.ps1 -Recreate -SmokeTest
```

## 2) Kør apps lokalt

Main Streamlit:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_main_streamlit.ps1 -Port 8501
```

Mini Streamlit:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_mini_streamlit.ps1 -Port 8502
```

Gradio:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_gradio.ps1 -Port 7860
```

Hvis Streamlit skal være synlig på dit lokale netværk:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_main_streamlit.ps1 -Port 8501 -Address 0.0.0.0 -Headless
powershell -ExecutionPolicy Bypass -File tools\run_mini_streamlit.ps1 -Port 8502 -Address 0.0.0.0 -Headless
```

## 3) Gør Streamlit offentligt tilgængelig (stabilt link)

Brug Streamlit Community Cloud med dit GitHub-repo.

1. Push dine seneste ændringer til GitHub.
2. Gå til `https://share.streamlit.io` og klik `New app`.
3. Opret app for main:
   - Repository: `Litterhosen/radio_splitter`
   - Branch: `main` (eller din ønskede branch)
   - Main file path: `app.py`
4. Opret app for mini (hvis du vil dele den separat):
   - Repository: `Litterhosen/radio_splitter`
   - Branch: `main`
   - Main file path: `mini_splitter/app.py`

Begge apps får egne offentlige URLs.

## 4) Gør Gradio offentligt tilgængelig (hurtigt link)

`Gradio/app_gradio.py` kører med `share=True`.

Når du starter Gradio via scriptet ovenfor, får du et offentligt `gradio.live` link i terminalen.

Vigtigt:
- Linket virker kun mens processen kører.
- Start appen igen for et nyt link.

## 5) Valgfri: Del lokal Streamlit via tunnel

Hvis du vil dele en lokal Streamlit med det samme uden cloud deploy:

```powershell
cloudflared tunnel --url http://localhost:8501
```

Du får et midlertidigt offentligt link. Gentag for port `8502` til mini-appen.
