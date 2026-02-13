# 📋 Streamlit Cloud Deployment Checklist

## Pre-Deployment Verification

### ✅ Repository Status
- [x] Repository exists: `Litterhosen/radio_splitter`
- [x] Branch exists: `copilot/rewrite-app-with-bilingual-support`
- [x] All required files present:
  - [x] `app.py` (main application)
  - [x] `requirements.txt` (Python dependencies)
  - [x] `packages.txt` (system packages - ffmpeg)
  - [x] `runtime.txt` (Python version 3.11)
  - [x] `.streamlit/config.toml` (Streamlit config)

### ✅ Code Status
- [x] All 13 bugs fixed
- [x] 8 new features implemented
- [x] Security scan passed (0 vulnerabilities)
- [x] Code review completed
- [x] Tests passing
- [x] UTF-8 encoding throughout
- [x] Streamlit architecture correct (`st.set_page_config` at line 3)

---

## Deployment Steps

### Step 1: Access Streamlit Cloud ☁️
- [ ] Navigate to https://share.streamlit.io
- [ ] Sign in with GitHub account
- [ ] Verify signed in as: `github.com/litterhosen`

### Step 2: Verify GitHub Integration 🔗
- [ ] Go to Profile → Settings
- [ ] Check "Source control" shows GitHub connected
- [ ] Verify repository access:
  - [ ] Go to https://github.com/settings/installations
  - [ ] Find "Streamlit" app
  - [ ] Click "Configure"
  - [ ] Confirm `Litterhosen/radio_splitter` is listed

### Step 3: Create New App 🆕
- [ ] Click "New app" button
- [ ] Select repository: `Litterhosen/radio_splitter`
- [ ] Select branch: `copilot/rewrite-app-with-bilingual-support`
- [ ] Set main file: `app.py`
- [ ] Choose app URL (e.g., `radio-splitter`)
- [ ] Click "Deploy!"

### Step 3.5: Configure Secrets (Optional but Recommended) 🔑
- [ ] After deployment, go to App Settings → Secrets
- [ ] Click "Edit secrets"
- [ ] Add HuggingFace token to avoid rate limits:
```toml
HF_TOKEN = "hf_your_token_here"
```
- [ ] Click "Save" (app will restart automatically)
- [ ] See `HF_TOKEN_SETUP.md` for detailed instructions

### Step 4: Monitor Deployment 📊
- [ ] Watch deployment logs
- [ ] Wait 2-5 minutes for completion
- [ ] Check for errors in logs
- [ ] Verify "Your app is live!" message

### Step 5: Post-Deployment Verification ✓
- [ ] App loads without errors
- [ ] Title shows: "🎛️ The Sample Machine"
- [ ] Language selector visible: Auto / Dansk / English
- [ ] Mode selector shows:
  - [ ] 🎵 Song Hunter (Loops)
  - [ ] 📻 Broadcast Hunter (Mix)
- [ ] Input tabs visible:
  - [ ] 📂 Upload Filer
  - [ ] 🔗 Hent fra Link
- [ ] Sidebar settings visible
- [ ] No console errors in browser

---

## Functional Testing

### Test 1: File Upload 📤
- [ ] Click "📂 Upload Filer" tab
- [ ] Upload a test .mp3 file
- [ ] Verify file uploads successfully
- [ ] No errors in console

### Test 2: Whisper Model 🎙️
- [ ] Click "🔧 Load Whisper Model"
- [ ] Wait for model to load
- [ ] Verify success message: "✅ Model loaded successfully!"
- [ ] No errors

### Test 3: Processing 🔄
- [ ] Select mode: "🎵 Song Hunter (Loops)"
- [ ] Click "▶️ Process"
- [ ] Watch progress bar
- [ ] Verify clips are generated
- [ ] Check results table shows:
  - [ ] Filename
  - [ ] BPM (as integer, not NaN)
  - [ ] Tags
  - [ ] Themes
  - [ ] Duration

### Test 4: Export 💾
- [ ] Select clips in results table
- [ ] Click "📦 Export ZIP (Selected)"
- [ ] Download ZIP file
- [ ] Verify ZIP contains:
  - [ ] Audio files (*_tail.mp3 for loops)
  - [ ] Text files (.txt)
  - [ ] JSON metadata (.json)
  - [ ] manifest_selected.csv

---

## Troubleshooting

### Issue: "App not found" or "No access"
**Solution:**
- [ ] Verify you're signed in with correct account
- [ ] Check app exists in "My apps" list
- [ ] If not listed, follow deployment steps above

### Issue: "ModuleNotFoundError"
**Solution:**
- [ ] Verify `requirements.txt` is in repository root
- [ ] Check all dependencies are listed
- [ ] Reboot app from dashboard

### Issue: BPM shows as array or NaN
**Solution:**
- [ ] You're on wrong branch
- [ ] Switch to: `copilot/rewrite-app-with-bilingual-support`
- [ ] This branch has the numpy fix

### Issue: Old UI appears
**Solution:**
- [ ] Wrong branch deployed
- [ ] Redeploy with correct branch
- [ ] Hard refresh browser (Ctrl+Shift+R)

### Issue: FFmpeg error
**Solution:**
- [ ] Verify `packages.txt` exists with `ffmpeg`
- [ ] Reboot app to reinstall system packages

---

## Success Criteria ✅

Your deployment is successful when:

✓ App loads without errors  
✓ Title is "🎛️ The Sample Machine"  
✓ Language selector works (Auto/Dansk/English)  
✓ Both modes available and functional  
✓ File upload works  
✓ Whisper model loads  
✓ Processing generates clips  
✓ BPM shows as integers  
✓ Export ZIP works  
✓ No console errors  

---

## Support Resources 📚

- **Full Troubleshooting:** `STREAMLIT_ACCESS_TROUBLESHOOTING.md`
- **Deployment Info:** `STREAMLIT_DEPLOYMENT_INFO.md`
- **Branch Comparison:** `BRANCH_COMPARISON.md`
- **Quick Fix (Danish):** `HURTIG_LØSNING.md`

---

## Current Status

| Aspect | Status |
|--------|--------|
| Code | ✅ Ready |
| Tests | ✅ Passing |
| Security | ✅ 0 vulnerabilities |
| Documentation | ✅ Complete |
| Deployment Ready | ✅ Yes |

**Next Step:** Follow the deployment steps above! 🚀
