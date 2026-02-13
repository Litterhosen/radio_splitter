# 🔑 HuggingFace Token Setup Guide

## Why Set a HuggingFace Token?

When the app downloads Whisper models from HuggingFace Hub without authentication, you'll see this warning:

```
Warning: You are sending unauthenticated requests to the HF Hub. 
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

Setting a HuggingFace token provides:
- ✅ **Higher rate limits** - No throttling on downloads
- ✅ **Faster downloads** - Better performance
- ✅ **No warning messages** - Clean deployment logs

## 🌐 Getting Your Token

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Sign in or create a free account
3. Click "New token" 
4. Give it a name (e.g., "Radio Splitter App")
5. Select **Read** role (sufficient for downloading models)
6. Click "Generate token"
7. Copy your token (starts with `hf_`)

## 🚀 Setup for Streamlit Cloud (Recommended)

### Step 1: Access Your App Settings
1. Go to [Streamlit Cloud Dashboard](https://share.streamlit.io)
2. Find your `radio_splitter` app
3. Click the **⋮** menu → **Settings**

### Step 2: Add the Secret
1. In Settings, go to the **Secrets** tab
2. Click **Edit secrets**
3. Add the following (paste your actual token):
```toml
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
4. Click **Save**
5. Your app will automatically restart

### Step 3: Verify
After restart, check the deployment logs. The warning should be gone! ✅

## 💻 Setup for Local Development

### Option 1: Environment Variable (Temporary)

**macOS/Linux:**
```bash
export HF_TOKEN="hf_your_token_here"
streamlit run app.py
```

**Windows PowerShell:**
```powershell
$env:HF_TOKEN="hf_your_token_here"
streamlit run app.py
```

**Windows Command Prompt:**
```cmd
set HF_TOKEN=hf_your_token_here
streamlit run app.py
```

### Option 2: Streamlit Secrets (Persistent)

1. Copy the example file:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

2. Edit `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "hf_your_token_here"
```

3. Run the app normally:
```bash
streamlit run app.py
```

**Note:** The `secrets.toml` file is gitignored and won't be committed.

### Option 3: System Environment Variable (Persistent)

**macOS/Linux** (add to `~/.bashrc`, `~/.zshrc`, or `~/.profile`):
```bash
export HF_TOKEN="hf_your_token_here"
```

**Windows** (System Environment Variables):
1. Search for "Environment Variables" in Start menu
2. Click "Edit system environment variables"
3. Click "Environment Variables..."
4. Under "User variables", click "New..."
5. Variable name: `HF_TOKEN`
6. Variable value: Your token
7. Click OK

## 🔒 Security Notes

- ✅ **DO** keep your token private
- ✅ **DO** use `.streamlit/secrets.toml` for local development
- ✅ **DO** add `secrets.toml` to `.gitignore` (already done)
- ❌ **DON'T** commit your token to git
- ❌ **DON'T** share your token publicly
- ❌ **DON'T** hardcode it in the source code

## ✅ Verification

To verify your token is being used:

1. Run the app
2. Click "🔧 Load Whisper Model"
3. Check the logs - you should **NOT** see the warning
4. Model download should be faster

## 🆘 Troubleshooting

### Token Not Working

**Problem:** Still seeing the warning after setting token.

**Solutions:**
1. Verify token format starts with `hf_`
2. Check for typos in the token
3. Ensure token has at least **Read** permission
4. For Streamlit Cloud: Make sure you clicked "Save" and app restarted
5. For local: Restart the app after setting environment variable

### Token Expired

**Problem:** Token not working anymore.

**Solution:**
1. Go to [HuggingFace tokens page](https://huggingface.co/settings/tokens)
2. Check if token is still active
3. Generate a new token if needed
4. Update your secrets/environment variable

## 📚 Additional Resources

- [HuggingFace Tokens Documentation](https://huggingface.co/docs/hub/security-tokens)
- [Streamlit Secrets Management](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
- [faster-whisper Documentation](https://github.com/guillaumekln/faster-whisper)

## ℹ️ Optional Feature

**Important:** Setting a HuggingFace token is **optional**. The app works perfectly fine without it, but you'll see a warning message and may experience slower downloads or rate limits during peak usage.

If you're just testing or using the app occasionally, you can ignore the warning.
