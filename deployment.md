# Stock Market Trend Prediction System — Deployment Guide

This guide provides step-by-step instructions to deploy the Stock Market Trend Prediction System to various modern hosting platforms.

---

## 1. Streamlit Community Cloud (Recommended & Easiest)
Streamlit Community Cloud is a free platform that allows you to deploy public Streamlit apps directly from a GitHub repository.

### Steps:
1. Push your project code to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in using your GitHub account.
3. Click the **"New app"** button.
4. Fill in the deployment details:
   - **Repository:** `your-username/Stock-Market-Trend-Prediction-System`
   - **Branch:** `main` (or your default branch)
   - **Main file path:** `streamlit_app.py`
5. Click **Deploy**. Your app will build (installing dependencies from `requirements.txt`) and be live in a few minutes!

---

## 2. Hugging Face Spaces (Streamlit SDK)
Hugging Face Spaces is a great hosting option offering free hardware.

### Steps:
1. Sign in to your account at [huggingface.co](https://huggingface.co/) and click **"New Space"**.
2. Name your Space and select the **Streamlit** SDK.
3. Choose the **Free CPU** tier (or upgraded hardware if preferred).
4. Select **Public** or **Private** visibility, then click **Create Space**.
5. Clone the space repository locally or upload files directly. Alternatively, add Hugging Face as a git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push -f hf main
   ```
6. Hugging Face will automatically detect `requirements.txt` and `streamlit_app.py` and run your app.

---

## 3. Docker Container Deployment (Cloud Run, Render, Railway, AWS, GCP)
You can package this application as a container image and run it on any provider that supports Docker.

### Local Verification:
To build and test the Docker image locally, execute:
```bash
# Build the image
docker build -t stock-market-predictor .

# Run the container locally (access at http://localhost:8501)
docker run -p 8501:8501 stock-market-predictor
```

### Deploying to Google Cloud Run:
1. Authenticate with Google Cloud and configure Docker:
   ```bash
   gcloud auth configure-docker
   ```
2. Tag and push your image to Google Container Registry (GCR) or Artifact Registry:
   ```bash
   docker tag stock-market-predictor gcr.io/YOUR_PROJECT_ID/stock-market-predictor
   docker push gcr.io/YOUR_PROJECT_ID/stock-market-predictor
   ```
3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy stock-market-predictor \
       --image gcr.io/YOUR_PROJECT_ID/stock-market-predictor \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated \
       --port 8501
   ```

### Deploying to Render:
1. Log in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **"New +"** and choose **"Web Service"**.
3. Connect your GitHub repository.
4. Select **Docker** as the environment (Render will automatically find the `Dockerfile` at the root).
5. Choose the free instance type and click **Deploy Web Service**.

### Deploying to Railway:
1. Log in to [Railway.app](https://railway.app/).
2. Create a **New Project** and connect your GitHub repository.
3. Railway automatically detects the `Dockerfile` and deploys it on port `8501`.

---

## 4. Configuration & File System Notes
- **Read-Only Environments:** The project has been fully updated to support read-only file systems (like GCP Cloud Run and serverless containers). If the system cannot write logs, cache data, or serialize trained models to the disk, it will log a warning and proceed completely **in-memory**.
- **Streamlit Settings:** Streamlit-specific settings are configured in `.streamlit/config.toml`. Telemetry is disabled (`gatherUsageStats = false`) and the server is set to run headlessly.
- **Port Binding:** The application runs on port `8501` by default and binds to host `0.0.0.0`. If your cloud provider expects the app to bind to a dynamic port environment variable (like `$PORT` on Heroku), configure your Docker command or start script accordingly.
