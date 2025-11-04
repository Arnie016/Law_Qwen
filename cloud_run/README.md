# Cloud Run Hackathon Deployment

Quick deploy script for Google Cloud Run hackathon entries.

## Setup

```bash
# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

## Deploy

```bash
chmod +x deploy.sh
./deploy.sh PROJECT_ID SERVICE_NAME REGION
```

Or manually:

```bash
# Build
gcloud builds submit --tag gcr.io/PROJECT_ID/SERVICE_NAME

# Deploy
gcloud run deploy SERVICE_NAME \
  --image gcr.io/PROJECT_ID/SERVICE_NAME \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```

## Categories

- **AI Studio**: Lightweight models, API endpoints
- **AI Agents (ADK)**: Agent workflows, multi-step reasoning
- **GPU (L4)**: Heavy models, use `--accelerator=type=nvidia-l4` flag

## GPU Example

```bash
gcloud run deploy SERVICE_NAME \
  --image gcr.io/PROJECT_ID/SERVICE_NAME \
  --platform managed \
  --region us-central1 \
  --accelerator=type=nvidia-l4,count=1 \
  --memory 16Gi
```

## Test

```bash
curl https://SERVICE_URL/
```


