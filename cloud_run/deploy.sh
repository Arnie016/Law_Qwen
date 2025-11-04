#!/bin/bash
# Cloud Run deploy script
# Usage: ./deploy.sh PROJECT_ID SERVICE_NAME REGION

PROJECT_ID=${1:-"your-project-id"}
SERVICE_NAME=${2:-"hackathon-app"}
REGION=${3:-"us-central1"}

# Build and deploy
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

gcloud run deploy ${SERVICE_NAME} \
  --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300

# Get URL
echo "Service deployed at:"
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)'


