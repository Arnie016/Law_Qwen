#!/bin/bash
# GitHub Setup Guide
# Instructions for pushing to GitHub

echo "=========================================="
echo "GITHUB SETUP GUIDE"
echo "=========================================="

cat << 'EOF'

OPTION 1: Create New GitHub Repo (Recommended)
==============================================

1. Go to: https://github.com/new
2. Repository name: amd-mi300x-legal-finetuning
3. Description: Fine-tuning Qwen 2.5 32B on legal datasets using 8x AMD MI300X GPUs
4. Visibility: Private (or Public)
5. Click "Create repository"

Then run these commands:

git remote add origin https://github.com/YOUR_USERNAME/amd-mi300x-legal-finetuning.git
git branch -M main
git push -u origin main

OPTION 2: Use GitHub CLI (If installed)
========================================

gh repo create amd-mi300x-legal-finetuning --private --source=. --remote=origin --push

OPTION 3: Manual Push (After creating repo)
============================================

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push
git push -u origin main

# If GitHub asks for credentials, use:
# Username: YOUR_GITHUB_USERNAME
# Password: Personal Access Token (not your password)
# Get token: https://github.com/settings/tokens

EOF

echo ""
echo "Current git status:"
cd /Users/hema/Desktop/AMD && git status --short | head -10

