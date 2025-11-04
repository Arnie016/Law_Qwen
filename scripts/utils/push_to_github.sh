#!/bin/bash
# Push to GitHub
# Run this after creating GitHub repo

set -e

REPO_URL="${1:-}"

if [ -z "$REPO_URL" ]; then
    echo "=========================================="
    echo "GITHUB PUSH GUIDE"
    echo "=========================================="
    echo ""
    echo "Step 1: Create GitHub Repo"
    echo "  Go to: https://github.com/new"
    echo "  Name: amd-mi300x-legal-finetuning"
    echo "  Click 'Create repository'"
    echo ""
    echo "Step 2: Push to GitHub"
    echo "  Run: $0 https://github.com/YOUR_USERNAME/amd-mi300x-legal-finetuning.git"
    echo ""
    echo "Or run commands manually:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
    echo "  git push -u origin main"
    echo ""
    exit 1
fi

echo "=========================================="
echo "PUSHING TO GITHUB"
echo "=========================================="
echo "Repository: $REPO_URL"
echo ""

# Add remote
echo "[1/3] Adding remote..."
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

# Set branch
echo "[2/3] Setting branch..."
git branch -M main

# Push
echo "[3/3] Pushing to GitHub..."
git push -u origin main

echo ""
echo "=========================================="
echo "✅ SUCCESS! Pushed to GitHub"
echo "=========================================="
echo "Repository: $REPO_URL"
echo ""

