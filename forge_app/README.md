# Forge App: AI Sprint Assistant

Jira Forge app that analyzes sprint progress and posts AI-generated summaries.

## Setup

```bash
# Install Forge CLI
npm install -g @forge/cli

# Login
forge login

# Create app (if not already created)
forge create

# Install dependencies
npm install
```

## Development

```bash
# Run tunnel for local dev
forge tunnel

# Deploy
forge deploy

# Install to Jira instance
forge install
```

## Features

- Button on sprint board
- Calls LLM to analyze sprint issues
- Posts summary comment

## Customization

1. Update `manifest.yml` with your app ID
2. Add LLM API key to environment variables
3. Adjust JQL query in `src/index.jsx` for your sprint structure
4. Customize comment posting logic

## Submission

1. Deploy app
2. Get install link from Forge dashboard
3. Create <5 min demo video
4. Submit to Codegeist 2025


