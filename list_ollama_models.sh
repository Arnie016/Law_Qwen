#!/bin/bash
# List all Ollama models
echo "=========================================="
echo "Ollama Models List"
echo "=========================================="

ollama list

echo ""
echo "=========================================="
echo "Model Details"
echo "=========================================="

# Get detailed info for each model
ollama list | awk 'NR>1 {print $1}' | while read model; do
    if [ ! -z "$model" ]; then
        echo ""
        echo "Model: $model"
        ollama show "$model" 2>/dev/null || echo "  Could not get details"
    fi
done

echo ""
echo "=========================================="
echo "Disk Usage"
echo "=========================================="

# Check Ollama models directory size
if [ -d ~/.ollama/models ]; then
    echo "Ollama models directory:"
    du -sh ~/.ollama/models 2>/dev/null || echo "Cannot check size"
    echo ""
    echo "Individual models:"
    ls -lh ~/.ollama/models/ 2>/dev/null | grep -v "^total" || echo "No models directory"
else
    echo "Ollama models directory not found at ~/.ollama/models"
fi

