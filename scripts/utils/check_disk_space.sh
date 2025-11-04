#!/bin/bash
# Check available disk space
echo "=========================================="
echo "Disk Space Check"
echo "=========================================="
df -h | head -3

echo ""
echo "Current directory usage:"
du -sh . 2>/dev/null || echo "Cannot check current directory"

echo ""
echo "Hugging Face cache size:"
du -sh ~/.cache/huggingface/ 2>/dev/null || echo "No cache yet"

echo ""
echo "=========================================="
echo "Available Space Summary"
echo "=========================================="
df -h / | tail -1 | awk '{print "Available: " $4}'

