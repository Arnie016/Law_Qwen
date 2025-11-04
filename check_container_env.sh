#!/bin/bash
# Check what's in the ROCm container
# Run inside: docker exec -it rocm /bin/bash

echo "=========================================="
echo "Python Environment Check"
echo "=========================================="

echo "Python version:"
python3 --version

echo ""
echo "Pip version:"
pip3 --version

echo ""
echo "Installed packages:"
pip3 list | head -20

echo ""
echo "=========================================="
echo "ROCm Check"
echo "=========================================="

if command -v rocminfo &> /dev/null; then
    echo "ROCm version:"
    rocminfo | grep -i "rocm version" | head -1
    echo ""
    echo "GPU count:"
    rocminfo | grep -c "Agent.*GPU" || echo "0"
    echo ""
    echo "GPU names:"
    rocminfo | grep "Name:" | head -5
else
    echo "rocminfo not found"
fi

echo ""
echo "=========================================="
echo "System Info"
echo "=========================================="

echo "OS:"
cat /etc/os-release | grep PRETTY_NAME

echo ""
echo "Disk space:"
df -h | head -3

echo ""
echo "RAM:"
free -h | head -2


