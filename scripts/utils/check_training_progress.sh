#!/bin/bash
# Check Training Progress on Server

echo "=========================================="
echo "CHECKING TRAINING PROGRESS"
echo "=========================================="
echo ""

# Check server 129.212.184.211 (new ROCm droplet)
SERVER="129.212.184.211"
echo "🔍 Checking server: $SERVER"
echo "----------------------------------------"

ssh -o ConnectTimeout=10 root@$SERVER 2>/dev/null << 'SSH_EOF' || {
    echo "❌ Cannot connect to $SERVER"
    echo "💡 Try: ssh root@$SERVER"
    exit 1
}

echo "✅ Connected"
echo ""

# Check Docker container
if docker ps -a | grep -q rocm; then
    echo "🐳 Docker container found"
    echo ""
    
    # Check processes
    echo "📊 Training processes:"
    docker exec rocm ps aux | grep -E "(python|training|grpo)" | grep -v grep || echo "  None found"
    echo ""
    
    # Check checkpoints
    echo "💾 Checkpoints:"
    docker exec rocm bash -c "ls -lh /root/qwen2.5-32b-law-*/checkpoint-*/ 2>/dev/null | tail -5" || \
    docker exec rocm bash -c "ls -lh /root/qwen2.5-32b-law-grpo/checkpoint-*/ 2>/dev/null | tail -5" || \
    echo "  None found"
    echo ""
    
    # Check GPU
    echo "🎮 GPU Status:"
    docker exec rocm python3 << 'PYEOF' 2>/dev/null || echo "  Cannot check"
import torch
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
else:
    print("  GPU not available")
PYEOF
    echo ""
    
    # Check logs
    echo "📝 Recent logs:"
    docker exec rocm bash -c "tail -30 /root/qwen2.5-32b-law-grpo/training.log 2>/dev/null || tail -30 /root/training.log 2>/dev/null || echo 'No logs found'"
    
else
    echo "⚠️  No Docker container found"
    echo ""
    
    # Check processes on host
    ps aux | grep -E "(python|training|grpo)" | grep -v grep || echo "No training processes"
    echo ""
    
    # Check checkpoints
    ls -lh /root/qwen2.5-32b-law-*/checkpoint-*/ 2>/dev/null | tail -5 || echo "No checkpoints"
fi

SSH_EOF

echo ""
echo "✅ Check complete!"
