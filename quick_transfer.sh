#!/bin/bash
# Quick Transfer Script
# Run this from your LOCAL machine

set -e

OLD_SERVER="134.199.192.60"
NEW_SERVER="${1:-}"  # Pass new server IP as argument
SSH_KEY="$HOME/.ssh/id_ed25519"

if [ -z "$NEW_SERVER" ]; then
    echo "Usage: $0 NEW_SERVER_IP"
    echo "Example: $0 192.168.1.100"
    exit 1
fi

echo "=========================================="
echo "TRANSFERRING TO NEW SERVER"
echo "=========================================="
echo "Old Server: $OLD_SERVER"
echo "New Server: $NEW_SERVER"
echo ""

# Step 1: Create backups on old server
echo "[1/5] Creating backups on old server..."
ssh -i "$SSH_KEY" root@$OLD_SERVER << 'SSH'
echo "Creating checkpoint backup..."
docker exec rocm bash -c "cd /root && tar -czf law_backup.tar.gz qwen2.5-32b-law-finetuned/ 2>/dev/null || echo 'No checkpoint dir'"
docker exec rocm bash -c "cd /root && tar -czf eval_results.tar.gz legal_eval_results.csv 2>/dev/null || echo 'No eval results'"
echo "✅ Backups created"
SSH

# Step 2: Transfer backups
echo "[2/5] Transferring backups..."
scp -i "$SSH_KEY" root@$OLD_SERVER:/root/law_backup.tar.gz root@$NEW_SERVER:/root/ 2>/dev/null || echo "⚠️  No checkpoint backup"
scp -i "$SSH_KEY" root@$OLD_SERVER:/root/eval_results.tar.gz root@$NEW_SERVER:/root/ 2>/dev/null || echo "⚠️  No eval results"

# Step 3: Transfer scripts
echo "[3/5] Transferring scripts..."
ssh -i "$SSH_KEY" root@$NEW_SERVER "mkdir -p /root/scripts"
scp -i "$SSH_KEY" -r /Users/hema/Desktop/AMD/*.py root@$NEW_SERVER:/root/scripts/ 2>/dev/null || echo "⚠️  Some scripts failed"
scp -i "$SSH_KEY" -r /Users/hema/Desktop/AMD/*.md root@$NEW_SERVER:/root/scripts/ 2>/dev/null || echo "⚠️  Some docs failed"
scp -i "$SSH_KEY" -r /Users/hema/Desktop/AMD/*.sh root@$NEW_SERVER:/root/scripts/ 2>/dev/null || echo "⚠️  Some scripts failed"

# Step 4: Extract on new server
echo "[4/5] Extracting on new server..."
ssh -i "$SSH_KEY" root@$NEW_SERVER << SSH
cd /root
if [ -f law_backup.tar.gz ]; then
    echo "Extracting checkpoint..."
    tar -xzf law_backup.tar.gz
    docker cp qwen2.5-32b-law-finetuned rocm:/qwen2.5-32b-law-finetuned || echo "⚠️  Docker cp failed - do manually"
fi
if [ -f eval_results.tar.gz ]; then
    echo "Extracting eval results..."
    tar -xzf eval_results.tar.gz
    docker cp legal_eval_results.csv rocm:/root/ || echo "⚠️  Docker cp failed - do manually"
fi
echo "✅ Extracted"
SSH

# Step 5: Verify
echo "[5/5] Verifying installation..."
ssh -i "$SSH_KEY" root@$NEW_SERVER << 'SSH'
echo "Checking files..."
docker exec rocm ls -lh /qwen2.5-32b-law-finetuned/checkpoint-*/ 2>/dev/null | head -5 || echo "⚠️  No checkpoint found"
docker exec rocm ls -lh /root/legal_eval_results.csv 2>/dev/null || echo "⚠️  No eval results"
ls -lh /root/scripts/*.py | head -5 || echo "⚠️  No scripts"
SSH

echo ""
echo "=========================================="
echo "✅ TRANSFER COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps on NEW server:"
echo "  1. ssh -i ~/.ssh/id_ed25519 root@$NEW_SERVER"
echo "  2. docker exec -it rocm /bin/bash"
echo "  3. Verify: ls -lh /qwen2.5-32b-law-finetuned/checkpoint-*/"
echo "  4. Test: python3 /root/scripts/load_law_model.py"
echo ""

