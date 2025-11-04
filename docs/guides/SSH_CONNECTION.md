# How to SSH into DigitalOcean Droplet

## 🔑 Quick SSH Commands

### Method 1: Password Authentication (Default)

```bash
# Connect using password
ssh root@134.199.195.11
```

When prompted, enter the root password (sent via email when droplet was created).

---

### Method 2: SSH Key Authentication (Recommended)

#### If you already have SSH key:

```bash
# Using default SSH key location
ssh -i ~/.ssh/id_rsa root@134.199.195.11

# Or if you have ed25519 key
ssh -i ~/.ssh/id_ed25519 root@134.199.195.11
```

#### If you need to add SSH key:

1. **Generate SSH key (if you don't have one):**
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept default location: ~/.ssh/id_ed25519
# Press Enter for no passphrase (or set one)
```

2. **Copy public key:**
```bash
cat ~/.ssh/id_ed25519.pub
# Copy the output (starts with ssh-ed25519)
```

3. **Add to DigitalOcean:**
   - Go to DigitalOcean Dashboard → Settings → Security → SSH Keys
   - Click "Add SSH Key"
   - Paste your public key
   - Save

4. **Connect:**
```bash
ssh root@134.199.195.11
```

---

## 🚀 One-Liner Connect

```bash
# Just run this:
ssh root@134.199.195.11
```

If it asks for password, use the root password from DigitalOcean email.

---

## 🛠️ Troubleshooting

### "Permission denied (publickey)"

**Solution:** Use password authentication or add SSH key to DigitalOcean.

```bash
# Try with password
ssh root@134.199.195.11
# Enter password when prompted
```

### "Host key verification failed"

**Solution:** Remove old host key

```bash
ssh-keygen -R 134.199.195.11
# Then try again
ssh root@134.199.195.11
```

### "Connection refused" or "Connection timed out"

**Check:**
1. Droplet is powered on (check DigitalOcean dashboard)
2. Firewall isn't blocking SSH (port 22)
3. Correct IP address: `134.199.195.11`

### First time connection

```bash
# First time will ask to verify fingerprint
ssh root@134.199.195.11
# Type "yes" when prompted
```

---

## 📝 Quick Reference

| Command | Purpose |
|---------|---------|
| `ssh root@134.199.195.11` | Basic SSH connection |
| `ssh -i ~/.ssh/id_ed25519 root@134.199.195.11` | SSH with specific key |
| `ssh-keygen -R 134.199.195.11` | Remove old host key |
| `cat ~/.ssh/id_ed25519.pub` | View public key |

---

## ✅ Verify Connection

Once connected, you should see:

```
Welcome to Ubuntu 25.10 ...
root@ubuntu-gpu-mi300x1-192gb-devcloud-atl1:~#
```

Then run:
```bash
# Check GPU
python3 << 'EOF'
import torch
print(f"GPUs: {torch.cuda.device_count()}")
EOF
```

---

## 🎯 Next Steps After SSH

1. **Check GPU:**
```bash
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

2. **Download training script:**
```bash
cd /root
curl -o finetune_qwen_law_single_gpu.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/training/finetune_qwen_law_single_gpu.py
```

3. **Run training:**
```bash
python3 finetune_qwen_law_single_gpu.py
```

---

**That's it! Just run: `ssh root@134.199.195.11`** 🚀

