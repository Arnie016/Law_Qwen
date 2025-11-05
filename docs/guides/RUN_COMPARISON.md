# Compare Base Model vs GRPO Fine-Tuned (checkpoint-1000)

## 🎯 Quick Run

### Option 1: Copy Script to Server and Run

```bash
# From local machine
scp scripts/evaluation/compare_base_vs_grpo.py root@129.212.184.211:/root/scripts/grpo/

# SSH into server
ssh root@129.212.184.211

# Enter Docker container
docker exec -it rocm /bin/bash

# Run comparison
cd /root/scripts/grpo
python3 compare_base_vs_grpo.py
```

### Option 2: Download and Run Directly

```bash
# SSH into server
ssh root@129.212.184.211
docker exec -it rocm /bin/bash

# Download script
cd /root/scripts/grpo
curl -o compare_base_vs_grpo.py \
    https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/evaluation/compare_base_vs_grpo.py

# Run comparison
python3 compare_base_vs_grpo.py
```

---

## 📊 What It Does

### Tests Both Models On:
1. 10 legal questions (contract law, constitutional law, etc.)
2. Same reward function used in training
3. Same generation parameters

### Compares:
- ✅ Reward scores (0-20 points)
- ✅ Response length
- ✅ Improvements per question
- ✅ Overall statistics

### Output:
- Console: Live comparison results
- JSON file: `comparison_results.json` (detailed results)

---

## 📈 Expected Output

```
📊 Average Rewards:
   Base Model:  9.50 / 20
   GRPO Model:  15.20 / 20
   Improvement: +5.70 points (+60.0%)

🏆 Wins:
   GRPO better: 8 / 10
   Base better: 1 / 10
   Ties:        1 / 10
```

---

## ✅ Quick Check

After running, you'll see:
- Base model average reward
- GRPO model average reward
- Improvement percentage
- Per-question breakdown

**Results saved to:** `/root/scripts/grpo/comparison_results.json`

---

## 🚀 Run Now

```bash
ssh root@129.212.184.211
docker exec -it rocm /bin/bash
cd /root/scripts/grpo
curl -o compare_base_vs_grpo.py https://raw.githubusercontent.com/Arnie016/Law_Qwen/main/scripts/evaluation/compare_base_vs_grpo.py
python3 compare_base_vs_grpo.py
```

**Takes ~10-15 minutes** (generates 20 responses total)

