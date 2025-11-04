# TikTok Video Datasets - Based on Available Space

Check your disk space first, then choose the right TikTok dataset.

## 🎬 Best TikTok Video Datasets

### 1. TikTok-10M (450 downloads) ⭐ **LARGEST**

**10 million TikTok videos - VERY LARGE:**

```bash
# Check disk space first!
df -h

# Download (use streaming for large datasets)
python3 << EOF
from datasets import load_dataset

# WARNING: This is HUGE - 10M videos
# Use streaming=True to avoid downloading everything
dataset = load_dataset("The-data-company/TikTok-10M", streaming=True)

# Sample a few
for i, sample in enumerate(dataset['train'].take(5)):
    print(f"Sample {i}: {sample}")
EOF
```

**Dataset:** `The-data-company/TikTok-10M`
- **Downloads:** 450 | **Likes:** 5
- **Size:** 10 MILLION videos (VERY LARGE - check space first!)
- **Link:** https://hf.co/datasets/The-data-company/TikTok-10M

---

### 2. TikTok Videos Analytics (187 downloads) ⭐ **RECOMMENDED**

**Smaller, manageable dataset:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("datahiveai/Tiktok-Videos")
print(f"Examples: {len(dataset['train'])}")
print(f"Sample: {dataset['train'][0]}")
EOF
```

**Dataset:** `datahiveai/Tiktok-Videos`
- **Downloads:** 187
- **Size:** 1K-10K videos (much smaller)
- **Includes:** Engagement metrics, metadata
- **Link:** https://hf.co/datasets/datahiveai/Tiktok-Videos

---

### 3. YouTube Shorts & TikTok Trends (351 downloads)

**Trends dataset (not full videos):**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("tarekmasryo/youtube-tiktok-trends-2025")
print(f"Examples: {len(dataset['train'])}")
EOF
```

**Dataset:** `tarekmasryo/youtube-tiktok-trends-2025`
- **Downloads:** 351 | **Likes:** 5
- **Size:** 10K-100K entries (trends data, not videos)
- **Link:** https://hf.co/datasets/tarekmasryo/youtube-tiktok-trends-2025

---

### 4. TikTok Comments (40 downloads)

**Comments dataset:**

```bash
python3 << EOF
from datasets import load_dataset

dataset = load_dataset("Claire925/tiktok-comments")
print(f"Examples: {len(dataset['train'])}")
EOF
```

**Dataset:** `Claire925/tiktok-comments`
- **Downloads:** 40
- **Size:** 10K-100K comments
- **Link:** https://hf.co/datasets/Claire925/tiktok-comments

---

## Check Disk Space First

**Before downloading:**

```bash
# Check available space
df -h

# Check Hugging Face cache size
du -sh ~/.cache/huggingface/ 2>/dev/null || echo "No cache yet"

# Quick check
df -h / | tail -1 | awk '{print "Available: " $4}'
```

---

## Recommendations Based on Space

### If you have < 100GB free:
- ❌ **Don't download TikTok-10M** (too large)
- ✅ Use `datahiveai/Tiktok-Videos` (smaller dataset)
- ✅ Use `tarekmasryo/youtube-tiktok-trends-2025` (trends data, not videos)

### If you have 100GB-500GB free:
- ✅ Try `datahiveai/Tiktok-Videos`
- ⚠️ Use `TikTok-10M` with `streaming=True` (don't download all)

### If you have > 500GB free:
- ✅ Can download `TikTok-10M` (but still use streaming for safety)

---

## Quick Download (Safe Options)

**Small dataset (recommended):**

```bash
pip install datasets

python3 << EOF
from datasets import load_dataset

# Small, manageable TikTok dataset
dataset = load_dataset("datahiveai/Tiktok-Videos")
print(f"Examples: {len(dataset['train'])}")
print(f"Columns: {dataset['train'].column_names}")
print(f"Sample: {dataset['train'][0]}")
EOF
```

**Trends data (no videos, just metadata):**

```bash
python3 << EOF
from datasets import load_dataset

# Trends data - much smaller
dataset = load_dataset("tarekmasryo/youtube-tiktok-trends-2025")
print(f"Examples: {len(dataset['train'])}")
EOF
```

---

## Summary

| Dataset | Size | Videos | Recommended For |
|---------|------|--------|-----------------|
| **TikTok-10M** | HUGE | 10M | Large research projects |
| **Tiktok-Videos** | Small | 1K-10K | ✅ Most users |
| **youtube-tiktok-trends** | Medium | Trends only | Analytics/trends |

---

## My Recommendation

**Start with `datahiveai/Tiktok-Videos`** - it's smaller and manageable, and includes engagement metrics.

**Check your space first:**
```bash
df -h
```

Then download based on available space!


