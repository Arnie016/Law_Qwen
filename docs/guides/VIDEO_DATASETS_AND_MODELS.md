# Video RL Fine-Tuning: Datasets & Models

## 🎬 What Are These Videos?

**Videos = 16 frames (images) played together = ~2 seconds of video**

- Each frame = image (512x512 pixels)
- 16 frames @ 8fps = 2 seconds
- Format: `[frame1, frame2, ..., frame16]` (list of images)

**Model generates:** Text → 16 frames → Video

---

## 📊 Underexplored Video Datasets (100-1000 Downloads)

### Best Options for RL Fine-Tuning:

1. **chungimungi/VideoDPO-10k** ⭐ **PERFECT FOR RL!**
   - Downloads: 1,400
   - **Purpose:** Video DPO dataset (preference data)
   - **Size:** 10K examples
   - **Why:** Already formatted for RL (DPO/GRPO)
   - **Link:** https://huggingface.co/datasets/chungimungi/VideoDPO-10k

2. **Rapidata/text-2-video-human-preferences** ✅ **Already Using**
   - Downloads: 1,400
   - **Purpose:** Human preferences for video generation
   - **Size:** 1K-10K examples
   - **Why:** Perfect for RL fine-tuning

3. **WenhaoWang/VideoUFO**
   - Downloads: 1,800
   - **Purpose:** Text-to-video dataset
   - **Size:** 1M-10M examples
   - **Why:** Large dataset, good for SFT then RL

4. **eagle0504/llava-video-text-dataset**
   - Downloads: 16
   - **Purpose:** LLaVA video-text pairs
   - **Size:** 4 examples (tiny, for testing)
   - **Why:** Very small, good for quick tests

5. **eagle0504/video-text-dataset**
   - Downloads: 10
   - **Purpose:** Video-text pairs
   - **Size:** 4 examples (tiny, for testing)
   - **Why:** Very small, good for quick tests

### Other Underexplored Options:

- **Wild-Heart/Disney-VideoGeneration-Dataset** (1.1K downloads)
  - 69 videos from Steamboat Willie
  - 6 seconds each
  - Good for style transfer

- **charlychan123/hoigen-filtered-videos** (1.3K downloads)
  - 28K filtered videos
  - Human-object interaction
  - Good for action videos

---

## 🤖 Best Open Source Video Models for Fine-Tuning

### Top Recommendations:

1. **stabilityai/stable-video-diffusion-img2vid-xt** ⭐ **BEST**
   - Downloads: 97.9K
   - **Type:** Image-to-video diffusion
   - **License:** Other (check terms)
   - **Why:** Most popular, well-documented, fine-tunable
   - **Link:** https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt

2. **ali-vilab/text-to-video-ms-1.7b** ✅ **Already Using**
   - Downloads: 2.9M
   - **Type:** Text-to-video direct
   - **License:** CC-BY-NC-4.0
   - **Why:** Text-to-video (no image needed)

3. **nvidia/Cosmos-1.0-Diffusion-7B-Video2World** 🔥 **NEW**
   - Downloads: 683
   - **Type:** Video diffusion (NVIDIA)
   - **License:** Other (check terms)
   - **Why:** Newer, NVIDIA quality, fine-tunable
   - **Link:** https://huggingface.co/nvidia/Cosmos-1.0-Diffusion-7B-Video2World

4. **nvidia/Cosmos-1.0-Diffusion-14B-Video2World**
   - Downloads: 50
   - **Type:** Larger version (14B)
   - **Why:** More capacity, better quality

### Model Comparison:

| Model | Downloads | Type | Fine-Tunable | Best For |
|-------|-----------|------|--------------|----------|
| **stable-video-diffusion-img2vid-xt** | 97.9K | Image→Video | ✅ Yes | Image-to-video |
| **text-to-video-ms-1.7b** | 2.9M | Text→Video | ✅ Yes | Text-to-video |
| **Cosmos-7B** | 683 | Video | ✅ Yes | High quality |
| **Cosmos-14B** | 50 | Video | ✅ Yes | Best quality |

---

## 🎯 Recommendations

### For RL Fine-Tuning:

**Best Dataset:** `chungimungi/VideoDPO-10k` (1.4K downloads)
- Already formatted for RL
- 10K examples
- DPO-ready format

**Best Model:** `stabilityai/stable-video-diffusion-img2vid-xt` (97.9K downloads)
- Most popular = more support
- Well-documented
- Easy to fine-tune

**Alternative:** `ali-vilab/text-to-video-ms-1.7b` (already using)
- Text-to-video direct
- Already working

### Why These Are Underexplored:

- **100-1000 downloads** = Not mainstream yet
- **Good potential** = Less competition
- **Fresh data** = Newer, less used
- **RL-ready** = Formatted for preference learning

---

## 🚀 Next Steps

1. **Try VideoDPO-10k dataset** (perfect for RL)
2. **Fine-tune Stable Video Diffusion** (most popular)
3. **Or continue with ModelScope** (already working)

**Your 205GB VRAM:** Perfect for any of these! 🎬

