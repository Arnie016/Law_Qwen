# Production-Grade Video RL Fine-Tuning: Expert Recommendations

## 🎯 Executive Summary

**Expert Feedback:** Improve rewards, use longer-context models, implement proper RL curriculum.

**Key Changes:**
- ✅ Better reward functions (temporal, spatial, semantic)
- ✅ Longer-context models (64-128 frames)
- ✅ Proper training curriculum
- ✅ ROCm optimizations
- ✅ Keyframe-then-inbetween strategy

---

## 🔧 Critical Fixes

### 1. Model Selection (Updated)

**❌ Original:** AnimateDiff-Lightning (2-6s, subject drift)

**✅ Better:** 
- **Longer-context backbone** (latent-video diffusion with 3D UNet)
- **Motion-module variant** trained on ≥64 frames
- **Use AnimateDiff-Lightning** for distillation only (after training)

**Recommendation:**
- Start with longer-context base model
- Fine-tune on 64-128 frames
- Distill to Lightning for fast inference

---

### 2. Dataset Strategy (Updated)

**❌ Original:** Only VideoDPO-10k (offline preferences)

**✅ Better:**
- **VideoDPO-10k** for offline DPO/IPO (preference shaping)
- **Online rollouts** for your specific prompts
- **Learned rewards** (don't overfit to dataset)

**Strategy:**
- Stage 1: Offline DPO on VideoDPO-10k (β=0.2)
- Stage 2: Online GRPO with custom rewards
- Stage 3: Length curriculum (32→48→64→96 frames)

---

### 3. Reward Functions (Completely Redesigned)

**❌ Original:** "Similar to chosen" (too vague)

**✅ Better:** Decomposed, testable rewards

---

## 🎯 Production-Grade Reward Functions

### Temporal & Stability Rewards

**1. t-LPIPS (Temporal LPIPS)**
```python
def t_lpips_reward(video_frames):
    """
    LPIPS between frame t and t+1 after optical-flow warping
    Encourages smooth, non-jittery motion
    """
    import lpips
    loss_fn = lpips.LPIPS(net='alex')
    
    rewards = []
    for t in range(len(video_frames) - 1):
        # Warp frame t+1 to frame t using optical flow
        warped = warp_frame(video_frames[t+1], flow[t])
        
        # Compute LPIPS (lower = smoother)
        lpips_score = loss_fn(video_frames[t], warped).item()
        
        # Reward smoothness (1 - normalized LPIPS)
        reward = max(0, 1.0 - lpips_score)
        rewards.append(reward)
    
    return np.mean(rewards)
```

**2. Flow Smoothness**
```python
def flow_smoothness_reward(video_frames):
    """
    Penalize high curl/divergence of optical flow
    Bonus for consistent velocity profiles
    """
    flows = compute_optical_flow(video_frames)
    
    rewards = []
    for flow in flows:
        # Compute curl and divergence
        curl = np.abs(np.curl(flow))
        div = np.abs(np.divergence(flow))
        
        # Penalize high curl/divergence
        penalty = np.mean(curl + div)
        
        # Reward smoothness
        reward = max(0, 1.0 - penalty / 10.0)
        rewards.append(reward)
    
    return np.mean(rewards)
```

**3. FVD-lite (Fréchet Video Distance)**
```python
def fvd_lite_reward(video_frames):
    """
    FVD using light video-I3D
    Compute on 16-frame crops for speed
    """
    from i3d import I3D
    
    model = I3D(num_classes=400)
    
    # Extract features from 16-frame crops
    features = []
    for i in range(0, len(video_frames), 16):
        crop = video_frames[i:i+16]
        feat = model.extract_features(crop)
        features.append(feat)
    
    # Compute FVD (lower = better)
    # Compare with reference distribution
    fvd_score = compute_fvd(features, reference_features)
    
    # Reward (lower FVD = higher reward)
    reward = max(0, 1.0 - fvd_score / 100.0)
    return reward
```

---

### Subject & Identity Coherence Rewards

**4. Identity Lock**
```python
def identity_lock_reward(video_frames):
    """
    Cosine similarity of face/subject embeddings across keyframes
    Penalize drift
    """
    from facenet_pytorch import InceptionResnetV1
    
    model = InceptionResnetV1(pretrained='vggface2')
    
    # Extract embeddings from keyframes
    keyframes = video_frames[::8]  # Every 8th frame
    embeddings = []
    
    for frame in keyframes:
        # Detect face/subject
        face = detect_face(frame)
        if face is not None:
            emb = model(face)
            embeddings.append(emb)
    
    if len(embeddings) < 2:
        return 0.0  # No faces detected
    
    # Compute cosine similarity between consecutive embeddings
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim.item())
    
    # Reward consistency (high similarity = high reward)
    reward = np.mean(similarities)
    return reward
```

**5. Structure Constancy**
```python
def structure_constancy_reward(video_frames):
    """
    SSIM of warped edges (Canny/Sobel) frame-to-frame
    Preserves shapes and outfits
    """
    import cv2
    
    rewards = []
    for t in range(len(video_frames) - 1):
        # Extract edges
        edges_t = cv2.Canny(video_frames[t], 50, 150)
        edges_t1 = cv2.Canny(video_frames[t+1], 50, 150)
        
        # Warp edges_t1 to edges_t using optical flow
        flow = compute_optical_flow(video_frames[t], video_frames[t+1])
        warped_edges = warp_frame(edges_t1, flow)
        
        # Compute SSIM
        ssim_score = ssim(edges_t, warped_edges)
        
        # Reward consistency
        rewards.append(ssim_score)
    
    return np.mean(rewards)
```

---

### Text-Video Alignment Rewards

**6. CLIP-VID Score**
```python
def clip_vid_reward(prompt, video_frames):
    """
    CLIP text-video similarity on sampled frames
    Sample evenly across clip (prevents front-loading)
    """
    import clip
    
    model, preprocess = clip.load("ViT-B/32")
    
    # Sample frames evenly
    num_samples = min(8, len(video_frames))
    sampled_frames = video_frames[::len(video_frames)//num_samples]
    
    # Encode text
    text_tokens = clip.tokenize([prompt])
    text_features = model.encode_text(text_tokens)
    
    # Encode frames
    similarities = []
    for frame in sampled_frames:
        frame_tensor = preprocess(frame).unsqueeze(0)
        frame_features = model.encode_image(frame_tensor)
        
        # Compute similarity
        sim = cosine_similarity(text_features, frame_features)
        similarities.append(sim.item())
    
    # Reward (average similarity)
    reward = np.mean(similarities)
    return reward
```

**7. Caption Consistency**
```python
def caption_consistency_reward(prompt, video_frames):
    """
    Auto-caption generated clip
    Compute text-text similarity with original prompt
    """
    from transformers import BlipForConditionalGeneration
    
    # Generate caption from video
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Sample frames and generate captions
    captions = []
    for frame in video_frames[::8]:  # Every 8th frame
        caption = model.generate(frame)
        captions.append(caption)
    
    # Combine captions
    combined_caption = " ".join(captions)
    
    # Compute similarity with original prompt
    from sentence_transformers import CrossEncoder
    model = CrossEncoder('cross-encoder/stsb-roberta-base')
    
    similarity = model.predict([prompt, combined_caption])
    
    # Reward similarity
    return similarity.item()
```

---

### Long-Horizon Quality Rewards

**8. Continuation Seam Score**
```python
def seam_consistency_reward(video_frames, overlap=16):
    """
    Generate in windows (64-frame chunks with 16-frame overlap)
    Train seam-critic to rate boundary consistency
    Reward low seam artifacts
    """
    window_size = 64
    
    if len(video_frames) < window_size:
        return 0.0
    
    # Extract overlapping windows
    windows = []
    for i in range(0, len(video_frames) - window_size, window_size - overlap):
        window = video_frames[i:i+window_size]
        windows.append(window)
    
    # Compute seam consistency between windows
    seam_scores = []
    for i in range(len(windows) - 1):
        # Extract overlap region
        overlap_region_1 = windows[i][-overlap:]
        overlap_region_2 = windows[i+1][:overlap]
        
        # Compute consistency (SSIM, LPIPS, etc.)
        consistency = compute_consistency(overlap_region_1, overlap_region_2)
        seam_scores.append(consistency)
    
    # Reward consistency
    reward = np.mean(seam_scores) if seam_scores else 0.0
    return reward
```

**9. Loop-Break Reward**
```python
def loop_break_reward(video_frames, prompt):
    """
    Detect near-periodic loops via FFT of per-patch motion magnitude
    Penalize repetitive cycles unless prompt demands "looping"
    """
    # Compute motion magnitude per patch
    motion_magnitude = []
    for t in range(len(video_frames) - 1):
        flow = compute_optical_flow(video_frames[t], video_frames[t+1])
        mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
        motion_magnitude.append(np.mean(mag))
    
    # FFT to detect periodicity
    fft = np.fft.fft(motion_magnitude)
    power = np.abs(fft)
    
    # Detect dominant frequency (periodic = bad unless "loop" in prompt)
    dominant_freq = np.argmax(power[1:]) + 1
    
    # Check if prompt mentions looping
    wants_loop = "loop" in prompt.lower() or "repeat" in prompt.lower()
    
    if wants_loop:
        # Reward periodicity
        reward = power[dominant_freq] / np.sum(power)
    else:
        # Penalize periodicity
        reward = 1.0 - (power[dominant_freq] / np.sum(power))
    
    return reward
```

---

### Cinematography & Composition Rewards

**10. Aesthetic Score**
```python
def aesthetic_reward(video_frames):
    """
    Frozen aesthetic predictor on every 8th frame
    Take mean + min (min catches ugly frames)
    """
    from aesthetic_predictor import AestheticPredictor
    
    model = AestheticPredictor()
    
    # Sample frames
    sampled_frames = video_frames[::8]
    
    scores = []
    for frame in sampled_frames:
        score = model.predict(frame)
        scores.append(score)
    
    # Reward: mean + min (penalize worst frame)
    reward = np.mean(scores) + np.min(scores)
    return reward / 2.0  # Normalize
```

**11. Framing Reward**
```python
def framing_reward(video_frames):
    """
    Encourage rule-of-thirds and horizon stability
    Bonus if subject stays inside safe boxes
    """
    rewards = []
    
    for frame in video_frames:
        # Detect rule-of-thirds
        thirds_score = check_rule_of_thirds(frame)
        
        # Check horizon stability
        horizon_score = check_horizon_stability(frame)
        
        # Check subject in safe zone
        safe_zone_score = check_safe_zone(frame)
        
        # Combined reward
        reward = (thirds_score + horizon_score + safe_zone_score) / 3.0
        rewards.append(reward)
    
    return np.mean(rewards)
```

---

### Length-Aware Rewards

**12. Novelty After K**
```python
def novelty_after_k_reward(video_frames, k=32):
    """
    Reward novel content beyond frame k
    Measured by feature-diversity Δ
    """
    if len(video_frames) <= k:
        return 0.0  # Too short
    
    # Extract features from first k frames
    early_features = extract_features(video_frames[:k])
    
    # Extract features from frames after k
    late_features = extract_features(video_frames[k:])
    
    # Compute diversity
    early_diversity = compute_diversity(early_features)
    late_diversity = compute_diversity(late_features)
    
    # Reward novel content (but with coherence tax)
    novelty = late_diversity - early_diversity
    
    # Coherence tax: penalize if too different
    coherence = compute_coherence(early_features, late_features)
    
    reward = novelty * coherence  # Balance novelty and coherence
    return reward
```

---

## 🚀 Complete Reward Function

```python
def comprehensive_reward_function(prompt, video_frames, weights=None):
    """
    Comprehensive reward function with all components
    """
    if weights is None:
        weights = {
            't_lpips': 1.0,
            'flow': 1.0,
            'fvd': 1.0,
            'identity': 1.0,
            'structure': 1.0,
            'clip': 1.0,
            'caption': 1.0,
            'seam': 1.0,
            'loop': 1.0,
            'aesthetic': 1.0,
            'framing': 1.0,
            'novelty': 1.0,
        }
    
    rewards = {
        't_lpips': t_lpips_reward(video_frames),
        'flow': flow_smoothness_reward(video_frames),
        'fvd': fvd_lite_reward(video_frames),
        'identity': identity_lock_reward(video_frames),
        'structure': structure_constancy_reward(video_frames),
        'clip': clip_vid_reward(prompt, video_frames),
        'caption': caption_consistency_reward(prompt, video_frames),
        'seam': seam_consistency_reward(video_frames),
        'loop': loop_break_reward(video_frames, prompt),
        'aesthetic': aesthetic_reward(video_frames),
        'framing': framing_reward(video_frames),
        'novelty': novelty_after_k_reward(video_frames, k=32),
    }
    
    # Weighted sum
    total_reward = sum(weights[k] * rewards[k] for k in rewards)
    
    return total_reward, rewards
```

---

## 🖥️ ROCm Optimizations

### PyTorch Setup

```python
import torch

# Use PyTorch ROCm >= 2.3 with bf16
assert torch.version.hip is not None, "Need ROCm PyTorch"

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use bf16
dtype = torch.bfloat16
```

### Memory Configuration (192GB VRAM)

```python
# Conservative start
config = {
    'resolution': (512, 512),
    'num_frames': 64,
    'batch_size': 6,
    'gradient_checkpointing': True,
    'bf16': True,
}

# Aggressive (if needed)
config_aggressive = {
    'resolution': (768, 768),
    'num_frames': 48,
    'batch_size': 2,
    'gradient_checkpointing': True,
    'bf16': True,
}
```

### FSDP Configuration

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# FSDP for multi-GPU
model = FSDP(
    model,
    mixed_precision=torch.distributed.fsdp.MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    ),
    cpu_offload=False,  # Keep on GPU (you have 192GB!)
)
```

---

## 📋 Training Recipe

### Stage 1: Offline DPO

```python
# DPO on VideoDPO-10k
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    beta=0.2,  # Low KL (keep close to base)
    learning_rate=1e-4,
    max_steps=1000,
)

trainer = DPOTrainer(
    model=model,
    ref_model=reference_model,  # Frozen base
    args=dpo_config,
    train_dataset=videodpo_dataset,
)
trainer.train()
```

### Stage 2: Online GRPO

```python
# GRPO with custom rewards
from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    learning_rate=5e-5,
    max_steps=5000,
    num_generations=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=your_dataset,
    reward_funcs=[comprehensive_reward_function],
)
trainer.train()
```

### Stage 3: Length Curriculum

```python
# Progressive length training
lengths = [32, 48, 64, 96, 128]
current_length = 32

for length in lengths:
    # Train on this length
    train_at_length(length, steps=1000)
    
    # Check metrics
    if metrics_clear_threshold():
        current_length = length
    else:
        break  # Don't advance if metrics not good
```

---

## 🎬 Keyframe-Then-Inbetween Strategy

```python
def keyframe_then_inbetween(prompt, num_frames=64):
    """
    Generate keyframes first, then in-between
    """
    # Step 1: Generate keyframes (every 8-12 frames)
    keyframe_interval = 8
    num_keyframes = num_frames // keyframe_interval
    
    keyframes = []
    for i in range(num_keyframes):
        keyframe = model.generate(
            prompt,
            num_frames=1,
            cfg_scale=7.0,  # Higher CFG for keyframes
        )
        keyframes.append(keyframe)
    
    # Step 2: Generate in-between frames
    inbetween_frames = []
    for i in range(len(keyframes) - 1):
        # Generate frames between keyframe i and i+1
        frames = model.generate_inbetween(
            keyframes[i],
            keyframes[i+1],
            num_frames=keyframe_interval - 1,
            cfg_scale=3.0,  # Lower CFG for in-betweens
            optical_flow_guidance=True,
        )
        inbetween_frames.extend(frames)
    
    # Combine
    final_video = combine_keyframes_and_inbetween(keyframes, inbetween_frames)
    
    return final_video
```

---

## 📊 Monitoring Metrics

```python
metrics = {
    't_lpips': [],  # Should decrease
    'fvd_lite': [],  # Should decrease
    'clip_vid': [],  # Should increase
    'seam_score': [],  # Should increase
    'identity_drift': [],  # Should decrease
    'loopiness': [],  # Should decrease
}
```

---

## ✅ TL;DR Starter Config (192GB VRAM)

```python
config = {
    # Model
    'resolution': (512, 512),
    'num_frames': 64,
    'batch_size': 6,
    
    # Training
    'dtype': torch.bfloat16,
    'gradient_checkpointing': True,
    'learning_rate': 5e-5,
    
    # RL
    'dpo_beta': 0.2,
    'grpo_generations': 4,
    
    # Strategy
    'keyframe_interval': 8,
    'overlap': 16,
    'length_curriculum': [32, 48, 64, 96],
}
```

---

## 🎯 Next Steps

1. **Implement reward functions** (start with t-LPIPS, CLIP-VID, seam score)
2. **Set up ROCm environment** (PyTorch >= 2.3, bf16)
3. **Load longer-context model** (64-128 frames)
4. **Start with offline DPO** (VideoDPO-10k)
5. **Move to online GRPO** (custom rewards)
6. **Implement length curriculum** (progressive training)

**Your 192GB VRAM:** Perfect for this! 🚀

