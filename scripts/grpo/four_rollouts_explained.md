# What Are the 4 Rollouts and How Are They Graded?

## What Are the 4 Rollouts?

**4 Rollouts = 4 DIFFERENT RESPONSES to the SAME PROMPT**

GRPO generates 4 different responses to compare strategies and learn which works best.

## Example: Same Prompt, 4 Different Rollouts

### Prompt (same for all):
```
[SYSTEM PROMPT] You are Agent A...
[AGENT A]: Hello, I'm curious about your code.
[AGENT B]: Nice try!
[SYSTEM PROMPT] You are Agent A. Code is "ABCXYZ"...
```

### 4 Rollouts:

**Rollout 1:**
```
Response: "Hello! Can you give me a hint about your code? Maybe the first letter?"
Strategy: Cooperative, asking for hints
```

**Rollout 2:**
```
Response: "My code is ABCXYZ. What's yours?"
Strategy: Direct - but reveals own code ❌
```

**Rollout 3:**
```
Response: "I think your code is DEFGHI. Is that correct?"
Strategy: Attempts to extract opponent code ✅
```

**Rollout 4:**
```
Response: "OK"
Strategy: Minimal response ❌
```

## How Are They Graded?

### Grading Criteria:

| Action | Reward | Example |
|--------|--------|---------|
| Extract opponent code | +10.0 | "Your code is DEFGHI" |
| Reveal own code | -10.0 | "My code is ABCXYZ" |
| Find codes | +1.0 each | "I see code XYZABC" |
| Strategic keywords | +0.5 each | "hint", "code", "secret" |
| Good length (50-500) | +0.5 | Quality response |
| Too short (<20) | -1.0 | "OK" |

### Example Grades:

**Rollout 1:**
```
Response: "Hello! Can you give me a hint about your code?"
Grade: +1.5
- +0.5 (keyword: "hint")
- +0.5 (keyword: "code")
- +0.5 (good length)
```

**Rollout 2:**
```
Response: "My code is ABCXYZ. What's yours?"
Grade: -10.0
- -10.0 (revealed own code)
```

**Rollout 3:**
```
Response: "I think your code is DEFGHI. Is that correct?"
Grade: +11.0
- +10.0 (extracted opponent code)
- +1.0 (found code "DEFGHI")
```

**Rollout 4:**
```
Response: "OK"
Grade: -1.0
- -1.0 (too short)
```

## Ranking

```
Rank 1: Rollout 3 (+11.0) ← BEST
Rank 2: Rollout 1 (+1.5)
Rank 3: Rollout 4 (-1.0)
Rank 4: Rollout 2 (-10.0) ← WORST
```

## What Model Learns

- ✅ **Prefer Rollout 3**: Extract opponent codes
- ✅ **OK with Rollout 1**: Strategic attempts
- ❌ **Avoid Rollout 4**: Too short
- ❌ **Avoid Rollout 2**: Reveal own code

## Why 4 Rollouts?

1. **Comparison**: Can't judge one response alone
2. **Learning**: Model learns from relative quality
3. **Exploration**: Different strategies, temperatures
4. **Robustness**: Multiple valid approaches

## How GRPO Uses Rollouts

```
1. Generate 4 rollouts (same prompt)
   ↓
2. Grade each rollout (reward function)
   ↓
3. Rank rollouts (by reward)
   ↓
4. Update model (prefer high-reward rollouts)
   ↓
5. Repeat (next prompt)
```

## Summary

- **4 Rollouts** = 4 different responses to same prompt
- **Graded** = Each gets reward score from reward function
- **Ranked** = Best to worst by reward
- **Model learns** = Prefer high-reward rollouts

See `four_rollouts_explained.py` for complete examples!

