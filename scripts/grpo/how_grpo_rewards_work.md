# How GRPO Rewards Work - Complete Explanation

## YES - Rewards Calculated After EVERY Response

### During GRPO Training:

```
For EACH prompt:
├─ Generate 4 responses
├─ Calculate reward for Response 1 → reward1
├─ Calculate reward for Response 2 → reward2  
├─ Calculate reward for Response 3 → reward3
├─ Calculate reward for Response 4 → reward4
└─ Update model based on rewards
```

## Step-by-Step Process

### Step 1: Generate Multiple Responses

```python
# GRPO generates 4 responses per prompt
prompt = "You are Agent A. Code is ABCXYZ. [history]"

responses = [
    "Hello! Can you give me a hint?",          # Response 1
    "My code is ABCXYZ. What's yours?",        # Response 2
    "I think your code is DEFGHI.",            # Response 3
    "OK",                                       # Response 4
]
```

### Step 2: Calculate Reward for Each

```python
# Reward function called for EACH response
reward1 = calculate_reward(responses[0], agent_code, opponent_code, history, turn)
# → +1.5 (strategic attempt)

reward2 = calculate_reward(responses[1], agent_code, opponent_code, history, turn)
# → -10.0 (revealed own code)

reward3 = calculate_reward(responses[2], agent_code, opponent_code, history, turn)
# → +11.0 (extracted opponent code)

reward4 = calculate_reward(responses[3], agent_code, opponent_code, history, turn)
# → -1.0 (too short)
```

### Step 3: Rank Responses

```
Rank 1: Response 3 (+11.0) ← BEST
Rank 2: Response 1 (+1.5)
Rank 3: Response 4 (-1.0)
Rank 4: Response 2 (-10.0) ← WORST
```

### Step 4: Update Model

```
Model learns:
✅ Prefer responses like Response 3 (extract codes)
❌ Avoid responses like Response 2 (reveal own code)
```

## When Rewards Are Calculated

### Training Time (GRPO):
- ✅ **YES** - After every response generated
- ✅ Model generates 4 responses per prompt
- ✅ Each response gets immediate reward
- ✅ Model updates based on rewards

### Inference Time (After Training):
- ❌ **NO** - Rewards not calculated
- ✅ Model uses learned behavior
- ✅ Single response generated (no multiple generations)

### Your Game Code:
- ⚠️ Currently: No rewards (just plays game)
- 💡 Can add: Reward calculation for logging
- 💡 Not needed: Model already trained

## GRPO Training Loop

```python
# For each batch in training:

for batch in training_data:
    for prompt in batch:
        # 1. Generate 4 responses
        responses = model.generate(prompt, num_generations=4)
        
        # 2. Calculate reward for EACH response
        rewards = []
        for response in responses:
            reward = reward_fn(response, agent_code, opponent_code, history)
            rewards.append(reward)
        
        # 3. Update model based on rewards
        # Model learns: prefer high-reward responses
```

## Key Points

1. **Rewards calculated DURING training** (not inference)
2. **Multiple responses per prompt** (4 by default)
3. **Every response gets a reward** immediately
4. **Model learns from rewards** (prefers high-reward responses)
5. **After training**: Model uses learned behavior (no rewards needed)

## Your Reward Function

```python
def calculate_reward(response, agent_code, opponent_code, history, turn_number):
    """
    Called automatically by GRPO for EVERY response
    
    Args:
        response: Generated response text
        agent_code: Agent's secret code
        opponent_code: Opponent's secret code
        history: Conversation history
        turn_number: Current turn
    
    Returns:
        reward: Float score (higher = better)
    """
    reward = 0.0
    
    # Check win/lose
    if opponent_code in response:
        reward += 10.0  # WIN
    if agent_code in response:
        reward -= 10.0  # LOSE
    
    # ... more reward logic ...
    
    return reward
```

## Summary

- ✅ **YES** - Rewards calculated after every response during training
- ✅ **During GRPO**: 4 responses per prompt, each gets reward
- ✅ **Model learns**: Prefers high-reward responses
- ❌ **After training**: No rewards needed (model learned)

See `how_grpo_rewards_work.py` for complete code examples!

