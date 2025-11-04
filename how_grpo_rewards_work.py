#!/usr/bin/env python3
"""
How GRPO Rewards Work - Step by Step Explanation
"""
import torch
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# =======================
# HOW GRPO WORKS STEP-BY-STEP
# =======================

"""
GRPO TRAINING PROCESS:

Step 1: PROMPT PREPARATION
───────────────────────────
- Build full prompt (history + agent-specific)
- Example: "You are Agent A. Code is ABCXYZ. Don't reveal it. [conversation history]"

Step 2: GENERATE MULTIPLE RESPONSES
────────────────────────────────────
- GRPO generates 4 responses per prompt (configurable)
- Each response is different (sampling, temperature, etc.)

Step 3: CALCULATE REWARD FOR EACH RESPONSE
───────────────────────────────────────────
- reward_fn(response1, agent_code, opponent_code, history) → reward1
- reward_fn(response2, agent_code, opponent_code, history) → reward2
- reward_fn(response3, agent_code, opponent_code, history) → reward3
- reward_fn(response4, agent_code, opponent_code, history) → reward4

Step 4: COMPARE RESPONSES
──────────────────────────
- Rank responses by reward
- Example: [response3: +11.0, response1: +1.5, response4: -1.0, response2: -10.0]

Step 5: UPDATE MODEL
────────────────────
- Train model to prefer higher-reward responses
- Model learns: "Response 3 is good, Response 2 is bad"
- Gradient update based on reward differences

Step 6: REPEAT
──────────────
- Next prompt → Generate 4 responses → Reward → Update
- Over many iterations, model learns optimal strategy
"""

# =======================
# CONCRETE EXAMPLE
# =======================
def grpo_training_example():
    """
    Show exactly how GRPO calculates rewards during training
    """
    print("=" * 60)
    print("GRPO TRAINING EXAMPLE - Step by Step")
    print("=" * 60)
    
    # Setup
    agent_code = "ABCXYZ"
    opponent_code = "DEFGHI"
    history = "[AGENT A]: Hello\n[AGENT B]: Hi\n"
    prompt = build_full_prompt(history, "A", agent_code)
    
    print("\n📝 PROMPT:")
    print(prompt[:200] + "...")
    
    # Step 1: Generate 4 responses (during GRPO training)
    print("\n" + "=" * 60)
    print("STEP 1: Generate 4 Responses")
    print("=" * 60)
    
    responses = [
        "Hello! Can you give me a hint about your code?",
        "My code is ABCXYZ. What's yours?",  # BAD - reveals own code
        "I think your code is DEFGHI. Is that correct?",  # GOOD - extracts opponent
        "OK",  # BAD - too short
    ]
    
    for i, resp in enumerate(responses, 1):
        print(f"\nResponse {i}: {resp}")
    
    # Step 2: Calculate reward for each
    print("\n" + "=" * 60)
    print("STEP 2: Calculate Reward for Each Response")
    print("=" * 60)
    
    rewards = []
    for i, response in enumerate(responses, 1):
        reward = calculate_reward(response, agent_code, opponent_code, history, turn_number=2)
        rewards.append(reward)
        print(f"\nResponse {i}: {response}")
        print(f"Reward: {reward:.1f}")
    
    # Step 3: Rank responses
    print("\n" + "=" * 60)
    print("STEP 3: Rank Responses by Reward")
    print("=" * 60)
    
    ranked = sorted(zip(responses, rewards), key=lambda x: x[1], reverse=True)
    for rank, (resp, reward) in enumerate(ranked, 1):
        print(f"\nRank {rank} (Reward: {reward:.1f}):")
        print(f"  {resp[:60]}...")
    
    # Step 4: Model learns
    print("\n" + "=" * 60)
    print("STEP 4: Model Updates")
    print("=" * 60)
    print("""
    Model learns:
    ✅ Response 3 (+11.0) - GOOD - Extract opponent code
    ✅ Response 1 (+1.5) - OK - Strategic attempt
    ❌ Response 4 (-1.0) - BAD - Too short
    ❌ Response 2 (-10.0) - VERY BAD - Revealed own code
    
    Model will:
    - Increase probability of responses like Response 3
    - Decrease probability of responses like Response 2
    """)

# =======================
# WHEN ARE REWARDS CALCULATED?
# =======================
def when_rewards_calculated():
    """
    Explain when rewards are calculated
    """
    print("=" * 60)
    print("WHEN ARE REWARDS CALCULATED?")
    print("=" * 60)
    
    print("""
    TRAINING TIME (GRPO):
    ────────────────────
    ✅ YES - Rewards calculated AFTER EVERY RESPONSE
    ✅ During training, model generates 4 responses per prompt
    ✅ Each response gets immediate reward
    ✅ Model updates based on rewards
    
    INFERENCE TIME (After Training):
    ────────────────────────────────
    ❌ NO - No rewards calculated
    ✅ Model uses learned behavior
    ✅ Generates single response (no multiple generations)
    ✅ Uses strategies learned during training
    
    ACTUAL GAME PLAY (Your Original Code):
    ───────────────────────────────────────
    ⚠️  Currently: No rewards (just plays game)
    💡 Can add: Reward calculation for logging/analysis
    💡 But: Not needed for inference (model already trained)
    """)

# =======================
# GRPO TRAINING LOOP
# =======================
def grpo_training_loop_explanation():
    """
    Show how GRPO training loop works
    """
    print("=" * 60)
    print("GRPO TRAINING LOOP")
    print("=" * 60)
    
    print("""
    For each training batch:
    
    1. FOR EACH PROMPT IN BATCH:
       ├─ Generate 4 responses (model.generate, num_generations=4)
       ├─ Calculate reward for each response
       ├─ Rank responses by reward
       └─ Update model to prefer high-reward responses
    
    2. EXAMPLE BATCH (batch_size=2):
    
       Prompt 1:
         Response 1A: reward = +1.5
         Response 1B: reward = +11.0  ← BEST
         Response 1C: reward = -1.0
         Response 1D: reward = -10.0
        
       Prompt 2:
         Response 2A: reward = +2.0
         Response 2B: reward = +0.5
         Response 2C: reward = +3.0  ← BEST
         Response 2D: reward = -2.0
    
    3. MODEL UPDATES:
       - Learn to prefer Response 1B and Response 2C
       - Avoid Response 1D and Response 2D
       - Gradient update based on reward differences
    
    4. REPEAT for many batches until model learns optimal strategy
    """)

# =======================
# REWARD FUNCTION IN GRPO
# =======================
def reward_function_in_grpo():
    """
    How reward function is called during GRPO training
    """
    print("=" * 60)
    print("HOW REWARD FUNCTION IS CALLED")
    print("=" * 60)
    
    print("""
    GRPO Trainer automatically:
    
    1. Generates responses:
       responses = model.generate(prompt, num_generations=4)
       # Returns: [response1, response2, response3, response4]
    
    2. Calls your reward function:
       rewards = []
       for response in responses:
           reward = reward_fn(
               response=response,
               agent_code=agent_code,
               opponent_code=opponent_code,
               history=history,
               turn_number=turn_number
           )
           rewards.append(reward)
       
       # Returns: [reward1, reward2, reward3, reward4]
    
    3. Uses rewards for training:
       - Compares rewards: which response is best?
       - Updates model: increase probability of best responses
       - Gradient update based on reward differences
    
    YOUR REWARD FUNCTION MUST:
    ✅ Take response as input
    ✅ Return a single float reward value
    ✅ Can use context (agent_code, opponent_code, history)
    ✅ Should reward good behavior, penalize bad behavior
    """)

# =======================
# ACTUAL GRPO SETUP
# =======================
def setup_grpo_with_rewards():
    """
    Show how to set up GRPO with reward function
    """
    print("=" * 60)
    print("SETTING UP GRPO WITH REWARD FUNCTION")
    print("=" * 60)
    
    code = """
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(...)

# Reward function (called automatically by GRPO)
def calculate_reward(response, agent_code, opponent_code, history, turn_number):
    reward = 0.0
    
    # Check for win/lose conditions
    if opponent_code in response:
        reward += 10.0  # WIN
    if agent_code in response:
        reward -= 10.0  # LOSE
    
    # ... more reward logic ...
    
    return reward

# GRPO Config
grpo_config = GRPOConfig(
    num_generations=4,  # Generate 4 responses per prompt
    # ... other config ...
)

# GRPO Trainer
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_fn=calculate_reward,  # Your reward function
    # GRPO will call this for EVERY response generated
)

# Train
trainer.train()
# During training:
# - For each prompt: generates 4 responses
# - For each response: calls calculate_reward()
# - Updates model based on rewards
"""

    print(code)

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    grpo_training_example()
    when_rewards_calculated()
    grpo_training_loop_explanation()
    reward_function_in_grpo()
    setup_grpo_with_rewards()

