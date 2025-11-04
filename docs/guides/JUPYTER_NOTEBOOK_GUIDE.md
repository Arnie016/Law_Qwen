# Using Jupyter Notebook on DigitalOcean Droplet

## 📊 What You Have

**Jupyter Server Running:**
- **URL:** http://129.212.184.211
- **Token:** rcvqGCRpfMthHoRVFQHBvtcdiU9wszu9gM3QCMuRZfM86fs8B
- **Container:** PyTorch available inside Docker container

---

## 🎯 What You Can Do with Jupyter

### 1. **Interactive Model Exploration**
```python
# Load and test models interactively
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

# Test prompts interactively
prompt = "What is negligence?"
response = generate_response(model, tokenizer, prompt)
print(response)
```

### 2. **Data Exploration**
```python
# Explore datasets before training
from datasets import load_dataset

dataset = load_dataset("lamblamb/pile_of_law_subset")
print(dataset)
print(dataset["train"][0])

# Visualize data statistics
import pandas as pd
df = pd.DataFrame(dataset["train"])
print(df.describe())
```

### 3. **Quick Experiments**
```python
# Test reward functions
def test_reward_function():
    responses = [
        "Negligence is failure to exercise reasonable care.",
        "I think negligence might be...",
        "Negligence: breach of duty causing harm. See Restatement § 282."
    ]
    for r in responses:
        reward = legal_reward_function(r, "What is negligence?")
        print(f"Reward: {reward:.2f} - {r[:50]}...")
```

### 4. **Training Monitoring**
```python
# Monitor training progress
import json
import matplotlib.pyplot as plt

# Load training logs
with open("training_logs.json") as f:
    logs = json.load(f)

# Plot loss over time
losses = [log["loss"] for log in logs]
plt.plot(losses)
plt.title("Training Loss")
plt.show()
```

### 5. **Model Comparison**
```python
# Compare base vs fine-tuned models side-by-side
base_response = generate_response(base_model, tokenizer, question)
finetuned_response = generate_response(finetuned_model, tokenizer, question)

print("BASE MODEL:")
print(base_response)
print("\nFINE-TUNED MODEL:")
print(finetuned_response)
```

### 6. **Dataset Preparation**
```python
# Prepare datasets for training
from datasets import load_dataset

dataset = load_dataset("pile-of-law/pile-of-law", "all", streaming=True)

# Sample and format
formatted = []
for example in dataset["train"].take(1000):
    formatted.append(format_instruction(example))

# Save for training
formatted_dataset = Dataset.from_list(formatted)
formatted_dataset.save_to_disk("./formatted_dataset")
```

---

## 🔧 How to Access Jupyter

### Option 1: Web Browser (Easiest)

1. **Open browser:**
   ```
   http://129.212.184.211
   ```

2. **Enter token:**
   ```
   rcvqGCRpfMthHoRVFQHBvtcdiU9wszu9gM3QCMuRZfM86fs8B
   ```

3. **Create notebook:**
   - Click "New" → "Python 3"
   - Start coding!

### Option 2: SSH Tunnel (More Secure)

```bash
# On your local machine:
ssh -L 8888:localhost:8888 root@129.212.184.211

# Then access:
http://localhost:8888
```

---

## ⚠️ Important Notes

### PyTorch Location
**IMPORTANT:** PyTorch is only available inside Docker container!

**Inside Jupyter Notebook:**
```python
# Check if PyTorch is available
import torch
print(torch.cuda.is_available())  # Should be True

# If False, you need to run notebook inside container
```

**To use PyTorch in Jupyter:**
1. Jupyter might already be running inside container
2. Or start Jupyter inside container:
```bash
docker exec -it rocm /bin/bash
# Inside container:
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

---

## 📝 Useful Notebook Templates

### Template 1: Model Testing
```python
# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model
model_name = "Qwen/Qwen2.5-32B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test prompt
prompt = "What is negligence?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Template 2: Dataset Exploration
```python
# %%
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("lamblamb/pile_of_law_subset")
print(f"Examples: {len(dataset['train'])}")

# Show sample
print(dataset["train"][0]["text"][:500])

# Statistics
lengths = [len(ex["text"]) for ex in dataset["train"][:1000]]
print(f"Avg length: {sum(lengths)/len(lengths):.0f} chars")
```

### Template 3: Training Progress
```python
# %%
import json
import matplotlib.pyplot as plt

# Load checkpoint
checkpoint_dir = "./qwen2.5-32b-law-finetuned/checkpoint-500"

# Check training state
with open(f"{checkpoint_dir}/trainer_state.json") as f:
    state = json.load(f)

# Plot loss
losses = [log["loss"] for log in state["log_history"]]
steps = [log["step"] for log in state["log_history"]]

plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
```

### Template 4: Reward Function Testing
```python
# %%
def legal_reward_function(response, question):
    # Your reward function here
    reward = 0.0
    # ... scoring logic ...
    return reward

# Test different responses
responses = [
    "Negligence is the failure to exercise reasonable care.",
    "I think negligence might be about being careful.",
    "Negligence: breach of duty of care causing harm. See Restatement § 282."
]

for r in responses:
    reward = legal_reward_function(r, "What is negligence?")
    print(f"{reward:.2f} - {r}")
```

---

## 🚀 Best Use Cases

### 1. **Before Training:**
- Explore datasets
- Test data formatting
- Verify model loading
- Check GPU availability

### 2. **During Training:**
- Monitor logs (if accessible)
- Check GPU usage
- Test reward functions
- Debug issues

### 3. **After Training:**
- Compare models
- Evaluate responses
- Analyze results
- Visualize improvements

### 4. **Quick Prototyping:**
- Test new reward functions
- Try different prompts
- Experiment with parameters
- Debug code interactively

---

## 💡 Pro Tips

1. **Use Markdown Cells:**
   - Document your process
   - Explain what each cell does
   - Keep notes for future reference

2. **Save Frequently:**
   - Notebooks auto-save, but export important ones
   - Export as `.py` for scripts

3. **GPU Monitoring:**
   ```python
   import torch
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
   ```

4. **Memory Management:**
   ```python
   # Clear GPU cache if needed
   torch.cuda.empty_cache()
   
   # Check memory usage
   print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

---

## 🔗 Integration with Training

**Use Jupyter for:**
- ✅ Interactive exploration
- ✅ Quick testing
- ✅ Data analysis
- ✅ Visualization

**Use Terminal/Docker for:**
- ✅ Long training runs
- ✅ Background processes
- ✅ Script execution
- ✅ Production workloads

---

## 📦 Example Workflow

1. **Explore in Jupyter:**
   ```python
   # Test dataset
   dataset = load_dataset("lamblamb/pile_of_law_subset")
   print(dataset["train"][0])
   ```

2. **Develop in Jupyter:**
   ```python
   # Test reward function
   reward = legal_reward_function(test_response, test_question)
   print(reward)
   ```

3. **Export to Script:**
   - File → Download → Notebook (.ipynb)
   - Or copy code to `.py` file

4. **Run Training in Terminal:**
   ```bash
   python3 scripts/training/finetune_qwen_law.py
   ```

5. **Analyze Results in Jupyter:**
   ```python
   # Compare models
   base_score = evaluate_model(base_model)
   finetuned_score = evaluate_model(finetuned_model)
   print(f"Improvement: {finetuned_score - base_score}")
   ```

---

**Your Jupyter URL:** http://129.212.184.211  
**Token:** rcvqGCRpfMthHoRVFQHBvtcdiU9wszu9gM3QCMuRZfM86fs8B

**Start exploring!** 🚀

