# How to Check GPUs and Space on AMD MI300X

Quick commands to check GPU count, memory, and disk space.

## Quick Commands

### Inside ROCm Container

```bash
# Enter container
docker exec -it rocm /bin/bash

# Run check script
bash /path/to/check_resources.sh
```

## Manual Commands

### 1. Check GPU Count & Info

```bash
# GPU count
rocminfo | grep -c "Agent.*GPU"

# GPU details
rocminfo | grep "Name:"

# Full GPU info
rocminfo
```

### 2. Check GPU Memory (Python)

```bash
python3 << EOF
import torch
print(f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
EOF
```

### 3. Check Disk Space

```bash
# Overall disk usage
df -h

# Current directory
du -sh .

# Specific directory
du -sh /path/to/directory
```

### 4. Check System RAM

```bash
free -h
```

### 5. Check GPU Usage (if available)

```bash
# ROCm system monitor
rocm-smi

# Or check via rocminfo
rocminfo
```

## Expected Values for MI300X

### Single GPU Instance
- **GPUs**: 1
- **GPU Memory**: ~192 GB HBM
- **GPU Name**: MI300X or similar

### 8x GPU Instance (like yours)
- **GPUs**: 8
- **GPU Memory**: ~192 GB HBM per GPU = ~1.5 TB total
- **GPU Name**: MI300X x8

## Quick One-Liners

```bash
# GPU count
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# GPU memory
python3 -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_properties(i).total_memory/1e9:.2f} GB') for i in range(torch.cuda.device_count())]"

# Disk space
df -h | head -2

# RAM
free -h | head -2
```

## Check GPU is Being Used

```bash
python3 << EOF
import torch
x = torch.randn(1000, 1000).cuda()
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
EOF
```


