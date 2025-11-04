# SSH Session Information (Lines 52-71)

## Jupyter Token
```
raCeIZmIgAsHDWQasGV9mKg2+W1jX9kxKgSKD5Z8G7RnWHFJy
```

## System Information
- **Host**: AMD DevCloud
- **Instance**: 7-gpu-mi300x8-1536gb-devcloud-atl1
- **Environment**: Jupyter + ROCm running under Docker container

## Docker Commands

### View Container Logs
```bash
docker logs rocm
```

### Access Interactive Shell
```bash
docker exec -it rocm /bin/bash
```
*(Use Ctrl+D to detach once finished)*

## Other Commands
- Delete message of the day: `rm -rf /etc/update-motd.d/99-one-click`
- Last login: Tue Nov 4 04:10:40 2025 from 137.132.26.64


