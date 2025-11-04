# Credit Usage Check

## Current Status

**Instance IP**: `134.199.192.60`
**Instance Name**: `7-gpu-mi300x8-1536gb-devcloud-atl1`

## Who's Credits Are Being Used?

- **The account that created/provisioned this instance is being billed**
- SSH access doesn't change billing - whoever owns the AMD DevCloud account is consuming credits
- If your friend created the instance and shared access → **your friend's credits are being used**

## How to Check

### Option 1: Check AMD DevCloud Dashboard
1. Log into AMD DevCloud portal: https://console.amd.com/
2. Check "Billing" or "Credits" section
3. See which account shows active instance `134.199.192.60`

### Option 2: Check Instance Metadata (from inside)
```bash
# Inside the instance, check:
curl -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/instance/
# Or check environment variables
env | grep -i amd
env | grep -i project
```

### Option 3: Ask Your Friend
- Check who created the instance in the Google Doc
- Confirm if they're okay with you using their credits

## Cost Estimate

- **MI300X x8 instance**: ~$15.92/hour
- **Your $200 credit**: Would last ~12.5 hours
- **Friend's credit**: Depends on their balance

## Recommendation

1. **Confirm with friend** before heavy usage
2. **Create your own instance** if you have your own $200 credit
3. **Monitor usage** if friend agrees to share


