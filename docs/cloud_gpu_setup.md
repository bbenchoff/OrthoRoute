# Cloud GPU Setup Guide for OrthoRoute

**Complete instructions for running OrthoRoute headless routing on Vast.ai or other cloud GPU providers**

**Last Updated:** November 15, 2025

---

## Step 1: Rent GPU Instance on Vast.ai

### Recommended Specifications

**For boards with <2,000 nets:**
- GPU: RTX 4090 (24 GB VRAM)
- Cost: ~$0.40/hr
- Sufficient for most boards

**For boards with 2,000-8,000 nets:**
- GPU: RTX 6000 Ada (48 GB VRAM) or A100 80GB
- Cost: ~$0.80-1.50/hr
- Needed for large backplanes

**For boards with >8,000 nets:**
- GPU: H100 80GB or A100 80GB
- Cost: ~$1.50-2.50/hr
- Maximum capacity

### On Vast.ai Website

1. Go to https://vast.ai/console/create/
2. **Filter instances:**
   - GPU Type: RTX 4090, RTX 6000 Ada, or A100
   - VRAM: ≥ 24 GB (48+ GB for large boards)
   - Disk Space: ≥ 20 GB
   - CUDA Version: 12.x or later
3. **Sort by price** ($/hr)
4. **Click "Rent"** on suitable instance
5. **Select:**
   - Image: `pytorch/pytorch:latest` (has CUDA + Python pre-installed)
   - Or: `nvidia/cuda:12.2.0-devel-ubuntu22.04`
6. **Click "Create"**

### Get SSH Connection Info

After instance starts (30-60 seconds):
1. Click on instance in dashboard
2. Copy SSH command shown (looks like):
   ```bash
   ssh -p 12345 root@ssh.vast.ai -L 8080:localhost:8080
   ```
3. Or use direct IP if shown

---

## Step 2: Connect and Setup Environment

### SSH into Instance

```bash
# Use the SSH command from Vast.ai dashboard
ssh -p 12345 root@ssh.vast.ai
```

**You should see a prompt like:**
```
root@C.27877234:~#
```

### Install System Dependencies

```bash
# Update package manager
apt-get update

# Install git and basic tools
apt-get install -y git tmux htop

# Verify CUDA is available
nvidia-smi
# Should show GPU info (e.g., RTX 4090, 24GB VRAM)

# Verify Python version
python3 --version
# Should be Python 3.8 or later
```

---

## Step 3: Clone OrthoRoute Repository

```bash
# Navigate to workspace
cd /workspace

# Clone repository
git clone https://github.com/bbenchoff/OrthoRoute.git
cd OrthoRoute

# Verify files
ls -la
# Should see: main.py, orthoroute/, logs/, etc.
```

**If using a private repository:**
```bash
# Option 1: Use HTTPS with token
git clone https://YOUR_TOKEN@github.com/YourUsername/OrthoRoute.git

# Option 2: Use SSH (need to add SSH key to GitHub first)
git clone git@github.com:YourUsername/OrthoRoute.git
```

---

## Step 4: Install Python Dependencies

### Check CUDA Version

```bash
nvcc --version
# Note the CUDA version (e.g., 12.2, 12.4, etc.)
```

### Install CuPy (GPU acceleration library)

**For CUDA 12.x:**
```bash
pip3 install cupy-cuda12x
```

**For CUDA 11.x:**
```bash
pip3 install cupy-cuda11x
```

**Verify CuPy installation:**
```bash
python3 -c "import cupy as cp; print(cp.__version__); print('GPU Available:', cp.cuda.is_available())"
# Should print: GPU Available: True
```

### Install Other Dependencies

```bash
# Install NumPy and SciPy
pip3 install numpy scipy

# Verify installations
python3 -c "import numpy; import scipy; print('NumPy:', numpy.__version__, 'SciPy:', scipy.__version__)"
```

**Complete dependency list:**
```bash
pip3 install cupy-cuda12x numpy scipy
```

**Note:** Don't install PyQt6 (GUI not needed for headless mode).

---

## Step 5: Upload Your ORP File

### From Your Local Machine

**Using SCP:**
```bash
# On your local machine (not on the Vast instance):
scp -P 12345 MainController.ORP root@ssh.vast.ai:/workspace/OrthoRoute/

# Replace:
#   12345 - with your actual port from Vast.ai
#   MainController.ORP - with your actual ORP filename
```

**Verify upload:**
```bash
# Back on the Vast instance:
cd /workspace/OrthoRoute
ls -lh *.ORP
# Should show your ORP file
```

### Alternative: Upload to Cloud Storage First

If ORP file is large:
```bash
# On local machine: Upload to temporary host
# curl -F "file=@MainController.ORP" https://file.io
# Gets back a URL

# On Vast instance: Download
wget https://file.io/XXXXXX -O MainController.ORP
```

---

## Step 6: Run OrthoRoute Headless Mode

### Using tmux (Recommended - survives SSH disconnects)

```bash
# Start new tmux session
tmux new -s routing

# Inside tmux, run OrthoRoute
cd /workspace/OrthoRoute
python3 main.py headless MainController.ORP

# Detach from tmux (keeps running in background):
# Press: Ctrl+b, then d

# Later, reattach to see progress:
tmux attach -t routing

# Kill session when done:
tmux kill-session -t routing
```

### Direct Run (Simpler but dies if SSH disconnects)

```bash
cd /workspace/OrthoRoute
python3 main.py headless MainController.ORP
```

### With Options

```bash
# Increase iterations for complex boards
python3 main.py headless MainController.ORP --max-iterations 150

# Force CPU mode if GPU runs out of memory
python3 main.py headless MainController.ORP --cpu-only

# Custom output filename
python3 main.py headless MainController.ORP -o CustomName.ORS
```

---

## Step 7: Monitor Progress

### Watch Live Console Output

**If using tmux:**
```bash
tmux attach -t routing
```

**If running directly:**
Already showing in your terminal.

### Tail Log Files

```bash
# In a second SSH session or tmux pane:
cd /workspace/OrthoRoute

# Watch latest log file
tail -f logs/run_*.log | grep "WARNING"

# Or just iteration summaries:
tail -f logs/run_*.log | grep "ITER.*nets="

# Or with watch command:
watch -n 2 'tail -5 logs/run_*.log'
```

### Monitor GPU Usage

```bash
# Watch GPU utilization every 5 seconds
nvidia-smi -l 5

# Or with watch:
watch -n 5 nvidia-smi
```

**What to look for:**
- GPU Utilization: Should be 80-100%
- GPU Memory: Should be stable (not growing infinitely)
- Power Usage: Should be near max (e.g., 350W for RTX 4090)

### Check Disk Space

```bash
# Iteration 1 on 8K nets creates LARGE log files
df -h

# If disk getting full, you can compress or delete old logs:
gzip logs/old_run_*.log
```

---

## Step 8: Handle Common Issues

### Out of Memory Error

**Error:**
```
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating X bytes
```

**Solutions:**

**A) Upgrade to larger GPU:**
- Kill current job: `pkill -f main.py`
- Destroy instance on Vast.ai
- Rent instance with more VRAM (48+ GB)
- Restart from Step 1

**B) Use CPU mode:**
```bash
pkill -f main.py
python3 main.py headless MainController.ORP --cpu-only
```

**C) Reduce batch size** (requires code change - not recommended)

### Process Killed / SSH Disconnected

**If you weren't using tmux:**
- Routing stopped when SSH died
- Must restart from scratch

**If you were using tmux:**
```bash
# Reconnect to Vast instance
ssh -p 12345 root@ssh.vast.ai

# Reattach to tmux session
tmux attach -t routing

# Routing should still be running!
```

### Instance Becomes Unresponsive

**If SSH hangs or times out:**
- Instance might have crashed
- Check Vast.ai dashboard - instance status
- If "stopped", you'll need to restart
- Unfortunately, routing progress lost (no checkpointing yet)

### Logs Too Large

**8K net routing can create 10+ GB log files:**

```bash
# Check log size
du -h logs/

# Compress old logs to save space
gzip logs/run_*.log

# Or delete very old logs
rm logs/run_2025111*.log
```

---

## Step 9: Download Results

### When Routing Completes

**You'll see:**
```
================================================================================
ROUTING COMPLETE!
================================================================================
Solution file: MainController.ORS
...
```

### Download ORS File to Local Machine

**Using SCP (from your local machine):**
```bash
scp -P 12345 root@ssh.vast.ai:/workspace/OrthoRoute/MainController.ORS ./

# Replace:
#   12345 - your Vast.ai port
#   MainController.ORS - your actual ORS filename
#   ./ - current directory (or specify path)
```

**Using cloud storage:**
```bash
# On Vast instance: Upload to file sharing service
curl -F "file=@MainController.ORS" https://file.io
# Returns download URL

# On local machine: Download
wget https://file.io/XXXXXX -O MainController.ORS
```

**Verify file integrity:**
```bash
# On local machine, check file is valid gzip:
gzip -t MainController.ORS && echo "File OK" || echo "File corrupted"

# Check file size (should be ~500KB - 5MB):
ls -lh MainController.ORS
```

---

## Step 10: Import into KiCad

**On your local machine:**

1. Open KiCad with your board
2. Launch OrthoRoute plugin
3. Press **Ctrl+I** (or File → Import Solution)
4. Select `MainController.ORS`
5. Review routing in preview
6. Click **"Apply to KiCad"** to commit traces/vias

---

## Complete Example Session

### Session Recording

```bash
# === ON LOCAL MACHINE ===

# 1. Export board
# (In KiCad OrthoRoute plugin: Ctrl+E → save MainController.ORP)

# 2. Upload to Vast
scp -P 12345 MainController.ORP root@ssh.vast.ai:/workspace/

# === ON VAST.AI INSTANCE ===

# 3. SSH in
ssh -p 12345 root@ssh.vast.ai

# 4. Setup
cd /workspace
git clone https://github.com/YourUser/OrthoRoute.git
cd OrthoRoute
pip3 install cupy-cuda12x numpy scipy

# 5. Verify GPU
nvidia-smi
python3 -c "import cupy; print('GPU:', cupy.cuda.is_available())"

# 6. Start tmux session
tmux new -s routing

# 7. Run routing
python3 main.py headless MainController.ORP

# 8. Detach from tmux (Ctrl+b, then d)

# 9. Monitor progress (optional)
tail -f logs/run_*.log | grep "ITER.*nets="

# 10. Wait for completion (check back in 4-8 hours)

# 11. Download result
exit  # Exit SSH

# === BACK ON LOCAL MACHINE ===

# 12. Download ORS file
scp -P 12345 root@ssh.vast.ai:/workspace/OrthoRoute/MainController.ORS ./

# 13. Import into KiCad (Ctrl+I)

# 14. Destroy Vast instance (stop billing)
# (In Vast.ai dashboard: click Destroy)
```

---

## Cost Estimation

### Typical Costs by Board Size

**Small board (100-500 nets):**
- Time: 10-30 minutes
- GPU: RTX 4090 @ $0.40/hr
- **Cost: $0.20**

**Medium board (500-2,000 nets):**
- Time: 30 minutes - 2 hours
- GPU: RTX 4090 @ $0.40/hr
- **Cost: $0.80**

**Large board (2,000-8,000 nets):**
- Time: 4-12 hours
- GPU: RTX 6000 Ada (48GB) @ $0.80/hr
- **Cost: $6-10**

**Huge board (8,000+ nets):**
- Time: 12-24 hours
- GPU: A100 80GB @ $1.50/hr
- **Cost: $18-36**

**vs. buying RTX 4090:** ~$1,600

**Break-even:** ~40 large routing jobs (or never, if you value your time)

---

## Tips & Tricks

### 1. Use tmux ALWAYS

```bash
# Start every session with:
tmux new -s routing

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t routing
```

**Why:** If SSH disconnects, routing keeps going. Saved me countless times.

### 2. Monitor Without Attaching

```bash
# See what's happening in tmux without attaching:
tmux capture-pane -t routing -p | tail -20
```

### 3. Multiple Sessions for Monitoring

```bash
# Window 1: Routing
tmux new -s routing
python3 main.py headless board.ORP

# Detach (Ctrl+b, d)

# Window 2: Monitoring
tmux new -s monitor
tail -f logs/run_*.log | grep "ITER.*nets="

# Detach (Ctrl+b, d)

# Switch between:
tmux attach -t routing
tmux attach -t monitor
```

### 4. Estimate Time Remaining

```bash
# From iteration timestamps, calculate rate:
# Example: ITER 10 at 10:30, ITER 20 at 11:45
# = 10 iterations in 75 minutes
# = 7.5 min/iteration
# If need 80 iterations total: (80-20) × 7.5 = 450 min = 7.5 hours
```

### 5. Verify GPU is Being Used

```bash
# Run this DURING routing:
nvidia-smi

# Look for:
#   GPU Util: 95-100%
#   Memory Usage: 20-30 GB (should be high)
#   Process: python3 main.py headless ...
```

**If GPU Util is 0%:** Routing is using CPU (slow!) - check CuPy installation.

### 6. Pre-test Small Board

Before routing huge board:
```bash
# Test with small ORP first:
python3 main.py headless TestBackplane.ORP

# Should complete in 20-30 min
# Verifies: GPU works, dependencies correct, no issues
```

### 7. Compress Logs to Save Disk

```bash
# While routing is running (in another terminal):
cd /workspace/OrthoRoute/logs
gzip run_2025*.log  # Compress old logs

# Or auto-compress with cron:
(crontab -l; echo "*/30 * * * * gzip /workspace/OrthoRoute/logs/*.log 2>/dev/null") | crontab -
```

---

## Troubleshooting

### "No module named 'cupy'"

**Problem:** CuPy not installed

**Fix:**
```bash
pip3 install cupy-cuda12x
```

### "CUDA initialization failed"

**Problem:** CUDA runtime mismatch

**Fix:**
```bash
# Check CUDA version
nvcc --version

# Install matching CuPy:
# CUDA 11.x: pip3 install cupy-cuda11x
# CUDA 12.x: pip3 install cupy-cuda12x
```

### "Permission denied" when cloning repo

**Problem:** Private repository

**Fix:**
```bash
# Generate SSH key on Vast instance:
ssh-keygen -t ed25519 -C "vast-gpu"
cat ~/.ssh/id_ed25519.pub
# Copy output, add to GitHub → Settings → SSH Keys

# Or use personal access token:
git clone https://YOUR_TOKEN@github.com/user/repo.git
```

### Routing uses CPU instead of GPU

**Check:**
```bash
python3 -c "import cupy; print('Available:', cupy.cuda.is_available())"
```

**If False:**
- CuPy not installed correctly
- CUDA version mismatch
- GPU drivers not loaded

**Force GPU mode:**
```bash
python3 main.py headless board.ORP --use-gpu
```

### Instance runs out of disk space

**Check space:**
```bash
df -h
```

**If <5 GB free:**
```bash
# Compress logs
gzip logs/*.log

# Or delete old logs
rm logs/run_2025111*.log

# Or mount external storage (Vast.ai option)
```

### Routing takes forever on CPU

**If forced to use `--cpu-only`:**
- 8K net board could take 48-72 hours
- Consider renting bigger GPU instead
- Or reduce grid resolution in ORP file

---

## Optimization Tips

### 1. Choose Right GPU for Your Board

| Board Size | Nets | VRAM Needed | Recommended GPU | Cost/hr |
|------------|------|-------------|-----------------|---------|
| Small | <500 | 8 GB | RTX 3080 | $0.25 |
| Medium | 500-2K | 16 GB | RTX 4090 | $0.40 |
| Large | 2K-6K | 24 GB | RTX 4090 | $0.40 |
| Huge | 6K-10K | 48 GB | RTX 6000 Ada | $0.80 |
| Massive | 10K+ | 80 GB | A100 80GB | $1.50 |

### 2. Batch Multiple Boards

```bash
# Route multiple boards in one session:
python3 main.py headless Board1.ORP
python3 main.py headless Board2.ORP
python3 main.py headless Board3.ORP

# Or in parallel (if enough VRAM):
python3 main.py headless Board1.ORP &
python3 main.py headless Board2.ORP &
wait
```

### 3. Auto-shutdown When Done

```bash
# Add to end of routing script:
python3 main.py headless board.ORP && shutdown -h now

# Instance stops automatically when complete
# Minimizes billing
```

---

## Quick Reference Card

**Setup:**
```bash
ssh -p PORT root@ssh.vast.ai
cd /workspace
git clone https://github.com/user/OrthoRoute.git
cd OrthoRoute
pip3 install cupy-cuda12x numpy scipy
```

**Upload file:**
```bash
# From local machine:
scp -P PORT board.ORP root@ssh.vast.ai:/workspace/OrthoRoute/
```

**Run routing:**
```bash
tmux new -s routing
python3 main.py headless board.ORP
# Ctrl+b, d to detach
```

**Monitor:**
```bash
tail -f logs/run_*.log | grep "ITER.*nets="
nvidia-smi -l 5
```

**Download result:**
```bash
# From local machine:
scp -P PORT root@ssh.vast.ai:/workspace/OrthoRoute/board.ORS ./
```

**Import to KiCad:**
```
Ctrl+I → select board.ORS → Apply to KiCad
```

---

## Expected Timeline (8K Net Board)

```
00:00 - Start instance, SSH in
00:05 - Clone repo, install dependencies
00:10 - Upload ORP file (depends on internet speed)
00:15 - Start routing in tmux
02:30 - Iteration 1 completes (greedy routing)
04:00 - Iteration 20 completes
08:00 - Iteration 50 completes
12:00 - Iteration 75 completes
14:00 - Convergence! (iteration 85-95)
14:05 - Download ORS file
14:10 - Destroy instance

Total: ~14 hours runtime, ~$12-15 cost
```

---

## Vast.ai Specific Notes

### Instance States

- **Loading:** Starting up (1-2 min)
- **Running:** Active and billable
- **Stopped:** Paused (not billable, but loses data)
- **Destroyed:** Terminated (stops billing)

### Billing

- Billed per **second** of runtime
- Continues billing until you **Destroy** instance
- Check dashboard frequently when job completes

### Data Persistence

- `/workspace` directory persists across stops
- `~/.ssh`, `/tmp` do NOT persist
- **Always destroy** when done (or you keep paying)

### Port Forwarding

SSH command includes port forwarding:
```bash
ssh -p 12345 root@ssh.vast.ai -L 8080:localhost:8080
```

You can ignore the `-L 8080:localhost:8080` part for headless routing.

---

## Other Cloud Providers

### RunPod

**Similar setup:**
```bash
# SSH command from RunPod dashboard
ssh root@X.X.X.X -p 22

# Rest is identical to Vast.ai
```

**Differences:**
- Easier UI
- Slightly more expensive (~$0.50/hr for RTX 4090)
- Better reliability
- Jupyter notebook support (not needed for headless)

### Lambda Labs

**Setup:**
```bash
ssh ubuntu@instance.lambdalabs.com
sudo apt-get install python3-pip
# Rest same as Vast.ai
```

**Differences:**
- More expensive (~$1.10/hr for A100)
- Very reliable
- Better for production workloads
- Fixed pricing (no bidding)

---

## Security Notes

### Protect Your ORP Files

ORP files contain your entire board design:
- Pad positions
- Net connectivity
- Design rules

**Don't:**
- Upload to public GitHub
- Share ORP files publicly
- Leave on instance after destroying

**Do:**
- Use private repositories
- Delete ORP/ORS from instance before destroying:
  ```bash
  rm /workspace/OrthoRoute/*.ORP
  rm /workspace/OrthoRoute/*.ORS
  ```
- Download and backup ORS files locally

### SSH Key Security

**Generate unique key for cloud instances:**
```bash
ssh-keygen -t ed25519 -f ~/.ssh/vast_key
# Use ~/.ssh/vast_key instead of default key
# If compromised, only affects cloud instances
```

---

## Post-Processing

### After Downloading ORS

**1. Verify file:**
```bash
ls -lh MainController.ORS
# Should be ~500KB - 5MB depending on board size
```

**2. Import to KiCad:**
- Ctrl+I in OrthoRoute plugin
- Select ORS file
- Review in preview

**3. Run DRC:**
- Check for violations
- Expect ~300-500 via barrel conflicts (known limitation)
- Zero trace-trace violations (should be clean)

**4. Manual cleanup (if needed):**
- Fix barrel conflicts by moving vias 0.1-0.2mm
- Typically 30-60 minutes for large boards

---

## FAQ

**Q: Can I close my laptop while routing?**
A: Yes, if using tmux! Routing continues on the cloud.

**Q: How do I know when it's done?**
A: Check tmux session or log files. Or set up email notification (advanced).

**Q: What if I run out of money mid-routing?**
A: Vast.ai stops instance, routing lost. Add credits before starting.

**Q: Can I pause and resume?**
A: Not currently. Checkpointing is a planned feature but not implemented.

**Q: GPU seems idle during routing?**
A: Check nvidia-smi. If 0%, CuPy isn't working. Use `--cpu-only` as fallback.

**Q: Can I route multiple boards in parallel?**
A: Yes, if enough VRAM. 2 small boards on 1 GPU works. Large boards need dedicated GPU.

---

**Last Updated:** November 15, 2025
**Tested On:** Vast.ai, RunPod, Lambda Labs
**GPU Tested:** RTX 4090, RTX 6000 Ada, A100 80GB
**Status:** Production-ready

