# Cloud Routing Workflow

## Overview

OrthoRoute supports **cloud-based routing** for large boards that exceed local VRAM capacity. This workflow allows you to:

1. **Export** your board geometry to a portable `.ORP` file
2. **Route** on a cloud GPU (Vast.ai, RunPod, Lambda Labs, etc.) using headless mode
3. **Import** the routing solution back into KiCad

This decouples the routing algorithm from KiCad, making it possible to rent powerful GPUs for $5-10 per routing job.

---

## File Formats

### `.ORP` - OrthoRoute PCB (Board Export)

Contains complete board geometry and all information needed by the routing algorithm:

- **Board metadata**: filename, bounds (x_min, y_min, x_max, y_max), layer count
- **Pads**: position, net assignment, drill size, layer mask
- **Nets**: net name, list of terminal positions
- **DRC rules**: clearance, track width, via diameter, via drill, minimum drill
- **Grid parameters**: discretization resolution
- **Format version**: for future compatibility

**Format**: JSON with optional gzip compression
**Naming**: Automatically derives from board file (e.g., `MainController.kicad_pcb` → `MainController.ORP`)

### `.ORS` - OrthoRoute Solution (Routing Results)

Contains the complete routing solution:

- **Per-net geometry**:
  - Trace segments: (layer, start_xy, end_xy, width)
  - Vias: (position_xy, layer_from, layer_to, diameter, drill)
- **Per-iteration metrics**: Array of convergence data for each PathFinder iteration:
  - Iteration number
  - Overuse count (edges with congestion > 1)
  - Nets successfully routed
  - Total overflow cost
  - Wirelength
  - Via count
  - Iteration runtime (seconds)
- **Final routing metadata**:
  - Total iterations, convergence status
  - Total routing time
  - Final quality metrics: wirelength, via count, overflow
  - Timestamp, OrthoRoute version
- **Format version**: for compatibility checking

**Format**: JSON with optional gzip compression
**Naming**: Automatically derives from `.ORP` file (e.g., `MainController.ORP` → `MainController.ORS`)

---

## Workflow

### Step 1: Export Board (Local Machine with KiCad)

1. Open your board in KiCad
2. Launch OrthoRoute plugin (`python main.py` or from KiCad's plugin manager)
3. In the plugin GUI, select **File → Export PCB**
4. Choose save location (e.g., `MainController.ORP`)
5. Upload the `.ORP` file to your cloud instance

**What gets exported:**
- Complete board geometry
- All pad positions and net assignments
- DRC constraints
- Layer stackup information
- Everything the routing algorithm needs (no KiCad dependency)

**What is NOT exported:**
- Existing traces (assumes clean board)
- Keepout zones (ignored for now)
- Visual styling, silkscreen, etc.

---

### Step 2: Route on Cloud GPU (Headless Mode)

#### Setting Up Cloud Instance

**Recommended Providers:**
- **Vast.ai**: ~$0.37/hr for RTX 5090 (32GB VRAM)
- **RunPod**: ~$0.40/hr for RTX 4090 (24GB VRAM)
- **Lambda Labs**: ~$1.10/hr for A100 (40GB VRAM)

**Requirements:**
- Python 3.8+
- CUDA 12.x
- CuPy (`pip install cupy-cuda12x`)
- NumPy, SciPy

**Setup Script:**
```bash
# Install dependencies
pip install cupy-cuda12x numpy scipy

# Upload OrthoRoute
scp -r orthoroute/ user@cloud-instance:/workspace/
scp MainController.ORP user@cloud-instance:/workspace/

# SSH into instance
ssh user@cloud-instance
cd /workspace
```

#### Running Headless Routing

```bash
python main.py --headless MainController.ORP
```

**What happens:**
1. Loads board geometry from `.ORP` file
2. Builds routing graph and initializes PathFinder algorithm
3. Routes all nets using GPU-accelerated Dijkstra
4. Iterates until convergence (or max iterations, default 200)
5. Saves solution to `MainController.ORS`
6. Generates debug logs and outputs (same as GUI mode)

**Expected runtime:** 10-25 hours for large boards (8000+ nets, 32 layers)

**Cost estimate:**
- RTX 5090 @ $0.37/hr × 15 hours = **~$5.55**
- RTX 4090 @ $0.40/hr × 20 hours = **~$8.00**

#### Monitoring Progress

Headless mode produces identical logs to GUI mode:

```bash
# Watch log output
tail -f logs/run_YYYYMMDD_HHMMSS.log

# Check convergence
grep "CONVERGENCE" logs/run_*.log

# Monitor VRAM usage
nvidia-smi -l 1
```

#### Checkpoint Support

Headless mode fully supports checkpoints (see `docs/checkpoint_guide.md`):

```bash
# Resume from checkpoint if interrupted
python main.py --headless MainController.ORP --resume-checkpoint checkpoint_iter50.pkl
```

---

### Step 3: Import Solution (Local Machine with KiCad)

1. Download `MainController.ORS` from cloud instance
2. Open your board in KiCad
3. Launch OrthoRoute plugin
4. In the plugin GUI, select **File → Import Solution**
5. Select the `.ORS` file
6. **Preview** the routing in the plugin window
7. Click **"Apply to KiCad"** to write traces/vias to the board

**What gets imported:**
- All routed traces (layer, start/end positions, width)
- All vias (position, layer pair, diameter, drill)
- Net assignments
- Routing quality metrics (displayed in GUI)

**Result:** Your board is now routed, ready for DRC and manufacturing!

---

## Command Reference

### Export PCB (GUI)
```
File → Export PCB → [choose location] → Save as MainController.ORP
```

### Headless Routing (CLI)
```bash
# Basic usage
python main.py --headless input.ORP

# With custom output location
python main.py --headless input.ORP --output solution.ORS

# With checkpoint resume
python main.py --headless input.ORP --resume-checkpoint checkpoint.pkl

# With iteration limit
python main.py --headless input.ORP --max-iterations 200
```

### Import Solution (GUI)
```
File → Import Solution → [select .ORS file] → Preview → Apply to KiCad
```

---

## Advantages

### vs. Local Routing
- **No VRAM limits**: Rent GPUs with 32GB+ VRAM for $5-10 per job
- **Faster turnaround**: Use latest/fastest GPUs without buying hardware
- **No babysitting**: Start job, close laptop, come back to finished routes

### vs. Separate CLI Tool
- **Single codebase**: Headless mode uses EXACTLY the same algorithm as GUI
- **Zero duplication**: GUI and headless never diverge
- **Guaranteed consistency**: Same PathFinder code, same parameters, same results

---

## Implementation Notes

### Coordinate System
- All coordinates are PCB millimeters (mm)
- Origin at board lower-left corner
- Consistent between export/import

### Layer Mapping
- Uses KiCad layer IDs (0 = F.Cu, 31 = B.Cu for 32-layer board)
- Via layer pairs stored explicitly in `.ORS`

### Parameter Serialization
- Routing algorithm parameters (PathFinder weights, history scaling, etc.) are **NOT** stored in `.ORP`
- Uses default parameters from `PathFinderConfig` in headless mode
- Future: Optional parameter override via CLI args

### Error Handling
- Graceful failure for corrupted files
- VRAM overflow detection (falls back to smaller batch sizes)
- Checkpoint auto-save on interruption

---

## Troubleshooting

### "Failed to load .ORP file"
- Check file is valid JSON (or gzipped JSON)
- Verify format version compatibility
- Re-export from GUI

### "Insufficient VRAM"
- Reduce batch size via CLI args (future feature)
- Rent larger GPU (32GB vs 24GB)
- Use checkpoint system to split work

### "Solution preview looks wrong"
- Check coordinate system matches export
- Verify layer mapping is correct
- Re-import with verbose logging

### "Routing didn't converge"
- Check logs for overflow nets
- May need parameter tuning (see `docs/tuning_guide.md`)
- Try increasing max iterations

---

## Future Enhancements

- [ ] Compression support for large `.ORP` files
- [ ] Parameter override via CLI (custom PathFinder weights)
- [ ] Multi-file batch routing
- [ ] Real-time progress streaming from cloud
- [ ] Automatic cloud instance provisioning
- [ ] Resume from partial `.ORS` (for iterative refinement)

---

## Cost Calculator

| Board Complexity | Nets | Layers | Est. Time | GPU         | Cost/Run |
|------------------|------|--------|-----------|-------------|----------|
| Small            | <500 | 4-8    | 1-2 hrs   | RTX 4090    | $0.40-0.80 |
| Medium           | 500-2000 | 8-16 | 5-10 hrs  | RTX 5090    | $1.85-3.70 |
| Large            | 2000-5000 | 16-32 | 10-20 hrs | RTX 5090    | $3.70-7.40 |
| Very Large       | 5000+ | 32     | 20-30 hrs | RTX 5090    | $7.40-11.10 |

*Prices assume Vast.ai spot pricing (subject to change)*

---

## See Also

- [Checkpoint Guide](checkpoint_guide.md) - Interrupt and resume routing
- [Tuning Guide](tuning_guide.md) - PathFinder parameter optimization
- [Congestion Ratio](congestion_ratio.md) - Understanding routing quality metrics
