"""
Generate iteration video from debug output captures.

Shows routing progress across iterations:
- Initial airwires
- Escape via planning
- Iteration 1, 2, 3... with stats

Usage:
    python viz/generate_iteration_video.py <run_folder>
    python viz/generate_iteration_video.py debug_output/run_20251104_205922
"""

import sys
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

VIDEO_WIDTH = 1920   # 1080p width
VIDEO_HEIGHT = 1080  # 1080p height
FPS = 30
HOLD_DURATION_SECS = 1  # Hold each frame for 1 second

OUTPUT_VIDEO = Path("iteration_video.mp4")  # Relative to viz/ directory


# =============================================================================
# ITERATION DATA PARSER
# =============================================================================

def parse_log_file(log_path: Path) -> Dict[int, Dict]:
    """
    Parse log file to extract iteration statistics.

    Returns:
        Dict mapping iteration number to stats:
        {
            1: {'routed': 8192, 'failed': 0, 'overuse': 1572593, ...},
            2: {'routed': 8192, 'failed': 0, 'overuse': 2842940, ...},
            ...
        }
    """
    logger.info(f"Parsing log file: {log_path}")

    iteration_data = {}

    if not log_path.exists():
        logger.warning(f"Log file not found: {log_path}")
        return iteration_data

    pattern = r'\[ITER (\d+)\] routed=(\d+) failed=(\d+) overuse=(\d+)'

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                iter_num = int(match.group(1))
                iteration_data[iter_num] = {
                    'routed': int(match.group(2)),
                    'failed': int(match.group(3)),
                    'overuse': int(match.group(4)),
                }

    logger.info(f"Parsed {len(iteration_data)} iterations from log")
    return iteration_data


def parse_screenshot_filename(filename: str) -> Tuple[int, str]:
    """
    Parse screenshot filename to extract sequence number and description.

    Examples:
        "01_board_with_airwires_20251104_205922_462.png" -> (1, "Showing airwires")
        "04_iteration_01_20251104_223759_792.png" -> (4, "Iteration 1 - Greedy")

    Returns:
        (sequence_num, description)
    """
    # Extract sequence number (first digits)
    seq_match = re.match(r'(\d+)_', filename)
    seq_num = int(seq_match.group(1)) if seq_match else 0

    # Determine description based on filename
    if 'airwires' in filename:
        desc = "Showing airwires"
    elif 'no_airwires' in filename:
        desc = "Planning escape vias"
    elif 'escapes' in filename:
        desc = "Planning escape vias"
    elif 'iteration_01' in filename or 'iteration_1_' in filename:
        desc = "Iteration 1 - Greedy"
    elif match := re.search(r'iteration_(\d+)', filename):
        iter_num = int(match.group(1))
        if iter_num == 1:
            desc = "Iteration 1 - Greedy"
        else:
            desc = f"Iteration {iter_num} - PathFinder"
    else:
        desc = "Routing"

    return seq_num, desc


def get_iteration_number(filename: str) -> int:
    """Extract iteration number from filename, or 0 if not an iteration."""
    match = re.search(r'iteration_(\d+)', filename)
    return int(match.group(1)) if match else 0


# =============================================================================
# TEXT OVERLAY
# =============================================================================

def create_composite_frame(screenshot_path: Path, board_name: str, state: str,
                           nets_routed: int, overuse: int, elapsed_time: str,
                           explanation: str, logo_path: Path) -> Image.Image:
    """
    Create composite frame with screenshot + text overlay matching the layout.

    Layout:
    - Top left: Logo (2.5x scaled)
    - Below logo: "OrthoRoute" + board name + stats
    - Right/center: PCB screenshot
    """
    # Load screenshot
    screenshot = Image.open(screenshot_path).convert('RGB')

    # Calculate scaling to fit screenshot on right side (leaving room for text on left)
    text_panel_width = 450
    pcb_area_width = VIDEO_WIDTH - text_panel_width
    pcb_area_height = VIDEO_HEIGHT

    # Scale screenshot to fit
    screenshot_aspect = screenshot.width / screenshot.height
    target_aspect = pcb_area_width / pcb_area_height

    if screenshot_aspect > target_aspect:
        # Width-constrained
        new_width = pcb_area_width
        new_height = int(pcb_area_width / screenshot_aspect)
    else:
        # Height-constrained
        new_height = pcb_area_height
        new_width = int(pcb_area_height * screenshot_aspect)

    screenshot = screenshot.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create base canvas (KiCad theme background color #001023)
    canvas = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 16, 35))

    # Paste screenshot on right side, centered
    pcb_x = text_panel_width + (pcb_area_width - new_width) // 2
    pcb_y = (pcb_area_height - new_height) // 2
    canvas.paste(screenshot, (pcb_x, pcb_y))

    # Create text overlay
    draw = ImageDraw.Draw(canvas)

    # Load fonts
    try:
        logo_font = ImageFont.truetype("arial.ttf", 36)
        title_font = ImageFont.truetype("arialbd.ttf", 28)
        stat_label_font = ImageFont.truetype("arial.ttf", 22)
        stat_value_font = ImageFont.truetype("arialbd.ttf", 32)
        explanation_font = ImageFont.truetype("arial.ttf", 24)
    except:
        logger.warning("Could not load fonts, using default")
        logo_font = ImageFont.load_default()
        title_font = ImageFont.load_default()
        stat_label_font = ImageFont.load_default()
        stat_value_font = ImageFont.load_default()
        explanation_font = ImageFont.load_default()

    # Add logo in upper left (1.25x scaled - half of net tour video size)
    y = 50
    if logo_path.exists():
        try:
            logo = Image.open(logo_path).convert('RGBA')
            # Scale logo by 1.25x (smaller for iteration video)
            scale_factor = 1.25
            new_width = int(logo.width * scale_factor)
            new_height = int(logo.height * scale_factor)
            logo = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)
            canvas.paste(logo, (30, y), logo)
            y += logo.height + 30  # Less spacing below logo
        except Exception as e:
            logger.warning(f"Failed to load logo: {e}")
            y = 100
    else:
        y = 100

    # Minimal spacing before text
    y += 20

    # Add "OrthoRoute" title
    draw.text((30, y), "OrthoRoute", font=logo_font, fill=(255, 255, 255))
    y += 45

    # Add board name
    draw.text((30, y), board_name, font=title_font, fill=(255, 255, 255))
    y += 60

    # Add stats (only Current state)
    draw.text((30, y), "Current state:", font=stat_label_font, fill=(200, 200, 200))
    y += 30
    draw.text((60, y), state, font=title_font, fill=(255, 255, 255))
    y += 50

    # No explanation text (removed per user request)

    return canvas


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def generate_iteration_video(run_folder: Path, output_path: Path):
    """
    Generate iteration video from debug output folder.

    Args:
        run_folder: Path to debug_output/run_* folder
        output_path: Output video path
    """
    logger.info(f"Generating iteration video from: {run_folder}")

    # Find matching log file (try exact match first, then fuzzy match by date)
    timestamp = run_folder.name.split('_', 1)[1]  # Extract timestamp from folder name
    log_file = Path('logs') / f"run_{timestamp}.log"

    # If exact match not found, try finding log file with same date
    if not log_file.exists():
        date_part = timestamp.split('_')[0]  # Extract YYYYMMDD
        logs_dir = Path('logs')
        if logs_dir.exists():
            matching_logs = list(logs_dir.glob(f"run_{date_part}_*.log"))
            if matching_logs:
                # Use the closest log file by timestamp
                log_file = sorted(matching_logs)[-1]  # Last log of that day
                logger.info(f"Using closest log file: {log_file.name}")

    # Parse iteration data from log
    iteration_data = parse_log_file(log_file)

    # Get all screenshots and sort by sequence number (not alphabetically!)
    screenshots = list(run_folder.glob('*.png'))
    # Sort by the leading number in filename
    screenshots.sort(key=lambda p: int(re.match(r'(\d+)_', p.name).group(1)))
    logger.info(f"Found {len(screenshots)} screenshots")

    if not screenshots:
        logger.error("No screenshots found!")
        return

    # Load logo (absolute path from script location)
    logo_path = Path(__file__).parent.parent / "graphics" / "icon200.png"

    # Board name (extract from first screenshot or use default)
    board_name = "Backplane Test Board"

    # Generate frames (relative to viz/ directory)
    frame_dir = Path("iteration_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Clear old frames
    for f in frame_dir.glob('*.png'):
        f.unlink()

    frame_idx = 0
    start_time = datetime.now()

    for screenshot_path in screenshots:
        seq_num, state_desc = parse_screenshot_filename(screenshot_path.name)
        iter_num = get_iteration_number(screenshot_path.name)

        logger.info(f"Processing: {screenshot_path.name} (seq={seq_num}, iter={iter_num})")

        # Get stats for this iteration
        if iter_num > 0 and iter_num in iteration_data:
            stats = iteration_data[iter_num]
            nets_routed = stats['routed']
            overuse = stats['overuse']
        else:
            nets_routed = 0
            overuse = 0

        # Calculate elapsed time (fake for now - could extract from timestamps)
        elapsed = timedelta(seconds=seq_num * 60)  # Rough estimate
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds

        # Create composite frame (no explanation text)
        composite = create_composite_frame(
            screenshot_path,
            board_name,
            state_desc,
            nets_routed,
            overuse,
            elapsed_str,
            "",  # No explanation
            logo_path
        )

        # Save just one frame (ffmpeg will handle duration)
        frame_path = frame_dir / f"frame_{frame_idx:06d}.png"
        composite.save(frame_path)
        frame_idx += 1

        logger.info(f"  Saved frame {frame_idx}")

    logger.info(f"Generated {frame_idx} total frames")

    # Encode with ffmpeg
    logger.info("Encoding video with ffmpeg...")

    import subprocess
    import shutil

    if not shutil.which('ffmpeg'):
        logger.error("ffmpeg not found! Cannot encode video.")
        logger.error(f"Frames are in: {frame_dir}")
        return

    # Use -r 1 to display each frame for 1 second (1 fps input = 1 sec per frame)
    # Then re-encode to 30fps for smooth playback
    cmd = [
        'ffmpeg',
        '-y',
        '-r', '1',  # 1 fps input = 1 second per frame
        '-i', str(frame_dir / 'frame_%06d.png'),
        '-r', str(FPS),  # Output at 30fps for compatibility
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"ffmpeg failed:\n{result.stderr}")
        raise RuntimeError("Video encoding failed")

    logger.info(f"Video saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Delete old log file if it exists and redirect logging to iteration_video.log
    log_file = Path(__file__).parent / "iteration_video.log"
    if log_file.exists():
        log_file.unlink()

    # Reconfigure logging to write to file
    import logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()  # Also print to console
        ]
    )

    if len(sys.argv) < 2:
        print("Usage: python generate_iteration_video.py <run_folder>")
        print("Example: python generate_iteration_video.py debug_output/run_20251104_205922")
        sys.exit(1)

    run_folder = Path(sys.argv[1])

    if not run_folder.exists():
        logger.error(f"Run folder not found: {run_folder}")
        sys.exit(1)

    if not run_folder.is_dir():
        logger.error(f"Not a directory: {run_folder}")
        sys.exit(1)

    # Generate video
    generate_iteration_video(run_folder, OUTPUT_VIDEO)

    logger.info("Done!")


if __name__ == '__main__':
    main()
