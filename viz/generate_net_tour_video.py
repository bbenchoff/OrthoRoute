"""
Generate a 4K video touring each net on the routed PCB.

Creates frames (one per net) showing:
- Right side: PCB preview scaled to 2160px height
- Left side: Board name, net counter, net name

Usage:
    python viz/generate_net_tour_video.py <path_to_kicad_pcb>
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# PyQt6 for offscreen rendering
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPainter, QImage, QColor, QPen, QBrush
from PyQt6.QtCore import QRectF, QPointF, Qt

# Import OrthoRoute's infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent))
from orthoroute.presentation.gui.kicad_colors import KiCadColorScheme
from orthoroute.infrastructure.kicad.rich_kicad_interface import RichKiCadInterface

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

VIDEO_WIDTH = 3840   # 4K width
VIDEO_HEIGHT = 2160  # 4K height
PCB_HEIGHT = 2160    # PCB preview fills full height
TEXT_PANEL_WIDTH = 1200  # Left panel for text overlay

FPS = 30
FRAME_DIR = Path("frames")  # Relative to viz/ directory
OUTPUT_VIDEO = Path("net_tour.mp4")  # Relative to viz/ directory


# =============================================================================
# BOARD LOADER
# =============================================================================

def load_board_from_kicad(pcb_path: str) -> Dict[str, Any]:
    """
    Load KiCad board using OrthoRoute's RichKicadInterface.

    Returns:
        board_data dict matching PCBViewer format
    """
    logger.info(f"Loading KiCad board: {pcb_path}")

    # Initialize KiCad interface
    kicad = RichKiCadInterface()

    # Connect to KiCad (this will start KiCad if not running)
    if not kicad.connect():
        raise RuntimeError("Failed to connect to KiCad")

    # Board must already be open in KiCad
    # (RichKiCadInterface doesn't have load_board method)

    # Get comprehensive board data
    board_data = kicad.get_board_data()

    if not board_data:
        raise RuntimeError("Failed to extract board data")

    logger.info(f"Loaded board: {board_data.get('filename', 'Unknown')}")
    logger.info(f"  Bounds: {board_data['bounds']}")
    logger.info(f"  Nets: {len(board_data['nets'])}")
    logger.info(f"  Tracks: {len(board_data['tracks'])}")
    logger.info(f"  Vias: {len(board_data.get('vias', []))}")
    logger.info(f"  Pads: {len(board_data['pads'])}")

    return board_data


# =============================================================================
# OFFSCREEN PCB RENDERER
# =============================================================================

class OffscreenPCBRenderer:
    """Renders PCB to QImage using OrthoRoute's PCBViewer logic"""

    def __init__(self, board_data: Dict[str, Any], width: int, height: int):
        self.board_data = board_data
        self.width = width
        self.height = height
        self.color_scheme = KiCadColorScheme()

        # Calculate zoom to fit board to specified height
        bounds = board_data['bounds']
        board_width = bounds[2] - bounds[0]
        board_height = bounds[3] - bounds[1]

        # Zoom to fit height
        self.zoom_factor = (height * 0.9) / board_height

        # Center board
        self.pan_x = (bounds[0] + bounds[2]) / 2
        self.pan_y = (bounds[1] + bounds[3]) / 2

        logger.info(f"PCB Renderer: zoom={self.zoom_factor:.2f}, pan=({self.pan_x:.1f}, {self.pan_y:.1f})")

    def render_frame(self, highlight_net: str = None, darken: bool = False) -> QImage:
        """
        Render PCB to QImage.

        Args:
            highlight_net: Net name to highlight (None = show all)
            darken: If True, darken all tracks except highlighted net

        Returns:
            QImage (RGBA)
        """
        image = QImage(self.width, self.height, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background - always use dark gray (same for opening and net tour)
        bg_color = QColor(20, 20, 20)
        painter.fillRect(0, 0, self.width, self.height, bg_color)

        # Set up coordinate transformation
        painter.translate(self.width / 2, self.height / 2)
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.translate(-self.pan_x, -self.pan_y)

        # Render elements
        self._draw_pads(painter, highlight_net, darken)  # Pads first (background)
        self._draw_tracks(painter, highlight_net, darken)
        self._draw_vias(painter, highlight_net, darken)

        painter.end()
        return image

    def _draw_tracks(self, painter: QPainter, highlight_net: str, darken: bool):
        """Draw tracks (matching main_window.py rendering)"""
        tracks = self.board_data.get('tracks', [])

        for track in tracks:
            track_net = track.get('net_name', track.get('net', ''))

            # Get layer and convert to string if needed
            layer_raw = track.get('layer', 0)
            if isinstance(layer_raw, int):
                if layer_raw == 0:
                    layer = 'F.Cu'
                elif layer_raw == 1:
                    layer = 'B.Cu'
                else:
                    layer = f'In{layer_raw-1}.Cu'
            else:
                layer = layer_raw

            # Determine color
            if darken:
                if highlight_net is not None and track_net == highlight_net:
                    # Highlighted track - bright yellow
                    color = QColor(255, 255, 0)
                    pen_width = track['width'] * 1.5  # Slightly thicker
                else:
                    # Background track - very dim
                    color = QColor(40, 40, 40)
                    pen_width = track['width'] * 0.8
            else:
                # Normal rendering - use proper layer colors like GUI
                if layer == 'F.Cu':
                    color = self.color_scheme.get_color('copper_front')
                elif layer == 'B.Cu':
                    color = self.color_scheme.get_color('copper_back')
                else:
                    # Internal layers
                    try:
                        if layer.startswith('In') and layer.endswith('.Cu'):
                            layer_num = int(layer[2:-3])
                            color_key = f'copper_in{layer_num}'
                            color = self.color_scheme.get_color(color_key)
                        else:
                            color = self.color_scheme.get_color('copper_inner')
                    except (ValueError, AttributeError):
                        color = self.color_scheme.get_color('copper_inner')
                pen_width = track['width']

            painter.setPen(QPen(color, pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(
                QPointF(track['start_x'], track['start_y']),
                QPointF(track['end_x'], track['end_y'])
            )

    def _draw_vias(self, painter: QPainter, highlight_net: str, darken: bool):
        """Draw vias"""
        vias = self.board_data.get('vias', [])

        for via in vias:
            via_net = via.get('net_name', via.get('net', ''))

            # Determine color
            if darken:
                if highlight_net is not None and via_net == highlight_net:
                    color = QColor(255, 255, 0)  # Yellow
                else:
                    color = QColor(30, 30, 30)
            else:
                color = self.color_scheme.get_color('via', QColor(200, 200, 200))

            painter.setPen(QPen(color, 0.1))
            painter.setBrush(QBrush(color))

            diameter = via['diameter']
            painter.drawEllipse(
                QPointF(via['x'], via['y']),
                diameter / 2, diameter / 2
            )

    def _draw_pads(self, painter: QPainter, highlight_net: str, darken: bool):
        """Draw pads (matching main_window.py rendering)"""
        pads = self.board_data.get('pads', [])

        for pad in pads:
            pad_net = pad.get('net_name', pad.get('net', ''))
            x = pad['x']
            y = pad['y']
            width = pad.get('width', pad.get('size_x', 1.0))
            height = pad.get('height', pad.get('size_y', 1.0))
            pad_type = pad.get('type', 'smd')
            layers = pad.get('layers', ['F.Cu'])
            drill_size = pad.get('drill', 0.0)

            # Determine color
            if darken:
                if highlight_net is not None and pad_net == highlight_net:
                    color = QColor(255, 200, 0)  # Slightly orange for pads
                else:
                    color = QColor(40, 40, 40)
            else:
                # Get appropriate pad color based on type and layer (like GUI)
                if pad_type == 'smd':
                    if any('B.' in layer for layer in layers):
                        color = self.color_scheme.get_color('pad_back')
                    else:
                        color = self.color_scheme.get_color('pad_front')
                elif pad_type == 'through_hole':
                    color = self.color_scheme.get_color('pad_through_hole')
                else:
                    color = self.color_scheme.get_color('pad_front')

            painter.setPen(QPen(color, 0.05))
            painter.setBrush(QBrush(color))

            # Draw pad based on type (like GUI)
            if pad_type == 'through_hole':
                # Draw through-hole pads as circles
                pad_size = max(width, height)
                pad_rect = QRectF(x - pad_size/2, y - pad_size/2, pad_size, pad_size)
                painter.drawEllipse(pad_rect)

                # Draw drill hole for through-hole pads
                if drill_size > 0 and not darken:
                    hole_color = QColor(255, 215, 0)  # Gold color
                    painter.setPen(QPen(hole_color, 0))
                    painter.setBrush(QBrush(hole_color))
                    hole_rect = QRectF(x - drill_size/2, y - drill_size/2, drill_size, drill_size)
                    painter.drawEllipse(hole_rect)
            else:
                # Draw SMD pads as rectangles
                pad_rect = QRectF(x - width/2, y - height/2, width, height)
                painter.drawRect(pad_rect)


# =============================================================================
# TEXT OVERLAY
# =============================================================================

def create_text_overlay(board_name: str, net_idx: int, total_nets: int, net_name: str,
                        width: int, height: int, logo_path: str = None) -> Image.Image:
    """
    Create text overlay for left panel.

    Args:
        board_name: Name of the PCB board
        net_idx: Current net index (0-based)
        total_nets: Total number of nets
        net_name: Name of current net
        width: Overlay width
        height: Overlay height
        logo_path: Path to logo image (optional)

    Returns:
        PIL Image (RGBA) with transparent background
    """
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Load fonts (fallback to default if not available)
    try:
        title_font = ImageFont.truetype("arial.ttf", 80)
        subtitle_font = ImageFont.truetype("arial.ttf", 60)
        net_font = ImageFont.truetype("arialbd.ttf", 100)  # Bold for net name
    except:
        logger.warning("Could not load TrueType fonts, using default")
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        net_font = ImageFont.load_default()

    # Text color
    text_color = (255, 255, 255, 255)  # White

    # Logo in upper left corner (2.5x larger)
    y_offset = 50
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert('RGBA')
            # Scale logo up by 2.5x
            scale_factor = 2.5
            new_width = int(logo.width * scale_factor)
            new_height = int(logo.height * scale_factor)
            logo = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Place in upper left with margin
            img.paste(logo, (50, y_offset), logo)
            y_offset += logo.height + 100  # Move text down below logo with more space
        except Exception as e:
            logger.warning(f"Failed to load logo: {e}")
            y_offset = 100
    else:
        y_offset = 100

    # Move text down below logo
    y_offset += 100

    # Board name
    draw.text((50, y_offset), board_name, font=title_font, fill=text_color)
    y_offset += 120

    # Net counter
    counter_text = f"Net {net_idx + 1} / {total_nets}"
    draw.text((50, y_offset), counter_text, font=subtitle_font, fill=text_color)
    y_offset += 100

    # Net name (word-wrap if too long)
    max_chars = 30
    if len(net_name) > max_chars:
        # Simple word wrap
        lines = [net_name[i:i+max_chars] for i in range(0, len(net_name), max_chars)]
        net_name_display = '\n'.join(lines[:3])  # Max 3 lines
    else:
        net_name_display = net_name

    draw.text((50, y_offset), net_name_display, font=net_font, fill=text_color)

    return img


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def generate_frames(board_data: Dict[str, Any], board_name: str, output_dir: Path):
    """
    Generate all frames for the net tour video.

    Args:
        board_data: Board data from load_board_from_kicad()
        board_name: Name of the board
        output_dir: Directory to save frames
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get nets from board_data (it's a dict, not a list)
    nets_dict = board_data['nets']

    # Filter to routable nets (nets with 2+ pads)
    routable_nets = [(name, data) for name, data in nets_dict.items()
                     if len(data.get('pads', [])) >= 2]

    total_nets = len(routable_nets)

    logger.info(f"Generating opening sequence + {total_nets} net frames...")

    # Calculate PCB panel width (right side of frame)
    pcb_width = VIDEO_WIDTH - TEXT_PANEL_WIDTH

    # Create PCB renderer
    renderer = OffscreenPCBRenderer(board_data, pcb_width, PCB_HEIGHT)

    # =============================================================================
    # OPENING SEQUENCE: Full board view in color for 2 seconds (60 frames at 30fps)
    # =============================================================================
    logger.info("Rendering opening sequence (2 seconds, full color board view)...")

    # Render full board in normal colors (not darkened, no highlight)
    opening_pcb_image = renderer.render_frame(highlight_net=None, darken=False)
    opening_pcb_image_pil = qimage_to_pil(opening_pcb_image)

    # Create opening text overlay using same function as net tour frames
    logo_path = Path(__file__).parent.parent / "graphics" / "icon200.png"

    # Use the same text overlay function but with empty net info for opening
    # This ensures consistent positioning with the net tour frames
    opening_overlay = Image.new('RGBA', (TEXT_PANEL_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(opening_overlay)

    # Add logo (same as net tour frames)
    y_offset = 50
    if os.path.exists(str(logo_path)):
        try:
            logo = Image.open(str(logo_path)).convert('RGBA')
            scale_factor = 2.5
            new_width = int(logo.width * scale_factor)
            new_height = int(logo.height * scale_factor)
            logo = logo.resize((new_width, new_height), Image.Resampling.LANCZOS)
            opening_overlay.paste(logo, (50, y_offset), logo)
            y_offset += logo.height + 100
        except Exception as e:
            logger.warning(f"Failed to load logo for opening: {e}")
            y_offset = 100
    else:
        y_offset = 100

    # Move text down below logo (same spacing as net tour frames)
    y_offset += 100

    # Add board name text at same position as net tour frames
    try:
        title_font = ImageFont.truetype("arial.ttf", 80)
    except:
        title_font = ImageFont.load_default()

    draw.text((50, y_offset), board_name, font=title_font, fill=(255, 255, 255, 255))

    # Composite opening frame with same dark background as net tour frames
    opening_frame = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), (20, 20, 20))
    opening_frame.paste(opening_overlay, (0, 0), opening_overlay)
    opening_frame.paste(opening_pcb_image_pil, (TEXT_PANEL_WIDTH, 0))

    # Save 60 copies of the opening frame (2 seconds at 30fps)
    for i in range(60):
        frame_path = output_dir / f"frame_{i:06d}.png"
        opening_frame.save(frame_path)
        if i % 20 == 0:
            logger.info(f"  Opening frame {i+1}/60")

    logger.info("Opening sequence complete!")

    # =============================================================================
    # NET TOUR SEQUENCE: One frame per net with highlighting
    # =============================================================================
    logger.info(f"Rendering net tour sequence ({total_nets} nets)...")

    for idx, (net_name, net_data) in enumerate(routable_nets):
        # Frame index starts at 60 (after opening sequence)
        frame_idx = 60 + idx
        if idx % 10 == 0:
            logger.info(f"Rendering frame {idx+1}/{total_nets}: {net_name}")

        # Render PCB with this net highlighted
        pcb_image = renderer.render_frame(highlight_net=net_name, darken=True)

        # Convert QImage to PIL Image
        pcb_image_pil = qimage_to_pil(pcb_image)

        # Create text overlay with logo
        logo_path = Path(__file__).parent.parent / "graphics" / "icon200.png"
        text_overlay = create_text_overlay(board_name, idx, total_nets, net_name,
                                          TEXT_PANEL_WIDTH, VIDEO_HEIGHT, str(logo_path))

        # Composite: text on left, PCB on right
        final_frame = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), (20, 20, 20))
        final_frame.paste(text_overlay, (0, 0), text_overlay)  # Use alpha channel
        final_frame.paste(pcb_image_pil, (TEXT_PANEL_WIDTH, 0))

        # Save frame (using frame_idx which accounts for 60-frame opening)
        frame_path = output_dir / f"frame_{frame_idx:06d}.png"
        final_frame.save(frame_path)

    logger.info(f"Saved opening sequence + {total_nets} net frames to {output_dir}")


def qimage_to_pil(qimage: QImage) -> Image.Image:
    """Convert QImage to PIL Image"""
    # Convert QImage to bytes
    qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(qimage.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

    # Create PIL image
    return Image.fromarray(arr, 'RGBA').convert('RGB')


def encode_video(frame_dir: Path, output_path: Path, fps: int = 30):
    """
    Encode frames to video using ffmpeg.

    Args:
        frame_dir: Directory containing frame_*.png files
        output_path: Output video path
        fps: Frames per second
    """
    import subprocess
    import shutil

    # Check if ffmpeg is available
    if not shutil.which('ffmpeg'):
        logger.warning("ffmpeg not found in PATH!")
        logger.warning("Frames are ready but cannot encode video.")
        logger.warning(f"To encode manually, run:")
        logger.warning(f'  ffmpeg -y -framerate {fps} -i "{frame_dir / "frame_%06d.png"}" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p "{output_path}"')
        return

    logger.info(f"Encoding video with ffmpeg at {fps} fps...")

    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-framerate', str(fps),
        '-i', str(frame_dir / 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',  # High quality
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
    # NOTE: Board must be open in KiCad before running this script!

    # Delete old log file if it exists and redirect logging to net_tour_video.log
    log_file = Path(__file__).parent / "net_tour_video.log"
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

    # Initialize Qt application (required for QPainter)
    app = QApplication(sys.argv)

    # Load board (from already-open KiCad)
    board_data = load_board_from_kicad("dummy_path")

    # Get board name from board_data
    board_name = board_data.get('filename', 'Unknown').replace('.kicad_pcb', '')

    # Generate frames
    generate_frames(board_data, board_name, FRAME_DIR)

    # Encode video
    encode_video(FRAME_DIR, OUTPUT_VIDEO, FPS)

    logger.info("Done!")


if __name__ == '__main__':
    main()
