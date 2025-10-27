"""
PathFinder Live Statistics Widget
Shows real-time routing progress, timing, and vital statistics
"""

import time
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QGroupBox, QGridLayout, QLCDNumber, QFrame
)
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

class PathFinderStatsWidget(QWidget):
    """Live statistics widget for PathFinder routing"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_time = None
        self.current_iteration = 0
        self.total_nets = 0
        self.successful_routes = 0
        self.failed_routes = 0
        self.congestion_count = 0
        self.overuse_history = []  # Track last 3 iterations: [(iter, overuse_sum, overuse_edges), ...]

        self.setup_ui()
        
        # Timer for live updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_elapsed_time)
        self.update_timer.setInterval(100)  # Update every 100ms
        
    def setup_ui(self):
        """Setup the statistics UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        
        title = QLabel("PathFinder Live Statistics")
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Progress Section
        progress_group = self.create_progress_section()
        layout.addWidget(progress_group)
        
        # Timing Section  
        timing_group = self.create_timing_section()
        layout.addWidget(timing_group)
        
        # Routing Stats Section
        routing_group = self.create_routing_stats_section()
        layout.addWidget(routing_group)
        
        # Congestion Section
        congestion_group = self.create_congestion_section()
        layout.addWidget(congestion_group)
        
        layout.addStretch()
        
    def create_progress_section(self) -> QGroupBox:
        """Create PathFinder iteration progress section"""
        group = QGroupBox("PathFinder Progress")
        layout = QVBoxLayout(group)
        
        # Current iteration
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iteration:"))
        self.iteration_label = QLabel("0 / 50")
        iter_layout.addWidget(self.iteration_label)
        iter_layout.addStretch()
        layout.addLayout(iter_layout)
        
        # Progress bar
        self.iteration_progress = QProgressBar()
        self.iteration_progress.setMaximum(50)  # Default max iterations
        self.iteration_progress.setValue(0)
        layout.addWidget(self.iteration_progress)
        
        # Overuse history display (last 3 iterations in table format)
        overuse_font = QFont("Courier New", 9)  # Monospace for table alignment
        self.overuse_label = QLabel("Waiting for routing...")
        self.overuse_label.setFont(overuse_font)
        self.overuse_label.setStyleSheet("QLabel { color: #333; background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        self.overuse_label.setWordWrap(True)
        layout.addWidget(self.overuse_label)
        
        return group
    
    def create_timing_section(self) -> QGroupBox:
        """Create timing statistics section"""
        group = QGroupBox("Timing")
        layout = QGridLayout(group)
        
        # Elapsed time with large LCD display
        layout.addWidget(QLabel("Elapsed:"), 0, 0)
        self.elapsed_lcd = QLCDNumber(8)
        self.elapsed_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.elapsed_lcd.setDigitCount(8)
        self.elapsed_lcd.display("00:00.00")
        layout.addWidget(self.elapsed_lcd, 0, 1, 1, 2)
        
        # Iteration time
        layout.addWidget(QLabel("Iter Time:"), 1, 0)
        self.iteration_time_label = QLabel("0.00s")
        layout.addWidget(self.iteration_time_label, 1, 1)
        
        # Estimated total
        layout.addWidget(QLabel("Est. Total:"), 1, 2)
        self.estimated_total_label = QLabel("--:--")
        layout.addWidget(self.estimated_total_label, 1, 3)
        
        return group
    
    def create_routing_stats_section(self) -> QGroupBox:
        """Create routing success statistics"""
        group = QGroupBox("Routing Statistics")
        layout = QGridLayout(group)
        
        # Total nets
        layout.addWidget(QLabel("Total Nets:"), 0, 0)
        self.total_nets_label = QLabel("0")
        layout.addWidget(self.total_nets_label, 0, 1)
        
        # Successful routes
        layout.addWidget(QLabel("Routed:"), 1, 0)
        self.success_label = QLabel("0")
        self.success_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        layout.addWidget(self.success_label, 1, 1)
        
        # Failed routes
        layout.addWidget(QLabel("Failed:"), 1, 2)
        self.failed_label = QLabel("0")
        self.failed_label.setStyleSheet("QLabel { color: red; }")
        layout.addWidget(self.failed_label, 1, 3)
        
        # Success rate
        layout.addWidget(QLabel("Success Rate:"), 2, 0)
        self.success_rate_label = QLabel("0.0%")
        layout.addWidget(self.success_rate_label, 2, 1)
        
        # Routes per second
        layout.addWidget(QLabel("Routes/sec:"), 2, 2)
        self.routes_per_sec_label = QLabel("0.0")
        layout.addWidget(self.routes_per_sec_label, 2, 3)
        
        return group
    
    def create_congestion_section(self) -> QGroupBox:
        """Create congestion statistics section"""
        group = QGroupBox("Congestion Analysis")
        layout = QGridLayout(group)
        
        # Congested edges
        layout.addWidget(QLabel("Congested Edges:"), 0, 0)
        self.congestion_count_label = QLabel("0")
        layout.addWidget(self.congestion_count_label, 0, 1)
        
        # Congestion progress bar
        layout.addWidget(QLabel("Congestion Level:"), 1, 0)
        self.congestion_progress = QProgressBar()
        self.congestion_progress.setMaximum(100)
        self.congestion_progress.setValue(0)
        layout.addWidget(self.congestion_progress, 1, 1, 1, 3)
        
        # Congestion trend
        layout.addWidget(QLabel("Trend:"), 2, 0)
        self.congestion_trend_label = QLabel("Stable")
        layout.addWidget(self.congestion_trend_label, 2, 1)
        
        return group
    
    def start_routing(self, total_nets: int, max_iterations: int = 50):
        """Start routing session"""
        self.start_time = time.time()
        self.total_nets = total_nets
        self.current_iteration = 0
        self.successful_routes = 0
        self.failed_routes = 0
        self.congestion_count = 0
        
        # Update UI
        self.total_nets_label.setText(str(total_nets))
        self.iteration_progress.setMaximum(max_iterations)
        self.iteration_progress.setValue(0)
        self.iteration_label.setText(f"0 / {max_iterations}")
        self.overuse_history = []  # Reset overuse history
        self.overuse_label.setText("Initializing routing...")
        
        # Start timer
        self.update_timer.start()
        
    def update_iteration(self, iteration: int, max_iterations: int, overuse_sum: int = 0, overuse_edges: int = 0):
        """Update current iteration progress with overuse data"""
        self.current_iteration = iteration
        self.iteration_progress.setValue(iteration)
        self.iteration_label.setText(f"{iteration} / {max_iterations}")

        # Track overuse history (keep last 3 iterations)
        self.overuse_history.append((iteration, overuse_sum, overuse_edges))
        if len(self.overuse_history) > 3:
            self.overuse_history.pop(0)

        # Build table display for last 3 iterations
        if self.overuse_history:
            lines = []
            lines.append("┌─────────┬──────────┬────────┐")
            lines.append("│  Iter   │ Overuse  │ Edges  │")
            lines.append("├─────────┼──────────┼────────┤")

            for iter_num, over_sum, over_edges in self.overuse_history:
                # Format with proper spacing for alignment
                iter_str = f"{iter_num:>5}"
                over_str = f"{over_sum:>8,}"
                edges_str = f"{over_edges:>6,}"
                lines.append(f"│  {iter_str}  │ {over_str} │ {edges_str} │")

            lines.append("└─────────┴──────────┴────────┘")

            table_text = "\n".join(lines)
            self.overuse_label.setText(table_text)
    
    def update_routing_stats(self, successful: int, failed: int):
        """Update routing success statistics"""
        self.successful_routes = successful
        self.failed_routes = failed
        
        self.success_label.setText(str(successful))
        self.failed_label.setText(str(failed))
        
        # Calculate success rate
        total_attempted = successful + failed
        if total_attempted > 0:
            success_rate = (successful / total_attempted) * 100
            self.success_rate_label.setText(f"{success_rate:.1f}%")
        else:
            self.success_rate_label.setText("0.0%")
        
        # Calculate routes per second
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                routes_per_sec = total_attempted / elapsed
                self.routes_per_sec_label.setText(f"{routes_per_sec:.1f}")
    
    def update_congestion(self, congested_edges: int, total_edges: int, trend: str = "Stable"):
        """Update congestion statistics"""
        self.congestion_count = congested_edges
        self.congestion_count_label.setText(str(congested_edges))
        
        # Update congestion progress bar
        if total_edges > 0:
            congestion_percent = min(100, (congested_edges / total_edges) * 100)
            self.congestion_progress.setValue(int(congestion_percent))
            
            # Color code based on congestion level
            if congestion_percent > 75:
                self.congestion_progress.setStyleSheet("QProgressBar::chunk { background-color: red; }")
            elif congestion_percent > 50:
                self.congestion_progress.setStyleSheet("QProgressBar::chunk { background-color: orange; }")
            elif congestion_percent > 25:
                self.congestion_progress.setStyleSheet("QProgressBar::chunk { background-color: yellow; }")
            else:
                self.congestion_progress.setStyleSheet("QProgressBar::chunk { background-color: green; }")
        
        self.congestion_trend_label.setText(trend)
    
    def update_elapsed_time(self):
        """Update elapsed time display"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            time_str = f"{minutes:02d}:{seconds:05.2f}"
            self.elapsed_lcd.display(time_str)
    
    def finish_routing(self, success: bool, final_message: str = ""):
        """Finish routing session"""
        self.update_timer.stop()

        # Keep the overuse table visible, just update iteration label
        if success:
            self.iteration_label.setText(self.iteration_label.text() + " ✓")
        else:
            self.iteration_label.setText(self.iteration_label.text() + " ✗")
    
    def reset(self):
        """Reset all statistics"""
        self.update_timer.stop()
        self.start_time = None
        self.current_iteration = 0
        self.successful_routes = 0
        self.failed_routes = 0
        self.congestion_count = 0
        
        # Reset UI
        self.iteration_label.setText("0 / 50")
        self.iteration_progress.setValue(0)
        self.overuse_history = []
        self.overuse_label.setText("Waiting for routing...")
        self.elapsed_lcd.display("00:00.00")
        self.success_label.setText("0")
        self.failed_label.setText("0")
        self.success_rate_label.setText("0.0%")
        self.routes_per_sec_label.setText("0.0")
        self.congestion_count_label.setText("0")
        self.congestion_progress.setValue(0)
        self.congestion_trend_label.setText("Stable")
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get current statistics as dictionary"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'elapsed_time': elapsed,
            'current_iteration': self.current_iteration,
            'total_nets': self.total_nets,
            'successful_routes': self.successful_routes,
            'failed_routes': self.failed_routes,
            'success_rate': self.successful_routes / max(1, self.successful_routes + self.failed_routes),
            'routes_per_second': (self.successful_routes + self.failed_routes) / max(0.1, elapsed),
            'congested_edges': self.congestion_count
        }