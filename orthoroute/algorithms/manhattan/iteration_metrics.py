"""
Iteration Metrics Logger

Tracks and logs detailed metrics for each routing iteration to CSV and Markdown files.
Provides comprehensive view of algorithm behavior, parameter evolution, and convergence.
"""

import os
import csv
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class IterationMetricsLogger:
    """
    Logs iteration metrics to both CSV (for analysis) and Markdown (for readability).
    Files are incrementally updated after each iteration.
    """

    # Define column order and headers
    COLUMNS = [
        'iteration',
        'timestamp',
        'duration_s',
        'overuse',
        'overuse_delta',
        'barrel_conflicts',
        'routed_nets',
        'failed_nets',
        'total_edges',
        'pres_fac',
        'pres_fac_mult',
        'hist_gain',
        'hist_cost_weight',
        'via_penalty',
        'hotset_size',
        'stagnant_iters',
        'stagnation_events',
        'plateau_kick_applied',
    ]

    HEADERS = {
        'iteration': 'Iter',
        'timestamp': 'Timestamp',
        'duration_s': 'Time(s)',
        'overuse': 'Overuse',
        'overuse_delta': 'Δ Overuse',
        'barrel_conflicts': 'Barrel Conflicts',
        'routed_nets': 'Routed',
        'failed_nets': 'Failed',
        'total_edges': 'Edges',
        'pres_fac': 'pres_fac',
        'pres_fac_mult': 'pres_mult',
        'hist_gain': 'hist_gain',
        'hist_cost_weight': 'hist_weight',
        'via_penalty': 'via_cost',
        'hotset_size': 'hotset',
        'stagnant_iters': 'stagnant',
        'stagnation_events': 'stag_events',
        'plateau_kick_applied': 'kick',
    }

    def __init__(self, debug_dir: str, board_info: Optional[Dict] = None):
        """
        Initialize metrics logger.

        Args:
            debug_dir: Directory to store metrics files (e.g., debug_output/run_YYYYMMDD_HHMMSS)
            board_info: Optional dict with board metadata (name, nets, layers, etc.)
        """
        self.debug_dir = debug_dir
        self.board_info = board_info or {}
        self.csv_path = os.path.join(debug_dir, "iteration_metrics.csv")
        self.md_path = os.path.join(debug_dir, "iteration_metrics.md")
        self.metrics_history: List[Dict] = []
        self.prev_overuse: Optional[int] = None

        # Ensure directory exists
        os.makedirs(debug_dir, exist_ok=True)

        # Initialize files
        self._init_csv()
        self._write_markdown_header()

        logger.info(f"[METRICS] Logging to {debug_dir}")

    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
        logger.info(f"[METRICS] Created CSV: {self.csv_path}")

    def _write_markdown_header(self):
        """Write markdown file header with board info."""
        with open(self.md_path, 'w') as f:
            f.write("# Routing Iteration Metrics\n\n")

            # Extract run ID from debug_dir path
            run_id = os.path.basename(self.debug_dir)
            f.write(f"**Run:** `{run_id}`\n\n")

            # Board configuration
            if self.board_info:
                f.write("## Configuration\n\n")
                for key, value in self.board_info.items():
                    f.write(f"- **{key}:** {value}\n")
                f.write("\n")

            f.write("## Iteration History\n\n")

        logger.info(f"[METRICS] Created Markdown: {self.md_path}")

    def log_iteration(self, metrics: Dict):
        """
        Log metrics for one iteration.

        Args:
            metrics: Dictionary with all metric values for this iteration
        """
        # Calculate delta from previous iteration
        if self.prev_overuse is not None:
            metrics['overuse_delta'] = metrics['overuse'] - self.prev_overuse
        else:
            metrics['overuse_delta'] = 0

        self.prev_overuse = metrics['overuse']

        # Ensure all columns are present (use defaults for missing)
        for col in self.COLUMNS:
            if col not in metrics:
                metrics[col] = 0 if col not in ['timestamp'] else ''

        # Store in history
        self.metrics_history.append(metrics)

        # Append to CSV (incremental)
        self._append_csv_row(metrics)

        # Regenerate markdown table (includes all iterations)
        self._write_markdown_table()

        logger.debug(f"[METRICS] Logged iteration {metrics['iteration']}: "
                    f"overuse={metrics['overuse']:,} failed={metrics['failed_nets']}")

    def _append_csv_row(self, metrics: Dict):
        """Append one row to CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(metrics)

    def _write_markdown_table(self):
        """Regenerate full markdown table with all iterations."""
        # Read existing header
        with open(self.md_path, 'r') as f:
            lines = f.readlines()

        # Find where table starts (after "## Iteration History")
        table_start_idx = None
        for i, line in enumerate(lines):
            if line.strip() == "## Iteration History":
                table_start_idx = i + 2  # Skip header and blank line
                break

        if table_start_idx is None:
            logger.error("[METRICS] Could not find table start in markdown")
            return

        # Rebuild file with updated table
        with open(self.md_path, 'w') as f:
            # Write header section (up to table start)
            f.writelines(lines[:table_start_idx])

            # Write table header
            f.write("| ")
            f.write(" | ".join(self.HEADERS[col] for col in self.COLUMNS))
            f.write(" |\n")

            # Write separator
            f.write("|")
            f.write("|".join("---" for _ in self.COLUMNS))
            f.write("|\n")

            # Write data rows
            for m in self.metrics_history:
                f.write("| ")
                row_values = []
                for col in self.COLUMNS:
                    val = m.get(col, '')

                    # Format based on column type
                    if col in ['overuse', 'barrel_conflicts', 'total_edges']:
                        row_values.append(f"{val:,}")
                    elif col in ['overuse_delta']:
                        sign = "+" if val > 0 else ""
                        row_values.append(f"{sign}{val:,}")
                    elif col in ['pres_fac', 'pres_fac_mult', 'hist_gain', 'hist_cost_weight', 'via_penalty']:
                        row_values.append(f"{val:.3f}")
                    elif col in ['duration_s']:
                        row_values.append(f"{val:.1f}")
                    elif col == 'plateau_kick_applied':
                        row_values.append("Y" if val else "")
                    else:
                        row_values.append(str(val))

                f.write(" | ".join(row_values))
                f.write(" |\n")

            # Add summary statistics
            f.write("\n## Summary\n\n")
            if self.metrics_history:
                last = self.metrics_history[-1]
                first = self.metrics_history[0]

                f.write(f"- **Total Iterations:** {len(self.metrics_history)}\n")
                f.write(f"- **Initial Overuse:** {first['overuse']:,}\n")
                f.write(f"- **Final Overuse:** {last['overuse']:,}\n")
                f.write(f"- **Reduction:** {first['overuse'] - last['overuse']:,} ({100*(first['overuse']-last['overuse'])/first['overuse']:.1f}%)\n")
                f.write(f"- **Failed Nets:** {last['failed_nets']}\n")

                total_time = sum(m['duration_s'] for m in self.metrics_history)
                f.write(f"- **Total Time:** {total_time:.1f}s ({total_time/60:.1f} min)\n")
                f.write(f"- **Avg Time/Iter:** {total_time/len(self.metrics_history):.1f}s\n")

    def get_summary(self) -> Dict:
        """Get summary statistics for this run."""
        if not self.metrics_history:
            return {}

        first = self.metrics_history[0]
        last = self.metrics_history[-1]
        total_time = sum(m['duration_s'] for m in self.metrics_history)

        return {
            'iterations': len(self.metrics_history),
            'initial_overuse': first['overuse'],
            'final_overuse': last['overuse'],
            'overuse_reduction': first['overuse'] - last['overuse'],
            'reduction_pct': 100 * (first['overuse'] - last['overuse']) / first['overuse'],
            'failed_nets': last['failed_nets'],
            'total_time_s': total_time,
            'avg_time_per_iter': total_time / len(self.metrics_history),
        }
