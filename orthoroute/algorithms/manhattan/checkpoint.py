"""
Checkpoint system for OrthoRoute PathFinder iterations.

Allows saving/loading routing state at any iteration to:
- Resume after crashes
- Experiment with different parameters from a saved state
- Analyze convergence behavior
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading routing checkpoints."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[CHECKPOINT] Initialized checkpoint manager: {self.checkpoint_dir}")

    def save_checkpoint(self, router, iteration: int, pres_fac: float,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save current routing state to checkpoint file.

        Args:
            router: PathFinderRouter instance
            iteration: Current iteration number
            pres_fac: Current pressure factor
            metadata: Optional metadata (overuse, timing, etc.)

        Returns:
            Path to saved checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_iter{iteration:03d}_{timestamp}.pkl"
        filepath = self.checkpoint_dir / filename

        try:
            # Extract state from router
            checkpoint = {
                'iteration': iteration,
                'pres_fac': pres_fac,
                'net_paths': dict(router.net_paths),  # Copy dict
                'timestamp': timestamp,
                'metadata': metadata or {},
            }

            # Save accounting state (convert GPU arrays to CPU if needed)
            if hasattr(router, 'accounting') and router.accounting is not None:
                acc = router.accounting
                checkpoint['accounting'] = {
                    'present': acc.present.get() if acc.use_gpu else acc.present.copy(),
                    'present_ema': acc.present_ema.get() if acc.use_gpu else acc.present_ema.copy(),
                    'history': acc.history.get() if acc.use_gpu else acc.history.copy(),
                    'capacity': acc.capacity.get() if acc.use_gpu else acc.capacity.copy(),
                    'use_gpu': acc.use_gpu,
                }

            # Save via usage state if it exists
            if hasattr(router, 'via_col_use'):
                checkpoint['via_state'] = {
                    'via_col_use': router.via_col_use.copy(),
                    'via_col_pres': router.via_col_pres.copy(),
                    'via_seg_use': router.via_seg_use.copy() if hasattr(router, 'via_seg_use') else None,
                    'via_seg_pres': router.via_seg_pres.copy() if hasattr(router, 'via_seg_pres') else None,
                }

            # Save config parameters (for reference)
            if hasattr(router, 'config'):
                checkpoint['config'] = {
                    'max_iterations': router.config.max_iterations,
                    'hist_cost_weight': router.config.hist_cost_weight,
                    'pres_fac_mult': router.config.pres_fac_mult,
                    'pres_fac_max': router.config.pres_fac_max,
                    'hist_gain': router.config.hist_gain,
                    'via_cost': router.config.via_cost,
                }

            # Save board geometry for instant resume (LARGE but eliminates 1-hour init)
            logger.info("[CHECKPOINT] Saving board geometry (graph, lattice) for instant resume...")
            checkpoint['geometry'] = {}

            # Save graph (CSRGraph)
            if hasattr(router, 'graph') and router.graph is not None:
                graph = router.graph
                # CSRGraph doesn't store N/E - calculate from arrays
                N = len(graph.indptr) - 1 if graph.indptr is not None else 0
                E = len(graph.indices) if graph.indices is not None else 0

                checkpoint['geometry']['graph'] = {
                    'indptr': graph.indptr.copy(),
                    'indices': graph.indices.copy(),
                    'base_costs': graph.base_costs.get() if hasattr(graph.base_costs, 'get') else graph.base_costs.copy(),
                    'N': N,
                    'E': E,
                }
                # Optional graph attributes
                if hasattr(graph, 'edge_layer'):
                    checkpoint['geometry']['graph']['edge_layer'] = graph.edge_layer.get() if hasattr(graph.edge_layer, 'get') else graph.edge_layer.copy()
                if hasattr(graph, 'edge_kind'):
                    checkpoint['geometry']['graph']['edge_kind'] = graph.edge_kind.get() if hasattr(graph.edge_kind, 'get') else graph.edge_kind.copy()

                logger.info(f"[CHECKPOINT] Saved graph: {N:,} nodes, {E:,} edges")

            # Save lattice (Lattice3D)
            if hasattr(router, 'lattice') and router.lattice is not None:
                lattice = router.lattice
                checkpoint['geometry']['lattice'] = {
                    'pitch': lattice.pitch,
                    'layers': lattice.layers,
                    'bounds': lattice.bounds,
                    'Nx': lattice.Nx,
                    'Ny': lattice.Ny,
                    'Nz': lattice.Nz,
                }
                # Save coordinate mapping if available
                if hasattr(lattice, 'idx_to_coord'):
                    checkpoint['geometry']['lattice']['idx_to_coord'] = lattice.idx_to_coord.copy()

            # Save pad mappings
            if hasattr(router, 'pad_to_node'):
                checkpoint['geometry']['pad_to_node'] = dict(router.pad_to_node)

            # Save escape routing
            if hasattr(router, '_escape_tracks'):
                checkpoint['geometry']['escape_tracks'] = router._escape_tracks
            if hasattr(router, '_escape_vias'):
                checkpoint['geometry']['escape_vias'] = router._escape_vias

            # Save via metadata
            if hasattr(router, '_via_edge_metadata') and router._via_edge_metadata is not None:
                via_meta = router._via_edge_metadata
                checkpoint['geometry']['via_metadata'] = {
                    'indices': via_meta['indices'].get() if hasattr(via_meta['indices'], 'get') else via_meta['indices'].copy(),
                    'xu': via_meta['xu'].get() if hasattr(via_meta['xu'], 'get') else via_meta['xu'].copy(),
                    'yu': via_meta['yu'].get() if hasattr(via_meta['yu'], 'get') else via_meta['yu'].copy(),
                    'z_lo': via_meta['z_lo'].get() if hasattr(via_meta['z_lo'], 'get') else via_meta['z_lo'].copy(),
                    'z_hi': via_meta['z_hi'].get() if hasattr(via_meta['z_hi'], 'get') else via_meta['z_hi'].copy(),
                }

            # Save grid dimensions
            if hasattr(router, '_Nx'):
                checkpoint['geometry']['_Nx'] = router._Nx
                checkpoint['geometry']['_Ny'] = router._Ny
                checkpoint['geometry']['_Nz'] = router._Nz

            # Write to file
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"[CHECKPOINT] Saved iteration {iteration} to {filepath.name} ({file_size_mb:.1f} MB)")

            return str(filepath)

        except Exception as e:
            logger.error(f"[CHECKPOINT] Failed to save iteration {iteration}: {e}")
            raise

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Load checkpoint from file.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary containing checkpoint state
        """
        try:
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)

            iteration = checkpoint.get('iteration', -1)
            logger.info(f"[CHECKPOINT] Loaded checkpoint from iteration {iteration}: {Path(filepath).name}")

            return checkpoint

        except Exception as e:
            logger.error(f"[CHECKPOINT] Failed to load checkpoint {filepath}: {e}")
            raise

    def restore_checkpoint(self, router, checkpoint: Dict[str, Any]):
        """
        Restore router state from checkpoint.

        Args:
            router: PathFinderRouter instance to restore
            checkpoint: Checkpoint dictionary from load_checkpoint()
        """
        try:
            # Restore iteration
            router.iteration = checkpoint['iteration']

            # Restore pres_fac (for resume to use correct pressure)
            if 'pres_fac' in checkpoint:
                router._checkpoint_pres_fac = checkpoint['pres_fac']

            # Restore paths
            router.net_paths = checkpoint['net_paths']

            # Restore accounting state
            if 'accounting' in checkpoint and router.accounting is not None:
                acc_data = checkpoint['accounting']

                if router.accounting.use_gpu:
                    import cupy as cp
                    router.accounting.present[:] = cp.asarray(acc_data['present'])
                    router.accounting.present_ema[:] = cp.asarray(acc_data['present_ema'])
                    router.accounting.history[:] = cp.asarray(acc_data['history'])
                    router.accounting.capacity[:] = cp.asarray(acc_data['capacity'])
                else:
                    router.accounting.present[:] = acc_data['present']
                    router.accounting.present_ema[:] = acc_data['present_ema']
                    router.accounting.history[:] = acc_data['history']
                    router.accounting.capacity[:] = acc_data['capacity']

            # Restore via state
            if 'via_state' in checkpoint:
                via_state = checkpoint['via_state']
                if hasattr(router, 'via_col_use'):
                    router.via_col_use[:] = via_state['via_col_use']
                    router.via_col_pres[:] = via_state['via_col_pres']
                    if via_state['via_seg_use'] is not None:
                        router.via_seg_use[:] = via_state['via_seg_use']
                        router.via_seg_pres[:] = via_state['via_seg_pres']

            logger.info(f"[CHECKPOINT] Restored router to iteration {router.iteration}")
            logger.info(f"[CHECKPOINT] Restored {len(router.net_paths)} net paths")

            # Restore board geometry for instant resume (NEW!)
            if 'geometry' in checkpoint:
                logger.info("[CHECKPOINT] Restoring board geometry (instant resume)...")
                geom = checkpoint['geometry']

                # Restore graph
                if 'graph' in geom:
                    from ..unified_pathfinder import CSRGraph
                    import numpy as np

                    graph_data = geom['graph']
                    router.graph = CSRGraph(
                        indptr=graph_data['indptr'],
                        indices=graph_data['indices'],
                        base_costs=graph_data['base_costs'],
                        N=graph_data['N'],
                        use_gpu=router.config.use_gpu
                    )
                    # Restore optional attributes
                    if 'edge_layer' in graph_data:
                        if router.config.use_gpu:
                            import cupy as cp
                            router.graph.edge_layer = cp.asarray(graph_data['edge_layer'])
                        else:
                            router.graph.edge_layer = graph_data['edge_layer']
                    if 'edge_kind' in graph_data:
                        if router.config.use_gpu:
                            import cupy as cp
                            router.graph.edge_kind = cp.asarray(graph_data['edge_kind'])
                        else:
                            router.graph.edge_kind = graph_data['edge_kind']

                    logger.info(f"[CHECKPOINT] Restored graph: {router.graph.N} nodes, {router.graph.E} edges")

                # Restore lattice
                if 'lattice' in geom:
                    from ..unified_pathfinder import Lattice3D

                    lat_data = geom['lattice']
                    router.lattice = Lattice3D(
                        bounds=tuple(lat_data['bounds']),
                        pitch=lat_data['pitch'],
                        layers=lat_data['layers']
                    )
                    # Restore dimensions
                    router.lattice.Nx = lat_data['Nx']
                    router.lattice.Ny = lat_data['Ny']
                    router.lattice.Nz = lat_data['Nz']
                    # Restore coordinate mapping
                    if 'idx_to_coord' in lat_data:
                        router.lattice.idx_to_coord = lat_data['idx_to_coord']

                    logger.info(f"[CHECKPOINT] Restored lattice: {router.lattice.Nx}×{router.lattice.Ny}×{router.lattice.Nz}")

                # Restore pad mappings
                if 'pad_to_node' in geom:
                    router.pad_to_node = geom['pad_to_node']
                    logger.info(f"[CHECKPOINT] Restored {len(router.pad_to_node)} pad mappings")

                # Restore escape routing
                if 'escape_tracks' in geom:
                    router._escape_tracks = geom['escape_tracks']
                if 'escape_vias' in geom:
                    router._escape_vias = geom['escape_vias']

                # Restore via metadata
                if 'via_metadata' in geom:
                    via_meta = geom['via_metadata']
                    if router.config.use_gpu:
                        import cupy as cp
                        router._via_edge_metadata = {
                            'indices': cp.asarray(via_meta['indices']),
                            'xu': cp.asarray(via_meta['xu']),
                            'yu': cp.asarray(via_meta['yu']),
                            'xy_coords': cp.stack([cp.asarray(via_meta['xu']), cp.asarray(via_meta['yu'])], axis=1),
                            'z_lo': cp.asarray(via_meta['z_lo']),
                            'z_hi': cp.asarray(via_meta['z_hi']),
                        }
                    else:
                        import numpy as np
                        router._via_edge_metadata = {
                            'indices': via_meta['indices'],
                            'xu': via_meta['xu'],
                            'yu': via_meta['yu'],
                            'xy_coords': np.stack([via_meta['xu'], via_meta['yu']], axis=1),
                            'z_lo': via_meta['z_lo'],
                            'z_hi': via_meta['z_hi'],
                        }

                # Restore grid dimensions
                if '_Nx' in geom:
                    router._Nx = geom['_Nx']
                    router._Ny = geom['_Ny']
                    router._Nz = geom['_Nz']

                logger.info("[CHECKPOINT] ✓ Board geometry restored - instant resume ready!")

            # Regenerate geometry from loaded paths for visualization
            if hasattr(router, '_generate_geometry_from_paths'):
                try:
                    tracks, vias = router._generate_geometry_from_paths()
                    # Update provisional geometry so GUI can display it
                    from ..unified_pathfinder import GeometryPayload
                    router._provisional_geometry = GeometryPayload(tracks, vias)
                    logger.info(f"[CHECKPOINT] Regenerated geometry: {len(tracks)} tracks, {len(vias)} vias")
                except Exception as e:
                    logger.warning(f"[CHECKPOINT] Failed to regenerate geometry: {e}")
                    logger.warning(f"[CHECKPOINT] Visualization will be empty until routing starts")

        except Exception as e:
            logger.error(f"[CHECKPOINT] Failed to restore checkpoint: {e}")
            raise

    def list_checkpoints(self) -> list:
        """List all available checkpoints in chronological order."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        return [str(cp) for cp in checkpoints]

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the most recent checkpoint file."""
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def cleanup_old_checkpoints(self, keep_last_n: int = 10):
        """
        Delete old checkpoints, keeping only the most recent N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) > keep_last_n:
            to_delete = checkpoints[:-keep_last_n]
            for cp in to_delete:
                Path(cp).unlink()
                logger.info(f"[CHECKPOINT] Deleted old checkpoint: {Path(cp).name}")
