"""Unified routing pipeline - single source of truth for both CLI and GUI paths."""
import logging
import os

logger = logging.getLogger(__name__)

def run_pipeline(board, router):
    """Run the complete routing pipeline with proper initialization order.
    
    This is the ONLY entry point for routing - both CLI and GUI must use this function.
    Nothing routes unless this pipeline runs in the correct sequence:
    1) Build lattice & CSR (or use already-built)
    2) Preflight validation 
    3) Map pads to lattice (no CSR mutation)
    4) Route nets (batch)
    
    Args:
        board: Board domain object with nets, components, etc.
        router: Routing engine instance (UnifiedPathFinder, etc.)
        
    Returns:
        Routing results from router.route_multiple_nets()
    """
    logger.info(f"[PIPELINE] Starting unified routing pipeline for {len(board.nets)} nets")
    
    # Step 1: Build lattice & CSR matrices
    logger.info(f"[PIPELINE] Step 1: Initialize graph with board data...")
    router.initialize_graph(board)
    
    # Log deterministic lattice count for CLI/GUI parity checks
    lattice_count = getattr(router, 'lattice_node_count', 'UNKNOWN')
    logger.info(f"[STEP5] Total deterministic lattice: {lattice_count} nodes")
    
    # Step 2: Preflight validation using shared checks
    logger.info(f"[PIPELINE] Step 2: Running preflight validation...")
    from ..algorithms.manhattan.graph_checks import preflight_graph
    if not preflight_graph(router.graph_state):
        raise RuntimeError("PREFLIGHT failed - graph validation errors detected")
    
    # Step 3: Add invariant assertions right after graph initialization
    N = router.lattice_node_count
    assert hasattr(router, 'node_coordinates_lattice'), "Missing node_coordinates_lattice"
    assert router.node_coordinates_lattice.shape[0] == N, f"Coordinate array {router.node_coordinates_lattice.shape[0]} != lattice nodes {N}"
    assert len(router.indptr_g) == N + 1, f"indptr length {len(router.indptr_g)} != N+1 ({N+1})"
    assert router.indptr_g[-1] == len(router.indices_g), f"indptr[-1] {router.indptr_g[-1]} != edges {len(router.indices_g)}"
    logger.info(f"[PIPELINE] Invariant assertions PASSED: {N} nodes, {len(router.indices_g)} edges")
    
    # Step 4: Map pads to lattice (degree-aware snap, no CSR mutation)
    logger.info(f"[PIPELINE] Step 3: Mapping pads to lattice...")
    router.map_all_pads(board)
    
    # Validate lattice freeze after pad mapping
    from ..algorithms.manhattan.graph_checks import validate_lattice_integrity
    if not validate_lattice_integrity(router.graph_state):
        raise RuntimeError("LATTICE INTEGRITY failed - lattice was mutated during pad mapping")
    
    # Step 5: Route all nets
    logger.info(f"[PIPELINE] Step 4: Routing {len(board.nets)} nets...")
    results = router.route_multiple_nets(board.nets)
    
    logger.info(f"[PIPELINE] Pipeline complete - returning routing results")
    return results


def create_unified_router(engine_type="unified_pathfinder", use_gpu=True):
    """DEPRECATED: Create a routing engine with fallback disabled during development.
    
    This function is deprecated. UnifiedPathFinder instances should ONLY be created
    in the plugin and passed through the pipeline to avoid the "second instance" bug.
    
    Args:
        engine_type: "unified_pathfinder" or "manhattan_router_rrg" 
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Configured routing engine instance
    """
    logger.warning(f"[UPF] DEPRECATED: create_unified_router() called - should use plugin-created instance instead")
    logger.info(f"[UPF] Selected router: {engine_type}")
    
    if engine_type == "unified_pathfinder":
        try:
            from ..algorithms.manhattan.unified_pathfinder import UnifiedPathFinder, PathFinderConfig
            logger.info("[UPF] Loading UnifiedPathFinder (improved coordinate handling)")
            router = UnifiedPathFinder(config=PathFinderConfig(), use_gpu=use_gpu)
            
            # Ban fallback during development to catch regressions
            if os.getenv("ORTHO_NO_FALLBACK", "1") == "1":
                logger.info("[UPF] Fallback disabled for development")
                if hasattr(router, 'disable_fallback'):
                    router.disable_fallback = True
            
            return router
            
        except Exception as e:
            logger.warning(f"[UPF] Failed to load UnifiedPathFinder: {e}")
            if os.getenv("ORTHO_NO_FALLBACK", "1") == "1":
                raise RuntimeError(f"UnifiedPathFinder required but failed to load: {e}")
            engine_type = "manhattan_router_rrg"  # fallback
    
    if engine_type == "manhattan_router_rrg":
        # DISABLE RRG during bring-up to avoid split codepaths masking bugs
        raise RuntimeError("RRG disabled during bring-up. Use unified_pathfinder.")
    
    raise ValueError(f"Unknown routing engine: {engine_type}")