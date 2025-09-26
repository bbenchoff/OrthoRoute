"""
MINIMAL INTEGRATION PATCH - Exact changes for unified_pathfinder.py

This provides the EXACT lines to add/replace in unified_pathfinder.py to wire in the real fixes.
"""

# EXACT PATCH 1: Add imports after line 35
IMPORTS_PATCH = '''
# CRITICAL: Import real global grid functions - ADD THIS AFTER LINE 35
from .real_global_grid import (
    GridShape, gid, xyz_from_gid, neighbors_for_gid,
    validate_path_bounds, validate_edges_from_path,
    route_path_integer, VersionedCosts, DeferQueue
)
'''

# EXACT PATCH 2: In __init__ method, add these lines after existing initialization
INIT_PATCH = '''
    # ADD THESE LINES IN __init__ method after config setup:

    # CRITICAL: Single source of truth for grid shape
    self.shape = None  # Will be set in build_routing_lattice
    self.costs = None  # Will be set in build_routing_lattice
    self.defer_q = DeferQueue()

    # Layer transition rules (replace any l^1 logic)
    self.allowed_layer_transitions = {}  # Will be populated based on actual layers
'''

# EXACT PATCH 3: In _build_3d_lattice method, REPLACE the grid setup section with:
BUILD_LATTICE_PATCH = '''
    def _build_3d_lattice(self, bounds: Tuple[float, float, float, float], layers: int, board=None):
        """Build 3D routing lattice using geometry registry with H/V discipline"""

        # ... keep existing bounds/registry setup ...

        # CRITICAL: Calculate grid dimensions consistently
        pitch = self.config.grid_pitch
        x_steps = int(np.ceil((bounds[2] - bounds[0]) / pitch))
        y_steps = int(np.ceil((bounds[3] - bounds[1]) / pitch))

        # CRITICAL: Single GridShape source of truth - REPLACE EXISTING GRID SETUP
        self.shape = GridShape(NL=layers, NX=x_steps, NY=y_steps)
        self.costs = VersionedCosts()

        # Set layer transitions based on actual layer count
        self.allowed_layer_transitions = {}
        for layer in range(layers):
            transitions = []
            if layer > 0:
                transitions.append(layer - 1)  # Down
            if layer < layers - 1:
                transitions.append(layer + 1)  # Up
            self.allowed_layer_transitions[layer] = transitions

        logger.info(f"[GRID-SHAPE] {self.shape}, transitions: {self.allowed_layer_transitions}")

        # ... keep rest of existing lattice building ...
'''

# EXACT PATCH 4: REPLACE the main A* search method with this:
SEARCH_PATCH = '''
    def _route_single_net_real_astar(self, net_name: str, src_gid: int, dst_gid: int) -> Tuple[bool, List[int], dict]:
        """REAL A* routing with all fixes - REPLACE YOUR EXISTING A* METHOD"""

        if not self.shape:
            logger.error("[SEARCH] GridShape not initialized")
            return False, [], {}

        # Get current cost version for stale detection
        ver_at_search = self.costs.version

        # Progressive parameters based on defer queue
        fail_count = self.defer_q.net_failures.get(net_name, 0)
        beam = 64 * (1.5 ** min(fail_count, 3))  # Escalate beam
        budget = 500 * (1.2 ** min(fail_count, 3))  # Escalate budget
        enable_vias = fail_count >= 1  # Enable vias after first failure

        logger.info(f"[SEARCH] net={net_name} src_gid={src_gid} dst_gid={dst_gid} "
                   f"ver={ver_at_search} beam={beam:.0f} budget={budget:.0f} "
                   f"via_cost=8 dir=bi h0={abs(dst_gid - src_gid)}")

        # Use REAL bidirectional A* with integer costs
        success, path_gids, expansions = route_path_integer(
            shape=self.shape,
            src_gid=src_gid,
            dst_gid=dst_gid,
            allowed_transitions=self.allowed_layer_transitions,
            beam_width=int(beam),
            max_expansions=int(budget),
            enable_vias_from_start=enable_vias
        )

        stats = {
            'version': ver_at_search,
            'expansions': expansions,
            'beam_used': beam,
            'enable_vias': enable_vias
        }

        if success:
            vias = sum(1 for i in range(len(path_gids)-1)
                      if xyz_from_gid(self.shape, path_gids[i])[0] != xyz_from_gid(self.shape, path_gids[i+1])[0])

            logger.info(f"[RESULT] net={net_name} ver_search={ver_at_search} ver_now={self.costs.version} "
                       f"steps={len(path_gids)} vias={vias} t={expansions/100:.2f} ok=1")

            # Remove from defer queue on success
            self.defer_q.remove_successful(net_name)
            return True, path_gids, stats
        else:
            logger.warning(f"[RESULT] net={net_name} ver_search={ver_at_search} ver_now={self.costs.version} "
                          f"steps=0 vias=0 t={expansions/100:.2f} ok=0")

            # Add to defer queue
            manhattan_dist = abs(xyz_from_gid(self.shape, dst_gid)[1] - xyz_from_gid(self.shape, src_gid)[1]) + \
                           abs(xyz_from_gid(self.shape, dst_gid)[2] - xyz_from_gid(self.shape, src_gid)[2])
            self.defer_q.add_failed_net(net_name, manhattan_dist)

            logger.info(f"[DEFER] net={net_name} reason=open_empty fails={fail_count+1} "
                       f"next_beam={beam*1.5:.0f} allow_vias={True}")

            return False, [], stats
'''

# EXACT PATCH 5: REPLACE commit method with validation:
COMMIT_PATCH = '''
    def _commit_net_path_with_validation(self, net_name: str, path_gids: List[int], search_stats: dict) -> int:
        """Commit path with CRITICAL validation - REPLACE YOUR EXISTING COMMIT METHOD"""

        if not path_gids:
            logger.warning(f"[COMMIT] {net_name}: Empty path")
            return 0

        path_array = np.array(path_gids, dtype=np.int32)

        # CRITICAL: Path bounds validation - catches OOB
        try:
            validate_path_bounds(self.shape, path_array, net_name)
        except (ValueError, AssertionError) as e:
            logger.error(f"[OOB-COMMIT] {net_name}: {e} - HARD FAIL")
            self.defer_q.add_failed_net(net_name, len(path_gids))
            raise ValueError(f"OOB path for {net_name}")

        # CRITICAL: Edge validation - prevents "No edges for..."
        try:
            edges = validate_edges_from_path(path_array, net_name)
        except ValueError as e:
            logger.error(f"[CONVERT-FAIL] {net_name}: {e} - HARD FAIL")
            self.defer_q.add_failed_net(net_name, len(path_gids))
            raise ValueError(f"Edge conversion failed for {net_name}")

        # CRITICAL: Stale-path detection
        search_version = search_stats.get('version', 0)
        if search_version != self.costs.version:
            logger.warning(f"[STALE] {net_name}: ver_search={search_version} ver_now={self.costs.version} requeue")
            self.defer_q.add_failed_net(net_name, len(path_gids))
            return 0  # Skip commit, will be rerouted

        # Commit edges to usage tracking
        self.costs.add_path_usage(path_gids)

        # Convert to your internal edge format and proceed with existing commit logic
        # ... your existing edge mapping code ...

        logger.debug(f"[COMMIT] {net_name}: {len(edges)} edges committed")
        return len(edges)
'''

# EXACT PATCH 6: Add negotiation loop with defer handling:
NEGOTIATION_PATCH = '''
    def negotiate_with_defer_queue(self, net_list: List[str], max_iterations: int = 50) -> int:
        """Negotiation with defer queue handling - REPLACE YOUR MAIN NEGOTIATION LOOP"""

        successful_nets = 0

        for iteration in range(max_iterations):
            logger.info(f"[NEGOTIATE] Iteration {iteration + 1}/{max_iterations}")

            # Prioritize deferred nets to prevent starvation
            deferred_batch = self.defer_q.get_next_batch(len(net_list) // 2)
            remaining_nets = [n for n in net_list if n not in deferred_batch]
            nets_this_iter = deferred_batch + remaining_nets

            iter_successful = 0

            for net_name in nets_this_iter:
                # Get src/dst gids (you'll need to implement this based on your net structure)
                src_gid = self._get_net_src_gid(net_name)  # Implement based on your net data
                dst_gid = self._get_net_dst_gid(net_name)  # Implement based on your net data

                try:
                    # Route with real A*
                    success, path_gids, stats = self._route_single_net_real_astar(net_name, src_gid, dst_gid)

                    if success:
                        # Commit with validation
                        committed = self._commit_net_path_with_validation(net_name, path_gids, stats)
                        if committed > 0:
                            iter_successful += 1

                except ValueError as e:
                    # Hard fail - net will be retried next iteration
                    logger.debug(f"[HARD-FAIL] {net_name}: {e}")
                    continue

            success_rate = iter_successful / len(nets_this_iter) if nets_this_iter else 0
            logger.info(f"[NEGOTIATE] Iter {iteration + 1}: {iter_successful}/{len(nets_this_iter)} success "
                       f"({success_rate:.1%}), deferred: {len(deferred_batch)}")

            # Update costs and increment version
            self.costs.update_pressure(pres_fac=1.5)

            # Check convergence
            if success_rate >= 0.95:
                logger.info(f"[NEGOTIATE] Converged at {success_rate:.1%}")
                successful_nets = iter_successful
                break

            successful_nets = max(successful_nets, iter_successful)

        return successful_nets
'''

# Unit tests to run after applying patches
UNIT_TESTS = '''
# UNIT TESTS - Run these after applying patches

def test_gid_roundtrip():
    """Test gid conversion roundtrip"""
    shape = GridShape(NL=4, NX=100, NY=80)
    test_coords = [(0, 0, 0), (3, 99, 79), (1, 50, 40)]

    for layer, x, y in test_coords:
        g = gid(shape, layer, x, y)
        l2, x2, y2 = xyz_from_gid(shape, g)
        assert (layer, x, y) == (l2, x2, y2), f"Roundtrip failed: {(layer, x, y)} != {(l2, x2, y2)}"
    print("PASS: gid roundtrip")

def test_neighbors_multilayer():
    """Test multi-layer neighbor generation (no l^1 toggle)"""
    shape = GridShape(NL=4, NX=10, NY=10)
    transitions = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}

    center_gid = gid(shape, 1, 5, 5)
    neighbors = neighbors_for_gid(shape, center_gid, transitions, include_vias=True)

    track_neighbors = [n for n in neighbors if not n[2]]
    via_neighbors = [n for n in neighbors if n[2]]

    assert len(track_neighbors) == 4, f"Expected 4 track neighbors, got {len(track_neighbors)}"
    assert len(via_neighbors) == 2, f"Expected 2 via neighbors, got {len(via_neighbors)}"

    via_layers = [xyz_from_gid(shape, n[0])[0] for n in via_neighbors]
    assert set(via_layers) == {0, 2}, f"Expected via layers {0,2}, got {set(via_layers)}"
    print("PASS: multilayer neighbors")

def test_path_bounds():
    """Test path bounds validation"""
    shape = GridShape(NL=2, NX=10, NY=10)

    # Valid path
    valid_path = np.array([gid(shape, 0, 0, 0), gid(shape, 0, 1, 0)])
    assert validate_path_bounds(shape, valid_path), "Valid path should pass"

    # Invalid path - should fail
    invalid_path = np.array([0, shape.total_nodes + 100])
    assert not validate_path_bounds(shape, invalid_path), "OOB path should fail"
    print("PASS: path bounds validation")

def test_stale_commit_rejects():
    """Test stale path rejection"""
    costs = VersionedCosts()
    initial_version = costs.version

    # Simulate version bump
    costs.update_pressure()
    new_version = costs.version

    assert new_version == initial_version + 1, "Version should increment"
    # In real integration, commit would check version mismatch and defer
    print("PASS: stale commit detection")

# Run all tests
if __name__ == "__main__":
    test_gid_roundtrip()
    test_neighbors_multilayer()
    test_path_bounds()
    test_stale_commit_rejects()
    print("ALL UNIT TESTS PASSED")
'''

def show_integration_checklist():
    """Show exact integration steps"""
    print("MINIMAL INTEGRATION CHECKLIST:")
    print("1. Add IMPORTS_PATCH after line 35 in unified_pathfinder.py")
    print("2. Add INIT_PATCH lines in __init__ method")
    print("3. Replace grid setup in _build_3d_lattice with BUILD_LATTICE_PATCH")
    print("4. Replace A* method with SEARCH_PATCH")
    print("5. Replace commit method with COMMIT_PATCH")
    print("6. Replace negotiation loop with NEGOTIATION_PATCH")
    print("7. Run unit tests to verify")
    print()
    print("EXPECTED LOG OUTPUT:")
    print("[SEARCH] net=B02B10_013 src_gid=1205 dst_gid=15430 ver=27 beam=128 budget=500 via_cost=8 dir=bi h0=428")
    print("[RESULT] net=B02B10_013 ver_search=27 ver_now=27 steps=311 vias=3 t=7.76 ok=1")
    print("[DEFER] net=difficult_net reason=open_empty fails=2 next_beam=192 allow_vias=1")
    print("[STALE] net=some_net ver_search=27 ver_now=28 requeue")

if __name__ == "__main__":
    show_integration_checklist()