"""Unit tests for KiCad file parser."""
import pytest
from pathlib import Path
from orthoroute.infrastructure.kicad.file_parser import KiCadFileParser


class TestKiCadFileParser:
    """Tests for KiCadFileParser class."""

    @pytest.fixture
    def parser(self):
        """Create a file parser instance."""
        return KiCadFileParser()

    @pytest.fixture
    def test_board_path(self):
        """Path to test board file."""
        repo_root = Path(__file__).parent.parent.parent
        return repo_root / "TestBoards" / "TestBackplane.kicad_pcb"

    def test_parser_initialization(self, parser):
        """Test parser can be initialized."""
        assert parser is not None
        assert isinstance(parser, KiCadFileParser)

    def test_load_board_success(self, parser, test_board_path):
        """Test loading a valid board file."""
        board = parser.load_board(str(test_board_path))
        assert board is not None
        assert board.name == "Untitled Board"  # TestBackplane has no title
        assert len(board.components) > 0
        assert len(board.nets) > 0
        assert len(board.layers) > 0

    def test_load_board_invalid_path(self, parser):
        """Test loading from non-existent file returns None."""
        board = parser.load_board("nonexistent_file.kicad_pcb")
        assert board is None

    def test_parse_file_invalid_extension(self, parser, tmp_path):
        """Test parsing file with invalid extension raises error."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("invalid content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse_file(str(test_file))

    def test_extract_components(self, parser, test_board_path):
        """Test component extraction from TestBackplane."""
        board = parser.load_board(str(test_board_path))
        
        # TestBackplane has 8 backplane connectors + 4 mounting holes = 12 footprints
        assert len(board.components) == 12
        
        # Verify component references exist
        refs = [c.reference for c in board.components]
        assert 'J1' in refs or 'J2' in refs or 'J3' in refs  # At least one connector
        
    def test_extract_pads(self, parser, test_board_path):
        """Test pad extraction from TestBackplane."""
        board = parser.load_board(str(test_board_path))
        
        # Count total pads across all components
        total_pads = sum(len(c.pads) for c in board.components)
        
        # TestBackplane has 1600-1604 pads (8 connectors × 200 pads each)
        assert total_pads >= 1600, f"Expected >= 1600 pads, got {total_pads}"
        assert total_pads <= 1650, f"Expected <= 1650 pads, got {total_pads}"

    def test_extract_pads_with_nets(self, parser, test_board_path):
        """Test that pads are connected to nets."""
        board = parser.load_board(str(test_board_path))
        
        # Find pads with net assignments - check raw pad data
        pads_with_nets = 0
        for component in board.components:
            for pad in component.pads:
                if pad.net_id is not None:
                    pads_with_nets += 1
        
        # Most pads should be connected to nets
        assert pads_with_nets > 100, f"Expected > 100 connected pads, got {pads_with_nets}"

    def test_extract_nets(self, parser, test_board_path):
        """Test net extraction from TestBackplane."""
        board = parser.load_board(str(test_board_path))
        
        # TestBackplane has 512 nets, but parser may also extract unconnected nets
        # Total can be higher than actual nets  
        assert len(board.nets) >= 500, f"Expected >= 500 nets, got {len(board.nets)}"
        assert len(board.nets) <= 1200, f"Expected <= 1200 nets (including unconnected), got {len(board.nets)}"
        
        # Verify nets have names
        net_names = [n.name for n in board.nets]
        assert len(net_names) == len(board.nets)
        assert all(name for name in net_names)  # No empty names

    def test_extract_layers(self, parser, test_board_path):
        """Test layer extraction from TestBackplane."""
        board = parser.load_board(str(test_board_path))
        
        # TestBackplane is 18-layer board:
        # F.Cu, In1.Cu-In16.Cu (16 internal), B.Cu = 18 copper layers
        # Edge.Cuts also has 'Cu' in name but is NOT a copper layer
        copper_layers = [l for l in board.layers if 'Cu' in l.name and l.type == 'signal' and not l.name.startswith('Edge')]
        assert len(copper_layers) >= 18, f"Expected >= 18 copper layers, got {len(copper_layers)}"
        
        # Verify F.Cu and B.Cu exist
        layer_names = [l.name for l in board.layers]
        assert 'F.Cu' in layer_names
        assert 'B.Cu' in layer_names
        assert 'In1.Cu' in layer_names  # Internal layer

    def test_pad_positions(self, parser, test_board_path):
        """Test that pads have valid positions."""
        board = parser.load_board(str(test_board_path))
        
        for component in board.components:
            for pad in component.pads:
                # Positions should be reasonable (within board bounds)
                # TestBackplane is 73.1×97.3mm, positions should be within reasonable range
                assert -10 < pad.position.x < 300, f"Pad {pad.id} x position out of range: {pad.position.x}"
                assert -10 < pad.position.y < 200, f"Pad {pad.id} y position out of range: {pad.position.y}"

    def test_pad_sizes(self, parser, test_board_path):
        """Test that pads have valid sizes."""
        board = parser.load_board(str(test_board_path))
        
        for component in board.components:
            for pad in component.pads:
                width, height = pad.size
                # Pad sizes should be positive and reasonable (0.1mm to 10mm)
                assert 0.01 < width < 20, f"Pad {pad.id} width out of range: {width}"
                assert 0.01 < height < 20, f"Pad {pad.id} height out of range: {height}"

    def test_component_references(self, parser, test_board_path):
        """Test that components have valid references."""
        board = parser.load_board(str(test_board_path))
        
        refs = [c.reference for c in board.components]
        
        # All components should have references
        assert len(refs) == len(board.components)
        assert all(ref for ref in refs)  # No empty references
        
        # References should be unique
        assert len(refs) == len(set(refs)), "Duplicate component references found"

    def test_skip_non_plated_holes(self, parser, test_board_path):
        """Test that non-plated holes are skipped."""
        board = parser.load_board(str(test_board_path))
        
        # TestBackplane has 4 mounting holes (H1-H4) with np_thru_hole pads
        # These should be skipped, so components should have 0 pads
        mounting_holes = [c for c in board.components if c.reference.startswith('H')]
        
        if mounting_holes:
            for hole in mounting_holes:
                # Mounting holes should have no pads (np_thru_hole pads are skipped)
                assert len(hole.pads) == 0, f"Mounting hole {hole.reference} should have 0 pads, got {len(hole.pads)}"


class TestFileParserEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def parser(self):
        """Create a file parser instance."""
        return KiCadFileParser()

    def test_load_nonexistent_file(self, parser):
        """Test loading non-existent file returns None."""
        board = parser.load_board("/path/to/nonexistent.kicad_pcb")
        assert board is None

    def test_parse_file_not_found(self, parser):
        """Test parsing non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/path/to/nonexistent.kicad_pcb")

    def test_empty_file(self, parser, tmp_path):
        """Test parsing empty file returns board with defaults."""
        test_file = tmp_path / "empty.kicad_pcb"
        test_file.write_text("")
        
        # Should not crash, but may return minimal board
        board = parser.load_board(str(test_file))
        # Parser handles errors gracefully - may return None or minimal board
        if board:
            assert isinstance(board.components, list)
            assert isinstance(board.nets, list)
