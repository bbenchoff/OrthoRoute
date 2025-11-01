#!/usr/bin/env python3
"""
Compare OrthoRoute JSON export with KiCad DRC report.

This script helps identify mismatches between what OrthoRoute thinks it exported
and what KiCad's DRC finds violations in.

Usage:
    python compare_drc_with_export.py <export.json> <drc.rpt>
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_drc_report(drc_path: Path) -> Dict:
    """Parse KiCad DRC .rpt file into structured data"""

    violations = []
    current_violation = None

    with open(drc_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # New violation type
            if line.startswith('[') and ']:' in line:
                if current_violation:
                    violations.append(current_violation)

                match = re.match(r'\[(\w+)\]: (.+)', line)
                if match:
                    current_violation = {
                        'type': match.group(1),
                        'description': match.group(2),
                        'items': []
                    }

            # Violation location/item
            elif line.startswith('@') and current_violation:
                # Extract coordinates and item info
                # Format: @(x mm, y mm): Type [NetName] on Layer
                coord_match = re.match(r'@\(([\d.]+) mm, ([\d.]+) mm\): (.+)', line)
                if coord_match:
                    x = float(coord_match.group(1))
                    y = float(coord_match.group(2))
                    item_desc = coord_match.group(3)

                    # Parse item description
                    # Examples:
                    # "Track [B01B05_013] on In13.Cu, length 7.2000 mm"
                    # "Blind/Buried Via [B01B05_013] on In13.Cu - In14.Cu"

                    item = {'x': x, 'y': y, 'description': item_desc}

                    # Extract net name
                    net_match = re.search(r'\[([^\]]+)\]', item_desc)
                    if net_match:
                        item['net'] = net_match.group(1)

                    # Extract type
                    if 'Track' in item_desc:
                        item['type'] = 'track'
                        # Extract layer
                        layer_match = re.search(r'on ([\w.]+)', item_desc)
                        if layer_match:
                            item['layer'] = layer_match.group(1)
                    elif 'Via' in item_desc:
                        item['type'] = 'via'
                        # Extract layer range
                        layer_match = re.search(r'on ([\w.]+) - ([\w.]+)', item_desc)
                        if layer_match:
                            item['from_layer'] = layer_match.group(1)
                            item['to_layer'] = layer_match.group(2)

                    current_violation['items'].append(item)

    # Add last violation
    if current_violation:
        violations.append(current_violation)

    return {'violations': violations}


def load_export_json(json_path: Path) -> Dict:
    """Load OrthoRoute export JSON"""
    with open(json_path, 'r') as f:
        return json.load(f)


def find_matching_geometry(violation_item: Dict, export_data: Dict, tolerance: float = 0.01) -> List[Dict]:
    """Find geometry in export that matches a DRC violation item"""

    matches = []
    item_type = violation_item.get('type')
    x = violation_item.get('x')
    y = violation_item.get('y')
    net = violation_item.get('net')

    if item_type == 'track':
        # Search tracks
        for track in export_data.get('tracks', []):
            # Check if violation point is on this track segment
            start = track['start']
            end = track['end']

            # Simple bounding box check
            x_min = min(start['x'], end['x']) - tolerance
            x_max = max(start['x'], end['x']) + tolerance
            y_min = min(start['y'], end['y']) - tolerance
            y_max = max(start['y'], end['y']) + tolerance

            if (x_min <= x <= x_max and y_min <= y <= y_max and
                track.get('net') == net):
                matches.append({
                    'type': 'track',
                    'data': track,
                    'distance': min(abs(x - start['x']) + abs(y - start['y']),
                                   abs(x - end['x']) + abs(y - end['y']))
                })

    elif item_type == 'via':
        # Search vias
        for via in export_data.get('vias', []):
            pos = via['position']
            dist = abs(pos['x'] - x) + abs(pos['y'] - y)

            if dist < tolerance and via.get('net') == net:
                matches.append({
                    'type': 'via',
                    'data': via,
                    'distance': dist
                })

    return sorted(matches, key=lambda m: m['distance'])


def analyze_violations(drc_data: Dict, export_data: Dict):
    """Analyze DRC violations against exported geometry"""

    print("\n" + "="*80)
    print("DRC VIOLATION ANALYSIS")
    print("="*80 + "\n")

    violations = drc_data.get('violations', [])

    # Categorize violations
    by_type = defaultdict(int)
    for v in violations:
        by_type[v['type']] += 1

    print(f"Total violations: {len(violations)}\n")
    print("Breakdown by type:")
    for vtype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {vtype:20s}: {count:4d}")

    # Detailed analysis of shorting_items (most critical)
    print("\n" + "="*80)
    print("DETAILED ANALYSIS: shorting_items")
    print("="*80 + "\n")

    shorting_violations = [v for v in violations if v['type'] == 'shorting_items']

    via_to_via = 0
    via_to_track = 0
    track_to_track = 0

    for v in shorting_violations[:50]:  # Analyze first 50
        items = v.get('items', [])
        if len(items) == 2:
            types = [item.get('type') for item in items]

            if types == ['via', 'via']:
                via_to_via += 1
                # Check if same coordinates
                if abs(items[0]['x'] - items[1]['x']) < 0.01 and abs(items[0]['y'] - items[1]['y']) < 0.01:
                    print(f"❌ VIA-VIA COLLISION at ({items[0]['x']:.2f}, {items[0]['y']:.2f})")
                    print(f"   Net 1: {items[0].get('net', 'unknown')}")
                    print(f"   Net 2: {items[1].get('net', 'unknown')}")

                    # Find in export
                    for item in items:
                        matches = find_matching_geometry(item, export_data)
                        if matches:
                            print(f"   Found in export: {matches[0]['data']}")
                        else:
                            print(f"   NOT FOUND in export!")
                    print()

            elif 'via' in types and 'track' in types:
                via_to_track += 1
                via_item = items[0] if items[0]['type'] == 'via' else items[1]
                track_item = items[1] if items[0]['type'] == 'via' else items[0]

                print(f"⚠️  VIA-TRACK COLLISION at ({via_item['x']:.2f}, {via_item['y']:.2f})")
                print(f"   Via net: {via_item.get('net', 'unknown')}")
                print(f"   Via layers: {via_item.get('from_layer', '?')} → {via_item.get('to_layer', '?')}")
                print(f"   Track net: {track_item.get('net', 'unknown')}")
                print(f"   Track layer: {track_item.get('layer', '?')}")

                # Check if via spans through track layer
                if 'from_layer' in via_item and 'to_layer' in via_item and 'layer' in track_item:
                    print(f"   ⚠️  Via spans through track's layer: {track_item['layer']}")
                print()

            elif types == ['track', 'track']:
                track_to_track += 1

    print(f"\nShorting violation breakdown:")
    print(f"  Via-to-Via collisions:   {via_to_via}")
    print(f"  Via-to-Track collisions: {via_to_track}")
    print(f"  Track-to-Track:          {track_to_track}")

    # Check clearance violations
    print("\n" + "="*80)
    print("CLEARANCE VIOLATIONS")
    print("="*80 + "\n")

    clearance_violations = [v for v in violations if v['type'] == 'clearance']
    print(f"Total: {len(clearance_violations)}")

    for v in clearance_violations[:10]:
        print(f"  {v.get('description', 'No description')}")

    return {
        'total': len(violations),
        'by_type': dict(by_type),
        'via_to_via': via_to_via,
        'via_to_track': via_to_track,
        'track_to_track': track_to_track
    }


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        print("\nUsage: python compare_drc_with_export.py <export.json> <drc.rpt>")
        sys.exit(1)

    export_path = Path(sys.argv[1])
    drc_path = Path(sys.argv[2])

    if not export_path.exists():
        print(f"Error: Export file not found: {export_path}")
        sys.exit(1)

    if not drc_path.exists():
        print(f"Error: DRC report not found: {drc_path}")
        sys.exit(1)

    print(f"Loading export: {export_path}")
    export_data = load_export_json(export_path)

    print(f"Loading DRC report: {drc_path}")
    drc_data = parse_drc_report(drc_path)

    print(f"\nExport metadata:")
    print(f"  Timestamp: {export_data['metadata']['timestamp']}")
    print(f"  Board: {export_data['metadata']['board_name']}")
    print(f"  Tracks: {export_data['metadata']['track_count']}")
    print(f"  Vias: {export_data['metadata']['via_count']}")

    # Analyze
    stats = analyze_violations(drc_data, export_data)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"OrthoRoute exported: {export_data['metadata']['track_count']} tracks, {export_data['metadata']['via_count']} vias")
    print(f"KiCad DRC found: {stats['total']} violations")

    if stats['via_to_via'] > 0:
        print(f"\n⚠️  CRITICAL: {stats['via_to_via']} via-to-via collisions (same coordinates)")
    if stats['via_to_track'] > 0:
        print(f"\n⚠️  CRITICAL: {stats['via_to_track']} via-to-track collisions (via spans through track layer)")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
