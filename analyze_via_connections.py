import json

# Load export data
with open(r'C:\Users\Benchoff\Documents\GitHub\OrthoRoute\debug_output\kicad_export_20251031_172921.json') as f:
    data = json.load(f)

tracks = data['tracks']
vias = data['vias']

print("=== VIA CONNECTION ANALYSIS ===\n")

# For each via, check if tracks connect to it on both endpoint layers
layer_map = {'F.Cu': 0, 'B.Cu': 17}
for i in range(1, 17):
    layer_map[f'In{i}.Cu'] = i

dangling_count = 0
examples = []

for via_idx, via in enumerate(vias):
    # Get via location and layers (handle both coordinate formats)
    if 'x' in via and 'y' in via:
        vx, vy = via['x'], via['y']
    elif 'start' in via:
        vx, vy = via['start']['x'], via['start']['y']
    else:
        continue

    from_layer = via['from_layer']
    to_layer = via['to_layer']
    net = via['net']

    # Find tracks on each layer that touch this via (within 0.01mm tolerance)
    tracks_on_from = [t for t in tracks
                      if t['net'] == net and t['layer'] == from_layer
                      and (abs(t['start']['x'] - vx) < 0.01 and abs(t['start']['y'] - vy) < 0.01
                           or abs(t['end']['x'] - vx) < 0.01 and abs(t['end']['y'] - vy) < 0.01)]

    tracks_on_to = [t for t in tracks
                    if t['net'] == net and t['layer'] == to_layer
                    and (abs(t['start']['x'] - vx) < 0.01 and abs(t['start']['y'] - vy) < 0.01
                         or abs(t['end']['x'] - vx) < 0.01 and abs(t['end']['y'] - vy) < 0.01)]

    # Via is dangling if no tracks on either endpoint layer
    if len(tracks_on_from) == 0 or len(tracks_on_to) == 0:
        dangling_count += 1
        if len(examples) < 10:
            examples.append({
                'net': net,
                'location': (vx, vy),
                'from': from_layer,
                'to': to_layer,
                'tracks_on_from': len(tracks_on_from),
                'tracks_on_to': len(tracks_on_to)
            })

print(f"Total vias: {len(vias)}")
print(f"Dangling vias (missing track on at least one endpoint): {dangling_count}")
print(f"Valid vias: {len(vias) - dangling_count}")
print(f"DRC violations: 1536")
print()

if examples:
    print("Example dangling vias:")
    for ex in examples:
        print(f"  {ex['net']}: ({ex['location'][0]:.1f}, {ex['location'][1]:.1f})")
        print(f"    {ex['from']} -> {ex['to']}")
        print(f"    Tracks on {ex['from']}: {ex['tracks_on_from']}, Tracks on {ex['to']}: {ex['tracks_on_to']}")
        print()
