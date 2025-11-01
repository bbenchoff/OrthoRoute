import json

# Load export data
with open(r'C:\Users\Benchoff\Documents\GitHub\OrthoRoute\debug_output\kicad_export_20251031_172921.json') as f:
    data = json.load(f)

vias = data['vias']
tracks = data['tracks']

# Categorize vias
escape_vias = [v for v in vias if v['from_layer'] == 'F.Cu']
routing_vias = [v for v in vias if v['from_layer'] != 'F.Cu']

print(f"=== VIA ANALYSIS ===")
print(f"Total vias: {len(vias)}")
print(f"  Escape vias (F.Cu -> internal): {len(escape_vias)}")
print(f"  Routing vias (internal -> internal): {len(routing_vias)}")
print(f"DRC violations: 1536")
print()

# Check multi-layer spans
print(f"=== MULTI-LAYER VIA SPANS ===")
layer_map = {'F.Cu': 0}
for i in range(1, 17):
    layer_map[f'In{i}.Cu'] = i

multi_layer_vias = []
for v in vias:
    from_idx = layer_map.get(v['from_layer'], 0)
    to_idx = layer_map.get(v['to_layer'], 0)
    span = abs(to_idx - from_idx)
    if span > 1:
        multi_layer_vias.append((v['net'], v['from_layer'], v['to_layer'], span))

print(f"Vias spanning >1 layer: {len(multi_layer_vias)}")
if multi_layer_vias:
    print("Examples:")
    for net, fr, to, span in multi_layer_vias[:10]:
        print(f"  {net}: {fr} -> {to} (span={span} layers)")
print()

# Find a specific DRC violation via
print(f"=== SPECIFIC DRC VIOLATION CHECK ===")
print("DRC says: (231.6, 58.4) B01B05_013 on In13.Cu - In14.Cu")
target_vias = [v for v in vias if abs(v['x'] - 231.6) < 0.5 and abs(v['y'] - 58.4) < 0.5]
print(f"Vias near (231.6, 58.4): {len(target_vias)}")
for v in target_vias:
    print(f"  {v['net']}: ({v['x']:.2f}, {v['y']:.2f}) {v['from_layer']} -> {v['to_layer']}")
