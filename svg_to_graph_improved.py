#!/usr/bin/env python3
"""
svg_to_graph_improved.py

Generates:
 - nodes.json        (list of nodes with id,x,y,label?)
 - edges.json        (list of [idA,idB])
 - nodes_edges.js    (browser-ready JS that sets `nodes` and `edges`)
 - room_to_node.json (mapping select-value -> nearest node id)

Uses real SVG coordinate space (viewBox or width/height). Adjust spacing/k as needed.
"""

import os, json, math
from xml.etree import ElementTree as ET

# Optional dependencies (if available) used for faster KNN
try:
    import numpy as np
    from scipy.spatial import cKDTree as KDTree
    HAVE_KDTREE = True
except Exception:
    HAVE_KDTREE = False

# PARAMETERS you can tweak
GRID_SPACING = 12.0    # spacing for interior sampled nodes (smaller => more nodes)
K_NEIGHBORS = 6        # K for KNN connectivity
MIN_DISTANCE_BETWEEN_NODES = 6.0  # dedupe threshold

SVG_FILE = 'idealab_floor_plan.svg'
OUT_DIR = '.'

def parse_svg(svg_file):
    paths = []
    for p in root.findall('.//{%s}path' % nsmap['svg']):
     d = p.get('d')
    tree = ET.parse(svg_file)
    root = tree.getroot()
    nsmap = {}
    # attempt to deduce namespace
    tag = root.tag
    if tag.startswith('{'):
        uri = tag.split('}')[0].strip('{')
        nsmap['svg'] = uri
    else:
        nsmap['svg'] = 'http://www.w3.org/2000/svg'

    # viewBox / width/height
    vb = root.get('viewBox')
    if vb:
        parts = list(map(float, vb.strip().split()))
        minx, miny, w, h = parts
    else:
        w = float(root.get('width') or 210)
        h = float(root.get('height') or 297)
        minx, miny = 0.0, 0.0

    # collect <text> with numeric x,y
    texts = []
    for t in root.findall('.//{%s}text' % nsmap['svg']):
        txt = ''.join(t.itertext()).strip()
        if not txt: continue
        x_attr = t.get('x') or t.get('dx')
        y_attr = t.get('y') or t.get('dy')
        if not x_attr or not y_attr: 
            # skip if coordinates not present
            continue
        try:
            x = float(str(x_attr).split()[0])
            y = float(str(y_attr).split()[0])
        except:
            continue
        texts.append({'text': txt, 'x': x, 'y': y})

    # collect <rect> obstacles (common shapes)
    rects = []
    for r in root.findall('.//{%s}rect' % nsmap['svg']):
        try:
            rx = float(r.get('x') or 0)
            ry = float(r.get('y') or 0)
            rw = float(r.get('width') or 0)
            rh = float(r.get('height') or 0)
            rects.append((rx, ry, rw, rh))
        except:
            continue

    return {'width': w, 'height': h, 'minx': minx, 'miny': miny, 'texts': texts, 'rects': rects}

def sample_grid(w, h, spacing, margin=6):
    pts = []
    y = margin
    while y < h - margin:
        x = margin
        while x < w - margin:
            pts.append((x, y))
            x += spacing
        y += spacing
    return pts

def dedupe_points(points, min_dist=4.0):
    kept = []
    for x,y,meta in points:
        skip=False
        for xx,yy,_m in kept:
            if math.hypot(xx-x, yy-y) < min_dist:
                skip=True; break
        if not skip:
            kept.append((x,y,meta))
    return kept

def build_nodes(svg_info):
    nodes = []
    # add text-labeled nodes first (so they have lower IDs)
    idx = 1
    for t in svg_info['texts']:
        nodes.append({'id': f'node{idx}', 'x': float(t['x']), 'y': float(t['y']), 'label': t['text']})
        idx += 1
    # add grid sampled points
    grid = sample_grid(svg_info['width'], svg_info['height'], GRID_SPACING)
    # convert to list of (x,y,meta)
    grid_pts = [(x,y,None) for (x,y) in grid]
    # combine and dedupe
    combined = []
    for n in nodes:
        combined.append((n['x'], n['y'], {'label': n.get('label')}))
    combined.extend(grid_pts)
    combined = dedupe_points(combined, min_dist=MIN_DISTANCE_BETWEEN_NODES)
    # reindex into nodes list (overwrite previous nodes to avoid duplicates)
    nodes = []
    idx = 1
    for x,y,meta in combined:
        lab = meta.get('label') if meta else None
        nodes.append({'id': f'node{idx}', 'x': float(x), 'y': float(y), 'label': lab})
        idx += 1
    return nodes
# --- Requires shapely (pip install shapely) ---
from shapely.geometry import LineString, box, Polygon

def svg_rects_to_polygons(rects):
    """
    rects: list of (x,y,w,h) from parse_svg()
    Returns list of shapely Polygons
    """
    polys = []
    for x,y,w,h in rects:
        try:
            p = box(x, y, x + w, y + h)
            polys.append(p)
        except Exception:
            continue
    return polys

def prune_edges_by_obstacles(nodes, edges, obstacle_polys, buffer_px=2.5):
    """
    nodes: list of node dicts with x,y
    edges: list of [idA, idB]
    obstacle_polys: list of shapely.Polygon
    buffer_px: expand obstacles slightly to avoid grazing edges
    returns: pruned_edges list
    """
    if not obstacle_polys:
        return edges

    # buffer each obstacle once (to simplify repeated operations)
    buffered = [p.buffer(buffer_px) for p in obstacle_polys]

    # quick lookup id->(x,y)
    node_by_id = { n['id']:(n['x'], n['y']) for n in nodes }

    pruned = []
    for a,b in edges:
        ax,ay = node_by_id[a]
        bx,by = node_by_id[b]
        seg = LineString([(ax,ay), (bx,by)])
        blocked = False
        # if midpoint inside any obstacle, definitely blocked (optional optimization)
        # mid = LineString([(ax,ay),(bx,by)]).interpolate(0.5, normalized=True)
        for obs in buffered:
            if seg.intersects(obs):
                blocked = True
                break
        if not blocked:
            pruned.append([a,b])
    return pruned


def build_edges(nodes, k=6):
    # simple KNN (Euclidean)
    coords = [(n['x'], n['y']) for n in nodes]
    N = len(nodes)
    edges = set()
    if HAVE_KDTREE:
        import numpy as np
        arr = np.array(coords)
        tree = KDTree(arr)
        dists, idxs = tree.query(arr, k=k+1)  # includes self
        for i in range(N):
            neighbors = idxs[i][1:]  # skip self
            for j in neighbors:
                a = nodes[i]['id']; b = nodes[j]['id']
                if a==b: continue
                if a < b: edges.add((a,b))
                else: edges.add((b,a))
    else:
        # naive O(N^2) for small N
        for i in range(N):
            dlist = []
            xi, yi = coords[i]
            for j in range(N):
                if i==j: continue
                xj,yj = coords[j]
                dlist.append((math.hypot(xi-xj, yi-yj), j))
            dlist.sort()
            for _, j in dlist[:k]:
                a = nodes[i]['id']; b = nodes[j]['id']
                if a==b: continue
                if a < b: edges.add((a,b))
                else: edges.add((b,a))
    return [list(e) for e in sorted(edges)]

def nearest_node(nodes, x,y):
    best=None; bd=1e9
    for n in nodes:
        d = math.hypot(n['x']-x, n['y']-y)
        if d < bd:
            bd=d; best=n
    return best

def write_outputs(nodes, edges, room_map):
    nodes_json = os.path.join(OUT_DIR,'nodes.json')
    edges_json = os.path.join(OUT_DIR,'edges.json')
    nodes_edges_js = os.path.join(OUT_DIR,'nodes_edges.js')
    room_map_file = os.path.join(OUT_DIR,'room_to_node.json')

    with open(nodes_json,'w') as f: json.dump(nodes, f, indent=2)
    with open(edges_json,'w') as f: json.dump(edges, f, indent=2)
    # write a simple JS file that defines nodes and edges variables
    with open(nodes_edges_js,'w') as f:
        f.write('// nodes_edges.js - generated\n')
        f.write('const nodes = ' + json.dumps(nodes, indent=2) + ';\n')
        f.write('const edges = ' + json.dumps(edges, indent=2) + ';\n')

    with open(room_map_file,'w') as f: json.dump(room_map, f, indent=2)
    print('Wrote:', nodes_json, edges_json, nodes_edges_js, room_map_file)

def main():
    svg_info = parse_svg(SVG_FILE)
    print('SVG size:', svg_info['width'], svg_info['height'], 'texts:', len(svg_info['texts']))
    nodes = build_nodes(svg_info)
    print('Generated nodes:', len(nodes))

    # build candidate edges using KNN
    edges = build_edges(nodes, k=K_NEIGHBORS)
    print('Initial edges:', len(edges))

    # build obstacle polygons from SVG rects (and optionally other shapes)
    obstacles = svg_rects_to_polygons(svg_info.get('rects', []))
    print('Detected obstacle rects -> polygons:', len(obstacles))

    # prune edges that cross obstacles (use buffer to be conservative)
    edges_pruned = prune_edges_by_obstacles(nodes, edges, obstacles, buffer_px=3.0)
    print('Edges after pruning:', len(edges_pruned))

    # continue using edges_pruned for output
    edges = edges_pruned

    print('Generated edges:', len(edges))
    # build room->node map: for every text label (normalized), map to nearest node
    room_map = {}
    for t in svg_info['texts']:
        label_norm = t['text'].strip()
        n = nearest_node(nodes, t['x'], t['y'])
        if n:
            room_map[label_norm] = n['id']
    write_outputs(nodes, edges, room_map)

if __name__ == '__main__':
    main()
