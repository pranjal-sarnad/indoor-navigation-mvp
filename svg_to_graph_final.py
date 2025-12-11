#!/usr/bin/env python3
"""
svg_to_graph_final.py
Generates nodes.json, edges.json, nodes_edges.js and room_to_node.json
from your SVG, with obstacle pruning (rectangles) and KNN connectivity.

Put this file in the same directory as idealab_floor_plan.svg and run:
python svg_to_graph_final.py
"""

import os, json, math
from xml.etree import ElementTree as ET

SVG_FILE = 'idealab_floor_plan.svg'
OUT_DIR = '.'

# tunables: lower GRID_SPACING -> more nodes (slower)
GRID_SPACING = 16.0   # start 16.0; reduce to 8.0 for more accuracy
K_NEIGHBORS = 6
MIN_NODE_DIST = 6.0   # dedupe radius
OBSTACLE_BUFFER = 3.0 # expand obstacles slightly

# optional acceleration: try to import numpy/scipy KDTree
try:
    import numpy as np
    from scipy.spatial import cKDTree as KDTree
    HAVE_KDTREE = True
except Exception:
    HAVE_KDTREE = False

# shapely for accurate geometry checks
try:
    from shapely.geometry import Point, LineString, box
    from shapely.ops import unary_union
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

def parse_svg(svg_file):
    """
    Improved SVG parser:
    - reads viewBox / width/height
    - extracts text positions (labels)
    - extracts obstacle bboxes from rect, circle, ellipse, polygon, polyline, path (approx)
      but only includes shapes that are filled or have an appreciable stroke width
    - filters out the outer frame and very large boxes that cover most of the SVG
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # namespace handling
    tag = root.tag
    if tag.startswith('{'):
        ns = tag.split('}')[0].strip('{')
    else:
        ns = 'http://www.w3.org/2000/svg'

    # read viewBox or width/height
    vb = root.get('viewBox')
    if vb:
        parts = list(map(float, vb.replace(',', ' ').split()))
        if len(parts) >= 4:
            minx, miny, w, h = parts[0], parts[1], parts[2], parts[3]
        else:
            minx, miny = 0.0, 0.0
            w = float(root.get('width') or 900)
            h = float(root.get('height') or 1200)
    else:
        minx, miny = 0.0, 0.0
        w = float(root.get('width') or 900)
        h = float(root.get('height') or 1200)

    # collect text elements (labels)
    texts = []
    for t in root.findall(f'.//{{{ns}}}text'):
        txt = ''.join(t.itertext()).strip()
        if not txt:
            continue
        x_attr = t.get('x') or t.get('dx')
        y_attr = t.get('y') or t.get('dy')
        if not x_attr or not y_attr:
            continue
        try:
            x = float(str(x_attr).split()[0])
            y = float(str(y_attr).split()[0])
        except:
            continue
        texts.append({'text': txt, 'x': x, 'y': y})

    # helpers to extract numeric lists and pair as points
    import re
    num_re = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
    def parse_points_attr(attr):
        nums = [float(x) for x in num_re.findall(attr)]
        if len(nums) % 2 == 1:
            nums = nums[:-1]
        return [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)] if nums else []

    def bbox_from_points(points):
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

    # collect raw bboxes (x1,y1,x2,y2)
    raw_boxes = []

    # 1) rects
    for r in root.findall(f'.//{{{ns}}}rect'):
        try:
            rx = float(r.get('x') or 0)
            ry = float(r.get('y') or 0)
            rw = float(r.get('width') or 0)
            rh = float(r.get('height') or 0)
            if rw > 0 and rh > 0:
                raw_boxes.append((rx, ry, rx + rw, ry + rh))
        except:
            continue

    # 2) circle / ellipse
    for c in root.findall(f'.//{{{ns}}}circle'):
        try:
            cx = float(c.get('cx') or 0); cy = float(c.get('cy') or 0); r0 = float(c.get('r') or 0)
            if r0 > 0:
                raw_boxes.append((cx - r0, cy - r0, cx + r0, cy + r0))
        except:
            continue
    for e in root.findall(f'.//{{{ns}}}ellipse'):
        try:
            cx = float(e.get('cx') or 0); cy = float(e.get('cy') or 0)
            rx = float(e.get('rx') or 0); ry = float(e.get('ry') or 0)
            if rx > 0 and ry > 0:
                raw_boxes.append((cx - rx, cy - ry, cx + rx, cy + ry))
        except:
            continue

    # 3) polygon / polyline -> include only if filled or stroke is wide enough
    for tagname in ('polygon', 'polyline'):
        for el in root.findall(f'.//{{{ns}}}{tagname}'):
            pts_attr = el.get('points') or ''
            if not pts_attr:
                continue
            # check visual attributes
            fill = (el.get('fill') or '').strip().lower()
            stroke_w = 0.0
            try:
                stroke_w = float(el.get('stroke-width') or 0)
            except:
                stroke_w = 0.0
            # include if filled or noticeable stroke
            if fill in ('', 'none', 'transparent') and stroke_w <= 0.5:
                # skip purely decorative outlines
                continue
            pts = parse_points_attr(pts_attr)
            if pts:
                xmin, ymin, xmax, ymax = bbox_from_points(pts)
                raw_boxes.append((xmin, ymin, xmax, ymax))

    # 4) path - approximate: extract numbers and pair them as coords to compute bbox
    #    only include if path has visible fill or stroke width
    for p in root.findall(f'.//{{{ns}}}path'):
        d = p.get('d') or ''
        if not d:
            continue
        fill = (p.get('fill') or '').strip().lower()
        stroke_w = 0.0
        try:
            stroke_w = float(p.get('stroke-width') or 0)
        except:
            stroke_w = 0.0
        # include path only if visibly filled or stroked
        if fill in ('', 'none', 'transparent') and stroke_w <= 0.5:
            continue
        nums = [float(x) for x in num_re.findall(d)]
        if len(nums) >= 2:
            pts = [(nums[i], nums[i+1]) for i in range(0, len(nums)-1, 2)]
            try:
                xmin, ymin, xmax, ymax = bbox_from_points(pts)
                raw_boxes.append((xmin, ymin, xmax, ymax))
            except:
                continue

    # convert to rect tuples (x,y,w,h)
    rects = []
    for (x1,y1,x2,y2) in raw_boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        rx = float(x1); ry = float(y1); rw = float(x2 - x1); rh = float(y2 - y1)
        rects.append((rx, ry, rw, rh))

    # FILTERING: remove outer-frame and overly-large boxes (tighter thresholds)
    svg_area = max(1.0, float(w) * float(h))
    # detect and remove any rect that touches or nearly touches all four edges (likely frame)
    filtered = []
    frame_margin = max(2.0, min(w,h) * 0.01)  # 1% margin or 2px
    largest_area = 0.0
    largest_idx = None
    for i,(rx,ry,rw,rh) in enumerate(rects):
        area = rw * rh
        if area > largest_area:
            largest_area = area; largest_idx = i

    for i,(rx,ry,rw,rh) in enumerate(rects):
        area = rw * rh
        # if this rect touches all four edges (frame-like), skip it
        if rx <= frame_margin and ry <= frame_margin and (rx+rw) >= (w - frame_margin) and (ry+rh) >= (h - frame_margin):
            # skip frame
            continue
        # skip boxes that cover too much area (tunable)
        if (area / svg_area) > 0.30:
            # if it's the single largest and not frame, we still skip it to avoid giant obstacle
            continue
        # skip tiny noise
        if area < 9.0:
            continue
        filtered.append((rx,ry,rw,rh))

    # if we filtered everything out but there are rects, fall back to keeping non-frame smaller ones
    if not filtered and rects:
        for (rx,ry,rw,rh) in rects:
            area = rw * rh
            if (area / svg_area) < 0.80 and area > 9.0:
                filtered.append((rx,ry,rw,rh))

    # final result: width, height, texts, rects
    return {'width': float(w), 'height': float(h), 'texts': texts, 'rects': filtered}


def sample_grid(w, h, spacing, margin=4):
    pts = []
    y = margin
    while y < h - margin + 1e-9:
        x = margin
        while x < w - margin + 1e-9:
            pts.append((x, y))
            x += spacing
        y += spacing
    return pts

def dedupe(points, min_dist=MIN_NODE_DIST):
    kept = []
    for x,y,meta in points:
        skip = False
        for xx,yy,_ in kept:
            if math.hypot(xx-x, yy-y) < min_dist:
                skip = True
                break
        if not skip:
            kept.append((x,y,meta))
    return kept

def rects_to_polygons(rects):
    if not HAVE_SHAPELY: return []
    polys = []
    for rx,ry,rw,rh in rects:
        polys.append(box(rx, ry, rx+rw, ry+rh))
    return polys

def remove_points_in_obstacles(points, obstacle_union, buffer_px=OBSTACLE_BUFFER):
    if not obstacle_union: return points
    filtered = []
    for x,y,meta in points:
        p = Point(x,y)
        if not p.within(obstacle_union.buffer(buffer_px)):
            filtered.append((x,y,meta))
    return filtered

def build_nodes(svg_info):
    pts = []
    # add labeled text positions first
    for t in svg_info['texts']:
        pts.append((float(t['x']), float(t['y']), {'label': t['text']}))
    # add grid samples
    grid = sample_grid(svg_info['width'], svg_info['height'], GRID_SPACING)
    for x,y in grid:
        pts.append((x,y,None))
    # dedupe
    pts = dedupe(pts, MIN_NODE_DIST)
    # remove those inside obstacles (handled later if shapely)
    return pts

def build_edges_from_nodes(nodes_list, k=K_NEIGHBORS):
    edges = set()
    N = len(nodes_list)
    coords = [(n['x'], n['y']) for n in nodes_list]
    if HAVE_KDTREE and N > 1:
        arr = np.array(coords)
        tree = KDTree(arr)
        kq = min(k+1, N)
        dists, idxs = tree.query(arr, k=kq)
        for i in range(N):
            for j in idxs[i][1:]:
                a = nodes_list[i]['id']; b = nodes_list[j]['id']
                if a == b: continue
                edges.add(tuple(sorted((a,b))))
    else:
        # naive
        for i in range(N):
            xi, yi = coords[i]
            dlist = []
            for j in range(N):
                if i==j: continue
                xj, yj = coords[j]
                dlist.append((math.hypot(xi-xj, yi-yj), j))
            dlist.sort()
            for _, j in dlist[:k]:
                a = nodes_list[i]['id']; b = nodes_list[j]['id']
                edges.add(tuple(sorted((a,b))))
    return [list(e) for e in sorted(edges)]

def prune_edges_by_obstacles(nodes_list, edges, obstacle_union, buffer_px=OBSTACLE_BUFFER):
    if not obstacle_union or not HAVE_SHAPELY:
        return edges
    buffered = obstacle_union.buffer(buffer_px)
    id_to_node = {n['id']:(n['x'], n['y']) for n in nodes_list}
    pruned = []
    for a,b in edges:
        ax,ay = id_to_node[a]; bx,by = id_to_node[b]
        seg = LineString([(ax,ay),(bx,by)])
        if seg.intersects(buffered):
            continue
        pruned.append([a,b])
    return pruned

def nearest_node(nodes_list, x,y):
    best=None; bd=1e9
    for n in nodes_list:
        d = math.hypot(n['x']-x, n['y']-y)
        if d < bd:
            bd = d; best = n
    return best

def main():
    print("Parsing SVG:", SVG_FILE)
    svg_info = parse_svg(SVG_FILE)
    print("SVG size:", svg_info['width'], svg_info['height'], "texts:", len(svg_info['texts']), "rects:", len(svg_info['rects']))

    pts = build_nodes(svg_info)
    print("Sampled candidate points:", len(pts))

    # build obstacle union if shapely available
    obstacle_polys = rects_to_polygons(svg_info['rects'])
    if obstacle_polys and HAVE_SHAPELY:
        obstacle_union = unary_union(obstacle_polys)
        print("Built obstacle union")
        pts = remove_points_in_obstacles(pts, obstacle_union, buffer_px=OBSTACLE_BUFFER)
        print("Points after removing inside obstacles:", len(pts))
    else:
        obstacle_union = None
        print("No obstacle pruning (shapely missing or no rects)")

    # build final node objects
    nodes = []
    idx = 1
    for x,y,meta in pts:
        lab = meta.get('label') if meta else None
        nodes.append({'id': f'node{idx}', 'x': float(x), 'y': float(y), 'label': lab})
        idx += 1
    print("Final nodes:", len(nodes))

    # build edges
    edges = build_edges_from_nodes(nodes, k=K_NEIGHBORS)
    print("Initial edges:", len(edges))

    # prune edges crossing obstacles
    edges_pruned = prune_edges_by_obstacles(nodes, edges, obstacle_union, buffer_px=OBSTACLE_BUFFER)
    print("Edges after pruning:", len(edges_pruned))

    # build room->node mapping from text labels (if any)
    room_map = {}
    for t in svg_info['texts']:
        p = nearest_node(nodes, float(t['x']), float(t['y']))
        if p:
            room_map[t['text'].strip()] = p['id']

    # write outputs
    with open(os.path.join(OUT_DIR,'nodes.json'),'w') as f: json.dump(nodes, f, indent=2)
    with open(os.path.join(OUT_DIR,'edges.json'),'w') as f: json.dump(edges_pruned, f, indent=2)
    with open(os.path.join(OUT_DIR,'nodes_edges.js'),'w') as f:
        f.write('// nodes_edges.js - generated\n')
        f.write('const nodes = ' + json.dumps(nodes, indent=2) + ';\n')
        f.write('const edges = ' + json.dumps(edges_pruned, indent=2) + ';\n')
    with open(os.path.join(OUT_DIR,'room_to_node.json'),'w') as f: json.dump(room_map, f, indent=2)

    print("Wrote nodes.json, edges.json, nodes_edges.js, room_to_node.json")

if __name__ == '__main__':
    main()
