
"""
svg_to_graph.py
Generates nodes.json, edges.json and nodes_edges.js from an SVG floorplan.

Usage:
  python svg_to_graph.py idealab_floor_plan.svg

Outputs:
  - nodes.json
  - edges.json
  - nodes_edges.js
  - route_example.json (optional example)
"""

import sys, os, json, math
from lxml import etree
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree
    HAVE_KDTREE = True
except Exception:
    HAVE_KDTREE = False

try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}

def parse_svg(path):
    tree = etree.parse(path)
    root = tree.getroot()
    return root

def extract_text_positions(root):
    pts = []
    for t in root.findall('.//svg:text', namespaces=SVG_NS):
        x = t.get('x'); y = t.get('y')
        if x is None or y is None: continue
        try:
            x = float(str(x).split()[0]); y = float(str(y).split()[0])
            label = ''.join(t.itertext()).strip()
            pts.append(('text', x, y, label))
        except:
            continue
    return pts

def extract_rect_centroids(root):
    pts = []
    for r in root.findall('.//svg:rect', namespaces=SVG_NS):
        x,r_y,w,h = r.get('x'), r.get('y'), r.get('width'), r.get('height')
        if x and r_y and w and h:
            try:
                cx = float(x) + float(w)/2.0
                cy = float(r_y) + float(h)/2.0
                pts.append(('rect', cx, cy, None))
            except:
                pass
    return pts

def extract_path_centroids(root):
    pts = []
    for p in root.findall('.//svg:path', namespaces=SVG_NS):
        d = p.get('d')
        if not d: continue
        nums = []
        for token in d.replace(',', ' ').split():
            try: nums.append(float(token))
            except: pass
        if len(nums) >= 4:
            xs = nums[0::2]; ys = nums[1::2]
            cx = sum(xs)/len(xs); cy = sum(ys)/len(ys)
            pts.append(('path', cx, cy, None))
    return pts

def extract_wall_polygons(root):
    polys = []
    # rects with stroke or large shapes -> treat as walls
    for r in root.findall('.//svg:rect', namespaces=SVG_NS):
        cls = (r.get('class') or r.get('id') or '').lower()
        stroke = r.get('stroke')
        if stroke or 'wall' in cls:
            try:
                x=float(r.get('x')); y=float(r.get('y')); w=float(r.get('width')); h=float(r.get('height'))
                polys.append(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)]))
            except:
                pass
    # paths: gather coordinates heuristically and buffer to create wall areas
    for p in root.findall('.//svg:path', namespaces=SVG_NS):
        cls = (p.get('class') or p.get('id') or '').lower()
        stroke = p.get('stroke')
        if stroke or 'wall' in cls:
            d = p.get('d')
            nums = []
            for token in d.replace(',', ' ').split():
                try: nums.append(float(token))
                except: pass
            if len(nums) >= 4:
                xs = nums[0::2]; ys = nums[1::2]
                pts = list(zip(xs, ys))
                if len(pts) >= 2:
                    poly = LineString(pts).buffer(float(p.get('stroke-width') or 4.0))
                    polys.append(poly)
    if not polys:
        return None
    return unary_union(polys)

def build_candidate_nodes(root):
    pts = []
    pts += extract_text_positions(root)
    pts += extract_rect_centroids(root)
    pts += extract_path_centroids(root)
    seen = set()
    nodes = []
    idx = 1
    for tag,x,y,label in pts:
        key = (round(x,2), round(y,2))
        if key in seen: continue
        seen.add(key)
        nodes.append({'id': f'node{idx}', 'x': float(x), 'y': float(y), 'source': tag, 'label': label})
        idx += 1
    return nodes

def connect_knn(nodes, k=4):
    coords = np.array([[n['x'], n['y']] for n in nodes])
    N = coords.shape[0]
    edges = set()
    if N <= 1:
        return []
    if HAVE_KDTREE:
        tree = KDTree(coords)
        dists, idxs = tree.query(coords, k=min(k+1, N))
        for i in range(N):
            neighbors = idxs[i][1:]
            for j in neighbors:
                a = nodes[i]['id']; b = nodes[int(j)]['id']
                if a!=b:
                    edges.add(tuple(sorted((a,b))))
    else:
        # fallback O(N^2)
        for i in range(N):
            dlist = []
            for j in range(N):
                if i==j: continue
                d = math.hypot(coords[i,0]-coords[j,0], coords[i,1]-coords[j,1])
                dlist.append((d,j))
            dlist.sort()
            for (d,j) in dlist[:k]:
                a = nodes[i]['id']; b = nodes[j]['id']
                edges.add(tuple(sorted((a,b))))
    return [list(e) for e in edges]

def prune_edges_by_walls(nodes, edges, wall_geom):
    if wall_geom is None:
        return edges
    node_by_id = {n['id']: n for n in nodes}
    good = []
    for a,b in edges:
        p1 = node_by_id[a]; p2 = node_by_id[b]
        seg = LineString([(p1['x'], p1['y']), (p2['x'], p2['y'])])
        if wall_geom.intersects(seg):
            continue
        good.append([a,b])
    return good

def write_outputs(nodes, edges):
    with open('nodes.json','w') as f: json.dump(nodes, f, indent=2)
    with open('edges.json','w') as f: json.dump(edges, f, indent=2)
    nodes_js = json.dumps([{'id':n['id'],'x':n['x'],'y':n['y']} for n in nodes], indent=2)
    edges_js = json.dumps(edges, indent=2)
    with open('nodes_edges.js','w') as f:
        f.write('// Generated by svg_to_graph.py\n')
        f.write('const nodes = ' + nodes_js + ';\n\n')
        f.write('const edges = ' + edges_js + ';\n')
    print('Wrote nodes.json, edges.json, nodes_edges.js')

def nearest_node(nodes, x,y):
    best=None; bd=1e12
    for n in nodes:
        d = math.hypot(n['x']-x, n['y']-y)
        if d < bd:
            bd = d; best = n
    return best, bd

def build_nx_and_path(nodes, edges, start_id, end_id):
    if not HAVE_NX:
        return None
    G = nx.Graph()
    for n in nodes: G.add_node(n['id'], x=n['x'], y=n['y'])
    for a,b in edges:
        na = next(n for n in nodes if n['id']==a)
        nb = next(n for n in nodes if n['id']==b)
        w = math.hypot(na['x']-nb['x'], na['y']-nb['y'])
        G.add_edge(a,b, weight=w)
    try:
        path = nx.shortest_path(G, start_id, end_id, weight='weight')
        return path
    except Exception:
        return None

def main(svg_file):
    if not os.path.exists(svg_file):
        print('SVG file not found:', svg_file); return
    root = parse_svg(svg_file)
    nodes = build_candidate_nodes(root)
    print('Found candidate nodes:', len(nodes))
    wall_geom = extract_wall_polygons(root)
    if wall_geom is None:
        print('No wall geometry detected (edges will not be pruned).')
    else:
        print('Wall geometry detected and merged.')
    edges = connect_knn(nodes, k=4)
    print('Initial edges (KNN):', len(edges))
    edges_pruned = prune_edges_by_walls(nodes, edges, wall_geom)
    print('Edges after pruning:', len(edges_pruned))
    write_outputs(nodes, edges_pruned)

    # write a trivial example route file using two heuristic points (center-ish)
    w = float(root.get('width') or 900)
    h = float(root.get('height') or 1200)
    start_px = (w*0.45, h*0.6)
    end_px   = (w*0.72, h*0.6)
    s_node, sd = nearest_node(nodes, start_px[0], start_px[1])
    e_node, ed = nearest_node(nodes, end_px[0], end_px[1])
    if s_node and e_node:
        path = build_nx_and_path(nodes, edges_pruned, s_node['id'], e_node['id'])
        if path:
            with open('route_example.json','w') as f:
                json.dump({'start_px': start_px, 'end_px': end_px, 'start_node': s_node, 'end_node': e_node, 'path': path}, f, indent=2)
            print('Wrote route_example.json (example path length:', len(path), ')')
        else:
            print('Could not compute example route with networkx (graph may be disconnected).')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python svg_to_graph.py path/to/your.svg')
        sys.exit(1)
    main(sys.argv[1])
