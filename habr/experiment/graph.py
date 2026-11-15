"""Скачать дорожный граф Москвы в пределах МКАД и сохранить в компактных массивах."""

from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)
CENTER = (55.7539, 37.6208)  # Красная площадь

CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "other"]
FREE_SPEED = {"motorway": 90, "trunk": 75, "primary": 60, "secondary": 50, "tertiary": 40, "residential": 30, "other": 25}


def road_class(highway) -> str:
    if isinstance(highway, list):
        highway = highway[0]
    base = str(highway).replace("_link", "")
    return base if base in CLASSES else "other"


def mkad_polygon() -> Polygon:
    gdf = ox.features_from_place("Москва", tags={"ref": "МКАД"})
    lines = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])]
    hull = lines.union_all().convex_hull
    return hull


def main() -> None:
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(DATA / "osmnx_cache")
    try:
        polygon = mkad_polygon()
        print("МКАД polygon area deg2:", polygon.area)
    except Exception as error:  # запасной вариант: круг радиусом 17 км
        print("fallback to circle:", error)
        polygon = None
    if polygon is not None:
        graph = ox.graph_from_polygon(polygon, network_type="drive", simplify=True)
    else:
        graph = ox.graph_from_point(CENTER, dist=17000, network_type="drive", simplify=True)
    graph = ox.truncate.largest_component(graph, strongly=True)
    graph = nx.convert_node_labels_to_integers(graph)
    lat = np.array([graph.nodes[n]["y"] for n in range(graph.number_of_nodes())])
    lon = np.array([graph.nodes[n]["x"] for n in range(graph.number_of_nodes())])
    best: dict[tuple[int, int], tuple[float, int]] = {}
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        cls = CLASSES.index(road_class(data.get("highway", "other")))
        length = float(data["length"])
        speed = FREE_SPEED[CLASSES[cls]]
        time = length / (speed / 3.6)
        key = (u, v)
        if key not in best or time < best[key][0] / 1:
            best[key] = (length, cls)
    us = np.array([k[0] for k in best], dtype=np.int32)
    vs = np.array([k[1] for k in best], dtype=np.int32)
    lengths = np.array([best[k][0] for k in best])
    classes = np.array([best[k][1] for k in best], dtype=np.int8)
    np.savez(DATA / "graph.npz", lat=lat, lon=lon, u=us, v=vs, length=lengths, cls=classes)
    print("nodes", len(lat), "edges", len(us))
    print("classes", {c: int((classes == i).sum()) for i, c in enumerate(CLASSES)})


if __name__ == "__main__":
    main()
