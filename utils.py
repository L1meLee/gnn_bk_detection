def add_edge(graph, start: int, end: int):
    if start == end:
        return
    try:
        graph.edge_ids(start, end)
    except Exception:
        graph.add_edges([start], [end])

    try:
        graph.edge_ids(end, start)
    except Exception:
        graph.add_edges([end], [start])