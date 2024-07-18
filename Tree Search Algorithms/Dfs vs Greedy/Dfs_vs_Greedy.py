import networkx as nx
import matplotlib.pyplot as plt
from heapq import heappush, heappop

def dfs_shortest_path(graph, start, goal):
    visited = set()
    stack = [(start, [start])]

    edges_expansion_order = []  # Store the edges in expansion order

    while stack:
        current, path = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            # Build the list of edges in expansion order
            edges_expansion_order.extend([(path[i], path[i + 1]) for i in range(len(path) - 1)])
            return path, edges_expansion_order

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None, edges_expansion_order

def greedy_shortest_path(graph, start, goal):
    visited = set()
    heap = [(0, start, [start])]

    while heap:
        cost, current, path = heappop(heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                neighbor_cost = graph[current][neighbor]['weight']
                heappush(heap, (neighbor_cost, neighbor, path + [neighbor]))

    return None

def main():
    # Create a weighted graph with Seattle city names and custom coordinates
    G = nx.Graph()
    cities = {
        'Seattle': (0, 0),
        'Bellevue': (3, 0),
        'Redmond': (5, 3),
        'Kirkland': (4, 5),
        'Renton': (1, 5),
        'Issaquah': (-1, 3),
        'Bothell': (6, 0),
        'Edmonds': (8, 2),
        'Kent': (2, 7),
        'Lynnwood': (7, 5),
    }

    # Add weighted edges with manipulated weights
    G.add_edge('Seattle', 'Bellevue', weight=4)
    G.add_edge('Seattle', 'Issaquah', weight=5)
    G.add_edge('Bellevue', 'Redmond', weight=4)
    G.add_edge('Redmond', 'Bothell', weight=3)
    G.add_edge('Kirkland', 'Renton', weight=7)
    G.add_edge('Kirkland', 'Kent', weight=7)
    G.add_edge('Renton', 'Issaquah', weight=6)
    G.add_edge('Issaquah', 'Bothell', weight=5)
    G.add_edge('Bothell', 'Edmonds', weight=5)
    G.add_edge('Edmonds', 'Kirkland', weight=6)
    G.add_edge('Kent', 'Lynnwood', weight=7)

    # Visualize the original graph with custom coordinates
    plt.figure(figsize=(8, 6))
    pos = cities
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Original Graph")
    plt.show()

    # Find and print paths
    source = 'Seattle'
    goal = 'Lynnwood'

    dfs_path, dfs_expansion_order = dfs_shortest_path(G, source, goal)
    greedy_path = greedy_shortest_path(G, source, goal)

    print("DFS Shortest Path:", dfs_path)
    print("DFS Expansion Order:", dfs_expansion_order)
    print("Greedy Shortest Path:", greedy_path)

    # Visualize the DFS shortest path
    plt.figure(figsize=(8, 6))
    edges_dfs = [(dfs_path[i], dfs_path[i + 1]) for i in range(len(dfs_path) - 1)]
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=10)
    nx.draw_networkx_edges(G, pos, edgelist=edges_dfs, edge_color='red', width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    total_weight_dfs = sum(G[dfs_path[i]][dfs_path[i + 1]]['weight'] for i in range(len(dfs_path) - 1))
    plt.suptitle(f"DFS Shortest Path (Total Weight: {total_weight_dfs})", y=0.95, fontsize=14)
    plt.show()

    # Visualize the Greedy shortest path
    plt.figure(figsize=(8, 6))
    edges_greedy = [(greedy_path[i], greedy_path[i + 1]) for i in range(len(greedy_path) - 1)]
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=10)
    nx.draw_networkx_edges(G, pos, edgelist=edges_greedy, edge_color='red', width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    total_weight_greedy = sum(G[greedy_path[i]][greedy_path[i + 1]]['weight'] for i in range(len(greedy_path) - 1))
    plt.suptitle(f"Greedy Shortest Path (Total Weight: {total_weight_greedy})", y=0.95, fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()
