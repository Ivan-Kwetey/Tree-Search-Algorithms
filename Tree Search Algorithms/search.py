from collections import deque
from queue import PriorityQueue
import matplotlib.pyplot as plt
import networkx as nx

"""This program is an implementation of three search algorithms (BFS, DFS, and A* that uses euclidean as a heuristic) 
applied to find the shortest path between two cities in a graph. The graph represents connections between cities, 
and the edges have associated weights. The program uses the NetworkX library for graph representation and 
visualization. The program is well-organized, allowing users to interactively explore the different search algorithms 
and visualize their outcomes on the graph. whenever a user inputs an invalid city name, instead of the program crashing,
it prompts them to enter a valid city from the map.

this program reads the data from an external .txt file. these files contain the city coordinates and also the 
weighted edges for the cities paths. the user is asked to choose source and destination cities. 

the visualisation function was written in a way to position nodes at specific points to mimic the map provided for 
this project, hence not random, the graph displayed indicates cities with their weighted edge for easy reference. 
the red path on the graph represents the paths that were visited or explored by each algorithm. source city nodes are 
colored red and goal cities are colored green, all other nodes are blue. 

Enjoy and thank you.

@author [Ivan Kwetey - NUID 002879874]
"""

"""Function to visualize a graph. This function visualizes a given graph using the specified node positions
and highlights a path with red edges. Edge weights are displayed on the edges."""


def visualize_graph(graph, path_edges=None, path_sum_of_weights=None, algorithm=None, goal_city=None, source_city=None):
    # Create a dictionary to specify the positions of each node
    node_positions = {node: (float(coord.split(',')[0]), float(coord.split(',')[1])) for node, coord in
                      nx.get_node_attributes(graph, 'pos').items()}

    # Uses the specified node positions for layout
    pos = node_positions

    # Makes goal city green color
    node_colors = ['green' if node == goal_city else 'lightblue' for node in graph.nodes()]

    # Giving source city a red color
    if source_city:
        node_colors[list(graph.nodes()).index(source_city)] = 'red'

    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_color=node_colors, node_size=700, font_size=8)

    # Adds edge labels with weights for all edges
    edge_labels = {(edge[0], edge[1]): graph[edge[0]][edge[1]]['weight'] for edge in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='black', font_size=8)

    if path_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='red', width=2)

        # Adds edge labels with weights for the path edges
        path_edge_labels = {(edge[0], edge[1]): graph[edge[0]][edge[1]]['weight'] for edge in path_edges}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=path_edge_labels, font_color='red', font_size=8)

    if path_sum_of_weights is not None:
        title = f"Sum of Edge Weights ({algorithm}): {path_sum_of_weights}"
        plt.suptitle(title)

    plt.show()


"""BFS algorithm to find the shortest path between a source and a goal city.
Uses a deque (frontier) to explore nodes in breadth-first order.
Visited nodes are tracked in the reached list. The result is visualized using the visualize_graph function."""


def bfs(G, source, goal):
    edges = set()
    frontier = deque([source])
    reached = []
    visited = {source}

    # To store visited nodes
    while frontier:
        current = frontier.popleft()

        # iterates over the neighbors (child) of the current node (current) using G.adj[current]
        for child in G.adj[current]:
            if child not in visited:
                edges.add((current, child))
                visited.add(child)
                frontier.append(child)
                reached.append(child)  # Add the visited node to the list

                if child == goal:
                    path_sum_of_weights = sum(G[edge[0]][edge[1]]['weight'] for edge in edges)
                    visualize_graph(G, edges, path_sum_of_weights=path_sum_of_weights, algorithm="BFS", goal_city=goal,
                                    source_city=source)

                    print("Visited Nodes (BFS):")
                    print('\n'.join(reached))  # Print the visited nodes on separate lines
                    return edges

    # If the goal node is not found during the BFS traversal, it visualizes the graph with the explored edges and
    # returns the set of edges.
    visualize_graph(G, edges, algorithm="BFS", source_city=source)
    return edges


"""Depth-First Search algorithm to find the shortest path between a source and a goal city.
Uses a deque (current_edge) to explore nodes in depth-first order.
Visited nodes are tracked in the visited set. The result is visualized using the visualize_graph function"""


def dfs(G, source, goal):
    def expand(node, visited_nodes):
        # Generate a list of unvisited child nodes for a given node
        return [(node, child) for child in G.adj[node] if child not in visited_nodes]

    edges = set()
    goal_found = False
    visited = {source}
    current_edge = deque(expand(source, visited))

    while not goal_found and current_edge:
        parent, current = current_edge.popleft()  # Dequeue the leftmost node from the current edge
        edges.add((parent, current))
        visited.add(current)

        print(f"Exploring: {current}")

        if current == goal:
            # If the goal node is found, visualize the path and return the set of traversed edges
            path_sum_of_weights = sum(G[edge[0]][edge[1]]['weight'] for edge in edges)
            visualize_graph(G, edges, path_sum_of_weights=path_sum_of_weights, algorithm="DFS", goal_city=goal,
                            source_city=source)

            goal_found = True
        else:
            expansion = expand(current, visited)
            current_edge.extendleft(expansion)

    if goal_found:
        return edges
    else:
        # If the goal is not found, visualize the traversal and return an empty list
        visualize_graph(G, edges, algorithm="DFS", source_city=source)
        return []


"""A* search algorithm to find the shortest path between a source and a goal city.
Uses a priority queue (paths) based on the estimated total cost.
The heuristic function h calculates the Euclidean distance between two cities' coordinates.
The result is visualized using the visualize_graph function. Using the euclidean_distance to calculate
the Euclidean distance between two sets of coordinates."""


# calculates the Euclidean distance between two sets of coordinates (coord1 and coord2)
def euclidean_distance(coord1, coord2):
    x1, y1 = map(float, coord1.split(','))
    x2, y2 = map(float, coord2.split(','))
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def astar(G, source, goal):
    # Heuristic function to estimate the cost from a node to the goal using the Euclidean distance between their
    # coordinates.
    def h(node):
        return euclidean_distance(G.nodes[node]['pos'], G.nodes[goal]['pos'])

    best_path = []  # A list to store the best path found.
    best_cost = 0  # A variable to store the cost of the best path found.

    if not G or not G.adj[source]:
        return best_path, best_cost

    best_cost = float('inf')
    paths = PriorityQueue()
    paths.put((0, [source]))

    while not paths.empty():
        cost, path = paths.get()
        current = path[-1]

        if current == goal:
            path_sum_of_weights = nx.path_weight(G, path, weight='weight')
            visualize_graph(G, [(path[i], path[i + 1]) for i in range(len(path) - 1)],
                            path_sum_of_weights=path_sum_of_weights, algorithm="A*", goal_city=goal, source_city=source)

            return path, path_sum_of_weights

        for child in G.adj[current]:
            if child not in path:
                f = cost + G[current][child]['weight'] + h(child)
                new_path = path[:]
                new_path.append(child)
                paths.put((f, new_path))

    visualize_graph(G, best_path, path_sum_of_weights=None, algorithm="A*", source_city=source)
    return best_path, best_cost


# Function to read city weighted edges
def load_edges_from_file(file_path):
    edges = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            edges.append((parts[0], parts[1], {'weight': int(parts[2])}))
    return edges


# Function to read city coordinates
def load_cities_from_file(file_path):
    cities = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split(': ')
        city = parts[0]
        coordinates = parts[1]
        cities.append((city, {"pos": coordinates}))
    return cities


def main():
    """Loads city data and edges from files, creates a graph, and visualizes it.
        Prompts the user to input source and goal cities.
        Applies BFS, DFS, and A* algorithms to find the shortest paths and visualizes the results."""

    # Load city data from the file
    cities = load_cities_from_file("cities_data.txt")

    # Create a graph representing connections between cities
    G = nx.Graph()

    # Loads edges from the file
    edges = load_edges_from_file("city_edges.txt")

    # Adds nodes and edges to the graph
    G.add_nodes_from(cities)
    G.add_edges_from(edges)

    # Visualize the graph
    visualize_graph(G)

    # Prompts the user to enter source and goal cities
    while True:
        source_city = input("Enter the source city: ").title()
        if source_city in G.nodes:
            break
        else:
            print("Invalid city. Please enter a valid city, also use '_' to space a two word name city.")

        # Prompt the user to enter goal city
    while True:
        goal_city = input("Enter the goal city: ").title()
        if goal_city in G.nodes:
            break
        else:
            print("Invalid city. Please enter a valid city, also use '_' to space a two word name city.")

    # Converts the graph to a NetworkX graph
    G_nx = nx.Graph(G)

    # BFS
    print("\nBFS:")
    result_bfs = bfs(G_nx, source_city, goal_city)
    print("Shortest Path (BFS):", result_bfs)

    # DFS
    print("\nDFS:")
    result_dfs = dfs(G_nx, source_city, goal_city)
    print("Shortest Path (DFS):", result_dfs)

    # A*
    print("\nA*:")
    result_astar = astar(G_nx, source_city, goal_city)
    print("Shortest Path (A*):", result_astar)


if __name__ == "__main__":
    main()
