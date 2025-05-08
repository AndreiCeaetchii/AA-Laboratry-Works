import time
import matplotlib.pyplot as plt
from GenerateGraphs import *


def dijkstra(adj, start):
    n = len(adj)
    dist = [float('inf')] * n
    dist[start] = 0
    visited = [False] * n
    nodes_to_process = [start]

    while nodes_to_process:
        u = None
        min_dist = float('inf')
        for node in nodes_to_process:
            if dist[node] < min_dist and not visited[node]:
                min_dist = dist[node]
                u = node

        if u is None:
            break

        visited[u] = True
        nodes_to_process.remove(u)

        for v, weight in adj[u].items():
            if not visited[v]:
                if dist[v] > dist[u] + weight:
                    dist[v] = dist[u] + weight
                    if v not in nodes_to_process:
                        nodes_to_process.append(v)
    return dist


def floyd_warshall(adj):
    n = len(adj)
    dist = [[float('inf')] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
        for j, weight in adj[i].items():
            dist[i][j] = weight

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


def add_weights(adj):
    weighted_adj = [{} for _ in range(len(adj))]
    for u in range(len(adj)):
        for v in adj[u]:
            # assign random weights between 1 and 10
            weighted_adj[u][v] = random.randint(1, 10)
    return weighted_adj


def test_dijkstra(adj):
    start_time = time.time()
    for i in range(len(adj)):
        dijkstra(adj, i)
    return time.time() - start_time


def test_floyd_warshall(adj):
    start_time = time.time()
    floyd_warshall(adj)
    return time.time() - start_time


def generate_all_weighted_graphs(n):
    graphs = {
        "Dense": add_weights(generate_dense_graph(n)),
        "Sparse": add_weights(generate_sparse_graph(n)),
    }
    return graphs


def plot_overall_results(sizes, results):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    for graph_type in results['Dijkstra']:
        if len(results['Dijkstra'][graph_type]) == len(sizes):
            plt.plot(sizes, results['Dijkstra'][graph_type], label=graph_type, marker='o')
    plt.title('Dijkstra Algorithm Performance')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    for graph_type in results['Floyd-Warshall']:
        if len(results['Floyd-Warshall'][graph_type]) == len(sizes):
            plt.plot(sizes, results['Floyd-Warshall'][graph_type], label=graph_type, marker='o')
    plt.title('Floyd-Warshall Algorithm Performance')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_individual_graph_results(sizes, results):
    graph_types = list(results['Dijkstra'].keys())

    for graph_type in graph_types:
        plt.figure(figsize=(10, 6))
        has_data = False

        if len(results['Dijkstra'].get(graph_type, [])) == len(sizes):
            plt.plot(sizes, results['Dijkstra'][graph_type], '-', marker='o', label='Dijkstra', color='violet')
            has_data = True

        if len(results['Floyd-Warshall'].get(graph_type, [])) == len(sizes):
            plt.plot(sizes, results['Floyd-Warshall'][graph_type], '-', marker='x', label='Floyd-Warshall',
                     color='slateblue')
            has_data = True

        if has_data:
            plt.title(f'Performance on {graph_type} Graph')
            plt.xlabel('Number of Nodes')
            plt.ylabel('Time (seconds)')
            plt.legend()
            plt.grid(True)
            plt.show()


def run_tests(max_nodes=500, step=10):
    sizes = list(range(1, max_nodes + 1, step))
    graph_types = [
        "Dense", "Sparse"
    ]

    results = {
        'Dijkstra': {gt: [] for gt in graph_types},
        'Floyd-Warshall': {gt: [] for gt in graph_types}
    }

    for size in sizes:
        print(f"Testing size: {size}")
        graphs = generate_all_weighted_graphs(size)

        for graph_type in graph_types:
            adj = graphs[graph_type]

            try:
                time_dijkstra = test_dijkstra(adj)
                print(f"Dijkstra  {graph_type}: {time_dijkstra:.2f}ms")
                results['Dijkstra'][graph_type].append(time_dijkstra)
            except Exception as e:
                print(f"Error with Dijkstra on {graph_type} graph (size {size}): {e}")
                results['Dijkstra'][graph_type].append(float('nan'))

            try:
                time_floyd = test_floyd_warshall(adj)
                print(f"Floyd-Warshall  {graph_type}: {time_floyd:.2f}ms")
                results['Floyd-Warshall'][graph_type].append(time_floyd)
            except Exception as e:
                print(f"Error with Floyd-Warshall on {graph_type} graph (size {size}): {e}")
                results['Floyd-Warshall'][graph_type].append(float('nan'))

    return sizes, results


# main execution
if __name__ == "__main__":
    sizes, results = run_tests(max_nodes=300, step=10)
    plot_overall_results(sizes, results)
    plot_individual_graph_results(sizes, results)