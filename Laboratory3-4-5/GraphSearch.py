import time
import matplotlib.pyplot as plt
from collections import deque
import sys
from GenerateGraphs import *

sys.setrecursionlimit(16000)


def bfs(adj):
    n = len(adj)
    visited = [False] * n
    result = []

    for start in range(n):
        if not visited[start]:
            queue = deque([start])
            visited[start] = True

            while queue:
                vertex = queue.popleft()
                result.append(vertex)

                for neighbor in adj[vertex]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

    return result


def dfs(adj):
    n = len(adj)
    visited = [False] * n
    result = []

    def dfs_visit(u):
        visited[u] = True
        result.append(u)
        for neighbor in adj[u]:
            if not visited[neighbor]:
                dfs_visit(neighbor)

    for i in range(n):
        if not visited[i]:
            dfs_visit(i)

    return result


def measure_time(algorithm, graph):
    start_time = time.time()
    algorithm(graph)
    end_time = time.time()
    return (end_time - start_time) * 1000


sizes = [i for i in range(1, 1000, 50)]

graph_types = {
    "Complete Graph": generate_complete_graph,
    "Dense Graph": generate_dense_graph,
    "Sparse Graph": generate_sparse_graph,
    "Tree Graph": generate_tree_graph,
    "Cyclic Graph": generate_cyclic_graph,
    "Acyclic Graph": generate_acyclic_graph,
    "Connected Graph": generate_connected_graph,
    "Disconnected Graph": generate_disconnected_graph,
    "Undirected Graph": generate_undirected_graph,
    "Directed Graph": generate_directed_graph,
    "Grid Graph": generate_grid_graph
}

results_bfs = {graph_type: [] for graph_type in graph_types}
results_dfs = {graph_type: [] for graph_type in graph_types}

for size in sizes:
    print(f"Testing graphs with {size} nodes...")
    for graph_type, generator in graph_types.items():
        graph = generator(size)

        bfs_time = measure_time(bfs, graph)
        results_bfs[graph_type].append(bfs_time)

        dfs_time = measure_time(dfs, graph)
        results_dfs[graph_type].append(dfs_time)

        print(f"  {graph_type}: BFS={bfs_time:.2f}ms, DFS={dfs_time:.2f}ms")

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
for graph_type, times in results_bfs.items():
    plt.plot(sizes, times, marker='o', label=graph_type)

plt.title('BFS Performance Across Graph Types')
plt.xlabel('Number of Nodes')
plt.ylabel('Time (ms)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
for graph_type, times in results_dfs.items():
    plt.plot(sizes, times, marker='o', label=graph_type)

plt.title('DFS Performance Across Graph Types')
plt.xlabel('Number of Nodes')
plt.ylabel('Time (ms)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

for graph_type in graph_types:
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, results_bfs[graph_type], marker='o', label='BFS', color='pink')
    plt.plot(sizes, results_dfs[graph_type], marker='x', label='DFS', color='mediumslateblue')

    plt.title(f'BFS vs DFS Performance on {graph_type}')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()