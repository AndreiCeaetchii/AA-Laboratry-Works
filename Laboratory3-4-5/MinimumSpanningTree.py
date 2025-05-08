import time
import matplotlib.pyplot as plt
from GenerateGraphs import *


def convert_to_weighted(adj):
    weighted = [{} for _ in range(len(adj))]
    for u in range(len(adj)):
        for v in adj[u]:
            if v not in weighted[u]:
                weight = random.randint(1, 10)
                weighted[u][v] = weight
                weighted[v][u] = weight
    return weighted


def kruskal(adj):
    n = len(adj)
    parent = list(range(n))

    def find(u):
        while u != parent[u]:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
            return True
        return False

    edges = []
    for u in range(n):
        for v, w in adj[u].items():
            if u < v:
                edges.append((w, u, v))

    edges.sort()
    mst_weight = 0
    for w, u, v in edges:
        if union(u, v):
            mst_weight += w
    return mst_weight


def prim(adj):
    n = len(adj)
    selected = [False] * n
    selected[0] = True
    mst_weight = 0

    for _ in range(n - 1):
        min_edge = float('inf')
        u, v = -1, -1
        for i in range(n):
            if selected[i]:
                for j, w in adj[i].items():
                    if not selected[j] and w < min_edge:
                        min_edge = w
                        u, v = i, j
        if v != -1:
            selected[v] = True
            mst_weight += min_edge
    return mst_weight


def test_kruskal(adj):
    start = time.time()
    kruskal(adj)
    return time.time() - start


def test_prim(adj):
    start = time.time()
    prim(adj)
    return time.time() - start


# generate all graph types with weights
def generate_all_weighted_graphs(n):
    return {
        "Dense": convert_to_weighted(generate_dense_graph(n)),
        "Sparse": convert_to_weighted(generate_sparse_graph(n))
    }


# run performance tests
def run_tests(max_nodes=300, step=10):
    sizes = list(range(10, max_nodes + 1, step))
    graph_types = [
        "Dense", "Sparse"
    ]
    results = {
        'Kruskal': {gt: [] for gt in graph_types},
        'Prim': {gt: [] for gt in graph_types}
    }

    for n in sizes:
        print(f"Testing size: {n}")
        graphs = generate_all_weighted_graphs(n)

        for gtype in graph_types:
            adj = graphs[gtype]
            try:
                t1 = test_kruskal(adj)
                print(f"Kruskal  {gtype}: {t1:.10f}ms")
                results['Kruskal'][gtype].append(t1)
            except:
                results['Kruskal'][gtype].append(float('nan'))
            try:
                t2 = test_prim(adj)
                print(f"Prim  {gtype}: {t2:.10f}ms")
                results['Prim'][gtype].append(t2)
            except:
                results['Prim'][gtype].append(float('nan'))

    return sizes, results


# plotting functions
def plot_overall_mst_results(sizes, results):
    plt.figure(figsize=(12, 8))

    # Kruskal overall
    plt.subplot(2, 1, 1)
    for graph_type in results['Kruskal']:
        if len(results['Kruskal'][graph_type]) == len(sizes):
            plt.plot(sizes, results['Kruskal'][graph_type], label=graph_type, marker='o')
    plt.title('Kruskal Algorithm Performance')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)

    # Prim overall
    plt.subplot(2, 1, 2)
    for graph_type in results['Prim']:
        if len(results['Prim'][graph_type]) == len(sizes):
            plt.plot(sizes, results['Prim'][graph_type], label=graph_type, marker='o')
    plt.title('Prim Algorithm Performance')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_individual_mst_results(sizes, results):
    graph_types = list(set(results['Kruskal'].keys()) | set(results['Prim'].keys()))

    for graph_type in graph_types:
        plt.figure(figsize=(10, 6))
        has_data = False

        if len(results['Kruskal'].get(graph_type, [])) == len(sizes):
            plt.plot(sizes, results['Kruskal'][graph_type], '-', marker='o', label='Kruskal', color='cornflowerblue')
            has_data = True

        if len(results['Prim'].get(graph_type, [])) == len(sizes):
            plt.plot(sizes, results['Prim'][graph_type], '-', marker='x', label='Prim', color='seagreen')
            has_data = True

        if has_data:
            plt.title(f'Performance on {graph_type} Graph')
            plt.xlabel('Number of Nodes')
            plt.ylabel('Time (seconds)')
            plt.legend()
            plt.grid(True)
            plt.show()


if __name__ == "__main__":
    sizes, results = run_tests(max_nodes=200, step=20)
    plot_overall_mst_results(sizes, results)
    plot_individual_mst_results(sizes, results)