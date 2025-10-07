import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

def are_isomorphic_manual(matrix1, matrix2):
    """
    Проверяет изоморфизм двух графов, заданных матрицами смежности,
    методом полного перебора перестановок.
    """
    A1 = np.array(matrix1)
    A2 = np.array(matrix2)
    n1, n2 = A1.shape[0], A2.shape[0]

    # Быстрые проверки-фильтры (инварианты)
    if n1 != n2:
        return False, None # Разное число вершин

    # Проверка последовательности степеней
    degrees1 = sorted(list(A1.sum(axis=1)))
    degrees2 = sorted(list(A2.sum(axis=1)))
    if degrees1 != degrees2:
        return False, None # Разные наборы степеней вершин

    # Полный перебор перестановок
    n = n1
    nodes = list(range(n))
    permutations = itertools.permutations(nodes)

    for p in permutations:
        # p - это кортеж, задающий отображение. p[i] = j означает,
        # что вершина i из графа 1 отображается в вершину j из графа 2.
        # Чтобы проверить A1[i][k] == A2[p[i]][p[k]], нужно создать инвертированную карту
        # или перестроить матрицу. Проще перестроить матрицу.
        
        is_match = True
        # Проверяем, совпадет ли A1 с A2 при текущей перестановке p
        for i in range(n):
            for j in range(n):
                # Связь между i и j в A1 должна быть такой же,
                # как связь между p[i] и p[j] в A2
                if A1[i, j] != A2[p[i], p[j]]:
                    is_match = False
                    break
            if not is_match:
                break
        
        # Если после всех проверок совпадение осталось, мы нашли изоморфизм
        if is_match:
            # Создаем словарь отображения G1 -> G2
            mapping = {i: p[i] for i in range(n)}
            return True, mapping # Графы изоморфны

    # Если ни одна перестановка не подошла
    return False, None

def read_graph(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    matrix = []
    for line in lines:
        row = [int(x) for x in line.split()]
        matrix.append(row)
    return matrix

def matrix_stats(A):
    A = np.array(A)
    n = A.shape[0]
    diag_nonzero = np.count_nonzero(np.diag(A))
    total_ones = int(A.sum())
    num_edges_by_matrix = total_ones // 2
    symmetric = np.array_equal(A, A.T)
    uniques = np.unique(A).tolist()
    degrees = list(map(int, A.sum(axis=1)))
    return {
        "n": n,
        "diag_nonzero": int(diag_nonzero),
        "total_ones": int(total_ones),
        "num_edges_by_matrix": int(num_edges_by_matrix),
        "symmetric": bool(symmetric),
        "unique_values": uniques,
        "degrees": degrees
    }

def build_nx_graph(adj):
    A = np.array(adj)
    n = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if A[i, j] != 0:
                G.add_edge(i, j)
    return G

def edges_set(G):
    return set((min(u,v), max(u,v)) for u,v in G.edges())

def draw_graph(adj_matrix, title, ax, seed=None, rad_scale=0.06):
    G = build_nx_graph(adj_matrix)
    pos = nx.spring_layout(G, seed=seed, iterations=100)

    for (u, v) in G.edges():
        rad = ((u * 31 + v * 17) % 7 - 3) * rad_scale
        p1, p2 = pos[u], pos[v]
        patch = FancyArrowPatch(
            p1, p2,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle='-',
            linewidth=1.6,
            alpha=0.9,
            color='gray'
        )
        ax.add_patch(patch)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=700, node_color="lightblue",
                           linewidths=1.2, edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax, font_weight="bold")
    ax.set_title(title)
    ax.set_axis_off()

def main():
    graph1 = read_graph('graph1.txt')
    graph2 = read_graph('graph2.txt')

    stats1 = matrix_stats(graph1)
    stats2 = matrix_stats(graph2)

    print("Graph1 stats:", stats1)
    print("Graph2 stats:", stats2)

    G1 = build_nx_graph(graph1)
    G2 = build_nx_graph(graph2)
    print("NetworkX edges: G1 =", G1.number_of_edges(), ", G2 =", G2.number_of_edges())
    
    print("\n--- Проверка на изоморфизм ---")
    are_iso, mapping = are_isomorphic_manual(graph1, graph2)

    if are_iso:
        print("Результат: графы ИЗОМОРФНЫ.")
        print("Пример отображения (G1 -> G2):", mapping)
    else:
        print("Результат: графы НЕ изоморфны.")
    print("--------------------------------------------------\n")

    # Отрисовка
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    draw_graph(graph1, f"Граф 1 (edges={G1.number_of_edges()})", ax1, seed=42)
    draw_graph(graph2, f"Граф 2 (edges={G2.number_of_edges()})", ax2, seed=7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
