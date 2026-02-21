from Diktyonphi import Graph, GraphType

def add_leaf(graph: Graph, v: int) -> Graph:
    if not graph.is_tree():
        raise ValueError("add_leaf: graph is not a tree")

    ids = set(graph.node_ids())
    if v not in ids:
        raise ValueError(f"add_leaf: vertex {v} is not in graph")

    u = max(ids) + 1  # если id плотные
    g2 = graph.copy()
    g2.add_edge(v, u)
    return g2

def generate_all_unlabeled_trees(n: int):

    def representative_vertices_by_type(T: Graph) -> list[int]:
        reps = {}
        for v in T.node_ids():
            t = T.canonical_code(v)
            if t not in reps:
                reps[t] = v
        return list(reps.values())
    
    def next_level(trees_n: dict[str, Graph]) -> dict[str, Graph]:
        out: dict[str, Graph] = {}

        for T in trees_n.values():
            reps = representative_vertices_by_type(T)
            for v in reps:
                T2 = add_leaf(T, v)
                c = T2.min_canonical_code()
                if c not in out:
                    out[c] = T2

        return out
    
    def generate_unlabeled_trees(N: int) -> dict[int, dict[str, Graph]]:
        if N < 1:
            raise ValueError("N must be >= 1")

        # n=1
        t1 = Graph(GraphType.UNDIRECTED)
        t1.add_node(0)
        trees: dict[int, dict[str, Graph]] = {1: {t1.min_canonical_code(): t1}}

        for k in range(1, N):
            trees[k+1] = next_level(trees[k])

        for k in range(1, N):
            trees.pop(k)

        return trees
    
    return generate_unlabeled_trees(n)

if __name__ == "__main__":
    k = 10
    for n in range(1, k):
        trees = generate_all_unlabeled_trees(n)
        count_of_orders = {}
        for T in trees[n].values():
            try:
                count_of_orders[T.aut_order()] += 1
            except:
                count_of_orders[T.aut_order()] = 1
        print(f"{len(count_of_orders)}, ", end="")
