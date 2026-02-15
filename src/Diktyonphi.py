import enum, subprocess, heapq
from typing import Dict, Hashable, Any, Optional, Iterator, Tuple, List
from itertools import product as prod
from PIL import Image
from paths import steps_dir

class GraphType(enum.Enum):
    """Typ orientace grafu: orientovaný nebo neorientovaný."""
    DIRECTED = 0
    UNDIRECTED = 1


class Edge:
    """Reprezentace hrany mezi dvěma vrcholy s atributy."""

    def __init__(self, src: 'Node', dest: 'Node', attrs: Dict[str, Any]):
        """
        Inicializace hrany z vrcholu `src` do vrcholu `dest` s danými atributy.

        :param src: Zdrojový vrchol.
        :param dest: Cílový vrchol.
        :param attrs: Slovník atributů hrany.
        """
        self.src = src
        self.dest = dest
        self._attrs = attrs

    def __getitem__(self, key: str) -> Any:
        """Vrací hodnotu atributu hrany podle klíče."""
        return self._attrs[key]

    def __setitem__(self, key: str, val: Any) -> None:
        """Nastaví atribut hrany podle klíče."""
        self._attrs[key] = val

    def __repr__(self):
        return f"Edge({self.src.id}→{self.dest.id}, {self._attrs})"


class Node:
    """Reprezentace vrcholu grafu s atributy a seznamem sousedů."""

    def __init__(self, graph: 'Graph', node_id: Hashable, attrs: Dict[str, Any]):
        """
        Inicializace vrcholu s daným identifikátorem a atributy.

        :param node_id: Identifikátor vrcholu.
        :param attrs: Slovník atributů vrcholu.
        """
        self.id = node_id
        self.graph = graph
        self._attrs = attrs
        self._neighbors: Dict[Hashable, Dict[str, Any]] = {}

    def __getitem__(self, item: str) -> Any:
        """Vrací atribut vrcholu podle klíče."""
        return self._attrs[item]

    def __setitem__(self, item: str, val: Any) -> None:
        """Nastaví atribut vrcholu podle klíče."""
        self._attrs[item] = val

    def to(self, dest: Hashable | 'Node') -> Edge:
        """
        Vrací hranu z tohoto vrcholu do vrcholu `dest`.

        :param dest: Cílový vrchol (ID nebo Node).
        :return: Instance Edge.
        :raises ValueError: Pokud hrana neexistuje.
        """
        dest_id = dest.id if isinstance(dest, Node) else dest
        if dest_id not in self._neighbors:
            return None
        return Edge(self, self.graph.node(dest_id), self._neighbors[dest_id])

    def connect_to(self, dest: Hashable | 'Node', attrs: Optional[Dict[str, Any]] = None):
        """Vytvoří hranu z tohoto vrcholu do vrcholu `dest`."""
        dest = dest if isinstance(dest, Node) else self.graph.node(dest)
        assert dest.graph == self.graph, f"Cílový vrchol {dest.id} není ve stejném grafu"
        assert dest.id in self.graph, f"Cílový vrchol {dest.id} není v grafu"
        self.graph.add_edge(self.id, dest.id, attrs if attrs is not None else {})

    def is_edge_to(self, dest: Hashable | 'Node') -> bool:
        """
        Zjistí, zda existuje hrana do vrcholu `dest`.

        :param dest: Cílový vrchol (ID nebo Node).
        :return: True pokud existuje, jinak False.
        """
        dest_id = dest.id if isinstance(dest, Node) else dest
        return dest_id in self._neighbors

    @property
    def neighbor_ids(self) -> Iterator[Hashable]:
        """Iterátor přes ID sousedních vrcholů."""
        return iter(self._neighbors)

    @property
    def neighbor_nodes(self) -> Iterator['Node']:
        """Iterátor přes sousední vrcholy."""
        for id in self.neighbor_ids:
            yield self.graph.node(id)

    @property
    def out_degree(self) -> int:
        """Počet hran vedoucích z vrcholu."""
        return len(self._neighbors)

    def __repr__(self):
        return f"Node({self.id}, {self._attrs})"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

class Graph:
    """Datová struktura grafu podporující orientované i neorientované grafy."""

    def __init__(self, type: GraphType):
        """
        Inicializace grafu daného typu.

        :param type: GraphType.DIRECTED nebo GraphType.UNDIRECTED
        """
        self.type = type
        self._nodes: Dict[Hashable, Node] = {}
        self._edges: Dict[Tuple, Dict] = {}

    def add_node(self, node_id: Hashable, attrs: Optional[Dict[str, Any]] = None) -> Node:
        """
        Přidá nový vrchol do grafu.

        :param node_id: Identifikátor vrcholu.
        :param attrs: Volitelný slovník atributů.
        :raises ValueError: Pokud vrchol již existuje.
        """
        if node_id in self._nodes:
            return
        return self._create_node(node_id, attrs if attrs is not None else {})

    def __contains__(self, node_id: Hashable) -> bool:
        """Vrací True, pokud graf obsahuje vrchol se zadaným ID."""
        return node_id in self._nodes

    def __len__(self) -> int:
        """Počet vrcholů v grafu."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        """Iterátor přes vrcholy."""
        return iter(self._nodes.values())

    def node_ids(self) -> Iterator[Hashable]:
        """Iterátor přes ID vrcholů."""
        return iter(self._nodes.keys())

    def node(self, node_id: Hashable) -> Node:
        """
        Vrátí instanci vrcholu podle ID.

        :param node_id: ID vrcholu.
        :return: Instanci Node.
        :raises KeyError: Pokud vrchol neexistuje.
        """
        return self._nodes[node_id]
    
    def edge(self, ids: Tuple[int, int]) -> Dict:
        """Vrací slovník atributů hrany"""
        return self._edges[ids]

    def _create_node(self, node_id: Hashable, attrs: Optional[Dict[str, Any]] = None) -> Node:
        """Interní metoda pro vytvoření vrcholu."""
        node = Node(self, node_id, attrs)
        self._nodes[node_id] = node
        return node

    def _set_edge(self, src_id: Hashable, target_id: Hashable, attrs: Dict[str, Any]) -> None:
        """Interní metoda vytvářející orientovanou hranu."""
        if target_id in self._nodes[src_id]._neighbors:
            return None
        self._nodes[src_id]._neighbors[target_id] = attrs

    def _edge_key(self, u, v):
        """
        Kanonický klíč hrany:
        - UNDIRECTED: (min(u,v), max(u,v))
        - DIRECTED:   (u,v)
        """
        if self.type == GraphType.UNDIRECTED:
            return (u, v) if u <= v else (v, u)
        return (u, v)

    def add_edge(self, src_id, dst_id, attrs=None):
        """
        Přidá hranu. U neorientovaného grafu ignoruje pořadí vrcholů.
        Vrcholy se vytvoří automaticky, pokud neexistují.
        """
        attrs = attrs if attrs is not None else {}

        if src_id not in self._nodes:
            self._create_node(src_id, {})
        if dst_id not in self._nodes:
            self._create_node(dst_id, {})

        key = self._edge_key(src_id, dst_id)

        if key in self._edges:
            return None

        self._set_edge(src_id, dst_id, attrs)
        if self.type == GraphType.UNDIRECTED:
            self._set_edge(dst_id, src_id, attrs)

        self._edges[key] = attrs
        return (self._nodes[src_id], self._nodes[dst_id])

    def del_edge(self, u, v) -> None:
        """
        Odstraní hranu mezi dvěma vrcholy.
        U neorientovaného grafu nezáleží na pořadí (u,v).
        """
        if u not in self._nodes:
            raise ValueError(f"Vrchol {u} neexistuje")
        if v not in self._nodes:
            raise ValueError(f"Vrchol {v} neexistuje")

        key = self._edge_key(u, v)
        if key not in self._edges:
            raise ValueError(f"Hrana mezi {u} a {v} neexistuje")

        self._nodes[u]._neighbors.pop(v, None)
        if self.type == GraphType.UNDIRECTED:
            self._nodes[v]._neighbors.pop(u, None)

        self._edges.pop(key)

    def del_node(self, node_id) -> None:
        """
        Odstraní vrchol a všechny incidentní hrany.
        (Důležité: odstraní i z _edges.)
        """
        if node_id not in self._nodes:
            return

        neighbors = list(self._nodes[node_id]._neighbors.keys())
        for nb in neighbors:
            self.del_edge(node_id, nb)

        self._nodes.pop(node_id)

    def set_node_id(self, old_id, new_id):
        """
        Změní ID vrcholu, zachová všechny hrany.
        """
        self._create_node(new_id, attrs=self.node(old_id)._attrs)
        for node in self._nodes:
            if self.node(node).to(old_id) and (node, old_id) not in self._edges:
                self.add_edge(node, new_id)
            if self.node(old_id).to(node) and (old_id, node) not in self._edges:
                self.add_edge(new_id, node)
        self.del_node(old_id)

    def in_degree(self, node_id) -> int:
        """Vrací počet vstupních hran do vrcholu."""
        number_of_ingoing_edges = 0
        for id in self._nodes:
            if node_id in self.node(id)._neighbors:
                number_of_ingoing_edges += 1
        return number_of_ingoing_edges
    
    def copy(self):
        """
        Vrátí kopii grafu.
        """
        graph = Graph(self.type)

        for node_id, node in self._nodes.items():
            graph._create_node(node_id, dict(node._attrs))

        for u, v in self._edges:
            graph.add_edge(u, v)

        return graph

    def __repr__(self):
        edges = sum(node.out_degree for node in self._nodes.values())
        if self.type == GraphType.UNDIRECTED:
            edges //= 2
        return f"Graph({self.type}, nodes: {len(self._nodes)}, edges: {edges})"

    def to_dot(self, label_attr: str = "label", weight_attr: str = "weight", fillcolor_attr: str = "fillcolor") -> str:
        """
        Generuje jednoduchou reprezentaci grafu ve formátu Graphviz (DOT).
        (Vygenerováno původně pomocí ChatGPT.)

        :return: Řetězec ve formátu DOT.
        """
        lines = []
        name = "G"
        connector = "->" if self.type == GraphType.DIRECTED else "--"

        # Hlavička grafu
        lines.append(
            f'digraph {name} {{' if self.type == GraphType.DIRECTED else f'graph {name} {{'
        )

        lines.append("""
            graph [dpi=180];
            node [
                fontsize=10,
                fontname="Helvetica-Bold",
                penwidth=1,
                width=0.2,
                height=0.15
            ];
            edge [penwidth=1];
        """)

        # Vrcholy
        for node_id in self.node_ids():
            node = self.node(node_id)
            label = node[label_attr] if label_attr in node._attrs else str(node_id)
            fillcolor = node[fillcolor_attr] if fillcolor_attr in node._attrs else "#ffffff"
            lines.append(f'    "{node_id}" [label="{label}", fillcolor="{fillcolor}", style="filled"];')

        # Hrany
        seen = set()
        for node_id in self.node_ids():
            node = self.node(node_id)
            for dst_id in node.neighbor_ids:
                if self.type == GraphType.UNDIRECTED and (dst_id, node_id) in seen:
                    continue
                seen.add((node_id, dst_id))
                edge = node.to(dst_id)
                label = edge[weight_attr] if weight_attr in edge._attrs else ""
                lines.append(f'    "{node_id}" {connector} "{dst_id}" [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    def export_to_png(
        self,
        image_name: str | None = "graph.png",
        code_type: str | None = None,
        dark: bool = False
    ) -> str:
        """
        Exportuje graf do formátu PNG a vrací úplnou cestu k souboru.

        DŮLEŽITÉ:
        - vždy zapisuje do outputs/...
        """
        if code_type is None:
            code_type = "graphs"

        out_dir = steps_dir(code_type)  # outputs/<code_type>/
        filename = out_dir / (image_name or "graph.png")

        dot_data = self.to_dot()

        try:
            subprocess.run(
                ["dot", "-Tpng", "-o", str(filename)],
                input=dot_data,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Graphviz 'dot' selhal: {e}") from e

        def invert(path: str, name: str):
            img = Image.open(path).convert("RGBA")
            img = img.crop((0, 0, img.width - 3, img.height - 3))
            px = img.load()
            new_img = img.copy()
            new_px = new_img.load()

            for h in range(img.height):
                for w in range(img.width):
                    if px[w, h] == (255, 255, 255, 255):
                        new_px[w, h] = (37, 37, 38, 255)

                    elif px[w, h] in [(255, 0, 0, 255), 
                                      (0, 0, 255, 255),
                                      (255, 0, 255, 255)]:
                        continue

                    else:
                        new_px[w, h] = (255 - px[w, h][0],
                                        255 - px[w, h][1],
                                        255 - px[w, h][2])

            new_img.save(name)

        if dark is True:
            invert(str(filename), str(filename))

        return str(filename)

    def __str__(self):
        """Stručná textová reprezentace vrcholů a hran."""
        return f"Vertexes: {list(self._nodes.keys())}, edges: {list(self._edges.keys())}"

    def dfs_pruchod(self, id=None, visited=None, result=None):
        """
        Jednoduchý DFS průchod grafem.

        Pokud není zadán počáteční vrchol, vezme se první existující.
        """
        if id is None:
            id = list(self._nodes.keys())[0]

        if visited is None:
            visited = set()
            result = []

        result.append(id)
        visited.add(id)

        for child in self.node(id)._neighbors:
            if child not in visited:
                self.dfs_pruchod(child, visited, result)
        return result
    
    def is_tree(self):
        """Ověřuje, zda je graf stromem."""
        if (len(self._edges) == len(self._nodes) - 1 and
            len(self.dfs_pruchod(list(self._nodes)[0])) == len(self._nodes)):
            return True
        return False

    def to_prufer(self, steps: bool = False):
        """
        Převede strom na Prüferův kód pomocí standardního algoritmu:
        - opakovaně hledá nejmenší list
        - odstraní list
        - zapíše jeho souseda do kódu
        - opakuje se, dokud nezůstaly dva vrcholy
        """
        if self.is_tree() is False:
            return
        
        code = []
        heap_listy = []
        tree = self.copy()

        if steps:
            base = steps_dir("prufer")
            # чистим старые step картинки
            for path in base.glob("to_code_*_step.png"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        # Najdeme všechny listy
        for node in tree._nodes:
            if tree.node(node).out_degree == 1:
                heapq.heappush(heap_listy, tree._nodes[node].id)

        step = 1
        # Opakujeme, dokud nezůstanou 2 vrcholy
        while len(tree._nodes) > 2:
            minimal_id = heapq.heappop(heap_listy)
            neighbor = list(tree.node(minimal_id).neighbor_nodes)[0].id
            code.append(neighbor)
            if steps:
                tree.node(minimal_id)._attrs["fillcolor"] = "#ff0000"
                tree.node(neighbor)._attrs["fillcolor"] = "#0000ff"
                tree.export_to_png(f"to_code_{step}_step.png", code_type="prufer", dark=True)
                tree.node(neighbor)._attrs["fillcolor"] = "#ffffff"
                step += 1
            tree.del_node(minimal_id)

            if tree.node(neighbor).out_degree == 1:
                heapq.heappush(heap_listy, neighbor)

        return code
    
    def is_graceful(self):
        """Testuje zda ohodnocení grafu je graciózní"""
        set_of_differencies = set()
        for edge in self._edges:
            difference = abs(edge[1] - edge[0])
            if difference in set_of_differencies or difference == 0:
                return False
            set_of_differencies.add(difference)
        edges = len(self._edges)
        return set_of_differencies == set(range(1, edges + 1))
    
    def to_sheppard(self, steps: bool = False):
        """
        Převede graciózní ohodnocený graf na sheppardův kód pomoci algoritmu:
        - hledá se hrana s minimálním ohodnocením
        - menší z její vrcholu se zapisuje do kódu
        - hrana se odstraňuje
        - opakuje se, dokud nebudou odstraněny všechny hrany
        """
        if self.is_graceful() is False:
            return
        
        graph = self.copy()
        if steps:
            base = steps_dir("sheppard")
            for path in base.glob("to_code_*_step.png"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

        edge_value = 1
        code = []
        while len(code) != len(self._edges):
            for edge in self._edges:
                if abs(edge[1] - edge[0]) == edge_value:
                    code.append(min(edge[1], edge[0]))
                    if steps:
                        graph.node(min(edge[0], edge[1]))._attrs["fillcolor"] = "#0000ff"
                        graph.node(max(edge[0], edge[1]))._attrs["fillcolor"] = "#ff0000"
                        graph.export_to_png(f"to_code_{edge_value}_step.png", code_type="sheppard", dark=True)
                        graph.del_edge(edge[0], edge[1])
                        graph.node(min(edge[0], edge[1]))._attrs["fillcolor"] = "#ffffff"
                        graph.node(max(edge[0], edge[1]))._attrs["fillcolor"] = "#ffffff"
            edge_value += 1

        return code
    
    def involute(self):
        """
        Inverzní transformace (involuce) vrcholů grafu:
        - transformuje vrcholy podle mapy vrchol → (max_vrchol - vrchol)
        """
        new_graph = Graph(self.type)
        max_vrchol = max(self._nodes)
        for u, v in self._edges:
            u2, v2 = max_vrchol - u, max_vrchol - v
            new_graph.add_edge(u2, v2)
        return new_graph
    
    def canonical_code(self, node_id, parent_id=None):
        children = [nid for nid in self.node(node_id).neighbor_ids
                    if parent_id is None or nid != parent_id]

        if not children:
            return "01"

        codes = [self.canonical_code(child, node_id) for child in children]
        codes.sort()
        code = "0" + "".join(codes) + "1"
        return code
                
def all_sheppard_codes(n):
    """
    Generátor „první poloviny“ Sheppardových kódů pro dané n.
    Celkový počet Sheppardových kódů je (n-1)!,
    vrací (n-1)! / 2 (lexikograficky první polovinu).
    """
    ranges = []
    total = 1

    for j in range(n - 1):
        L = n - 1 - j
        total *= L
        ranges.append(range(0, L))

    limit = int(total / 2)
    count = 0

    for code in prod(*ranges):
        if count >= limit:
            break
        yield code
        count += 1

def from_prufer(code: List[int], steps: bool = False) -> Graph:
    """
    Rekonstrukce stromu z Prüferova kódu (0-based).
    Vrcholy: 0..n-1, kde n = len(code) + 2,
    Kód: délky n-2, hodnoty 0..n-1
    """
    code = [int(x) for x in code]
    n = len(code) + 2

    if any(x < 0 or x > n - 1 for x in code):
        raise ValueError(f"Invalid Prüfer code")

    if steps:
        base = steps_dir("prufer")
        for path in base.glob("graph_*_step.png"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    tree = Graph(GraphType.UNDIRECTED)

    degrees = [1] * n
    for el in code:
        degrees[el] += 1

    min_leaf = 0
    while min_leaf < n and degrees[min_leaf] != 1:
        min_leaf += 1
    p = min_leaf

    step = 1
    for el in code:
        tree.add_edge(p, el)

        if steps:
            tree.node(el)._attrs["fillcolor"] = "#0000ff"
            tree.node(p)._attrs["fillcolor"] = "#ff0000"
            tree.export_to_png(f"to_graph_{step}_step.png", code_type="prufer", dark=True)
            tree.node(el)._attrs["fillcolor"] = "#ffffff"
            tree.node(p)._attrs["fillcolor"] = "#ffffff"
            step += 1

        degrees[p] = 0
        degrees[el] -= 1

        if degrees[el] == 1 and el < min_leaf:
            p = el
        else:
            while min_leaf < n and degrees[min_leaf] != 1:
                min_leaf += 1
            p = min_leaf

    u = degrees.index(1)
    degrees[u] = 0
    v = degrees.index(1)
    tree.add_edge(u, v)

    if steps:
        tree.node(u)._attrs["fillcolor"] = "#ff00ff"
        tree.node(v)._attrs["fillcolor"] = "#ff00ff"
        tree.export_to_png(f"to_graph_{step}_step.png", code_type="prufer", dark=True)
        tree.node(u)._attrs["fillcolor"] = "#ffffff"
        tree.node(v)._attrs["fillcolor"] = "#ffffff"

    return tree

def from_sheppard(code: list[int], steps: bool = False, complete: bool = False) -> Graph:
    if any(code[i] > len(code) - i - 1 
           or code[i] < 0 
           or type(code[i]) is not int for i in range(len(code))):
        raise ValueError(f"Invalid Sheppard code")

    if steps:
        base = steps_dir("sheppard")
        for path in base.glob("to_graph_*_step.png"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    graph = Graph(GraphType.UNDIRECTED)

    for i, u in enumerate(code):
        v = u + i + 1
        graph.add_edge(u, v)

        if steps:
            graph.node(u)._attrs["fillcolor"] = "#0000ff"
            graph.node(v)._attrs["fillcolor"] = "#ff0000"
            graph.export_to_png(f"to_graph_{i + 1}_step.png", code_type="sheppard", dark=True)
            graph.node(u)._attrs["fillcolor"] = "#ffffff"
            graph.node(v)._attrs["fillcolor"] = "#ffffff"

    if complete:
        for i in range(len(code)):
            graph.add_node(i)

    return graph


def sheppard_uses_all_vertices(code, n: int) -> bool:
    """
    Rychlá kontrola, zda Sheppardův kód využívá všech n vrcholů.
    Pokud ano, může se jednat o strom (ale nemusí).
    Stačí ověřit, že množina {j, j+i} má velikost n.
    """
    seen = set()

    for i, j in enumerate(code, start=1):  # i = 1..n-1
        u = j
        v = j + i
        seen.add(u)
        seen.add(v)

    return len(seen) == n

def involute_prufer_code(code: List[int]) -> List[int]:
    """
    Pruferův kód ohodnocení, získaného přes involuci:
    - podle kódu sestaví strom
    - najde jeho involutorní páru
    - vrací kód involutorního ohodnocení
    """
    involute_code = from_prufer(code).involute().to_prufer()
    return involute_code

def prufer_lex_rank(code: tuple[int, ...], n: int) -> int:
    """
    Lexikografický rang (počínaje 1) v prostoru {0..n-1}^(n-2), n = len(code) - 2.
    Počítá se pomocí mocnin: sum((x_i-1) * n^(k-1-i)) + 1
    """
    k = len(code)
    r0 = 0
    for i, x in enumerate(code):
        power = k - 1 - i
        r0 += x * (n ** power)
    return r0 + 1

if __name__ == "__main__":
    graph = Graph(GraphType.UNDIRECTED)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(2, 4)
    graph.add_edge(0, 3)
    print(graph._edges)
    graph.del_edge(2, 3)
