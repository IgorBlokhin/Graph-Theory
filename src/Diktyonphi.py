import enum, subprocess, heapq, time, os, glob
from typing import Dict, Hashable, Any, Optional, Iterator, Tuple, List
from itertools import product as prod
from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from invert_color import invert

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
        self._edges: List[Tuple] = []

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

    def add_edge(self, src_id: Hashable, dst_id: Hashable,
                 attrs: Optional[Dict[str, Any]] = None) -> Tuple[Node, Node]:
        """
        Přidá novou hranu mezi dvěma vrcholy. Vrcholy se vytvoří automaticky,
        pokud neexistují.

        :param src_id: Zdrojový vrchol.
        :param dst_id: Cílový vrchol.
        :param attrs: Volitelný slovník atributů hrany.
        :return: (zdrojový vrchol, cílový vrchol)
        """
        if self.type == GraphType.UNDIRECTED:
            if (src_id, dst_id) in self._edges or (dst_id, src_id) in self._edges:
                return None

        if self.type == GraphType.DIRECTED:
            if (src_id, dst_id) in self._edges:
                return None

        attrs = attrs if attrs is not None else {}
        if src_id not in self._nodes:
            self._create_node(src_id, {})
        if dst_id not in self._nodes:
            self._create_node(dst_id, {})
        self._set_edge(src_id, dst_id, attrs)
        if self.type == GraphType.UNDIRECTED:
            self._set_edge(dst_id, src_id, attrs)
        self._edges.append((self.node(src_id).id, self.node(dst_id).id))
        return (self._nodes[src_id], self._nodes[dst_id])

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

    def del_node(self, node_id) -> None:
        """Odstraní vrchol a všechny hrany, které na něj vedou."""
        for neighbor_id in self._nodes:
            if node_id in self.node(neighbor_id)._neighbors:
                self.node(neighbor_id)._neighbors.pop(node_id)
        self._nodes.pop(node_id)

    def del_edge(self, node_id_1, node_id_2) -> None:
        """
        Odstraní hranu mezi dvěma vrcholy.

        :raises ValueError: Pokud vrcholy nebo hrana neexistují.
        """
        if self.__contains__(node_id_1) is False:
            raise ValueError(f"Vrchol {node_id_1} neexistuje")
        elif self.__contains__(node_id_2) is False:
            raise ValueError(f"Vrchol {node_id_2} neexistuje")
        elif node_id_1 not in self.node(node_id_2)._neighbors and node_id_2 not in self.node(node_id_1)._neighbors:
            raise ValueError(f"Hrana mezi {node_id_1} a {node_id_2} neexistuje")

        if node_id_2 in self.node(node_id_1)._neighbors:
            self.node(node_id_1)._neighbors.pop(node_id_2)
        if node_id_1 in self.node(node_id_2)._neighbors:
            self.node(node_id_2)._neighbors.pop(node_id_1)

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
                penwidth=1
                width=0.5
                height=0.35
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

    def export_to_png(self, image_name: str | None = "graph.png", code_type: str | None = None, dark: bool = False) -> str:
        """
        Экспортирует граф в PNG и возвращает полный путь к файлу.
        Если filename не задан, сохраняет в 'graph.png' рядом с модулем.
        """
        base_dir = Path(__file__).resolve().parent  # папка, где лежит модуль с графом
        try:
            (base_dir / code_type).mkdir()
        except FileExistsError:
            pass
        filename = base_dir / code_type
        filename = filename / image_name

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

        if dark is True:
            invert(str(Path(filename)), filename)

        return str(Path(filename))


    def _repr_svg_(self):
        """
        Vráti SVG reprezentaci grafu pro Jupyter Notebook
        (IPython inline protokol).
        """
        return self.to_image().data

    def to_image(self):
        """
        Vrátí graf jako SVG (vhodné pro zobrazení v IPythonu).
        """
        from IPython.display import SVG
        dot_data = self.to_dot()
        try:
            process = subprocess.run(
                ['dot', '-Tsvg'],
                input=dot_data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return SVG(data=process.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Graphviz 'dot' selhal: {e} s chybou: {e.stderr}"
            ) from e

    def __str__(self):
        """Stručná textová reprezentace vrcholů a hran."""
        return f"Vertexes: {sorted(list(self._nodes.keys()))}, edges: {sorted(self._edges)}"

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
        """
        Ověřuje, zda je graf stromem. Graf není stromem, pokud počet
        hran se nerovná počtu vrcholů, snižinému o 1 nebo pokud má víc
        než jednu komponentu souvislosti, což ověřujeme pomoci algoritmu
        prohledávání do hloubky
        """
        if (len(self._edges) != len(self._nodes) - 1 or
            len(self.dfs_pruchod(list(self._nodes)[0])) != len(self._nodes) or 
            not self._edges):
            return False
        return True

    def to_prufer(self):
        """
        Převede strom na Prüferův kód pomocí standardního algoritmu:
        - opakovaně hledá nejmenší list
        - odstraní list
        - zapíše jeho souseda do kódu
        """
        code = []
        heap_listy = []
        tree = self.copy()

        # Najdeme všechny listy
        for node in tree._nodes:
            if tree.node(node).out_degree == 1:
                heapq.heappush(heap_listy, tree._nodes[node].id)

        # Opakujeme, dokud nezůstanou 2 vrcholy
        while len(tree._nodes) > 2:
            minimal_id = heapq.heappop(heap_listy)
            neighbor = list(tree.node(minimal_id).neighbor_nodes)[0].id
            code.append(neighbor)
            tree.del_node(minimal_id)

            if tree.node(neighbor).out_degree == 1:
                heapq.heappush(heap_listy, neighbor)

        return code
    
    def is_graceful(self):
        set_of_differencies = set()
        for edge in self._edges:
            difference = abs(edge[1] - edge[0])
            if difference in set_of_differencies or difference == 0:
                return False
            set_of_differencies.add(difference)
        edges = len(self._edges)
        return set_of_differencies == set(range(1, edges + 1))
    
    def involute(self):
        """
        Inverzní transformace (involuce) vrcholů grafu:
        - transformuje vrcholy podle mapy vrchol → (max_vrchol - vrchol + 1)
        """
        new_graph = Graph(self.type)
        max_vrchol = max(self._nodes)  # это будет n-1
        for u, v in self._edges:
            u2, v2 = max_vrchol - u, max_vrchol - v
            new_graph.add_edge(u2, v2)
        return new_graph


def all_sheppard_codes(n):
    """
    Generátor „první poloviny“ Sheppardových kódů pro dané n.
    Celkový počet Sheppardových kódů je (n-1)!,
    zde vracíme nejvýše (n-1)! / 2 (lexikograficky první polovinu).
    """
    ranges = []
    total = 1

    for j in range(n - 1):
        L = n - 1 - j          # длины: n-1, n-2, ..., 1
        total *= L
        ranges.append(range(0, L))  # 0..L-1

    limit = total // 2
    count = 0

    for code in prod(*ranges):
        if count >= limit:
            break
        yield code
        count += 1

def from_prufer(code: List[int], steps: bool = False) -> Graph:
    """
    Rekonstrukce stromu z Prüferova kódu (0-based).
    Vršoly: 0..n-1, kde n = len(code) + 2
    Kód: délky n-2, hodnoty 0..n-1
    """
    if steps:
        BASE_DIR = Path(__file__).resolve().parent / "prufer"
        for path in BASE_DIR.glob("graph_*_step.png"):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        try:
            os.remove("final_graph.png")
        except FileNotFoundError:
            pass

    code = [int(x) for x in code]
    n = len(code) + 2

    tree = Graph(GraphType.UNDIRECTED)

    # степени (0-based индексация)
    degrees = [1] * n
    for el in code:
        # el должен быть в 0..n-1
        degrees[el] += 1

    # находим минимальный лист
    min_leaf = 0
    while min_leaf < n and degrees[min_leaf] != 1:
        min_leaf += 1
    p = min_leaf

    step = 1
    for el in code:
        # добавляем ребро (p, el)
        tree.add_edge(p, el)

        if steps:
            tree.node(el)._attrs["fillcolor"] = "#ff0000"
            tree.node(p)._attrs["fillcolor"] = "#0000ff"
            tree.export_to_png(f"graph_{step}_step.png", code_type="prufer", dark=True)
            tree.node(el)._attrs["fillcolor"] = "#ffffff"
            tree.node(p)._attrs["fillcolor"] = "#ffffff"
            step += 1

        # "удаляем" лист p
        degrees[p] = 0
        degrees[el] -= 1

        # выбираем следующий минимальный лист
        if degrees[el] == 1 and el < min_leaf:
            p = el
        else:
            while min_leaf < n and degrees[min_leaf] != 1:
                min_leaf += 1
            p = min_leaf

    # соединяем последние два листа
    u = degrees.index(1)
    degrees[u] = 0
    v = degrees.index(1)
    tree.add_edge(u, v)

    if steps:
        tree.node(u)._attrs["fillcolor"] = "#00ff00"
        tree.node(v)._attrs["fillcolor"] = "#00ff00"
        tree.export_to_png(f"graph_{step}_step.png", code_type="prufer", dark=True)

    return tree

def from_sheppard(code: list, steps: bool = False) -> Graph:
    """
    Rekonstrukce grafu ze Sheppardova kódu.
    Každá pozice kódu i obsahuje číslo j,
    což znamená hranu (j, j+i+1).

    Vrcholy které nebyly nalezeny doplníme.
    """
    if steps:
        BASE_DIR = Path(__file__).resolve().parent / "sheppard"
        for path in BASE_DIR.glob("graph_*_step.png"):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        try:
            os.remove("final_graph.png")
        except FileNotFoundError:
            pass

    graph = Graph(GraphType.UNDIRECTED)
    for i in range(len(code)):
        graph.add_edge(code[i], code[i] + i + 1)
        if steps:
            graph.node(code[i])._attrs["fillcolor"] = "#ff0000"
            graph.node(code[i] + i + 1)._attrs["fillcolor"] = "#0000ff"
            graph.export_to_png(f"graph_{i + 1}_step.png", "sheppard", dark=True)
            graph.node(code[i])._attrs["fillcolor"] = "#ffffff"
            graph.node(code[i] + i + 1)._attrs["fillcolor"] = "#ffffff"

    # добиваем вершины 0..n-1, где n = len(code)+1
    for v in range(0, len(code) + 1):
        graph.add_node(v)

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

def check_batch_sheppard(codes_batch, n):
    """
    На вход: список Sheppard-кодов длины n-1.
    На выход: ПЛОСКИЙ список Prüfer-кодов:
        [pr1, pr1_inv, pr2, pr2_inv, ...]
    где каждый pr* — это tuple длины n-2.
    """
    results = []

    for shep in codes_batch:
        # 1) используют ли все вершины?
        if not sheppard_uses_all_vertices(shep, n):
            continue

        # 2) строим дерево
        tree = from_sheppard(shep)

        # 3) проверка: это действительно дерево (связный граф на n вершинах)
        # dfs_pruchod должен пройти все n вершин
        if len(tree.dfs_pruchod()) != n:
            continue

        # 4) если дошли сюда — Sheppard-код задаёт грациозную разметку дерева
        pr = tuple(tree.to_prufer())
        inv_pr = tuple(tree.involute().to_prufer())

        # добавляем ОБА Prüfer-кода как отдельные записи
        results.append(pr)
        results.append(inv_pr)

    return results


# ---- pomocná funkce pro dávky ----
def take_batch(it, batch_size):
    """Vezme z iterátoru další dávku o velikosti batch_size (nebo méně)."""
    return list(islice(it, batch_size))

import os

def prufer_lex_rank(code: tuple[int, ...], n: int) -> int:
    """
    Лексикографический ранг (начиная с 1) в пространстве {1..n}^k, k=len(code).
    Считает через степени: sum((x_i-1) * n^(k-1-i)) + 1
    """
    k = len(code)
    r0 = 0
    for i, x in enumerate(code):
        power = k - 1 - i
        r0 += x * (n ** power)
    return r0 + 1

def sort_and_index_file(filepath: str, n: int) -> None:
    """
    Читает файл, где каждая строка — один Prüfer-код как числа через пробел:
        "x1 x2 ... x_{n-2}"

    Также поддерживает строки с уже добавленным индексом:
        "<rank> x1 x2 ... x_{n-2}"

    Сортирует по лексикографическому рангу в Prüfer-пространстве и перезаписывает файл:
        "<rank>  x1 x2 ... x_{n-2}"

    Индекс выровнен по правому краю.
    """
    if not os.path.isfile(filepath):
        print(f"[index] Soubor nenalezen: {filepath}")
        return

    k = n - 2
    codes: list[tuple[int, ...]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Пытаемся вытащить все целые из строки
            parts = s.split()
            try:
                nums = [int(p) for p in parts]
            except ValueError:
                # если попалась нечисловая строка — пропускаем
                continue

            # Различаем:
            #  - [x1..xk]                 -> len == k
            #  - [rank, x1..xk]           -> len == k+1
            if len(nums) == k:
                code = tuple(nums)
            elif len(nums) == k + 1:
                code = tuple(nums[1:])
            else:
                continue

            # Проверяем диапазон значений Prüfer-кода
            if any(not (0 <= x <= n-1) for x in code):
                continue

            codes.append(code)

    if not codes:
        print(f"[index] Žádné platné kódy v souboru: {filepath}")
        return

    # Сортировка по рангу
    codes.sort(key=lambda c: prufer_lex_rank(c, n))

    # Ширина для красивого выравнивания индекса
    max_rank = prufer_lex_rank(codes[-1], n)
    w = len(str(max_rank))

    with open(filepath, "w", encoding="utf-8") as f:
        for code in codes:
            r = prufer_lex_rank(code, n)
            f.write(f"{r:>{w}}  " + " ".join(map(str, code)) + "\n")

    print(f"[index] Hotovo: seřazeno + index přepsán: {filepath}")

# ================== HLAVNÍ FUNKCE ==================

def graceful_codes_from_sheppard(
    n,
    output_file=None,
    workers=2,
    batch_size=12000,
    max_inflight=12,
    heartbeat_sec=4.0,
    buf_write_every=30_000,
    sort=False,
    max_file_mb=50,   # Лимит размера ОДНОГО файла в мегабайтах
):
    """
    Параллельный проход по Sheppard-кодам для n:
      - sheppard_uses_all_vertices(code, n)
      - дерево (len(dfs_pruchod()) == n)
      - tree.is_graceful()

    Для КАЖДОГО найденного дерева записываем:
      - его Prüfer-код
      - Prüfer-код дерева-инволюции

    Формат вывода:
      - создаётся несколько файлов:
            {base_name}_part_001.txt
            {base_name}_part_002.txt
            ...
      - каждая строка: один Prüfer-код в виде "(..., ..., ...)".
      - как только размер текущего файла превышает max_file_mb МБ,
        открывается новый файл (номер части ++).
    """

    # --- Куда писать и какое базовое имя использовать для частей ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    if output_file is None:
        out_dir = OUTPUTS_DIR / "sheppard" / f"n={n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if sort:
            base_name = f"graceful_sheppard_{n}_sort"
        else:
            base_name = f"graceful_sheppard_{n}"
    else:
        # Если задан полный путь, разбиваем его на каталог + базовое имя
        output_file = os.path.abspath(output_file)
        out_dir, base = os.path.split(output_file)
        os.makedirs(out_dir, exist_ok=True)
        base_name, _ = os.path.splitext(base)

    # Лимит размера одной части в байтах
    limit_bytes = max_file_mb * 1024 * 1024

    def make_part_path(part_idx: int) -> str:
        """Построить путь к файлу для части с номером part_idx."""
        return os.path.join(out_dir, f"{base_name}_part_{part_idx:03d}.txt")

    total_sheppard = 0  # сколько Sheppard-кодов проверено
    total_prufer = 0    # сколько Prüfer-кодов записано

    t0 = time.perf_counter()
    last_print = t0
    last_total_sheppard = 0

    print(
        f"[main] (Sheppard) n={n}, workers={workers}, batch_size={batch_size}, "
        f"max_inflight={max_inflight}, max_file_mb={max_file_mb}",
        flush=True,
    )

    codes_iter = all_sheppard_codes(n)  # генерируем ПЕРВУЮ половину Sheppard-кодов

    buf = []

    # --- Параллельный проход + запись с делением на части ---
    with ProcessPoolExecutor(max_workers=workers) as ex:
        part_idx = 1
        out_path = make_part_path(part_idx)
        out = open(out_path, "w", encoding="utf-8")
        current_size = 0  # сколько байт уже записано в текущий файл

        def flush_buf():
            """
            Сбросить буфер в файл.
            Формат: числа, разделённые пробелом, одна строка = один Prüfer-код.
            """
            nonlocal out, current_size, part_idx, out_path, buf
            if not buf:
                return

            # формируем текст без скобок и запятых
            lines = [" ".join(str(x) for x in pr) for pr in buf]
            text = "\n".join(lines) + "\n"

            data = text.encode("utf-8")
            size = len(data)

            if current_size + size > limit_bytes and current_size > 0:
                out.close()
                part_idx += 1
                out_path = make_part_path(part_idx)
                out = open(out_path, "w", encoding="utf-8")
                current_size = 0

            out.write(text)
            current_size += size
            buf.clear()

        try:
            inflight = set()
            fut_sizes = {}  # future -> сколько Sheppard-кодов было в батче

            # --- начальная загрузка очереди ---
            while len(inflight) < max_inflight:
                batch = take_batch(codes_iter, batch_size)
                if not batch:
                    break
                fut = ex.submit(check_batch_sheppard, batch, n)
                inflight.add(fut)
                fut_sizes[fut] = len(batch)

            # --- основной цикл обработки результатов ---
            while inflight:
                done_any = False

                for fut in as_completed(list(inflight), timeout=heartbeat_sec):
                    inflight.remove(fut)
                    batch_size_done = fut_sizes.pop(fut, 0)

                    # эти Sheppard-коды мы уже проверили
                    total_sheppard += batch_size_done

                    # тут приходит список Prüfer-кодов [pr1, pr1_inv, pr2, pr2_inv, ...]
                    prufer_list = fut.result()
                    total_prufer += len(prufer_list)

                    # складываем Prüfer-коды в буфер
                    for pr in prufer_list:
                        buf.append(pr)
                        if len(buf) >= buf_write_every:
                            flush_buf()

                    # подкидываем новую порцию
                    batch = take_batch(codes_iter, batch_size)
                    if batch:
                        new_fut = ex.submit(check_batch_sheppard, batch, n)
                        inflight.add(new_fut)
                        fut_sizes[new_fut] = len(batch)

                    now = time.perf_counter()
                    done_any = True

                # --- heartbeat, pokud algoritmus stale běží ---
                now = time.perf_counter()
                if (not done_any) or (now - last_print >= heartbeat_sec):
                    elapsed = now - t0
                    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    avg = total_sheppard / elapsed if elapsed > 0 else 0.0
                    inst = (
                        (total_sheppard - last_total_sheppard) / (now - last_print)
                        if now > last_print else 0.0
                    )
                    print(
                        f"\r[hb] (Sheppard) {total_sheppard:,} Š-kódů | "
                        f"{total_prufer:,} Prüfer-kódů | "
                        f"průměrně: {avg:,.0f}/s | aktuálně: {inst:,.0f}/s | "
                        f"uplynulo: {elapsed_str} | inflight={len(inflight)}",
                        end="",
                        flush=True,
                    )
                    last_print = now
                    last_total_sheppard = total_sheppard

            # --- добросброс буфера ---
            flush_buf()

        finally:
            out.close()

    elapsed = time.perf_counter() - t0
    print()
    print(
        f"[done] (Sheppard) Zkontrolováno Š-kódů: {total_sheppard:,} | "
        f"zapsaných Prüfer-kódů (včetně involucí): {total_prufer:,} | "
        f"průměrně ~{total_sheppard/elapsed:,.0f}/s | "
        f"výstup v adresáři: {out_dir}, soubory {base_name}_part_XXX.txt",
        flush=True,
    )

    # --- Локальная сортировка КАЖДОГО файла-части, если sort=True ---
    if sort:
        pattern = os.path.join(out_dir, f"{base_name}_part_*.txt")
        part_files = sorted(glob.glob(pattern))

        print(
            f"[sort] sort=True, třídím jednotlivé části "
            f"({len(part_files)} souborů)...",
            flush=True,
        )

        for path in part_files:
            print(f"[sort] -> {os.path.basename(path)}", flush=True)
            sort_and_index_file(path, n)


if __name__ == "__main__":
    # Příklad: spuštění Sheppardového průchodu pro stromy s n vrcholy
    graceful_codes_from_sheppard(n=11, workers=5, max_file_mb=50)
