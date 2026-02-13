from Diktyonphi import GraphType, Node, Graph, Edge
import Diktyonphi as phi
import glob, time, os
from pathlib import Path
from paths import data_dir
from concurrent.futures import ProcessPoolExecutor, as_completed

def find_center(tree: Graph):
    copy_tree = tree.copy()
    if len(copy_tree._nodes) in [0, 1, 2]:
        return list(copy_tree.node_ids())
    leaves = [copy_tree.node(id).id for id in copy_tree.node_ids() if copy_tree.node(id).out_degree==1]

    def delete_leaves(leaves):
        new_leaves = []
        for leaf in leaves:
            neighbor_id = next(copy_tree.node(leaf).neighbor_ids)
            copy_tree.del_node(leaf)
            if copy_tree.node(neighbor_id).out_degree == 1:
                new_leaves.append(neighbor_id)

        if len(copy_tree._nodes) in [0, 1, 2]:
            return list(copy_tree.node_ids())
        
        return delete_leaves(new_leaves)

    return delete_leaves(leaves)

def min_canonical_code(tree: Graph):
    center = find_center(tree)  # список ids
    if len(center) == 0:
        return ""  # или raise, но пусть будет безопасно

    if len(center) == 1:
        c = center[0]
        return tree.canonical_code(c, None)

    # два центра: кодируем “в обе стороны” через parent
    c1, c2 = center[0], center[1]
    code1 = tree.canonical_code(c1, c2)
    code2 = tree.canonical_code(c2, c1)
    return min(code1, code2)   # лексикографически

def degree_signature(g: Graph):
    # для каждого id берём out_degree и сортируем
    return tuple(sorted(g.node(i).out_degree for i in g.node_ids()))

def is_isomorphic(tree1: Graph, PATTERN_DEGREE, PATTERN_CANONICAL):
    if degree_signature(tree1) != PATTERN_DEGREE:
        return False
    tree1_code = min_canonical_code(tree1)
    if tree1_code != PATTERN_CANONICAL:
        return False
    return True

def check_batch_sheppard(codes_batch, n, involution=None, PATTERN_DEGREE=None, PATTERN_CANONICAL=None):
    """
    На вход: список Sheppard-кодов длины n-1.
    На выход: ПЛОСКИЙ список Prüfer-кодов:
        [pr1, pr1_inv, pr2, pr2_inv, ...]
    где каждый pr* — это tuple длины n-2.
    """
    results = []

    for shep in codes_batch:
        # 1) используют ли все вершины?
        if not phi.sheppard_uses_all_vertices(shep, n):
            continue

        # 2) строим дерево
        tree = phi.from_sheppard(shep)
        if len(tree.dfs_pruchod()) != n:
            continue

        # 3) если задан паттерн, проверить соответсвует ли ему полученное дерево
        if PATTERN_DEGREE is not None and PATTERN_CANONICAL is not None:
            if is_isomorphic(tree, PATTERN_DEGREE, PATTERN_CANONICAL) == False:
                continue

        # 4) если дошли сюда — Sheppard-код задаёт грациозную разметку дерева
        pr = tuple(tree.to_prufer())
        if involution:
            inv_pr = tuple(tree.involute().to_prufer())

        # добавляем ОБА Prüfer-кода как отдельные записи
        results.append(pr)
        if involution:
            results.append(inv_pr)

    return results

def graceful_codes_from_sheppard(
    n=10,
    pattern: "Graph" = None,
    output_dir: str | Path | None = None,
    output_file=None,
    workers=2,
    batch_size=12000,
    max_inflight=12,
    heartbeat_sec=4.0,
    buf_write_every=30_000,
    sort=False,
    involution=True,
    max_file_mb=50,   # Лимит размера ОДНОГО файла в мегабайтах
):
    """
    Параллельный проход по Sheppard-кодам для n:
      - sheppard_uses_all_vertices(code, n)
      - дерево (len(dfs_pruchod()) == n)
      - tree.is_graceful()

    Для КАЖДОГО найденного дерева записываем:
      - его Prüfer-код
      - Prüfer-код дерева-инволюции (если involution=True)

    Формат вывода:
      - создаётся несколько файлов:
            {base_name}_part_001.txt
            {base_name}_part_002.txt
            ...
      - каждая строка: один Prüfer-код в виде "... ... ...".
      - как только размер текущего файла превышает max_file_mb МБ,
        открывается новый файл (номер части ++).
    """

    # --- Определяем out_dir (КУДА сохраняем) ---
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = data_dir() / f"n={n}"
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- Определяем base_name (КАК называем файлы) ---
    if output_file is None:
        base_name = f"graceful_sheppard_{n}_sort" if sort else f"graceful_sheppard_{n}"
    else:
        # Если задан output_file — используем его имя (без расширения) как base_name
        # ВАЖНО: каталог берём из output_dir (если задан), иначе всё равно out_dir выше.
        base_name = Path(os.path.abspath(output_file)).stem

    # Лимит размера одной части в байтах
    limit_bytes = int(max_file_mb) * 1024 * 1024

    def make_part_path(part_idx: int) -> Path:
        """Путь к файлу очередной части."""
        return out_dir / f"{base_name}_part_{part_idx:03d}.txt"

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

    codes_iter = phi.all_sheppard_codes(n)  # генерируем ПЕРВУЮ половину Sheppard-кодов
    buf = []

    # --- Параллельный проход + запись с делением на части ---
    with ProcessPoolExecutor(max_workers=workers) as ex:
        part_idx = 1
        out_path = make_part_path(part_idx)
        out = open(out_path, "w", encoding="utf-8")
        current_size = 0  # сколько байт уже записано в текущий файл

        if pattern:
            PATTERN_DEG = degree_signature(pattern)
            PATTERN_CODE = min_canonical_code(pattern)

        def flush_buf():
            """
            Сбросить буфер в файл.
            Формат: числа, разделённые пробелом, одна строка = один Prüfer-код.
            """
            nonlocal out, current_size, part_idx, out_path, buf
            if not buf:
                return

            lines = [" ".join(str(x) for x in pr) for pr in buf]
            text = "\n".join(lines) + "\n"

            data = text.encode("utf-8")
            size = len(data)

            # если переполняем текущую часть — открываем новую
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
                batch = phi.take_batch(codes_iter, batch_size)
                if not batch:
                    break
                if pattern:
                    fut = ex.submit(check_batch_sheppard, batch, n, involution, PATTERN_DEG, PATTERN_CODE)
                else:
                    fut = ex.submit(check_batch_sheppard, batch, n, involution)
                inflight.add(fut)
                fut_sizes[fut] = len(batch)

            # --- основной цикл обработки результатов ---
            while inflight:
                done_any = False

                for fut in as_completed(list(inflight), timeout=heartbeat_sec):
                    inflight.remove(fut)
                    batch_size_done = fut_sizes.pop(fut, 0)

                    total_sheppard += batch_size_done

                    prufer_list = fut.result()
                    total_prufer += len(prufer_list)

                    for pr in prufer_list:
                        buf.append(pr)
                        if len(buf) >= buf_write_every:
                            flush_buf()

                    batch = phi.take_batch(codes_iter, batch_size)
                    if batch:
                        if pattern:
                            new_fut = ex.submit(check_batch_sheppard, batch, n, involution, PATTERN_DEG, PATTERN_CODE)
                        else:
                            new_fut = ex.submit(check_batch_sheppard, batch, n, involution)
                        inflight.add(new_fut)
                        fut_sizes[new_fut] = len(batch)

                    done_any = True

                # --- heartbeat ---
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

            flush_buf()

        finally:
            out.close()

    elapsed = time.perf_counter() - t0
    print()

    written = (
        f"zapsaných Prüfer-kódů (včetně involucí): {total_prufer:,} | "
        if involution else
        f"zapsaných Prüfer-kódů (bez involucí): {total_prufer:,} | "
    )

    print(
        f"[done] (Sheppard) Zkontrolováno Š-kódů: {total_sheppard:,} | "
        f"{written}"
        f"průměrně ~{total_sheppard/elapsed:,.0f}/s | "
        f"výstup v adresáři: {out_dir} | soubory: {base_name}_part_XXX.txt",
        flush=True,
    )

    # --- Сортировка частей, если sort=True ---
    if sort:
        pattern_glob = str(out_dir / f"{base_name}_part_*.txt")
        part_files = sorted(glob.glob(pattern_glob))

        print(
            f"[sort] sort=True, třídím jednotlivé části "
            f"({len(part_files)} souborů)...",
            flush=True,
        )

        for path in part_files:
            print(f"[sort] -> {os.path.basename(path)}", flush=True)
            phi.sort_and_index_file(path, n)

if __name__ == "__main__":
    n=12
    graceful_codes_from_sheppard(n=n,
                                 workers=6, 
                                 involution=False, 
                                 batch_size=12000, 
                                 output_dir=fr"C:\Users\Igor\Desktop\Python-programs\bachelors\data\n={n}",
                                 output_file=f"graceful_sheppard_no_involution{n}.txt"
    )
