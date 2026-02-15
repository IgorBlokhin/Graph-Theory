from Diktyonphi import Graph
import Diktyonphi as phi
import glob, time, os
from itertools import islice
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
    center = find_center(tree)
    if len(center) == 0:
        return ""

    if len(center) == 1:
        c = center[0]
        return tree.canonical_code(c, None)

    c1, c2 = center[0], center[1]
    code1 = tree.canonical_code(c1, c2)
    code2 = tree.canonical_code(c2, c1)
    return min(code1, code2)

def degree_signature(g: Graph):
    return tuple(sorted(g.node(i).out_degree for i in g.node_ids()))

def is_isomorphic_degree_signature(tree1: Graph, PATTERN_DEGREE):
    if degree_signature(tree1) != PATTERN_DEGREE:
        return False
    return True

def is_isomorphic_binary_code(tree1: Graph, PATTERN_CANONICAL):
    if min_canonical_code(tree1) != PATTERN_CANONICAL:
        return False
    return True

def check_batch_sheppard(codes_batch, n, involution=None, PATTERN_DEGREE=None, PATTERN_CANONICAL=None):
    """
    Zpracuje dávku Sheppardových kódů a vrátí odpovídající
    Prüferovy kódy stromů, které splňují zadané podmínky.
    
    Parametry:
        codes_batch (list): seznam Sheppardových kódů délky n-1.
        n (int): počet vrcholů.
        involution (bool): zda se má přidat i involutorní kód.
        PATTERN_DEGREE (tuple | None): multimnožina stupňů vzorového stromu.
        PATTERN_CANONICAL (str | None): kanonický kód vzorového stromu.
    
    Návratová hodnota:
        list[tuple[int, ...]]: seznam Prüferových kódů.
    """
    results = []

    for shep in codes_batch:
        if not phi.sheppard_uses_all_vertices(shep, n):
            continue

        tree = phi.from_sheppard(shep)
        if len(tree.dfs_pruchod()) != n:
            continue

        if PATTERN_DEGREE is not None and PATTERN_CANONICAL is not None:
            if is_isomorphic_degree_signature(tree, PATTERN_DEGREE) == False:
                continue

            if is_isomorphic_binary_code(tree, PATTERN_CANONICAL) == False:
                continue
            
        pr = tuple(tree.to_prufer())
        if involution:
            inv_pr = tuple(tree.involute().to_prufer())

        results.append(pr)
        if involution:
            results.append(inv_pr)

    return results

def sort_and_index_file(filepath: str, n: int) -> None:
    """
    Čte soubor, kde každý řádek představuje jeden Prüferův kód jako čísla oddělená mezerou:
        „x1 x2 ... x_{n-2}“

    Podporuje také řádky s již přidaným indexem:
        „<rank> x1 x2 ... x_{n-2}“

    Seřadí podle lexikografického pořadí v Prüferově prostoru a přepíše soubor:
        „<rank>  x1 x2 ... x_{n-2}“

    Index je zarovnán k pravému okraji.
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

            parts = s.split()
            try:
                nums = [int(p) for p in parts]
            except ValueError:
                continue

            if len(nums) == k:
                code = tuple(nums)
            elif len(nums) == k + 1:
                code = tuple(nums[1:])
            else:
                continue

            if any(not (0 <= x <= n-1) for x in code):
                continue

            codes.append(code)

    if not codes:
        print(f"[index] Žádné platné kódy v souboru: {filepath}")
        return

    codes.sort(key=lambda c: phi.prufer_lex_rank(c, n))

    max_rank = phi.prufer_lex_rank(codes[-1], n)
    w = len(str(max_rank))

    with open(filepath, "w", encoding="utf-8") as f:
        for code in codes:
            r = phi.prufer_lex_rank(code, n)
            f.write(f"{r:>{w}}  " + " ".join(map(str, code)) + "\n")

    print(f"[index] Hotovo: seřazeno + index přepsán: {filepath}")

def take_batch(it, batch_size):
    """Vezme z iterátoru další dávku o velikosti batch_size (nebo méně)."""
    return list(islice(it, batch_size))

def graceful_prufer_codes_n(
    n=10,
    pattern: Graph | None = None,
    output_dir: Path | None = None,
    output_file: str | None = None,
    workers: int | None = 2,
    batch_size: int | None = 12000,
    max_inflight: int | None = 12,
    heartbeat_sec: int | None = 4,
    buf_write_every: int | None = 30_000,
    sort: bool | None = False,
    involution: bool | None = True,
    max_file_mb: int | None = 50,
):
    """
    Hlavní paralelní procedura pro generování graciózních 
    Prüferových kódů pro daný počet vrcholů n.

    Vstup:
        n (int): počet vrcholů stromu.
        pattern (Graph | None): volitelný vzorový strom pro filtraci.
        output_dir (Path | None): cílový adresář pro výstupní soubory.
        output_file (str | None): základní název výstupních souborů.
        workers (int): počet paralelních procesů.
        batch_size (int): velikost dávky Sheppardových kódů.
        max_inflight (int): maximální počet paralelně zpracovávaných dávek.
        heartbeat_sec (int): interval výpisu průběžných statistik.
        buf_write_every (int): velikost interního bufferu pro zápis.
        sort (bool): zda mají být výstupní soubory lexikograficky seřazeny.
        involution (bool): zda se mají generovat i involutorní kódy.
        max_file_mb (int): maximální velikost jednoho výstupního souboru.

    Výstup:
        Funkce vrací None.
        Výsledkem jsou textové soubory obsahující všechny nalezené
        graciózní Prüferovy kódy v daném rozsahu.
    """

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = data_dir() / f"n={n}"
        out_dir.mkdir(parents=True, exist_ok=True)

    if output_file is None:
        base_name = f"graceful_sheppard_{n}_sort" if sort else f"graceful_sheppard_{n}"
    else:
        base_name = Path(os.path.abspath(output_file)).stem

    limit_bytes = int(max_file_mb) * 1024 * 1024

    def make_part_path(part_idx: int) -> Path:
        """Cesta k souboru."""
        return out_dir / f"{base_name}_part_{part_idx:03d}.txt"

    total_sheppard = 0
    total_prufer = 0

    t0 = time.perf_counter()
    last_print = t0
    last_total_sheppard = 0

    print(
        f"[main] (Sheppard) n={n}, workers={workers}, batch_size={batch_size}, "
        f"max_inflight={max_inflight}, max_file_mb={max_file_mb}",
        flush=True,
    )

    codes_iter = phi.all_sheppard_codes(n)
    buf = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        part_idx = 1
        out_path = make_part_path(part_idx)
        out = open(out_path, "w", encoding="utf-8")
        current_size = 0

        if pattern:
            PATTERN_DEG = degree_signature(pattern)
            PATTERN_CODE = min_canonical_code(pattern)

        def flush_buf():
            """
            Zapsat buffer do souboru.
            Formát: čísla oddělená mezerou, jeden řádek = jeden Prüferův kód.
            """
            nonlocal out, current_size, part_idx, out_path, buf
            if not buf:
                return

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
            fut_sizes = {}

            while len(inflight) < max_inflight:
                batch = take_batch(codes_iter, batch_size)
                if not batch:
                    break
                if pattern:
                    fut = ex.submit(check_batch_sheppard, batch, n, involution, PATTERN_DEG, PATTERN_CODE)
                else:
                    fut = ex.submit(check_batch_sheppard, batch, n, involution)
                inflight.add(fut)
                fut_sizes[fut] = len(batch)

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

                    batch = take_batch(codes_iter, batch_size)
                    if batch:
                        if pattern:
                            new_fut = ex.submit(check_batch_sheppard, batch, n, involution, PATTERN_DEG, PATTERN_CODE)
                        else:
                            new_fut = ex.submit(check_batch_sheppard, batch, n, involution)
                        inflight.add(new_fut)
                        fut_sizes[new_fut] = len(batch)

                    done_any = True
                    
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
            sort_and_index_file(path, n)

if __name__ == "__main__":
    n=11
    graceful_prufer_codes_n(n=n,
                            workers=6, 
                            involution=True, 
                            batch_size=12000, 
                            output_dir=fr"C:\Users\Igor\Desktop",
                            output_file=f"graceful_prufer_{n}.txt"
    )
