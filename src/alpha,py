import re
from pathlib import Path
from collections import deque
from typing import List, Dict, Tuple, Optional


_int_re = re.compile(r"-?\d+")

def prufer_to_tree_adj(code: List[int], n: int) -> Dict[int, List[int]]:
    deg = [1] * n
    for x in code:
        deg[x] += 1

    leaves = [i for i in range(n) if deg[i] == 1]
    leaves.sort()

    adj = {i: [] for i in range(n)}

    def pop_smallest_leaf() -> int:
        return leaves.pop(0)

    for x in code:
        leaf = pop_smallest_leaf()
        adj[leaf].append(x)
        adj[x].append(leaf)

        deg[leaf] -= 1
        deg[x] -= 1

        if deg[x] == 1:
            # вставка x в отсортированный список leaves
            lo, hi = 0, len(leaves)
            while lo < hi:
                mid = (lo + hi) // 2
                if leaves[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            leaves.insert(lo, x)

    u, v = leaves
    adj[u].append(v)
    adj[v].append(u)
    return adj

def bipartition(adj: Dict[int, List[int]], n: int) -> Tuple[List[int], List[int]]:
    color = [-1] * n
    q = deque([0])
    color[0] = 0

    while q:
        v = q.popleft()
        for u in adj[v]:
            if color[u] == -1:
                color[u] = 1 - color[v]
                q.append(u)

    part0 = [i for i, c in enumerate(color) if c == 0]
    part1 = [i for i, c in enumerate(color) if c == 1]
    return part0, part1

def is_alpha_for_labeled_tree(adj: Dict[int, List[int]], n: int) -> bool:
    part0, part1 = bipartition(adj, n)
    max0, min0 = max(part0), min(part0)
    max1, min1 = max(part1), min(part1)
    return (max0 < min1) or (max1 < min0)

def filter_alpha_prufer_codes_to_file(
    input_path_or_dir: str | Path,
    *,
    expected_len: int,                 # длина Prüfer-кода = n-2
    files_pattern: str = "*_part_*.txt",
    drop_leading_index_if_present: bool = True,

    # --- ВЫХОД ---
    # Вариант 1: сохранить в один файл
    output_path: str | Path | None = None,

    # Вариант 2: сохранить в несколько частей (если output_path=None)
    output_dir: str | Path | None = None,
    output_base: str = "alpha_codes",
    max_file_mb: int = 50,

    buf_write_every: int = 50_000,
) -> Dict[str, int]:
    """
    Читает ваши файлы с грациозными Prüfer-кодами, оставляет только α-разметки,
    и СРАЗУ сохраняет их в указанный файл/файлы.

    - Если output_path задан (например ".../alpha_n12.txt") -> пишем в ОДИН файл.
    - Иначе пишем в output_dir (или рядом с входом) в несколько файлов *_part_001.txt...,
      с ограничением max_file_mb на файл.

    Возвращает мета-статистику.
    """
    input_path_or_dir = Path(input_path_or_dir)
    n = expected_len + 2

    # список входных файлов
    if input_path_or_dir.is_dir():
        files = sorted(input_path_or_dir.glob(files_pattern))
        in_base_dir = input_path_or_dir
    else:
        files = [input_path_or_dir]
        in_base_dir = input_path_or_dir.parent

    if not files:
        raise FileNotFoundError(f"No files matched {files_pattern} in {input_path_or_dir}")

    meta = {
        "files": len(files),
        "lines_read": 0,
        "codes_ok": 0,
        "codes_bad": 0,
        "codes_stripped_index": 0,
        "alpha_yes": 0,
        "alpha_no": 0,
        "out_files": 0,
        "out_path_mode": 0,  # 1=single file, 2=multi-part
    }

    # --- подготовка вывода ---
    buf: List[str] = []

    if output_path is not None:
        # ОДИН выходной файл
        out_fp = Path(output_path)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_fp.open("w", encoding="utf-8")
        meta["out_files"] = 1
        meta["out_path_mode"] = 1

        def flush():
            if not buf:
                return
            out_f.write("".join(buf))
            buf.clear()
            out_f.flush()

        def close_out():
            flush()
            out_f.close()

    else:
        # МНОГОЧАСТНЫЙ выход (как раньше)
        out_dir = Path(output_dir) if output_dir is not None else in_base_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        part_idx = 1
        out_fp = out_dir / f"{output_base}_part_{part_idx:03d}.txt"
        out_f = out_fp.open("w", encoding="utf-8")
        meta["out_files"] = 1
        meta["out_path_mode"] = 2

        def flush():
            nonlocal out_f, out_fp, part_idx
            if not buf:
                return
            out_f.write("".join(buf))
            buf.clear()
            out_f.flush()

            # ограничение на размер
            if out_fp.stat().st_size >= max_file_mb * 1024 * 1024:
                out_f.close()
                part_idx += 1
                out_fp = out_dir / f"{output_base}_part_{part_idx:03d}.txt"
                out_f = out_fp.open("w", encoding="utf-8")
                meta["out_files"] += 1

        def close_out():
            nonlocal out_f
            flush()
            out_f.close()

    # --- основной проход ---
    file_number = 1
    for fp in files:
        print(f"File number is {file_number}")
        file_number += 1
        with fp.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                meta["lines_read"] += 1

                nums = list(map(int, _int_re.findall(line)))
                if drop_leading_index_if_present and len(nums) == expected_len + 1:
                    meta["codes_stripped_index"] += 1
                    nums = nums[1:]

                if len(nums) != expected_len:
                    meta["codes_bad"] += 1
                    continue

                meta["codes_ok"] += 1
                code = nums

                adj = prufer_to_tree_adj(code, n)
                if is_alpha_for_labeled_tree(adj, n):
                    meta["alpha_yes"] += 1
                    buf.append(" ".join(map(str, code)) + "\n")
                    if len(buf) >= buf_write_every:
                        flush()
                else:
                    meta["alpha_no"] += 1

    close_out()
    return meta

n = 10
meta = filter_alpha_prufer_codes_to_file(
    input_path_or_dir=rf"C:\Users\Igor\Desktop\Python-programs\bachelors\data\n={n}",
    files_pattern=f"graceful_sheppard_{n}_part_*.txt",   # <-- шаблон
    expected_len=n-2,
    output_path=rf"C:\Users\Igor\Desktop\Python-programs\bachelors\data\n={n}\alpha_graceful_{n}.txt",
)

print(meta)
