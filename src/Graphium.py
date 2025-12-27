from pathlib import Path
from itertools import product
from collections import namedtuple
from typing import Optional, List

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import dearpygui.dearpygui as dpg
import Diktyonphi as phi  # твой модуль с from_prufer / from_sheppard

# =========================================================
#                ИНИЦИАЛИЗАЦИЯ DPG + ШРИФТ
# =========================================================

dpg.create_context()


def load_font():
    """
    Загружает TTF и добавляет диапазоны Unicode:
    - латиница
    - кириллица
    - Latin Extended-A (č, ř, ě, š, ž, ů, …)
    """
    font_path = Path(__file__).with_name("DejaVuSans.ttf")
    print("Font path:", font_path)
    print("Font exists:", font_path.exists())

    if not font_path.exists():
        print("[font] WARNING: font file not found, using default DearPyGui font.")
        return None

    try:
        with dpg.font_registry():
            with dpg.font(str(font_path), 18) as font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
                dpg.add_font_range(0x0100, 0x017F)  # Latin Extended-A

        print("[font] Loaded font with Unicode ranges successfully.")
        return font
    except Exception as e:
        print(f"[font] ERROR: {e}")
        return None


default_font = load_font()

# =========================================================
#                      ПАРСИНГ КОДА
# =========================================================

def parse_code(text: str):
    """'1 2 3', '1,2,3', '1; 2; 3' -> [1, 2, 3]."""
    for sep in [",", ";", ":", "\t"]:
        text = text.replace(sep, " ")
    return [int(x) for x in text.split() if x.strip()]

# =========================================================
#                       ПЕРЕВОДЫ
# =========================================================

TRANSLATIONS = {
    # ----------------------------------------------------------
    # ČEŠTINA
    # ----------------------------------------------------------
    "cs": {
        # --- Window titles ---
        "main_window": "Hlavní okno",

        # --- Language selector ---
        "lang_label": "Jazyk:",
        "lang_items": ["Čeština", "Русский", "English"],

        # --- Tabs ---
        "tab_prufer": "Prüferův kód",
        "tab_sheppard": "Sheppardův kód",
        "tab_space": "Prostor Prüferových kódů",

        # ===================== Prüfer TAB ======================
        "prufer_input_label": "Zadejte Prüferův kód (hodnoty musí být od 1 do n a délka kódu n-2):",
        "prufer_button": "Zobrazit strom",
        "prufer_hint": "Výsledný strom se zobrazí níže.",
        "prufer_error_invalid": "Neplatný Prüferův kód.",
        "prufer_error_empty": "Nebyl zadán žádný kód.",
        "prufer_error_export": "Chyba při generování obrázku stromu.",
        "show_steps": "Zobrazit kroky",
        "choose_step": "Vyberte krok: ",

        # ===================== Sheppard TAB ====================
        "sheppard_input_label": "Zadejte Sheppardův kód:",
        "sheppard_button": "Zobrazit graf",
        "sheppard_hint": "Výsledný graf se zobrazí níže.",
        "sheppard_error_invalid": "Neplatný Sheppardův kód.",
        "sheppard_error_empty": "Nebyl zadán žádný kód.",
        "sheppard_error_export": "Chyba při generování obrázku stromu.",

        # ===================== Prüfer SPACE TAB ================
        "space_title": "Prostor Prüferových kódů",
        "space_n_label": "Vyberte počet vrcholů n:",
        "space_fixed_axis_label": "Fixovaná osa (pouze pro n = 6):",
        "space_fixed_value_label": "Hodnota na fixované ose:",

        "space_button": "Otevřít 3D vizualizaci",

        # --- 3D viewer strings ---
        "space_window_title": "Prostor Prüferových kódů",
        "space_info_window_title": "Informace o stromu",

        "space_fig_title": "Prostor Prüferových kódů pro n={n}",
        "space_fig_title_4d": "Prostor Prüferových kódů pro n={n} (řez 4D: osa {axis} = {value})",

        "space_click_hint": "Klikněte na bod v prostoru Prüferových kódů pro zobrazení struktury stromu.",

        "space_code_label": "Prüferův kód",
        "space_n_label_info": "n",
        "space_edges_label": "Počet hran",
        "space_edges_list_label": "Hrany",
    },

    # ----------------------------------------------------------
    # РУССКИЙ
    # ----------------------------------------------------------
    "ru": {
        # --- Window titles ---
        "main_window": "Главное окно",

        # --- Language selector ---
        "lang_label": "Язык:",
        "lang_items": ["Čeština", "Русский", "English"],

        # --- Tabs ---
        "tab_prufer": "Код Прюфера",
        "tab_sheppard": "Код Шеппарда",
        "tab_space": "Пространство Прюфера",

        # ===================== Prüfer TAB ======================
        "prufer_input_label": "Введите код Прюфера (значения от 1 до n при длине кода n-2):",
        "prufer_button": "Показать дерево",
        "prufer_hint": "Изображение дерева появится ниже.",
        "prufer_error_invalid": "Некорректный код Прюфера.",
        "prufer_error_empty": "Код не введён.",
        "prufer_error_export": "Ошибка при создании изображения дерева.",
        "show_steps": "Показать шаги",
        "choose_step": "Выберите шаг: ",

        # ===================== Sheppard TAB ====================
        "sheppard_input_label": "Введите код Шеппарда:",
        "sheppard_button": "Показать граф",
        "sheppard_hint": "Изображение графа появится ниже.",
        "sheppard_error_invalid": "Некорректный код Шеппарда.",
        "sheppard_error_empty": "Код не введён.",
        "sheppard_error_export": "Ошибка при создании изображения дерева.",

        # ===================== Prüfer SPACE TAB ================
        "space_title": "Пространство кодов Прюфера",
        "space_n_label": "Выберите число вершин n:",
        "space_fixed_axis_label": "Фиксируемая ось (только для n = 6):",
        "space_fixed_value_label": "Значение фиксируемой оси:",

        "space_button": "Открыть 3D визуализацию",

        # --- 3D viewer strings ---
        "space_window_title": "Пространство Прюфера",
        "space_info_window_title": "Информация o дереве",

        "space_fig_title": "Пространство кодов Прюфера при n = {n}",
        "space_fig_title_4d": "4D-срез: пространство Прюфера при n = {n}, ось {axis} = {value}",

        "space_click_hint": "Щёлкните по точке в пространстве Прюфера, чтобы увидеть структуру дерева.",

        "space_code_label": "Код Прюфера",
        "space_n_label_info": "n",
        "space_edges_label": "Число рёбер",
        "space_edges_list_label": "Рёбра",
    },

    # ----------------------------------------------------------
    # ENGLISH
    # ----------------------------------------------------------
    "en": {
        # --- Window titles ---
        "main_window": "Main window",

        # --- Language selector ---
        "lang_label": "Language:",
        "lang_items": ["Čeština", "Русский", "English"],

        # --- Tabs ---
        "tab_prufer": "Prüfer code",
        "tab_sheppard": "Sheppard code",
        "tab_space": "Prüfer space",

        # ===================== Prüfer TAB ======================
        "prufer_input_label": "Enter Prüfer code (values must be from 1 to n and the length of the code n-2):",
        "prufer_button": "Show tree",
        "prufer_hint": "The tree image will appear below.",
        "prufer_error_invalid": "Invalid Prüfer code.",
        "prufer_error_empty": "No Prüfer code entered.",
        "prufer_error_export": "Failed to generate tree image.",
        "show_steps": "Show steps",
        "choose_step": "Choose step: ",

        # ===================== Sheppard TAB ====================
        "sheppard_input_label": "Enter Sheppard code:",
        "sheppard_button": "Show graph",
        "sheppard_hint": "The graph image will appear below.",
        "sheppard_error_invalid": "Invalid Sheppard code.",
        "sheppard_error_empty": "No Sheppard code entered.",
        "sheppard_error_export": "Failed to generate tree image.",

        # ===================== Prüfer SPACE TAB ================
        "space_title": "Prüfer code space",
        "space_n_label": "Choose number of vertices n:",
        "space_fixed_axis_label": "Fixed axis (only for n = 6):",
        "space_fixed_value_label": "Value on fixed axis:",

        "space_button": "Open 3D viewer",

        # --- 3D viewer strings ---
        "space_window_title": "Prüfer space",
        "space_info_window_title": "Tree info",

        "space_fig_title": "Prüfer space for n={n}",
        "space_fig_title_4d": "Prüfer space for n={n} (4D slice: axis {axis} = {value})",

        "space_click_hint": "Click a point in the Prüfer space to view the tree structure here.",

        "space_code_label": "Prüfer code",
        "space_n_label_info": "n",
        "space_edges_label": "Number of edges",
        "space_edges_list_label": "Edges",
    },
}

current_lang = "cs"
current_codes = {
                "prufer": {"code": None, "by_steps": None},
                "sheppard": {"code": None, "by_steps": None}
                }
t = TRANSLATIONS[current_lang]

def apply_language(lang: str):
    global current_lang
    current_lang = lang
    t = TRANSLATIONS[lang]

    dpg.set_item_label("main_window", t["main_window"])

    dpg.set_value("lang_label_text", t["lang_label"])
    dpg.configure_item("lang_combo", items=t["lang_items"])

    if lang == "cs":
        dpg.set_value("lang_combo", "Čeština")
    elif lang == "ru":
        dpg.set_value("lang_combo", "Русский")
    else:
        dpg.set_value("lang_combo", "English")

    dpg.set_item_label("prufer_tab", t["tab_prufer"])
    dpg.set_item_label("sheppard_tab", t["tab_sheppard"])
    dpg.set_item_label("space_tab", t["tab_space"])

    # Prüfer
    dpg.set_value("prufer_input_label", t["prufer_input_label"])
    dpg.set_item_label("prufer_button", t["prufer_button"])
    dpg.set_value("prufer_hint", t["prufer_hint"])
    if dpg.does_item_exist("show_steps_prufer"):
        dpg.set_item_label("show_steps_prufer", t["show_steps"])
    if dpg.does_item_exist("choose_step_prufer"):
        dpg.set_value("choose_step_prufer", t["choose_step"])

    # Sheppard
    dpg.set_value("sheppard_input_label", t["sheppard_input_label"])
    dpg.set_item_label("sheppard_button", t["sheppard_button"])
    dpg.set_value("sheppard_hint", t["sheppard_hint"])
    if dpg.does_item_exist("show_steps_sheppard"):
        dpg.set_item_label("show_steps_sheppard", t["show_steps"])
    if dpg.does_item_exist("choose_step_sheppard"):
        dpg.set_value("choose_step_sheppard", t["choose_step"])

    # Prüfer space
    dpg.set_value("space_n_label", t["space_n_label"])
    dpg.set_value("space_fixed_axis_label", t["space_fixed_axis_label"])
    dpg.set_value("space_fixed_value_label", t["space_fixed_value_label"])
    dpg.set_item_label("space_button", t["space_button"])


def on_language_change(sender, app_data, user_data):
    if app_data == "Čeština":
        apply_language("cs")
    elif app_data == "Русский":
        apply_language("ru")
    else:
        apply_language("en")


# =========================================================
#       Prüfer / Sheppard → дерево → PNG для вкладок
# =========================================================

def check_if_code_is_new(code, code_type: str|str, type_on_steps: bool = False):
    global current_codes
    key = "by_steps" if type_on_steps else "code"
    if current_codes[code_type][key] == code:
        return False
    current_codes[code_type][key] = code
    return True

def show_graph(sender, app_data, user_data):
    code_type = user_data
    text = dpg.get_value(f"{code_type}_input")
    try:
        code = parse_code(text)
    except ValueError:
        return
    
    if check_if_code_is_new(code, code_type) is False:
        return

    if dpg.does_item_exist(f"show_steps_{code_type}"):
        dpg.delete_item(f"show_steps_{code_type}")
    if dpg.does_item_exist(f"step_number_{code_type}"):
        dpg.delete_item(f"step_number_{code_type}")
    if dpg.does_item_exist(f"choose_step_{code_type}"):
        dpg.delete_item(f"choose_step_{code_type}")
    clear_step(code_type)

    if code_type == "sheppard":
        graph = phi.from_sheppard(code)
    else:
        graph = phi.from_prufer(code)
    img_path = graph.export_to_png(code_type=code_type, dark=True)
    width, height, channels, data = dpg.load_image(img_path)

    if dpg.does_item_exist(f"{code_type}_image"):
        dpg.delete_item(f"{code_type}_image")
    if dpg.does_item_exist(f"{code_type}_texture"):
        dpg.delete_item(f"{code_type}_texture")

    with dpg.texture_registry():
        dpg.add_static_texture(width, height, data, tag=f"{code_type}_texture")

    dpg.add_image(f"{code_type}_texture", label="main", parent=f"{code_type}_tab", tag=f"{code_type}_image")
    dpg.add_button(tag=f"show_steps_{code_type}", 
                   label=TRANSLATIONS[current_lang]["show_steps"], 
                   parent=f"{code_type}_tab", 
                   callback=on_show_steps, 
                   user_data={"code_type": code_type, 
                              "code": code})

def clear_step(code_type: str, DoNotDeleteMain: bool = False):
    # children[1] — это "item slot 1" = контейнер для дочерних элементов
    for child in dpg.get_item_children(f"{code_type}_tab", 1):
        if dpg.get_item_type(child) == "mvAppItemType::mvImage":
            if dpg.get_item_label(child) == "main" and DoNotDeleteMain is True:
                continue
            dpg.delete_item(child)

def on_show_steps(sender, app_data, user_data):
    code_data=user_data
    if check_if_code_is_new(code_data["code"], code_data["code_type"], type_on_steps=True) is False:
        return
    if dpg.does_item_exist(f"choose_step_{code_data['code_type']}"):
        dpg.delete_item(f"choose_step_{code_data['code_type']}")
    dpg.add_text(TRANSLATIONS[current_lang]["choose_step"], parent=f"{code_data['code_type']}_tab", tag=f"choose_step_{code_data['code_type']}")
    dpg.add_slider_int(label="", 
                       min_value=1, 
                       max_value=(len(code_data["code"]) + 1 if code_data["code_type"] == "prufer" else len(code_data["code"])), 
                       default_value=1, 
                       tag=f"step_number_{code_data['code_type']}", 
                       parent=f"{code_data['code_type']}_tab", 
                       callback=change_step, 
                       user_data=code_data['code_type'])
    if code_data["code_type"] == "prufer":
        phi.from_prufer(code_data["code"], steps=True)
    else:
        phi.from_sheppard(code_data["code"], steps=True)

def change_step(sender, app_data, user_data):
    dpg.set_y_scroll("main_window", 10**9)
    step_number = dpg.get_value(f"step_number_{user_data}")
    clear_step(user_data, DoNotDeleteMain=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), user_data)
    path = os.path.join(path, f"graph_{step_number}_step.png")
    print(path)
    width, height, channels, data = dpg.load_image(path)
    with dpg.texture_registry():
        if dpg.does_item_exist(path):
            dpg.delete_item(path)
        dpg.add_static_texture(width, height, data, tag=path)

    dpg.add_image(path, parent=f"{user_data}_tab")
    dpg.set_y_scroll("main_window", 10**9)

# =========================================================
#           3D-ПРОСТРАНСТВО PRÜFER-КОДОВ (MATPLOTLIB)
# =========================================================

Point = namedtuple("Point", ["code", "x", "y", "z"])

def _scale_coord(v: int, n: int) -> float:
    """
    Переводим v из {1, ..., n} в [0, 1].
    1  -> 0.0
    n  -> 1.0
    """
    if n <= 1:
        return 0.0
    return (v - 1) / (n - 1)

def generate_prufer_space_points(
    n: int,
    fixed_axis: Optional[int] = None,
    fixed_value: Optional[int] = None,
) -> List[Point]:
    n = int(n)
    if n < 3 or n > 6:
        raise ValueError("n must be between 3 and 6")

    if fixed_axis is not None:
        fixed_axis = int(fixed_axis)
        fixed_value = int(fixed_value)

    L = n - 2
    coords_range = range(1, n + 1)
    points: List[Point] = []

    for code in product(coords_range, repeat=L):
        code = list(code)

        if L == 4:
            if fixed_axis is None or fixed_value is None:
                continue
            if code[fixed_axis] != fixed_value:
                continue
            free_axes = [i for i in range(4) if i != fixed_axis]
            c_x = code[free_axes[0]]
            c_y = code[free_axes[1]]
            c_z = code[free_axes[2]]
        else:
            padded = code + [1] * (3 - L)
            c_x, c_y, c_z = padded[:3]

        x = _scale_coord(c_x, n)
        y = _scale_coord(c_y, n)
        z = _scale_coord(c_z, n)

        points.append(Point(tuple(code), x, y, z))

    return points


def draw_axes(ax, length=1.2, labels=None):
    """
    Большие цветные стрелки осей.
    labels: список или кортеж из трёх строк, например ["1", "2", "3"].
    """
    if labels is None:
        labels = ("X", "Y", "Z")

    # X
    ax.quiver(0, 0, 0, length, 0, 0, color='r', linewidth=2)
    ax.text(length + 0.2, 0, 0, str(labels[0]), color='r', fontsize=12)

    # Y
    ax.quiver(0, 0, 0, 0, length, 0, color='g', linewidth=2)
    ax.text(0, length + 0.2, 0, str(labels[1]), color='g', fontsize=12)

    # Z
    ax.quiver(0, 0, 0, 0, 0, length, color='b', linewidth=2)
    ax.text(0, 0, length + 0.2, str(labels[2]), color='b', fontsize=12)

def plot_prufer_space_with_info(
    n: int,
    fixed_axis: Optional[int] = None,
    fixed_value: Optional[int] = None,
):
    points = generate_prufer_space_points(n, fixed_axis, fixed_value)
    if not points:
        print("No points generated — check n / fixed_axis / fixed_value.")
        return

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    codes = [p.code for p in points]

    L = n - 2  # длина Prüfer-кода

    # по умолчанию считаем, что на X,Y,Z стоят позиции 1,2,3 кода
    axis_positions = [1, 2, 3]

    if L == 4 and fixed_axis is not None and fixed_value is not None:
        # те же free_axes, что и в generate_prufer_space_points
        free_axes = [i for i in range(4) if i != fixed_axis]
        # индексы в человекочитаемом виде: 1..4
        axis_positions = [i + 1 for i in free_axes]

    # --- Окно 1: 3D-пространство ---
    fig_space = plt.figure(t["space_window_title"], figsize=(7, 6))
    ax3d = fig_space.add_subplot(111, projection="3d")

    # оси с номерами позиций Prüfer-кода
    axis_labels = [str(p) for p in axis_positions]
    draw_axes(ax3d, length=1.0, labels=axis_labels)

    sc = ax3d.scatter(xs, ys, zs, s=30, picker=True)

    # убираем тики и подписи
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    if (n - 2) == 4 and fixed_axis is not None and fixed_value is not None:
    # +1, чтобы в заголовке было 1..4, а не 0..3
        title = t["space_fig_title_4d"].format(
            n=n,
            axis=fixed_axis + 1,
            value=fixed_value
        )
    else:
        title = t["space_fig_title"].format(n=n)

    ax3d.set_title(title)

    # --- Окно 2: инфо + картинка дерева ---
    fig_info = plt.figure(t["space_info_window_title"], figsize=(6, 8))
    gs = fig_info.add_gridspec(2, 1, height_ratios=[1, 3])

    ax_text = fig_info.add_subplot(gs[0])
    ax_img = fig_info.add_subplot(gs[1])
    ax_text.axis("off")
    ax_img.axis("off")

    ax_text.text(
        0.0,
        1.0,
        t["space_click_hint"],
        va="top",
        ha="left",
        fontsize=10,
    )
    fig_info.tight_layout()

    def on_pick(event):
        ind = event.ind[0]
        code = codes[ind]
        code_list = list(code)

        tree = phi.from_prufer(code_list)

        if hasattr(tree, "_edges"):
            edges = list(tree._edges)
            edges_str = "\n".join(f"{u}–{v}" for (u, v) in edges)
            m = len(edges)
        else:
            edges_str = "(no _edges attribute)"
            m = 0

        img_path = tree.export_to_png()
        img = plt.imread(img_path)

        ax_text.clear()
        ax_img.clear()
        ax_text.axis("off")
        ax_img.axis("off")

        txt = (
            f'{t["space_code_label"]}: {code_list}\n'
            f'{t["space_n_label_info"]} = {n}\n'
            f'{t["space_edges_label"]}: {m}\n\n'
            f'{t["space_edges_list_label"]}:\n{edges_str}'
        )
        ax_text.text(
            0.0,
            1.0,
            txt,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )
        ax_img.imshow(img)

        fig_info.tight_layout()
        fig_info.canvas.draw_idle()

    fig_space.canvas.mpl_connect("pick_event", on_pick)

    plt.show()


# =========================================================
#            CALLBACK ДЛЯ ТАБА "PRÜFER SPACE"
# =========================================================

def on_space_n_change(sender, app_data, user_data):
    """Показываем/скрываем фиксацию координаты только для n=6."""
    n = int(app_data)
    show = (n == 6)
    for tag in ["space_fixed_axis_label", "space_fixed_axis",
                "space_fixed_value_label", "space_fixed_value"]:
        dpg.configure_item(tag, show=show)


def on_space_open_viewer(sender, app_data, user_data):
    n = dpg.get_value("space_n")
    try:
        n = int(n)
    except (TypeError, ValueError):
        return

    fixed_axis = None
    fixed_value = None

    if n == 6:
        axis_val = dpg.get_value("space_fixed_axis")
        fixed_val = dpg.get_value("space_fixed_value")
        try:
            axis_val = int(axis_val)
            fixed_val = int(fixed_val)
        except (TypeError, ValueError):
            return

        fixed_axis = axis_val - 1   # <-- 1..4 → 0..3
        fixed_value = fixed_val

    plot_prufer_space_with_info(n, fixed_axis, fixed_value)


# =========================================================
#                        GUI
# =========================================================

with dpg.window(label="Graphium", tag="main_window", width=1000, height=700):

    with dpg.group(horizontal=True):
        dpg.add_text("", tag="lang_label_text")
        dpg.add_combo(
            items=["Čeština", "Русский", "English"],
            tag="lang_combo",
            default_value="Čeština",
            width=140,
            callback=on_language_change
        )

    dpg.add_separator()

    with dpg.tab_bar():

        # --- Prüfer tab ---
        with dpg.tab(label="Prüfer", tag="prufer_tab"):
            dpg.add_text("", tag="prufer_input_label")
            dpg.add_input_text(tag="prufer_input", width=300, default_value="")
            dpg.add_button(label="Show Prüfer tree", tag="prufer_button", callback=show_graph, user_data="prufer")
            dpg.add_separator()
            dpg.add_text("", tag="prufer_hint")

        # --- Sheppard tab ---
        with dpg.tab(label="Sheppard", tag="sheppard_tab"):
            dpg.add_text("", tag="sheppard_input_label")
            dpg.add_input_text(tag="sheppard_input", width=300, default_value="")
            dpg.add_button(label="Show Sheppard tree", tag="sheppard_button", callback=show_graph, user_data="sheppard")
            dpg.add_separator()
            dpg.add_text("", tag="sheppard_hint")

        # --- Prüfer space tab ---
        with dpg.tab(label="Prüfer space", tag="space_tab"):
            dpg.add_text("", tag="space_n_label")
            dpg.add_combo(
                items=[3, 4, 5, 6],
                tag="space_n",
                default_value=4,
                callback=on_space_n_change
            )

            dpg.add_text("", tag="space_fixed_axis_label")
            dpg.add_combo(
                items=[1, 2, 3, 4],
                tag="space_fixed_axis",
                default_value=1
            )

            dpg.add_text("", tag="space_fixed_value_label")
            dpg.add_combo(
                items=[1, 2, 3, 4, 5, 6],
                tag="space_fixed_value",
                default_value=1
            )

            dpg.add_spacer(height=10)
            dpg.add_button(label="Open 3D viewer", tag="space_button",
                           callback=on_space_open_viewer)

# по умолчанию n=4 → фиксация координат скрыта
for tag in ["space_fixed_axis_label", "space_fixed_axis",
            "space_fixed_value_label", "space_fixed_value"]:
    dpg.configure_item(tag, show=False)


# =========================================================
#                     ЗАПУСК ПРИЛОЖЕНИЯ
# =========================================================

dpg.create_viewport(title="Graphium", width=1020, height=740)

apply_language("cs")

dpg.setup_dearpygui()
dpg.show_viewport()

if default_font is not None:
    dpg.bind_font(default_font)

dpg.start_dearpygui()
dpg.destroy_context()
