from __future__ import annotations

from itertools import product
from collections import namedtuple
from typing import Optional, List
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import dearpygui.dearpygui as dpg

import Diktyonphi as phi  # твой модуль с from_prufer / from_sheppard
from paths import steps_dir, resource_path


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
    font_path = resource_path("assets/font/DejaVuSans.ttf")
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
        print("[font] Loaded font successfully.")
        return font
    except Exception as e:
        print(f"[font] ERROR: {e}")
        return None

# =========================================================
#                      ПАРСИНГ КОДА
# =========================================================

def parse_code(text: str):
    """'1 2 3', '1,2,3', '1; 2; 3' -> [1, 2, 3]."""
    for sep in [",", ";", ":", "\t"]:
        text = text.replace(sep, " ")
    return [int(x) for x in text.split() if x.strip()]

def parse_edges(text: str):
    """"'1 2 2 4', '1,2,2,4', '1;2;2;4' -> [(1, 2), (2, 4)]."""
    for sep in [",", ";", ":", "\t"]:
        text = text.replace(sep, " ")
    list_of_vertices = text.split()
    list_of_edges = []
    for i in range(len(list_of_vertices)):
        if i % 2 == 0:
            list_of_edges.append((int(list_of_vertices[i]), int(list_of_vertices[i + 1])))
    return list_of_edges

# =========================================================
#                       ПЕРЕВОДЫ
# =========================================================

TRANSLATIONS = {
    "cs": {
        "main_window": "Hlavní okno",
        "lang_label": "Jazyk:",
        "lang_items": ["Čeština", "Русский", "English"],
        "prufer_tab": "Prüferův kód",
        "sheppard_tab": "Sheppardův kód",
        "from_prufer_to_graph": "Graf z kódu",
        "from_sheppard_to_graph": "Graf z kódu",
        "from_graph_to_sheppard": "Kód z grafu",
        "from_graph_to_prufer": "Kód z grafu",
        "space_tab": "Prostor Prüferových kódů",
        "prufer_input_label": "Zadejte Prüferův kód (hodnoty musí být od 0 do n-1 a délka kódu n-2):",
        "prufer_button": "Zobrazit strom",
        "prufer_hint": "Výsledný strom se zobrazí níže.",
        "prufer_code_announcment_label": "Prüferův kód:",
        "sheppard_code_announcment_label": "Sheppardův kód:",
        "show_code_button": "Ukazat kód",
        "prufer_error_invalid": "Neplatný Prüferův kód.",
        "prufer_error_empty": "Nebyl zadán žádný kód.",
        "prufer_error_export": "Chyba při generování obrázku stromu.",
        "tree_input_label": "Zadejte hrany stromu:",
        "graph_input_label": "Zadejte hrany graphu:",
        "show_sheppard_code_button": "Ukazat kód",
        "show_prufer_code_button": "Ukazat kód",
        "show_steps": "Zobrazit kroky",
        "choose_step": "Vyberte krok: ",
        "sheppard_input_label": "Zadejte Sheppardův kód:",
        "sheppard_button": "Zobrazit graf",
        "sheppard_hint": "Výsledný graf se zobrazí níže.",
        "sheppard_error_invalid": "Neplatný Sheppardův kód.",
        "sheppard_error_empty": "Nebyl zadán žádný kód.",
        "sheppard_error_export": "Chyba při generování obrázku stromu.",
        "graceful_error": "Ohodnocení není graciózní, proto pro něj neexistuje Sheppardův kód.",
        "space_title": "Prostor Prüferových kódů",
        "space_n_label": "Vyberte počet vrcholů n:",
        "space_fixed_axis_label": "Fixovaná osa (pouze pro n = 6):",
        "space_fixed_value_label": "Hodnota na fixované ose:",
        "space_button": "Otevřít 3D vizualizaci",
        "space_window_title": "Prostor Prüferových kódů",
        "space_info_window_title": "Informace o stromu",
        "space_fig_title": "Prostor Prüferových kódů pro n={n}",
        "space_fig_title_4d": "Prostor Prüferových kódů pro n={n} (řez 4D: osa {axis} = {value})",
        "space_click_hint": "Klikněte na bod v prostoru Prüferových kódů pro zobrazení struktury stromu.",
        "space_code_label": "Prüferův kód",
        "space_n_label_info": "n",
        "space_edges_label": "Počet hran",
        "space_edges_list_label": "Hrany",
        "prufer_error": "V zadaném Prüferovém kódu je chyba.",
        "sheppard_error": """Neplatný Sheppardův kód.

Zadaný kód nebo kód ze zadaného grafu porušuje základní pravidla platnosti:
1. Všechny hodnoty musí být celá nezáporná čísla (0, 1, 2, ...).
2. Na žádné pozici nesmí být číslo větší než počet prvků napravo od této pozice.
Z toho plyne:
Na poslední pozici může být pouze 0.
Na předposlední pozici pouze 0 nebo 1.
Na třetí pozici od konce pouze 0, 1 nebo 2, atd.
Obecně: na pozici i může být jen číslo z intervalu 0 až n - i, kde n je délka kódu.
Pokud je některé číslo mimo tento povolený rozsah, kód požaduje neexistující volbu a nelze jej dekódovat.
Zkontrolujte prosím zadaný kód a opravte neplatné hodnoty.""",
    },
    "ru": {
        "main_window": "Главное окно",
        "lang_label": "Язык:",
        "lang_items": ["Čeština", "Русский", "English"],
        "prufer_tab": "Код Прюфера",
        "sheppard_tab": "Код Шеппарда",
        "from_prufer_to_graph": "Граф по коду",
        "from_sheppard_to_graph": "Граф по коду",
        "from_graph_to_sheppard": "Кoд по графу",
        "from_graph_to_prufer": "Кoд по графу",
        "space_tab": "Пространство Прюфера",
        "prufer_input_label": "Введите код Прюфера (значения от 0 до n-1 при длине кода n-2):",
        "prufer_button": "Показать дерево",
        "prufer_hint": "Изображение дерева появится ниже.",
        "prufer_code_announcment_label": "Код Прюфера:",
        "sheppard_code_announcment_label": "Код Шеппарда:",
        "tree_input_label": "Задайте peбpa дерева:",
        "graph_input_label": "Задайте ребра графа:",
        "show_sheppard_code_button": "Показать код",
        "show_prufer_code_button": "Показать код",
        "prufer_error_invalid": "Некорректный код Прюфера.",
        "prufer_error_empty": "Код не введён.",
        "prufer_error_export": "Ошибка при создании изображения дерева.",
        "show_steps": "Показать шаги",
        "choose_step": "Выберите шаг: ",
        "sheppard_input_label": "Введите код Шеппарда:",
        "sheppard_button": "Показать граф",
        "sheppard_hint": "Изображение графа появится ниже.",
        "graceful_error": "Введенная разметка не является грациозной, поэтому нельзя найти ее код Шеппарда.",
        "sheppard_error_invalid": "Некорректный код Шеппарда.",
        "sheppard_error_empty": "Код не введён.",
        "sheppard_error_export": "Ошибка при создании изображения дерева.",
        "space_title": "Пространство кодов Прюфера",
        "space_n_label": "Выберите число вершин n:",
        "space_fixed_axis_label": "Фиксируемая ось (только для n = 6):",
        "space_fixed_value_label": "Значение фиксируемой оси:",
        "space_button": "Открыть 3D визуализацию",
        "space_window_title": "Пространство Прюфера",
        "space_info_window_title": "Информация o дереве",
        "space_fig_title": "Пространство кодов Прюфера при n = {n}",
        "space_fig_title_4d": "4D-срез: пространство Прюфера при n = {n}, ось {axis} = {value}",
        "space_click_hint": "Щёлкните по точке в пространстве Прюфера, чтобы увидеть структуру дерева.",
        "space_code_label": "Код Прюфера",
        "space_n_label_info": "n",
        "space_edges_label": "Число рёбер",
        "space_edges_list_label": "Рёбра",
        "prufer_error": "B заданном коде Прюфера есть ошибка.",
        "sheppard_error": """Недопустимый код Шеппарда.

Введенный код или код, получеенный из введенноо графа, нарушает основные правила допустимости:
1. Все значения должны быть целыми неотрицательными числами (0, 1, 2, ...).
2. Ни в одной позиции число не может быть больше, чем количество элементов справа от этой позиции.
Из этого следует:
В последней позиции может быть только 0.
В предпоследней позиции может быть только 0 или 1.
В третьей позиции от конца может быть только 0, 1 или 2 и т. д.
В общем случае: в позиции i может быть только число из интервала от 0 до n - i, где n — длина кода.
Если какое-либо число выходит за пределы этого допустимого диапазона, код требует несуществующего выбора и не может быть декодирован.
Пожалуйста, проверьте введенный код и исправьте недопустимые значения.""",
    },
    "en": {
        "main_window": "Main window",
        "lang_label": "Language:",
        "lang_items": ["Čeština", "Русский", "English"],
        "prufer_tab": "Prüfer code",
        "sheppard_tab": "Sheppard code",
        "from_prufer_to_graph": "Graph from code",
        "from_sheppard_to_graph": "Graph from code",
        "from_graph_to_sheppard": "Code from graph",
        "from_graph_to_prufer": "Code from graph",
        "tab_space": "Prüfer space",
        "prufer_input_label": "Enter Prüfer code (values must be from 0 to n-1 and the length of the code n-2):",
        "prufer_button": "Show tree",
        "tree_input_label": "Set tree edges:",
        "show_sheppard_code_button": "Show code",
        "show_prufer_code_button": "Show code",
        "graph_input_label": "Set graph edges:",
        "prufer_hint": "The tree image will appear below.",
        "prufer_error_invalid": "Invalid Prüfer code.",
        "prufer_error_empty": "No Prüfer code entered.",
        "prufer_error_export": "Failed to generate tree image.",
        "prufer_code_announcment_label": "The Prüfer code is:",
        "sheppard_code_announcment_label": "The Sheppard code is:",
        "graceful_error": "The labeling you have entered is not a graceful labeling, so there is no Sheppard code for it.",
        "show_steps": "Show steps",
        "choose_step": "Choose step: ",
        "sheppard_input_label": "Enter Sheppard code:",
        "sheppard_button": "Show graph",
        "sheppard_hint": "The graph image will appear below.",
        "sheppard_error_invalid": "Invalid Sheppard code.",
        "sheppard_error_empty": "No Sheppard code entered.",
        "sheppard_error_export": "Failed to generate tree image.",
        "space_title": "Prüfer code space",
        "space_n_label": "Choose number of vertices n:",
        "space_fixed_axis_label": "Fixed axis (only for n = 6):",
        "space_fixed_value_label": "Value on fixed axis:",
        "space_button": "Open 3D viewer",
        "space_window_title": "Prüfer space",
        "space_info_window_title": "Tree info",
        "space_fig_title": "Prüfer space for n={n}",
        "space_fig_title_4d": "Prüfer space for n={n} (4D slice: axis {axis} = {value})",
        "space_click_hint": "Click a point in the Prüfer space to view the tree structure here.",
        "space_code_label": "Prüfer code",
        "space_n_label_info": "n",
        "space_edges_label": "Number of edges",
        "space_edges_list_label": "Edges",
        "prufer_error": "There is an error in the Prüfer code.",
        "sheppard_error": """Invalid Sheppard code.

The entered code or the code from the entered graph violates the basic rules of validity:
1. All values must be non-negative integers (0, 1, 2, ...).
2. No position may contain a number greater than the number of elements to the right of that position.
It follows that:
The last position can only contain 0.
Only 0 or 1 can be in the penultimate position.
Only 0, 1, or 2 can be in the third position from the end, etc.
In general: only a number from the interval 0 to n - i can be in position i, where n is the length of the code.
If any number is outside this allowed range, the code requests a non-existent option and cannot be decoded.
Please check the entered code and correct any invalid values.""",
    },
}

current_lang = "cs"
t = TRANSLATIONS[current_lang]

current_codes = {
    "prufer": {"code": None, "by_steps": None},
    "sheppard": {"code": None, "by_steps": None},
}


def apply_language(lang: str):
    global current_lang, t
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

    dpg.set_item_label("prufer_tab", t["prufer_tab"])
    dpg.set_item_label("sheppard_tab", t["sheppard_tab"])
    dpg.set_item_label("from_prufer_to_graph", t["from_prufer_to_graph"])
    dpg.set_item_label("from_sheppard_to_graph", t["from_sheppard_to_graph"])
    dpg.set_item_label("from_graph_to_prufer", t["from_graph_to_prufer"])
    dpg.set_item_label("from_graph_to_sheppard", t["from_graph_to_sheppard"])
    # dpg.set_item_label("space_tab", t["tab_space"])

    # Prüfer
    dpg.set_value("prufer_input_label", t["prufer_input_label"])
    dpg.set_item_label("prufer_button", t["prufer_button"])
    dpg.set_value("prufer_hint", t["prufer_hint"])
    if dpg.does_item_exist("show_steps_prufer"):
        dpg.set_item_label("show_steps_prufer", t["show_steps"])
    if dpg.does_item_exist("choose_step_prufer"):
        dpg.set_value("choose_step_prufer", t["choose_step"])
    if dpg.does_item_exist("choose_step_to_prufer"):
        dpg.set_value("choose_step_to_prufer", t["choose_step"])
    if dpg.does_item_exist("tree_input_label"):
        dpg.set_value("tree_input_label", t["tree_input_label"])
    if dpg.does_item_exist("graph_input_label"):
        dpg.set_value("graph_input_label", t["graph_input_label"])
    if dpg.does_item_exist("show_prufer_code_button"):
        dpg.set_item_label("show_prufer_code_button", t["show_prufer_code_button"])
    if dpg.does_item_exist("edge_error_label"):
        dpg.set_value("edge_error_label", t["edge_error_label"])
    if dpg.does_item_exist("prufer_code_announcment_label"):
        dpg.set_value("prufer_code_announcment_label", t["prufer_code_announcment_label"])
    if dpg.does_item_exist("sheppard_code_announcment_label"):
        dpg.set_value("sheppard_code_announcment_label", t["sheppard_code_announcment_label"])

    # Sheppard
    dpg.set_value("sheppard_input_label", t["sheppard_input_label"])
    dpg.set_item_label("sheppard_button", t["sheppard_button"])
    dpg.set_value("sheppard_hint", t["sheppard_hint"])
    if dpg.does_item_exist("show_steps_sheppard"):
        dpg.set_item_label("show_steps_sheppard", t["show_steps"])
    if dpg.does_item_exist("show_steps_to_sheppard"):
        dpg.set_item_label("show_steps_to_sheppard", t["show_steps"])
    if dpg.does_item_exist("choose_step_sheppard"):
        dpg.set_value("choose_step_sheppard", t["choose_step"])
    if dpg.does_item_exist("show_sheppard_code_button"):
        dpg.set_item_label("show_sheppard_code_button", t["show_sheppard_code_button"])
    if dpg.does_item_exist("choose_step_to_sheppard"):
        dpg.set_value("choose_step_to_sheppard", t["choose_step"])
    if dpg.does_item_exist("graceful_error"):
        dpg.set_value("graceful_error", t["graceful_error"])
    if dpg.does_item_exist("from_graph_to_sheppard_error"):
        dpg.set_value("from_graph_to_sheppard_error", t["sheppard_error"])

    if dpg.does_item_exist("prufer_error_text"):
        dpg.set_value("prufer_error_text", t["prufer_error"])
    if dpg.does_item_exist("sheppard_error_text"):
        dpg.set_value("sheppard_error_text", t["sheppard_error"])
    if dpg.does_item_exist("to_sheppard_error_text"):
        dpg.set_value("to_sheppard_error_text", t["sheppard_error"])

    # Space
    # dpg.set_value("space_n_label", t["space_n_label"])
    # dpg.set_value("space_fixed_axis_label", t["space_fixed_axis_label"])
    # dpg.set_value("space_fixed_value_label", t["space_fixed_value_label"])
    # dpg.set_item_label("space_button", t["space_button"])


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

def check_if_code_is_new(code, code_type: str, type_on_steps: bool = False):
    key = "by_steps" if type_on_steps else "code"
    if current_codes[code_type][key] == code:
        return False
    current_codes[code_type][key] = code
    return True

def clear_step(code_type: str, DoNotDeleteMain: bool = False):
    for child in dpg.get_item_children(f"{code_type}_tab", 1):
        if dpg.get_item_type(child) == "mvAppItemType::mvImage":
            if dpg.get_item_label(child) == "main" and DoNotDeleteMain is True:
                continue
            dpg.delete_item(child)

def show_graph(sender, app_data, user_data):
    code_type = user_data
    text = dpg.get_value(f"{code_type}_input")

    try:
        code = parse_code(text)
    except ValueError:
        return

    if check_if_code_is_new(code, code_type) is False:
        return
    
    if dpg.does_item_exist(f"{code_type}_error_text"):
        dpg.delete_item(f"{code_type}_error_text")
    # чистим старое
    for tag in (f"{code_type}_image", f"{code_type}_texture"):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)

    # убрать старые UI элементы шагов
    for tag in (f"show_steps_{code_type}", 
                f"step_number_{code_type}", 
                f"choose_step_{code_type}",
                f"{code_type}_step_image",
                f"{code_type}_step_texture"):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    clear_step(code_type)

    # строим граф
    if code_type == "sheppard":
        try:
            graph = phi.from_sheppard(code, complete=True)
        except:
            dpg.add_text(TRANSLATIONS[current_lang]["sheppard_error"],
                         parent="from_sheppard_to_graph",
                         tag="sheppard_error_text")
            print('error')
    else:
        try:
            graph = phi.from_prufer(code)
        except:
            dpg.add_text(TRANSLATIONS[current_lang]["prufer_error"], 
                         parent="from_prufer_to_graph",
                         tag="prufer_error_text")

    # экспорт картинки "main"
    # ВАЖНО: хорошо бы, чтобы export_to_png тоже писал в outputs_dir(),
    # но пока используем как есть (вернёт путь)
    print(code, code_type)
    img_path = graph.export_to_png(image_name=f"main_to_{code_type}.png", code_type=code_type, dark=True)
    width, height, channels, data = dpg.load_image(str(img_path))

    # текстура
    with dpg.texture_registry():
        dpg.add_static_texture(width, height, data, tag=f"{code_type}_texture")

    dpg.add_image(
        f"{code_type}_texture",
        label="main",
        parent=f"from_{code_type}_to_graph",
        tag=f"{code_type}_image"
    )

    dpg.add_button(
        tag=f"show_steps_{code_type}",
        label=TRANSLATIONS[current_lang]["show_steps"],
        parent=f"from_{code_type}_to_graph",
        callback=on_show_steps,
        user_data={"code_type": code_type, "code": code},
    )

def on_show_steps(sender, app_data, user_data):
    code_data = user_data
    code_type = code_data["code_type"]
    if check_if_code_is_new(code_data["code"], code_type, type_on_steps=True) is False:
        return

    if dpg.does_item_exist(f"show_steps_{code_type}"):
        dpg.delete_item(f"show_steps_{code_type}")
    if dpg.does_item_exist(f"choose_step_{code_type}"):
        dpg.delete_item(f"choose_step_{code_type}")

    dpg.add_text(
        TRANSLATIONS[current_lang]["choose_step"],
        parent=f"from_{code_type}_to_graph",
        tag=f"choose_step_{code_type}"
    )

    max_step = (len(code_data["code"]) + 1 if code_type == "prufer" else len(code_data["code"]))
    dpg.add_slider_int(
        label="",
        min_value=1,
        max_value=max_step,
        default_value=1,
        tag=f"step_number_{code_type}",
        parent=f"from_{code_type}_to_graph",
        callback=change_step,
        user_data=code_type,
    )

    # Генерим step-картинки.
    # ВАЖНО: чтобы это работало в exe, phi должен сохранять step PNG в writable папку,
    # например outputs/prufer/... и outputs/sheppard/...
    if code_type == "prufer":
        phi.from_prufer(code_data["code"], steps=True)
    else:
        phi.from_sheppard(code_data["code"], steps=True)

    change_step(None, None, code_type)  # показать 1-й шаг сразу

def on_show_steps_to(sender, app_data, user_data):
    edge_data = user_data
    code_type = edge_data["code_type"]

    if dpg.does_item_exist(f"choose_step_to_{code_type}"):
        dpg.delete_item(f"choose_step_to_{code_type}")
    if dpg.does_item_exist(f"show_steps_to_{code_type}"):
        dpg.delete_item(f"show_steps_to_{code_type}")

    dpg.add_text(
        TRANSLATIONS[current_lang]["choose_step"],
        parent=f"from_graph_to_{code_type}",
        tag=f"choose_step_to_{code_type}"
    )

    max_step = (len(edge_data["edges"]) - 1 if code_type == "prufer" else len(edge_data["edges"]))
    dpg.add_slider_int(
        label="",
        min_value=1,
        max_value=max_step,
        default_value=1,
        tag=f"step_number_to_{code_type}",
        parent=f"from_graph_to_{code_type}",
        callback=change_step_to,
        user_data=code_type,
    )

    graph = phi.Graph(phi.GraphType.UNDIRECTED)
    for edge in edge_data["edges"]:
        graph.add_edge(edge[0], edge[1])

    if code_type == "prufer":
        graph.to_prufer(steps=True)
    else:
        graph.to_sheppard(steps=True)

    change_step_to(None, None, code_type)

def change_step_to(sender, app_data, user_data):
    code_type = user_data
    dpg.set_y_scroll("main_window", 10**9)

    step_number = dpg.get_value(f"step_number_to_{code_type}")
    clear_step(code_type, DoNotDeleteMain=True)

    # ИЩЕМ step-картинки здесь:
    # outputs/prufer/graph_1_step.png и т.п.
    path = steps_dir(code_type) / f"to_code_{step_number}_step.png"

    if not path.exists():
        # чтобы не падало
        dpg.add_text(f"[missing] {path}", parent=f"{code_type}_tab")
        return

    width, height, channels, data = dpg.load_image(str(path))

    # один стабильный тег текстуры на вкладку (не плодим миллион)
    texture_tag = f"to_{code_type}_step_texture"
    image_tag = f"to_{code_type}_step_image"

    if dpg.does_item_exist(texture_tag):
        dpg.delete_item(texture_tag)
    if dpg.does_item_exist(image_tag):
        dpg.delete_item(image_tag)

    with dpg.texture_registry():
        dpg.add_static_texture(width, height, data, tag=texture_tag)

    dpg.add_image(texture_tag, parent=f"from_graph_to_{code_type}", tag=image_tag)

    dpg.set_y_scroll("main_window", 10**9)

def change_step(sender, app_data, user_data):
    code_type = user_data
    dpg.set_y_scroll("main_window", 10**9)

    step_number = dpg.get_value(f"step_number_{code_type}")
    clear_step(code_type, DoNotDeleteMain=True)

    # ИЩЕМ step-картинки здесь:
    # outputs/prufer/graph_1_step.png и т.п.
    path = steps_dir(code_type) / f"graph_{step_number}_step.png"

    if not path.exists():
        # чтобы не падало
        dpg.add_text(f"[missing] {path}", parent=f"{code_type}_tab")
        return

    width, height, channels, data = dpg.load_image(str(path))

    # один стабильный тег текстуры на вкладку (не плодим миллион)
    texture_tag = f"{code_type}_step_texture"
    image_tag = f"{code_type}_step_image"

    if dpg.does_item_exist(texture_tag):
        dpg.delete_item(texture_tag)
    if dpg.does_item_exist(image_tag):
        dpg.delete_item(image_tag)

    with dpg.texture_registry():
        dpg.add_static_texture(width, height, data, tag=texture_tag)

    dpg.add_image(texture_tag, parent=f"from_{code_type}_to_graph", tag=image_tag)

    dpg.set_y_scroll("main_window", 10**9)

def on_show_code_button(sender, app_data, user_data):
    code_type = user_data
    text_edges = dpg.get_value(f"{code_type}_graph_input")
    try:
        edges = parse_edges(text_edges)
    except:
        return
    
    if dpg.does_item_exist(f"{code_type}_code_label"):
        dpg.delete_item(f"{code_type}_code_label")
    if dpg.does_item_exist(f"step_number_to_{code_type}"):
        dpg.delete_item(f"step_number_to_{code_type}")
    if dpg.does_item_exist(f"to_{code_type}_step_texture"):
        dpg.delete_item(f"to_{code_type}_step_texture")
    if dpg.does_item_exist(f"to_{code_type}_step_image"):
        dpg.delete_item(f"to_{code_type}_step_image")
    if dpg.does_item_exist(f"show_steps_to_{code_type}"):
        dpg.delete_item(f"show_steps_to_{code_type}")
    if dpg.does_item_exist(f"{code_type}_code_announcment_label"):
        dpg.delete_item(f"{code_type}_code_announcment_label")
    if dpg.does_item_exist(f"{code_type}_code_label"):
        dpg.delete_item(f"{code_type}_code_label")
    if dpg.does_item_exist(f"choose_step_to_{code_type}"):
        dpg.delete_item(f"choose_step_to_{code_type}")
    if dpg.does_item_exist("to_sheppard_error_text"):
        dpg.delete_item("to_sheppard_error_text")
    if dpg.does_item_exist("graceful_error"):
        dpg.delete_item("graceful_error")
    for tag in (f"to_{code_type}_image", f"to_{code_type}_texture"):
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    
    graph = phi.Graph(phi.GraphType.UNDIRECTED)
    for edge in edges:
        graph.add_edge(edge[0], edge[1])

    if code_type == "prufer":
        code_list = graph.to_prufer()
        code = str(code_list)
    else:
        code_list = graph.to_sheppard()
        if code_list is None:
            dpg.add_text(TRANSLATIONS[current_lang]["graceful_error"],
                         parent="from_graph_to_sheppard",
                         tag="graceful_error")
            return
        code = str(code_list)

    print(edges)
    dpg.add_text(TRANSLATIONS[current_lang][f"{code_type}_code_announcment_label"], 
                 tag=f"{code_type}_code_announcment_label", 
                 parent=f"from_graph_to_{code_type}")
    dpg.add_text(code, tag=f"{code_type}_code_label", parent=f"from_graph_to_{code_type}")
    if code_type == "sheppard":
        if any(graph.node(id).id > len(graph._edges) for id in graph.node_ids()):
            dpg.add_text(TRANSLATIONS[current_lang]["sheppard_error"],
                         parent="from_graph_to_sheppard",
                         tag="to_sheppard_error_text")

    graph_image = graph.export_to_png(f"to_{code_type}_main.png", code_type=user_data, dark=True)
    width, height, channels, data = dpg.load_image(str(graph_image))

    with dpg.texture_registry():
        dpg.add_static_texture(width, height, data, tag=f"to_{code_type}_texture")

    dpg.add_image(
        f"to_{code_type}_texture",
        label="main",
        parent=f"from_graph_to_{code_type}",
        tag=f"to_{code_type}_image"
    )

    dpg.add_button(
        tag=f"show_steps_to_{code_type}",
        label=TRANSLATIONS[current_lang]["show_steps"],
        parent=f"from_graph_to_{code_type}",
        callback=on_show_steps_to,
        user_data={"code_type": code_type, "edges": edges},
    )

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
            callback=on_language_change,
        )

    dpg.add_separator()

    with dpg.tab_bar():
        with dpg.tab(label="", tag="prufer_tab"):
            with dpg.tab_bar():
                with dpg.tab(label="From code to graph", tag="from_prufer_to_graph"):
                    dpg.add_text("", tag="prufer_input_label")
                    dpg.add_input_text(tag="prufer_input", width=300, default_value="")
                    dpg.add_button(label="", tag="prufer_button",
                                callback=show_graph, user_data="prufer")
                    dpg.add_separator()
                    dpg.add_text("", tag="prufer_hint")
                with dpg.tab(label="From graph to code", tag="from_graph_to_prufer"):
                    dpg.add_text("", tag="tree_input_label")
                    dpg.add_input_text(tag="prufer_graph_input", width=300, default_value="")
                    dpg.add_button(label="", tag="show_prufer_code_button",
                                   callback=on_show_code_button, user_data="prufer")

        with dpg.tab(label="", tag="sheppard_tab"):
            with dpg.tab_bar():
                with dpg.tab(label="From code to graph", tag="from_sheppard_to_graph"):
                    dpg.add_text("", tag="sheppard_input_label")
                    dpg.add_input_text(tag="sheppard_input", width=300, default_value="")
                    dpg.add_button(label="", tag="sheppard_button",
                                callback=show_graph, user_data="sheppard")
                    dpg.add_separator()
                    dpg.add_text("", tag="sheppard_hint")
                with dpg.tab(label="From graph to code", tag="from_graph_to_sheppard"):
                    dpg.add_text("", tag="graph_input_label")
                    dpg.add_input_text(tag="sheppard_graph_input", width=300, default_value="")
                    dpg.add_button(label="", tag="show_sheppard_code_button",
                                   callback=on_show_code_button, user_data="sheppard")

# =========================================================
#                     ЗАПУСК ПРИЛОЖЕНИЯ
# =========================================================

def main():
    default_font = load_font()
    dpg.create_viewport(title="Graphium", width=1020, height=740)
    apply_language("cs")
    dpg.setup_dearpygui()
    dpg.show_viewport()

    if default_font is not None:
        dpg.bind_font(default_font)

    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
