"""
Moduł I/O do wczytywania geometrii, materiałów i obciążeń kratownicy z pliku JSON.
"""
import json
import numpy as np
from pathlib import Path


def load_truss(json_path, return_dict=False):
    """
    Wczytuje kratownicę z pliku JSON i przelicza wielkości na bazowe jednostki SI:
        m, N, Pa, m^2
    Brak wymaganych parametrów lub błędne dane wywołują błąd ValueError, KeyError lub TypeError.
    """
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {json_path}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    u = data.get("units", {})
    if not isinstance(u, dict):
        raise TypeError("Sekcja 'units' musi być słownikiem.")

    l_unit = str(u.get("length", "m")).lower()
    f_unit = str(u.get("force", "N")).lower()
    e_unit = str(u.get("modulus", "Pa")).lower()
    a_unit = str(u.get("area", "m2")).lower()

    scale_l_dict = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
    scale_f_dict = {"n": 1.0, "kn": 1e3, "mn": 1e6}
    scale_e_dict = {"pa": 1.0, "kpa": 1e3, "mpa": 1e6, "gpa": 1e9}
    scale_a_dict = {"m2": 1.0, "cm2": 1e-4, "mm2": 1e-6}

    if l_unit not in scale_l_dict:
        raise ValueError(f"Nieobsługiwana jednostka długości: '{l_unit}'")
    if f_unit not in scale_f_dict:
        raise ValueError(f"Nieobsługiwana jednostka siły: '{f_unit}'")
    if e_unit not in scale_e_dict:
        raise ValueError(f"Nieobsługiwana jednostka modułu Younga: '{e_unit}'")
    if a_unit not in scale_a_dict:
        raise ValueError(f"Nieobsługiwana jednostka pola przekroju: '{a_unit}'")

    scale_l = scale_l_dict[l_unit]
    scale_f = scale_f_dict[f_unit]
    scale_e = scale_e_dict[e_unit]
    scale_a = scale_a_dict[a_unit]

    raw_materials = data.get("materials", {})
    if not isinstance(raw_materials, dict):
        raise TypeError("Sekcja 'materials' musi być słownikiem.")

    mat_E = {}
    for k, v in raw_materials.items():
        if not isinstance(v, dict) or "E" not in v:
            raise KeyError(f"Brak zdefiniowanego modułu 'E' w materiale '{k}'.")
        e_val = float(v["E"]) * scale_e
        if e_val <= 0:
            raise ValueError(f"Moduł Younga 'E' w materiale '{k}' musi być dodatni.")
        mat_E[str(k)] = e_val

    raw_sections = data.get("sections", {})
    if not isinstance(raw_sections, dict):
        raise TypeError("Sekcja 'sections' musi być słownikiem.")

    sec_A = {}
    for k, v in raw_sections.items():
        if not isinstance(v, dict) or "A" not in v:
            raise KeyError(f"Brak zdefiniowanego pola 'A' w sekcji '{k}'.")
        a_val = float(v["A"]) * scale_a
        if a_val <= 0:
            raise ValueError(f"Pole przekroju 'A' w sekcji '{k}' musi być dodatnie.")
        sec_A[str(k)] = a_val

    raw_nodes = data.get("nodes", [])
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("Sekcja 'nodes' musi być niepustą listą.")

    nodes, node_map, fixed_dofs, loads = [], {}, [], []

    for i, n in enumerate(raw_nodes):
        if not isinstance(n, dict):
            raise TypeError(f"Węzeł o indeksie {i} musi być słownikiem.")

        if "id" not in n:
            raise KeyError(f"Węzeł o indeksie {i} nie posiada wymaganego pola 'id'.")

        nid_str = str(n["id"])
        if nid_str in node_map:
            raise ValueError(f"Zduplikowany identyfikator węzła (id={n['id']}).")
        node_map[nid_str] = i

        if "x" not in n or "y" not in n:
            raise KeyError(f"Węzeł id={n['id']} nie posiada zdefiniowanych współrzędnych 'x' i 'y'.")

        nodes.append([float(n["x"]) * scale_l, float(n["y"]) * scale_l])

        fix_x = bool(n.get("fixed_x", False))
        fix_y = bool(n.get("fixed_y", False))

        if "support" in n:
            supp = str(n["support"]).lower()
            if supp in ["pinned", "fixed", "stala", "stała", "nieprzesuwna"]:
                fix_x, fix_y = True, True
            elif supp in ["roller", "roller_x", "przesuwna", "przesuwna_x", "roller-x"]:
                fix_x, fix_y = False, True
            elif supp in ["roller_y", "przesuwna_y", "roller-y"]:
                fix_x, fix_y = True, False

        if "fixed" in n:
            f = n["fixed"]
            if isinstance(f, (list, tuple)) and len(f) == 2:
                fix_x, fix_y = bool(f[0]), bool(f[1])
            elif isinstance(f, str):
                fix_x = "x" in f.lower()
                fix_y = "y" in f.lower()

        if fix_x:
            fixed_dofs.append(2 * i)
        if fix_y:
            fixed_dofs.append(2 * i + 1)

        if "load" in n and n["load"] is not None:
            ld = n["load"]
            if isinstance(ld, (list, tuple)) and len(ld) == 2:
                fx, fy = float(ld[0]), float(ld[1])
            elif isinstance(ld, (int, float)):
                fx, fy = 0.0, float(ld)
            else:
                raise ValueError(f"Pole 'load' w węźle id={n['id']} musi być dwuelementową listą [Fx, Fy].")
        else:
            fx = float(n.get("load_x", 0.0))
            fy = float(n.get("load_y", 0.0))

        loads.extend([fx * scale_f, fy * scale_f])

    raw_elements = data.get("elements", [])
    if not isinstance(raw_elements, list) or not raw_elements:
        raise ValueError("Sekcja 'elements' musi być niepustą listą.")

    elements, E_list, A_list = [], [], []

    for i, e in enumerate(raw_elements):
        if not isinstance(e, dict):
            raise TypeError(f"Element o indeksie {i} musi być słownikiem.")

        if "nodes" in e:
            pair = e["nodes"]
        elif "node_start" in e and "node_end" in e:
            pair = [e["node_start"], e["node_end"]]
        else:
            raise KeyError(f"Element index={i} nie ma zdefiniowanych węzłów połączeniowych ('nodes' lub 'node_start'/'node_end').")

        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Węzły elementu index={i} muszą stanowić parę [start, end].")

        n1_str, n2_str = str(pair[0]), str(pair[1])

        if n1_str not in node_map or n2_str not in node_map:
            raise KeyError(f"Element index={i} odwołuje się do nieistniejącego węzła w połączeniu {pair}.")

        if n1_str == n2_str:
            raise ValueError(f"Element index={i} łączy węzeł {pair[0]} sam ze sobą.")

        idx1, idx2 = node_map[n1_str], node_map[n2_str]

        p1 = np.array(nodes[idx1])
        p2 = np.array(nodes[idx2])
        length = np.linalg.norm(p2 - p1)
        if length == 0.0:
            raise ValueError(f"Element index={i} (węzły {pair}) ma zerową długość.")

        elements.append([idx1, idx2])

        if "E" in e:
            e_val = float(e["E"]) * scale_e
        elif "material" in e:
            mat_name = str(e["material"])
            if mat_name not in mat_E:
                raise KeyError(f"Element index={i} odwołuje się do niezdefiniowanego materiału '{mat_name}'.")
            e_val = mat_E[mat_name]
        else:
            raise ValueError(f"Element index={i} (węzły {pair}) nie posiada zdefiniowanego 'E' ani 'material'.")

        if e_val <= 0:
            raise ValueError(f"Moduł Younga E dla elementu index={i} musi być dodatni.")

        if "A" in e:
            a_val = float(e["A"]) * scale_a
        elif "section" in e:
            sec_name = str(e["section"])
            if sec_name not in sec_A:
                raise KeyError(f"Element index={i} odwołuje się do niezdefiniowanej sekcji '{sec_name}'.")
            a_val = sec_A[sec_name]
        else:
            raise ValueError(f"Element index={i} (węzły {pair}) nie posiada zdefiniowanego 'A' ani 'section'.")

        if a_val <= 0:
            raise ValueError(f"Pole przekroju A dla elementu index={i} musi być dodatnie.")

        E_list.append(e_val)
        A_list.append(a_val)

    nodes_arr = np.array(nodes, dtype=float)
    elements_arr = np.array(elements, dtype=int)
    E_arr = np.array(E_list, dtype=float)
    A_arr = np.array(A_list, dtype=float)
    loads_arr = np.array(loads, dtype=float)
    fixed_dofs_unique = sorted(list(set(fixed_dofs)))
    title = str(data.get("title", "Kratownica 2D"))

    if return_dict:
        return {
            "nodes": nodes_arr,
            "elements": elements_arr,
            "E": E_arr,
            "A": A_arr,
            "loads": loads_arr,
            "fixed_dofs": fixed_dofs_unique,
            "title": title,
            "units": u,
            "raw_data": data
        }

    return (
        nodes_arr,
        elements_arr,
        E_arr,
        A_arr,
        loads_arr,
        fixed_dofs_unique,
        title
    )