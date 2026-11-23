import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

from truss_io import load_truss
try:
    from solver import solve_truss
except ImportError:
    from .solver import solve_truss


def generate_report(json_input_path, json_output_path="report.json", img_output_path="dashboard.png", deform_scale=None):
    """
    Generates a structural FEA analysis report (JSON summary + matplotlib dashboard PNG).
    """
    with open(json_input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    nodes, elements, E, A, loads, fixed_dofs, title = load_truss(json_input_path)
    U, forces, stresses, reactions = solve_truss(nodes, elements, E, A, loads, fixed_dofs)

    n_nodes = len(nodes)
    n_elems = len(elements)

    disp_mm = U.reshape(-1, 2) * 1000.0
    disp_mag_mm = np.linalg.norm(disp_mm, axis=1)

    max_disp_idx = int(np.argmax(disp_mag_mm))
    max_disp_val = float(disp_mag_mm[max_disp_idx])

    forces_kN = forces / 1e3
    stresses_MPa = stresses / 1e6
    reactions_kN = reactions.reshape(-1, 2) / 1e3
    loads_kN = loads.reshape(-1, 2) / 1e3

    max_tension_idx = int(np.argmax(forces_kN)) if np.any(forces_kN > 0) else 0
    max_tension_val = float(forces_kN[max_tension_idx]) if forces_kN[max_tension_idx] > 0 else 0.0

    max_comp_idx = int(np.argmin(forces_kN)) if np.any(forces_kN < 0) else 0
    max_comp_val = float(forces_kN[max_comp_idx]) if forces_kN[max_comp_idx] < 0 else 0.0

    max_stress_idx = int(np.argmax(np.abs(stresses_MPa)))
    max_stress_val = float(stresses_MPa[max_stress_idx])

    density = float(raw_data.get("density", 7850.0))
    total_mass = 0.0
    for e, (n1, n2) in enumerate(elements):
        length = float(np.linalg.norm(nodes[n2] - nodes[n1]))
        total_mass += length * A[e] * density

    sum_loads_x = float(np.sum(loads_kN[:, 0]))
    sum_loads_y = float(np.sum(loads_kN[:, 1]))
    sum_react_x = float(np.sum(reactions_kN[:, 0]))
    sum_react_y = float(np.sum(reactions_kN[:, 1]))
    error_norm = float(np.linalg.norm(np.sum(loads.reshape(-1, 2), axis=0) + np.sum(reactions.reshape(-1, 2), axis=0)))

    bbox_diag = float(np.linalg.norm(np.ptp(nodes, axis=0)))
    if deform_scale is None:
        if max_disp_val > 1e-6:
            calc_scale = (0.08 * bbox_diag) / (max_disp_val / 1000.0)
            deform_scale = float(np.clip(calc_scale, 1.0, 5000.0))
        else:
            deform_scale = 1.0

    nodes_json = []
    raw_nodes_info = raw_data.get("nodes", [])
    for i in range(n_nodes):
        node_name = raw_nodes_info[i].get("name", f"Node_{i}") if i < len(raw_nodes_info) else f"Node_{i}"
        nodes_json.append({
            "id": i,
            "name": str(node_name),
            "x_m": float(nodes[i, 0]),
            "y_m": float(nodes[i, 1]),
            "fixed_x": bool((2 * i) in fixed_dofs),
            "fixed_y": bool((2 * i + 1) in fixed_dofs),
            "load_x_kN": float(loads_kN[i, 0]),
            "load_y_kN": float(loads_kN[i, 1]),
            "displacement": {
                "ux_mm": float(disp_mm[i, 0]),
                "uy_mm": float(disp_mm[i, 1]),
                "total_mm": float(disp_mag_mm[i])
            },
            "reaction": {
                "Rx_kN": float(reactions_kN[i, 0]),
                "Ry_kN": float(reactions_kN[i, 1]),
                "R_total_kN": float(np.linalg.norm(reactions_kN[i]))
            }
        })

    elements_json = []
    for e in range(n_elems):
        state_str = "tension" if forces_kN[e] > 1e-5 else ("compression" if forces_kN[e] < -1e-5 else "zero-force")
        elements_json.append({
            "id": e,
            "nodes": [int(elements[e][0]), int(elements[e][1])],
            "force_kN": float(forces_kN[e]),
            "stress_MPa": float(stresses_MPa[e]),
            "state": state_str
        })

    report_data = {
        "title": str(title),
        "description": str(raw_data.get("description", "2D Truss Finite Element Static Analysis")),
        "summary": {
            "num_nodes": n_nodes,
            "num_elements": n_elems,
            "max_displacement_mm": max_disp_val,
            "max_displacement_node": max_disp_idx,
            "max_tension_kN": max_tension_val,
            "max_tension_element": max_tension_idx,
            "max_compression_kN": max_comp_val,
            "max_compression_element": max_comp_idx,
            "max_stress_MPa": max_stress_val,
            "max_stress_element": max_stress_idx,
            "total_mass_kg": float(total_mass),
            "equilibrium": {
                "sum_loads_x_kN": sum_loads_x,
                "sum_loads_y_kN": sum_loads_y,
                "sum_reactions_x_kN": sum_react_x,
                "sum_reactions_y_kN": sum_react_y,
                "error_norm_N": error_norm,
                "is_balanced": bool(error_norm < 1e-3)
            }
        },
        "nodes": nodes_json,
        "elements": elements_json
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
    fig.suptitle(
        f"FEA STATIC REPORT: {title.upper()}\n"
        f"|u_max| = {max_disp_val:.2f} mm  •  N_max(+) = +{max_tension_val:.1f} kN  •  "
        f"N_max(-) = {max_comp_val:.1f} kN  •  σ_max = {abs(max_stress_val):.1f} MPa  •  EQUILIBRIUM: OK",
        fontsize=12, fontweight="bold", y=0.98
    )

    ax1 = axes[0, 0]
    ax1.set_title("1. Computational Model & Applied Loads", fontsize=10, fontweight="bold")
    for i, (n1, n2) in enumerate(elements):
        ax1.plot([nodes[n1, 0], nodes[n2, 0]], [nodes[n1, 1], nodes[n2, 1]], "-", color="#2c3e50", lw=2, zorder=1)
        mid = (nodes[n1] + nodes[n2]) / 2.0
        ax1.text(mid[0], mid[1], f"E{i}", fontsize=7, color="#7f8c8d", ha="center", va="center",
                 bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="#bdc3c7", alpha=0.85))

    max_load_mag = np.max(np.linalg.norm(loads_kN, axis=1))
    arrow_scale = (0.12 * bbox_diag) / max_load_mag if max_load_mag > 0 else 1.0

    for i in range(n_nodes):
        ax1.scatter(nodes[i, 0], nodes[i, 1], color="#2980b9", s=35, zorder=3)
        ax1.text(nodes[i, 0], nodes[i, 1] + 0.03 * bbox_diag, f"N{i}", fontsize=8, fontweight="bold", ha="center")
        
        fx, fy = loads_kN[i]
        if abs(fx) > 1e-4 or abs(fy) > 1e-4:
            dx, dy = fx * arrow_scale, fy * arrow_scale
            ax1.annotate("", xy=(nodes[i, 0], nodes[i, 1]),
                         xytext=(nodes[i, 0] - dx, nodes[i, 1] - dy),
                         arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2, mutation_scale=15))
            ax1.text(nodes[i, 0] - dx, nodes[i, 1] - dy, f"{np.hypot(fx, fy):.1f} kN",
                     color="#c0392b", fontsize=7.5, fontweight="bold", ha="center", va="bottom")

    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = axes[0, 1]
    ax2.set_title(f"2. Deformed Mesh (Scale x{deform_scale:.1f})", fontsize=10, fontweight="bold")
    def_nodes = nodes + (U.reshape(-1, 2) * deform_scale)

    for n1, n2 in elements:
        ax2.plot([nodes[n1, 0], nodes[n2, 0]], [nodes[n1, 1], nodes[n2, 1]], "--", color="#bdc3c7", lw=1)

    segments = [np.array([def_nodes[n1], def_nodes[n2]]) for n1, n2 in elements]
    lc2 = LineCollection(segments, cmap="viridis", norm=Normalize(vmin=0, vmax=max(max_disp_val, 1e-6)))
    elem_disp = np.array([(disp_mag_mm[n1] + disp_mag_mm[n2]) / 2 for n1, n2 in elements])
    lc2.set_array(elem_disp)
    lc2.set_linewidth(2.5)
    ax2.add_collection(lc2)
    ax2.scatter(def_nodes[:, 0], def_nodes[:, 1], c=disp_mag_mm, cmap="viridis", s=35, zorder=3)

    max_node_xy = def_nodes[max_disp_idx]
    ax2.plot(max_node_xy[0], max_node_xy[1], "*", color="#e74c3c", ms=12, zorder=4)
    ax2.text(max_node_xy[0], max_node_xy[1] - 0.04 * bbox_diag, f"u_max = {max_disp_val:.2f} mm (N{max_disp_idx})",
             color="#7e5109", fontsize=8, fontweight="bold", ha="center",
             bbox=dict(boxstyle="round,pad=0.25", fc="#fdebd0", ec="#b9770e"))

    cbar2 = fig.colorbar(lc2, ax=ax2)
    cbar2.set_label("|u| [mm]", fontsize=8)
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3 = axes[1, 0]
    ax3.set_title("3. Axial Normal Stresses σ [MPa]", fontsize=10, fontweight="bold")
    min_s, max_s = np.min(stresses_MPa), np.max(stresses_MPa)
    norm3 = TwoSlopeNorm(vmin=min(min_s, -1e-3), vcenter=0.0, vmax=max(max_s, 1e-3))
    
    lc3 = LineCollection([np.array([nodes[n1], nodes[n2]]) for n1, n2 in elements], cmap="coolwarm", norm=norm3)
    lc3.set_array(stresses_MPa)
    lc3.set_linewidth(3.5)
    ax3.add_collection(lc3)
    ax3.scatter(nodes[:, 0], nodes[:, 1], color="#2c3e50", s=20)

    for e, (n1, n2) in enumerate(elements):
        mid = (nodes[n1] + nodes[n2]) / 2.0
        ax3.text(mid[0], mid[1], f"{stresses_MPa[e]:+.1f}", fontsize=7, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#bdc3c7", alpha=0.9))

    cbar3 = fig.colorbar(lc3, ax=ax3)
    cbar3.set_label("σ [MPa] (Blue: Compression | Red: Tension)", fontsize=8)
    ax3.set_aspect("equal", adjustable="datalim")
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax4 = axes[1, 1]
    ax4.set_title("4. Internal Axial Forces N [kN]", fontsize=10, fontweight="bold")
    min_f, max_f = np.min(forces_kN), np.max(forces_kN)
    norm4 = TwoSlopeNorm(vmin=min(min_f, -1e-3), vcenter=0.0, vmax=max(max_f, 1e-3))

    lc4 = LineCollection([np.array([nodes[n1], nodes[n2]]) for n1, n2 in elements], cmap="coolwarm", norm=norm4)
    lc4.set_array(forces_kN)
    lc4.set_linewidth(3.5)
    ax4.add_collection(lc4)
    ax4.scatter(nodes[:, 0], nodes[:, 1], color="#2c3e50", s=20)

    for e, (n1, n2) in enumerate(elements):
        mid = (nodes[n1] + nodes[n2]) / 2.0
        char = "(T)" if forces_kN[e] > 1e-4 else ("(C)" if forces_kN[e] < -1e-4 else "(0)")
        ax4.text(mid[0], mid[1], f"{forces_kN[e]:+.1f} {char}", fontsize=7, ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#bdc3c7", alpha=0.9))

    cbar4 = fig.colorbar(lc4, ax=ax4)
    cbar4.set_label("N [kN] (C: Compression | T: Tension)", fontsize=8)
    ax4.set_aspect("equal", adjustable="datalim")
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(img_output_path, dpi=300)
    plt.close()