import os
import glob
import numpy as np

from truss_io import load_truss
from solver import solve_truss
from Truss_reporter import generate_report


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def show_menu():
    print("==================================================")
    print("          Truss Solver - KH Industries            ")
    print("==================================================")


def print_summary(nodes, elements, U, forces, stresses, reactions, loads, title="Truss 2D"):
    """Prints a formatted summary of FEA results in English."""
    print("=" * 60)
    print(f" STATIC REPORT: {title}")
    print("=" * 60)

    print("\n1. NODE DISPLACEMENTS:")
    print(f"{'Node':<8}{'X [m]':<10}{'Y [m]':<10}{'ux [mm]':<12}{'uy [mm]':<12}{'|u| [mm]':<12}")
    print("-" * 64)
    for i, (x, y) in enumerate(nodes):
        ux_mm = U[2*i] * 1e3
        uy_mm = U[2*i+1] * 1e3
        u_mag = np.hypot(ux_mm, uy_mm)
        print(f"{i:<8}{x:<10.3f}{y:<10.3f}{ux_mm:<12.4f}{uy_mm:<12.4f}{u_mag:<12.4f}")

    print("\n2. AXIAL FORCES AND STRESSES IN ELEMENTS:")
    print(f"{'Elem':<6}{'Nodes':<10}{'Force [kN]':<16}{'State':<18}{'Stress [MPa]':<18}")
    print("-" * 68)
    for e, (n1, n2) in enumerate(elements):
        f_val = forces[e]
        f_kn = f_val / 1e3
        s_mpa = stresses[e] / 1e6
        state = "Tension (+)" if f_val > 1e-6 else ("Compression (-)" if f_val < -1e-6 else "Zero-force (0)")
        print(f"{e:<6}{f'{n1}->{n2}':<10}{f_kn:<16.2f}{state:<18}{s_mpa:<18.2f}")

    print("\n3. SUPPORT REACTIONS:")
    print(f"{'Node':<8}{'Rx [kN]':<15}{'Ry [kN]':<15}{'R_resultant [kN]':<18}")
    print("-" * 56)
    has_reaction = False
    for i in range(len(nodes)):
        rx_val = reactions[2*i]
        ry_val = reactions[2*i+1]
        rx = rx_val / 1e3
        ry = ry_val / 1e3
        if abs(rx_val) > 1e-6 or abs(ry_val) > 1e-6:
            has_reaction = True
            r_mag = np.hypot(rx, ry)
            print(f"{i:<8}{rx:<15.2f}{ry:<15.2f}{r_mag:<18.2f}")
    if not has_reaction:
        print("  No support reactions.")

    print("\n4. STATIC EQUILIBRIUM CHECK:")
    sum_rx = np.sum(reactions[0::2]) / 1e3
    sum_ry = np.sum(reactions[1::2]) / 1e3
    sum_fx = np.sum(loads[0::2]) / 1e3
    sum_fy = np.sum(loads[1::2]) / 1e3
    print(f"  Sum of external forces: Fx = {sum_fx:.2f} kN, Fy = {sum_fy:.2f} kN")
    print(f"  Sum of reactions:       Rx = {sum_rx:.2f} kN, Ry = {sum_ry:.2f} kN")
    print(f"  Balance Fx + Rx = {sum_fx + sum_rx:+.2e} kN (Expected 0)")
    print(f"  Balance Fy + Ry = {sum_fy + sum_ry:+.2e} kN (Expected 0)")
    print("=" * 60)


def main():
    while True:
        clear_screen()
        show_menu()

        json_files = glob.glob("*.json")
        json_files = [f for f in json_files if not f.endswith("_report.json")]

        print("\nAvailable geometry files:")
        if json_files:
            for idx, file_name in enumerate(json_files, 1):
                print(f"  [{idx}] {file_name}")
        else:
            print("  (No .json files found in current folder)")

        print("\nOptions:")
        print("  [W] Enter file path")
        print("  [Q] Quit")

        choice = input("\nChoose an option or file no. : ").strip()

        if choice.lower() == "q":
            print("\nExiting program.")
            break

        selected_file = None

        if choice.lower() == "w":
            selected_file = input("Enter path to file: ").strip()
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(json_files):
                selected_file = json_files[idx]
            else:
                print("\nInvalid selection index.")
        else:
            print("\nInvalid option. Please try again.")

        if selected_file:
            if not os.path.exists(selected_file):
                print(f"\nFile '{selected_file}' does not exist.")
            else:
                base_name = os.path.splitext(os.path.basename(selected_file))[0]
                out_json = f"{base_name}_report.json"
                out_img = f"{base_name}_dashboard.png"

                print(f"\nProcessing: {selected_file} ...")
                try:
                    nodes, elements, E, A, loads, fixed_dofs, title = load_truss(selected_file)
                    U, forces, stresses, reactions = solve_truss(nodes, elements, E, A, loads, fixed_dofs)

                    generate_report(
                        json_input_path=selected_file,
                        json_output_path=out_json,
                        img_output_path=out_img,
                    )
                    
                    print("\n" + "=" * 50)
                    print("SUCCESS!")
                    print(f"   ├─ JSON Report: {out_json}")
                    print(f"   └─ PNG Dashboard: {out_img}")
                    print("=" * 50 + "\n")

                    print_summary(nodes, elements, U, forces, stresses, reactions, loads, title)

                except Exception as e:
                    print(f"\nSolver execution failed:\n{e}")

        input("\nPress ENTER to continue...")


if __name__ == "__main__":
    main()