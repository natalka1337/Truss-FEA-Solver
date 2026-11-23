import sys
import numpy as np

try:
    from truss_io import load_truss
except ImportError:
    from .truss_io import load_truss


def solve_truss(nodes, elements, E, A, loads, fixed_dofs):
    """
    Rozwiązuje zadanie statyki kratownicy 2D metodą elementów skończonych.
    
    Zwraca:
    - U: wektor przemieszczeń węzłów [m]
    - forces: wektor sił osiowych w prętach [N]
    - stresses: wektor naprężeń osiowych w prętach [Pa]
    - reactions: wektor reakcji podporowych [N]
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    E = np.asarray(E, dtype=float)
    A = np.asarray(A, dtype=float)
    loads = np.asarray(loads, dtype=float).flatten()
    fixed_dofs = np.unique(np.asarray(fixed_dofs, dtype=int))

    n_nodes = len(nodes)
    n_elems = len(elements)
    total_dofs = 2 * n_nodes

    if len(loads) != total_dofs:
        raise ValueError(f"wektor obciazen ({len(loads)}) dofs ({total_dofs}).")

    K = np.zeros((total_dofs, total_dofs))

    for e, (n1, n2) in enumerate(elements):
        d = nodes[n2] - nodes[n1]
        L = np.hypot(d[0], d[1])
        if L <= 0:
            raise ValueError(f"Element {e} is 0 length")
        c, s = d / L

        ke = (E[e] * A[e] / L) * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += ke[i, j]

    free = np.setdiff1d(np.arange(total_dofs), fixed_dofs)
    U = np.zeros(total_dofs)

    if len(free) > 0:
        try:
            U[free] = np.linalg.solve(K[np.ix_(free, free)], loads[free])
        except np.linalg.LinAlgError:
            raise ValueError("Matrix failed")

    forces = np.zeros(n_elems)
    stresses = np.zeros(n_elems)
    for e, (n1, n2) in enumerate(elements):
        d = nodes[n2] - nodes[n1]
        L = np.hypot(d[0], d[1])
        c, s = d / L
        ue = np.array([U[2*n1], U[2*n1+1], U[2*n2], U[2*n2+1]])
        forces[e] = (E[e] * A[e] / L) * np.dot([-c, -s, c, s], ue)
        stresses[e] = forces[e] / A[e]

    reactions = K @ U - loads
    reactions[free] = 0.0
    reactions[np.abs(reactions) < 1e-9] = 0.0

    return U, forces, stresses, reactions


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "simple_triangle.json"
    nodes, elements, E, A, loads, fixed_dofs, title = load_truss(json_path)
    U, forces, stresses, reactions = solve_truss(nodes, elements, E, A, loads, fixed_dofs)