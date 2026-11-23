# TrussSolver

A truss linear-elastic FEA solver using the direct stiffness method. Computes nodal displacements, element axial forces, normal stresses, support reactions, and verifies static equilibrium. Includes automated export to JSON and 4-panel matplotlib dashboards.

## Repository Structure

```text
├── examples/
│   ├── simple_triangle.json   # 3-bar benchmark
│   ├── zbiezne.json           # Fully triangulated model
│   └── rozbiezne.json         # Non-convergent node offset model
├── output/
│   ├── rozbiezne_report.json
│   ├── rozbiezne_dashboard.json
|   ├── zbiezne_report.json
│   └── zbiezne_dasboard.png
├── truss_io.py                # JSON input parser and SI unit converter
├── solver.py                  # Core FEA solver (K assembly, Ku = F solve)
├── Truss_reporter.py          # Dashboard plot and JSON report generator
├── main.py                    # Interactive CLI runner
└── README.md
```

## Input Format

Models are specified in JSON. Displacements and forces are calculated in SI units regardless of input dimensions:

```json
{
  "title": "Triangle Benchmark",
  "units": { "length": "m", "force": "kN", "modulus": "GPa", "area": "cm2" },
  "materials": { "steel": { "E": 210.0 } },
  "sections": { "std": { "A": 25.0 } },
  "nodes": [
    { "id": 0, "x": 0.0, "y": 0.0, "support": "pinned", "load": [0.0, 0.0] },
    { "id": 1, "x": 4.0, "y": 0.0, "support": "roller_x", "load": [0.0, 0.0] },
    { "id": 2, "x": 2.0, "y": 2.0, "load": [0.0, -100.0] }
  ],
  "elements": [
    { "id": 0, "nodes": [0, 1], "material": "steel", "section": "std" },
    { "id": 1, "nodes": [0, 2], "material": "steel", "section": "std" },
    { "id": 2, "nodes": [1, 2], "material": "steel", "section": "std" }
  ]
}
```

Supports: `pinned`, `roller_x` (free in X, fixed in Y), `roller_y` (fixed in X, free in Y), or explicit booleans `fixed_x` / `fixed_y`.

## Case Study: Node Convergence vs. Joint Offset

Comparison between an idealized triangulated truss with concurrent member axes (`zbiezne.json`) and the same layout where diagonal connections are offset along the chords (`rozbiezne.json`), introducing quadrilaterals and joint eccentricity. Both cases use identical boundary conditions and a 50 kN horizontal load.

| Parameter | Concurrent Nodes (`zbiezne.json`) | Offset Nodes (`rozbiezne.json`) | Delta |
| :--- | :---: | :---: | :---: |
| Nodes / Elements | 10 / 21 | 14 / 25 | +4 / +4 |
| Total mass | 1237.1 kg | 1097.2 kg | -11.3% |
| Max displacement \|u\| | 0.867 mm (node 8) | 3.119 mm (node 12) | +259.8% (3.6x) |
| Max compression force | -45.73 kN (elem 6) | -69.44 kN (elem 8) | +51.8% |
| Max tension force | +15.94 kN (elem 14) | +42.69 kN (elem 14) | +167.8% (2.7x) |
| Peak normal stress | 18.29 MPa | 27.78 MPa | +51.8% |
| Residual equilibrium error | < 1e-9 N | < 1e-9 N | Exact |

### Observations

- Offsetting diagonal brace attachments by 0.4 m along the chord breaks triangulation and increases deflection by 260% (from 0.87 mm to 3.12 mm).
- Peak tension increases by 168% and compression by 52%, requiring heavier profiles to prevent yielding or member buckling despite a nominal 11% reduction in initial member length.
- Concurrent member centrelines at joints are essential for structural stiffness and force distribution.
