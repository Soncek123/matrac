"""
apollonian_gasket.py

Generate and display (and optionally save) an Apollonian gasket using
the complex-form Descartes (Soddy) circle theorem.

Usage:
    python apollonian_gasket.py            # show a default gasket
    python apollonian_gasket.py -o out.png --max-circles 1000 --min-radius 0.002

Dependencies:
    numpy, matplotlib
Install with: pip install numpy matplotlib
"""

import math
import cmath
import argparse
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# -------------------------
# Geometry / Descartes code
# -------------------------
def descartes_solutions(k1, z1, k2, z2, k3, z3):
    """
    Return the two (k4, z4) solutions (curvature and complex center)
    from the complex Descartes theorem for three given circles.
    k's are curvatures (1/r). z's are complex centers.
    """
    # curvatures
    s = k1 + k2 + k3
    prod = k1 * k2 + k2 * k3 + k3 * k1
    # two curvature solutions
    k4_a = s + 2 * math.sqrt(prod)
    k4_b = s - 2 * math.sqrt(prod)

    # centers (complex)
    S = k1 * z1 + k2 * z2 + k3 * z3
    T = k1 * k2 * z1 * z2 + k2 * k3 * z2 * z3 + k3 * k1 * z3 * z1
    sqrt_term = cmath.sqrt(T)

    # Note: denominator must match curvature used in numerator's branch.
    z4_a = (S + 2 * sqrt_term) / k4_a
    z4_b = (S - 2 * sqrt_term) / k4_b

    return (float(k4_a), z4_a), (float(k4_b), z4_b)


def add_circle_if_new(circles, k, z, eps=1e-10):
    """
    Add circle (k, z) to list circles if it is not already present.
    circles is a list of dicts: {'k':..., 'z':complex, 'r':...}
    """
    # clean invalid / degenerate
    if abs(k) < 1e-12:
        return False
    r = abs(1.0 / k)
    # Avoid extremely tiny or nan centers
    if not (math.isfinite(r) and math.isfinite(z.real) and math.isfinite(z.imag)):
        return False
    for c in circles:
        if abs(c['k'] - k) < eps and abs(c['z'] - z) < 1e-8:
            return False
    circles.append({'k': float(k), 'z': complex(z), 'r': r})
    return True


# -------------------------
# Build initial configuration
# -------------------------
def initial_three_equal(radius=1.0):
    """
    Return a list with three equal tangent circles (radius given)
    arranged as an equilateral triangle, and the enclosing outer circle.
    """
    r = float(radius)
    # centers of three mutually tangent equal circles:
    c1 = complex(0.0, 0.0)
    c2 = complex(2 * r, 0.0)
    c3 = complex(r, math.sqrt(3) * r)

    circles = []
    k_small = 1.0 / r
    add_circle_if_new(circles, k_small, c1)
    add_circle_if_new(circles, k_small, c2)
    add_circle_if_new(circles, k_small, c3)

    # compute outer circle tangent to all three: solve for Z and R.
    # For circles i and j: |Z - ci|^2 - |Z - cj|^2 = (R - ri)^2 - (R - rj)^2
    # This simplifies to linear equations for Z = (x,y).
    ci = [c1, c2, c3]
    ri = [r, r, r]
    A = []
    b = []
    for i in range(2):
        x0, y0 = ci[0].real, ci[0].imag
        xi, yi = ci[i + 1].real, ci[i + 1].imag
        ri0, rii = ri[0], ri[i + 1]
        A.append([2 * (xi - x0), 2 * (yi - y0)])
        b.append((xi * xi + yi * yi - rii * rii) - (x0 * x0 + y0 * y0 - ri0 * ri0))
    A = np.array(A)
    b = np.array(b)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    Z = complex(sol[0], sol[1])
    R = ri[0] + abs(Z - ci[0])
    k_outer = -1.0 / R  # negative curvature for enclosing circle
    add_circle_if_new(circles, k_outer, Z)

    return circles


def initial_three_unequal(r1=1.0, r2=0.8, r3=0.6):
    """
    Build three mutually tangent circles of radii r1, r2, r3,
    and their enclosing outer circle.
    Returns a list of 4 circles (3 inner + 1 outer).
    """
    # Place circle1 at origin
    c1 = complex(0.0, 0.0)
    # Place circle2 on x-axis tangent to circle1
    c2 = complex(r1 + r2, 0.0)
    # Compute circle3 position tangent to both 1 and 2
    x = (r1 + r3) * (r1 + r3) - (r2 + r3) * (r2 + r3) + (r1 + r2) * (r1 + r2)
    x /= 2 * (r1 + r2)
    y = math.sqrt(max(0.0, (r1 + r3) ** 2 - x ** 2))
    c3 = complex(x, y)

    circles = []
    add_circle_if_new(circles, 1 / r1, c1)
    add_circle_if_new(circles, 1 / r2, c2)
    add_circle_if_new(circles, 1 / r3, c3)

    # Find outer circle tangent to all three
    ci = [c1, c2, c3]
    ri = [r1, r2, r3]
    A, b = [], []
    for i in range(2):
        x0, y0 = ci[0].real, ci[0].imag
        xi, yi = ci[i + 1].real, ci[i + 1].imag
        ri0, rii = ri[0], ri[i + 1]
        A.append([2 * (xi - x0), 2 * (yi - y0)])
        b.append((xi * xi + yi * yi - rii * rii) - (x0 * x0 + y0 * y0 - ri0 * ri0))
    A, b = np.array(A), np.array(b)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    Z = complex(sol[0], sol[1])
    R = ri[0] + abs(Z - ci[0])
    k_outer = -1.0 / R
    add_circle_if_new(circles, k_outer, Z)
    return circles


# -------------------------
# Packing algorithm
# -------------------------
def build_apollonian(circles, max_circles=1000, min_radius=1e-3, max_iterations=200000):
    """
    Given an initial list of circles, iteratively apply Descartes theorem
    to fill gaps. Returns the augmented circles list.
    - max_circles: stop after this many circles total
    - min_radius: don't add circles smaller than this
    """
    checked = set()  # cache triple keys we already attempted
    iterations = 0

    def triple_key(i, j, k):
        return tuple(sorted((i, j, k)))

    while iterations < max_iterations and len(circles) < max_circles:
        added_in_pass = 0
        n = len(circles)
        # iterate triples (combinatorially); practical for a few thousand circles.
        for (i, j, k) in combinations(range(n), 3):
            key = triple_key(i, j, k)
            if key in checked:
                continue
            checked.add(key)
            cA, cB, cC = circles[i], circles[j], circles[k]
            # Quick tangency check: skip triples that are obviously not mutually tangent
            good = True
            for (ci, cj) in ((cA, cB), (cB, cC), (cC, cA)):
                dist = abs(ci['z'] - cj['z'])
                expected = ci['r'] + cj['r']
                # allow small slack (relative to radius)
                if abs(dist - expected) > max(1e-3, 0.05 * min(ci['r'], cj['r'])):
                    good = False
                    break
            if not good:
                continue

            try:
                sols = descartes_solutions(cA['k'], cA['z'], cB['k'], cB['z'], cC['k'], cC['z'])
            except Exception:
                continue

            for (k_new, z_new) in sols:
                # consider only real-valued curvature (imaginary part negligible)
                if isinstance(k_new, complex) and abs(k_new.imag) > 1e-9:
                    continue
                k_new = float(k_new.real) if isinstance(k_new, complex) else float(k_new)
                if abs(k_new) < 1e-12:
                    continue
                r_new = abs(1.0 / k_new)
                if r_new < min_radius:
                    continue
                if add_circle_if_new(circles, k_new, z_new):
                    added_in_pass += 1
                    iterations += 1
                    if len(circles) >= max_circles or iterations >= max_iterations:
                        break
            if len(circles) >= max_circles or iterations >= max_iterations:
                break
        if added_in_pass == 0:
            break  # no more circles to add
    return circles


# -------------------------
# Plotting utilities
# -------------------------
def plot_circles(circles, show=True, outpath=None, figsize=(8, 8), cmap='viridis'):
    """
    Plot circles. If outpath provided, save to file.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_axis_off()

    # color by radius (log scale)
    radii = [c['r'] for c in circles]
    log_r = np.log(np.array(radii) + 1e-16)
    norm = (log_r - log_r.min()) / max(1e-12, (log_r.max() - log_r.min()))
    cmap_obj = plt.get_cmap(cmap)

    # draw largest first (so smaller ones are visible on top)
    for c, t in sorted(zip(circles, norm), key=lambda pair: -pair[0]['r']):
        circ = Circle((c['z'].real, c['z'].imag), radius=c['r'],
                      facecolor=cmap_obj(t), edgecolor='k', linewidth=0.3)
        ax.add_patch(circ)

    # autoscale
    xs = [c['z'].real for c in circles]
    ys = [c['z'].imag for c in circles]
    rs = [c['r'] for c in circles]
    xmin = min(x - r for x, r in zip(xs, rs))
    xmax = max(x + r for x, r in zip(xs, rs))
    ymin = min(y - r for y, r in zip(ys, rs))
    ymax = max(y + r for y, r in zip(ys, rs))
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - 0.05 * dx, xmax + 0.05 * dx)
    ax.set_ylim(ymin - 0.05 * dy, ymax + 0.05 * dy)

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    plt.close(fig)


# -------------------------
# CLI & main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Render an Apollonian gasket (circle packing).")
    parser.add_argument("--unequal", nargs=3, type=float, metavar=("R1", "R2", "R3"),
                        help="use three unequal initial circles with given radii")
    parser.add_argument("--initial-radius", type=float, default=1.0, help="radius of the three initial equal circles")
    parser.add_argument("--max-circles", type=int, default=800, help="maximum number of circles to generate")
    parser.add_argument("--min-radius", type=float, default=0.003, help="smallest radius to include")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="output filename (PNG/SVG). If omitted, shows interactively.")
    parser.add_argument("--no-show", action="store_true", help="don't display the interactive window")
    parser.add_argument("--cmap", type=str, default="viridis", help="matplotlib colormap for circles")
    args = parser.parse_args()

    # build
    if args.unequal:
        r1, r2, r3 = args.unequal
        circles = initial_three_unequal(r1, r2, r3)
    else:
        circles = initial_three_equal(radius=args.initial_radius)

    print(f"Initial circles: {len(circles)}")
    circles = build_apollonian(circles, max_circles=args.max_circles, min_radius=args.min_radius)
    print(f"Total circles after packing: {len(circles)}")

    # plot
    show = not args.no_show
    plot_circles(circles, show=show, outpath=args.out, cmap=args.cmap)


if __name__ == "__main__":
    main()
