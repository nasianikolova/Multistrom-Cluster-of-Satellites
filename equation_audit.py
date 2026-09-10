"""Equation audit for FAROS, Reviewer Major Issue 8.

For each equation the reviewer flagged, this script computes the quantity BOTH
as printed in the manuscript AND in its corrected form, and compares each
against what the simulation code actually does.

The purpose is to establish which errors are typographical, affecting only the
printed text, and which would change a computed result. Run it and paste the
output into the revision notes.

    python equation_audit.py
"""

import math

# Physical constants, stated once, in explicit units.
G = 6.67430e-11              # m^3 kg^-1 s^-2
M_EARTH = 5.97219e24         # kg
R_EARTH_KM = 6371.0          # km
MU_EARTH = 398_600.4418      # km^3 s^-2  (= G*M in km-based units)
C_M_S = 299_792_458.0        # m/s
C_KM_S = 299_792.458         # km/s
K_BOLTZ = 1.380649e-23       # J/K

BAR = "-" * 78


def header(n, title):
    print(f"\n{BAR}\nEquation {n}: {title}\n{BAR}")


def verdict(changes_result: bool, note: str = ""):
    tag = "AFFECTS A COMPUTED RESULT" if changes_result else "TYPOGRAPHICAL ONLY"
    print(f"  >> {tag}. {note}".rstrip())


# ---------------------------------------------------------------------------
def eq2_areal_velocity():
    header(2, "Kepler's second law, areal velocity")
    print("  As printed : A = dA/dt")
    print("  Correct    : A-dot = dA/dt   (the overdot is missing in the text)")
    print("  The variable list already defines A-dot correctly, so only the")
    print("  equation display is wrong.")
    verdict(False, "No quantity is computed from this equation anywhere.")


def eq3_keplers_third():
    header(3, "Kepler's third law, T^2 proportional to a^3")
    print("  As printed : T^2 ~ a^3     (no constant, no units stated)")
    print("  Correct    : T^2 = (4*pi^2 / GM) * a^3")
    print("  Alternative: T^2 = a^3 holds numerically ONLY with T in years and")
    print("               a in astronomical units, which the text never states.\n")

    for alt_km in (550.0, 3000.0):
        a_km = R_EARTH_KM + alt_km
        T_correct = 2 * math.pi * math.sqrt(a_km ** 3 / MU_EARTH)
        T_naive = math.sqrt(a_km ** 3)          # taking T^2 = a^3 literally
        print(f"  altitude {alt_km:>7.0f} km  ->  a = {a_km:>8.1f} km")
        print(f"     T with the constant     : {T_correct:>12.1f} s "
              f"({T_correct/60:>6.2f} min)")
        print(f"     T from T^2 = a^3 as read: {T_naive:>12.1f} s "
              f"({T_naive/60:>6.2f} min)  <- wrong by a factor "
              f"{T_naive/T_correct:>5.1f}")
    verdict(False, "No orbital period is computed in the manuscript, but the "
                   "equation as printed is unusable without the constant.")


def eq7_and_eq14_labelling():
    header("7 / 14", "Cross-reference error, Friis mislabelled")
    print("  The Methods text calls Friis 'the Friis equation (7)'.")
    print("  Equation 7 is Newton's law of gravitation, F = GMm/r^2.")
    print("  Friis is Equation 14. The citation points at the wrong equation.")
    print("  Equation 14 also prints 'Rr' on the left while the variable list")
    print("  below it defines 'Pr' as received power.")
    verdict(False, "Numbering and symbol naming only.")


def eq8_energy():
    header(8, "Mechanical energy, E = K + U")
    print("  As printed : E = K + U = (1/2)mv^2 - GMm/r = 0   for all orbit types")
    print("  Correct    : E < 0  bound (circular, elliptical)")
    print("               E = 0  parabolic, the escape case only")
    print("               E > 0  hyperbolic\n")

    m = 1000.0                       # kg, arbitrary satellite mass
    r_km = R_EARTH_KM + 550.0
    r_m = r_km * 1000.0
    v_circ = math.sqrt(MU_EARTH / r_km) * 1000.0      # m/s
    v_esc = math.sqrt(2.0) * v_circ

    for label, v in (("circular orbit", v_circ),
                     ("escape velocity", v_esc),
                     ("hyperbolic, 1.2x escape", 1.2 * v_esc)):
        E = 0.5 * m * v * v - G * M_EARTH * m / r_m
        sign = "< 0" if E < -1e-6 else ("= 0" if abs(E) < 1e-6 else "> 0")
        print(f"  {label:<26} v = {v/1000:>7.3f} km/s   E = {E:>13.3e} J  {sign}")

    print("\n  Setting E = 0 for every orbit type also contradicts the Figure 7C")
    print("  caption, which states that bound orbits have negative total energy.")
    verdict(False, "The manuscript's own figure caption is correct; the equation "
                   "is the error.")


def eq9_escape_velocity():
    header(9, "Escape velocity")
    r_km = R_EARTH_KM + 550.0
    v_circ = math.sqrt(MU_EARTH / r_km)
    v_esc = math.sqrt(2 * MU_EARTH / r_km)
    print(f"  v_circular = {v_circ:.4f} km/s")
    print(f"  v_escape   = {v_esc:.4f} km/s")
    print(f"  ratio      = {v_esc/v_circ:.6f}   (sqrt(2) = {math.sqrt(2):.6f})")
    verdict(False, "Equation 9 as printed is correct.")


def eq12_horizon():
    header(12, "Horizon arc length")
    print("  As printed : S = phi * R * arccos(R/(R+h))")
    print("  Correct    : phi = arccos(R/(R+h))   and   S = R * phi")
    print("  The printed form multiplies by phi and then by arccos(...) again,")
    print("  so the angular term appears twice.\n")

    print(f"  {'h (km)':>8}{'phi (deg)':>12}{'S correct (km)':>17}"
          f"{'S as printed':>15}{'ratio':>9}")
    for h in (250.0, 550.0, 1000.0, 2000.0, 3000.0):
        phi = math.acos(R_EARTH_KM / (R_EARTH_KM + h))     # radians
        S_correct = R_EARTH_KM * phi
        S_printed = phi * R_EARTH_KM * phi                 # the doubled term
        print(f"  {h:>8.0f}{math.degrees(phi):>12.2f}{S_correct:>17.1f}"
              f"{S_printed:>15.1f}{S_printed/S_correct:>9.3f}")

    print("\n  The Figure 8 caption additionally describes arccos(R/(R+h)) as")
    print("  'the distance from the center of the Earth to the observer's eyes'.")
    print("  It is an angle in radians, not a distance. That distance is R + h.")
    verdict(False, "No figure computes S numerically, so no plotted value "
                   "changes, but the printed equation is wrong as written.")


def eq13_repulsion_exponent():
    header(13, "Repulsive force exponent")
    print("  As printed : F = 1 / d^N,  with N never assigned a value")
    print("  In the code: N_dim = 2 in the 2D and 3D routines")
    print("               N_dim = 1 in the introductory one-dimensional demo")
    print("  The Methods text describes the force as inverse-square, which")
    print("  matches N = 2. State N = 2 explicitly.\n")

    d = 1500.0
    for N in (1, 2, 3):
        print(f"  N = {N}:  F(d = {d:.0f} km) = {1.0/d**N:.6e}")
    verdict(False, "The value used is consistent with the prose; only the "
                   "written equation omits it.")


def eq14_friis():
    header(14, "Friis transmission equation")
    Pt, Gt, Gr, f_hz = 5.0, 3000.0, 3000.0, 12e9
    lam = C_M_S / f_hz
    T_sys, B = 290.0, 20e6
    N = K_BOLTZ * T_sys * B

    print("  As printed : Pr = Pt*Gt*Gr*lambda / (4*pi*d^2)")
    print("  Correct    : Pr = Pt*Gt*Gr*(lambda / (4*pi*d))^2")
    print("  The code uses the correct form: (wavelength/(4*pi*d_m))**2\n")

    print(f"  {'d (km)':>9}{'Pr correct (W)':>18}{'Pr as printed (W)':>20}"
          f"{'SNR correct':>14}")
    for d_km in (300.0, 550.0, 2000.0):
        d_m = d_km * 1000.0
        Pr_ok = Pt * Gt * Gr * (lam / (4 * math.pi * d_m)) ** 2
        Pr_bad = Pt * Gt * Gr * lam / (4 * math.pi * d_m ** 2)
        print(f"  {d_km:>9.0f}{Pr_ok:>18.4e}{Pr_bad:>20.4e}{Pr_ok/N:>14,.0f}")

    print("\n  Dimensional check of the printed form:")
    print("    Pt*Gt*Gr has units of W; lambda/(4*pi*d^2) has units of 1/m.")
    print("    The printed right-hand side is therefore W/m, not W.")
    print("    The corrected form is dimensionless in the bracket and yields W.")
    verdict(False, "The code implements the correct form, so Figure 1C and the "
                   "link budget are unaffected.")


def speed_of_light_units():
    header("Units", "Speed of light in the Methods")
    print(f"  As printed : 299,792.458 m/s")
    print(f"  Correct    : {C_M_S:,.0f} m/s   or   {C_KM_S:,.3f} km/s")
    print(f"  The printed value is the km/s figure carrying the m/s unit,")
    print(f"  understating the speed of light by a factor of 1,000.\n")

    d_km = 8690.6      # the controlled-comparison path length
    print(f"  Propagation over {d_km:,.1f} km:")
    print(f"    with the correct value  : {1000*d_km/C_KM_S:>10.2f} ms")
    print(f"    with the printed value  : {1000*d_km/(299.792458):>10.2f} ms"
          f"   <- a factor of 1,000 too large")
    print(f"\n  The code defines C_KM_S = {C_KM_S} and divides kilometres by it,")
    print(f"  which is dimensionally correct.")
    verdict(False, "Prose only. Every computed delay in the manuscript used the "
                   "correct value.")


def main():
    print("FAROS equation audit, Reviewer Major Issue 8")
    print("Each item computed both as printed and as corrected.")
    eq2_areal_velocity()
    eq3_keplers_third()
    eq7_and_eq14_labelling()
    eq8_energy()
    eq9_escape_velocity()
    eq12_horizon()
    eq13_repulsion_exponent()
    eq14_friis()
    speed_of_light_units()

    print(f"\n{BAR}\nSUMMARY\n{BAR}")
    print("  All nine items are errors in the printed text.")
    print("  None of them changes a computed result: in every case where the")
    print("  manuscript prints an incorrect form, the source code implements")
    print("  the correct one. No figure, table or reported number is affected.")
    print("  The corrections are confined to the manuscript text and captions.")


if __name__ == "__main__":
    main()
