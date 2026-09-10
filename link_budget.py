"""Link budget for the FAROS constellation.

Computes the derived values quoted in Table 1 and Figure 2C of the manuscript:
effective isotropic radiated power, received power, the thermal noise floor,
signal-to-noise ratio against range, and the maximum range permitted by the
required carrier-to-noise ratio with margin.

The maximum permitted range is the quantity that determines whether the link
budget or the geometric inter-satellite link limit is the binding constraint
during graph construction.

Usage:  python link_budget.py
"""

import math

# Physical constants, in explicit units.
C_M_S = 299_792_458.0        # m/s
C_KM_S = 299_792.458         # km/s
K_BOLTZ = 1.380649e-23       # J/K

# Link parameters, matching Table 1 of the manuscript.
PT_W = 5.0                   # transmit power, W
GT = 3000.0                  # transmit antenna gain, linear
GR = 3000.0                  # receive antenna gain, linear
FREQ_HZ = 12e9               # carrier frequency, Ku band
T_SYS_K = 290.0              # system noise temperature, K
BW_HZ = 20e6                 # bandwidth, Hz

# Constraints applied during graph construction.
REQUIRED_CN_DB = 10.0        # required carrier-to-noise ratio
MARGIN_DB = 3.0              # link margin
ISL_RANGE_KM = 2500.0        # geometric inter-satellite link limit

LAMBDA_M = C_M_S / FREQ_HZ
BAR = "-" * 66


def db(x):
    return 10.0 * math.log10(x)


def received_power_w(distance_km):
    """Friis transmission equation."""
    d_m = distance_km * 1000.0
    return PT_W * GT * GR * (LAMBDA_M / (4.0 * math.pi * d_m)) ** 2


def noise_power_w():
    """Thermal noise floor, N = kTB."""
    return K_BOLTZ * T_SYS_K * BW_HZ


def snr_db(distance_km):
    return db(received_power_w(distance_km) / noise_power_w())


def max_range_km(required_db):
    """Range at which the SNR falls to the required value.

    Free-space loss scales as the square of distance, so the SNR falls by
    20 dB per decade of range. Solving from a known reference point.
    """
    ref_km = 1000.0
    return ref_km * 10.0 ** ((snr_db(ref_km) - required_db) / 20.0)


def main():
    n_w = noise_power_w()

    print("FAROS link budget")
    print(BAR)
    print(f"  {'Transmit power':<32}{PT_W:>10.1f} W      {db(PT_W):>8.2f} dBW")
    print(f"  {'Transmit antenna gain':<32}{GT:>10.0f}        {db(GT):>8.2f} dBi")
    print(f"  {'Receive antenna gain':<32}{GR:>10.0f}        {db(GR):>8.2f} dBi")
    print(f"  {'EIRP':<32}{'':>10}        {db(PT_W * GT):>8.2f} dBW")
    print(f"  {'Pt x Gt x Gr':<32}{'':>10}        {db(PT_W * GT * GR):>8.2f} dBW")
    print(f"  {'Carrier frequency':<32}{FREQ_HZ/1e9:>10.1f} GHz")
    print(f"  {'Wavelength':<32}{LAMBDA_M*100:>10.2f} cm")
    print(f"  {'System noise temperature':<32}{T_SYS_K:>10.0f} K")
    print(f"  {'Bandwidth':<32}{BW_HZ/1e6:>10.0f} MHz    {db(BW_HZ):>8.2f} dBHz")
    print(f"  {'Noise power, kTB':<32}{n_w:>10.3e} W    {db(n_w):>8.2f} dBW")

    print(f"\n{BAR}\nSignal-to-noise ratio against range\n{BAR}")
    print(f"  {'Range (km)':>12}{'Pr (W)':>16}{'SNR (linear)':>16}{'SNR (dB)':>12}")
    for d in (300.0, 550.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0):
        pr = received_power_w(d)
        print(f"  {d:>12,.0f}{pr:>16.3e}{pr/n_w:>16,.0f}{snr_db(d):>12.2f}")

    print(f"\n{BAR}\nMaximum permitted range\n{BAR}")
    for req, label in ((REQUIRED_CN_DB, f"C/N = {REQUIRED_CN_DB:.0f} dB, no margin"),
                       (REQUIRED_CN_DB + MARGIN_DB,
                        f"C/N = {REQUIRED_CN_DB:.0f} dB + {MARGIN_DB:.0f} dB margin")):
        print(f"  {label:<38}{max_range_km(req):>12,.0f} km")

    permitted = max_range_km(REQUIRED_CN_DB + MARGIN_DB)
    print(f"\n  Geometric inter-satellite link limit:{ISL_RANGE_KM:>14,.0f} km")
    binding = "geometric" if ISL_RANGE_KM < permitted else "link budget"
    print(f"  Binding constraint during graph construction: {binding}")
    if ISL_RANGE_KM < permitted:
        print("  The link-budget gate therefore admits every link the geometric")
        print("  limit already allows, and alters no reported path.")


if __name__ == "__main__":
    main()
