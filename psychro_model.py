"""
psychro_model.py
Calculs psychrométriques conformes à l'ASHRAE Fondamentaux 2017.
Unités internes : SI (°C, Pa, kg/kg as).
"""

import math
import numpy as np
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════
# Constantes physiques
# ═══════════════════════════════════════════════════════════════════════
PATM_MER = 101_325.0   # Pa  – pression atmosphérique au niveau de la mer
CPa      = 1.006        # kJ/(kg·K) – chaleur spécifique air sec
CPv      = 1.86         # kJ/(kg·K) – chaleur spécifique vapeur d'eau
hfg0     = 2501.0       # kJ/kg     – chaleur latente de vaporisation à 0 °C
Ra       = 287.055      # J/(kg·K)  – constante des gaz, air sec

# Alias anglais conservé pour compatibilité
PATM_SEA = PATM_MER


# ═══════════════════════════════════════════════════════════════════════
# Classe état psychrométrique
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class EtatPsychro:
    Tbs : float    # Température bulbe sec           [°C]
    W   : float    # Teneur en eau (humidité absolue)[kg/kg as]
    HR  : float    # Humidité relative               [0–1]
    Tbh : float    # Température bulbe humide        [°C]
    Tro : float    # Température de rosée            [°C]
    h   : float    # Enthalpie spécifique            [kJ/kg as]
    v   : float    # Volume spécifique               [m³/kg as]
    P   : float = PATM_MER   # Pression atm.         [Pa]

# Alias anglais (compatibilité app.py)
PsychroState = EtatPsychro


# ═══════════════════════════════════════════════════════════════════════
# Équations ASHRAE 2017
# ═══════════════════════════════════════════════════════════════════════

def altitude_vers_pression(alt_m: float) -> float:
    """Altitude [m] → pression atmosphérique [Pa] (ASHRAE éq. 3)."""
    return 101_325.0 * (1.0 - 2.25577e-5 * alt_m) ** 5.2559

altitude_to_pressure = altitude_vers_pression  # alias


def psat(T_C: float) -> float:
    """
    Pression de vapeur saturante [Pa].
    Branche eau liquide  : ASHRAE Fondamentaux 2017, Ch.1, éq. (6).
    Branche glace        : formule de Buck (1981) — robuste numériquement.
    Erreur < 0.05 % sur [-40 ; +60 °C].

    Plage de validité : -40 °C ≤ T_C ≤ 80 °C.
    En dehors, les valeurs sont extrapolées — précision non garantie.
    """
    if T_C < -40.0 or T_C > 80.0:
        import warnings
        warnings.warn(
            f"psat({T_C:.1f} °C) : hors plage de validité [-40 ; 80 °C]. "
            "Les résultats peuvent être inexacts.",
            stacklevel=2,
        )
    if T_C <= 0.0:           # glace
        # Buck (1981) : précision 0.04 % sur [-40 ; 0 °C]
        return 611.2 * math.exp(22.46 * T_C / (272.62 + T_C))
    else:                    # eau liquide — ASHRAE éq. (6)
        T = T_C + 273.15
        C = (-5.8002206e3, 1.3914993, -4.8640239e-2,
              4.1764768e-5, -1.4452093e-8, 6.5459673)
        return math.exp(C[0]/T + C[1] + C[2]*T + C[3]*T**2
                        + C[4]*T**3 + C[5]*math.log(T))


def W_sat(T: float, P: float = PATM_MER) -> float:
    """Teneur en eau à saturation [kg/kg as]."""
    ps = psat(T)
    return 0.621945 * ps / (P - ps)


def W_de_Tbs_HR(Tbs: float, HR: float, P: float = PATM_MER) -> float:
    """W [kg/kg as] depuis Tbs et HR [0–1]."""
    ps = psat(Tbs)
    pw = HR * ps
    return 0.621945 * pw / (P - pw)


def W_de_Tbs_Tbh(Tbs: float, Tbh: float, P: float = PATM_MER) -> float:
    """W [kg/kg as] depuis Tbs et Tbh. ASHRAE éq. 35/36."""
    Ws_bh = W_sat(Tbh, P)
    if Tbs >= 0:
        return ((2501.0 - 2.381 * Tbh) * Ws_bh - 1.006 * (Tbs - Tbh)) \
               / (2501.0 + 1.805 * Tbs - 4.186 * Tbh)
    else:
        return ((2830.0 - 0.240 * Tbh) * Ws_bh - 1.006 * (Tbs - Tbh)) \
               / (2830.0 + 1.860 * Tbs - 2.100 * Tbh)


def HR_de_Tbs_W(Tbs: float, W: float, P: float = PATM_MER) -> float:
    """HR [0–1] depuis Tbs et W."""
    pw = W * P / (0.621945 + W)
    ps = psat(Tbs)
    return min(pw / ps, 1.0)


def Tro_de_W(W: float, P: float = PATM_MER) -> float:
    """Température de rosée [°C] par Newton-Raphson."""
    if W <= 0:
        return -50.0
    pw = W * P / (0.621945 + W)
    Tr = 243.04 * math.log(pw / 610.78) / (17.625 - math.log(pw / 610.78))
    for _ in range(50):
        f  = psat(Tr) - pw
        df = (psat(Tr + 0.005) - psat(Tr - 0.005)) / 0.01
        if abs(df) < 1e-14:
            break
        Tr -= f / df
        if abs(f) < 0.05:
            break
    return Tr


def Tbh_de_Tbs_W(Tbs: float, W: float, P: float = PATM_MER,
                 tol: float = 1e-6) -> float:
    """Température de bulbe humide [°C] par Newton-Raphson."""
    Tbh = Tbs - (W_sat(Tbs, P) - W) * 2200
    Tbh = max(min(Tbh, Tbs), -50.0)
    for _ in range(60):
        W_c = W_de_Tbs_Tbh(Tbs, Tbh, P)
        err = W_c - W
        dW  = (W_de_Tbs_Tbh(Tbs, Tbh + 0.005, P) - W_c) / 0.005
        if abs(dW) < 1e-14:
            break
        Tbh -= err / dW
        if abs(err) < tol:
            break
    return Tbh


def enthalpie(Tbs: float, W: float) -> float:
    """Enthalpie spécifique [kJ/kg as]. ASHRAE éq. 32."""
    return CPa * Tbs + W * (hfg0 + CPv * Tbs)


def volume_specifique(Tbs: float, W: float, P: float = PATM_MER) -> float:
    """Volume spécifique [m³/kg as]. ASHRAE éq. 28."""
    return (Ra / P) * (Tbs + 273.15) * (1.0 + W / 0.621945)


# ═══════════════════════════════════════════════════════════════════════
# Calcul d'état complet depuis deux paramètres quelconques
# ═══════════════════════════════════════════════════════════════════════

def calculer_etat(param1: str, val1: float,
                  param2: str, val2: float,
                  P: float = PATM_MER) -> EtatPsychro:
    """
    Calcule l'état psychrométrique complet depuis deux paramètres.
    Noms acceptés :
        'Tdb'/'Tbs'  – température bulbe sec  [°C]
        'RH'/'HR'    – humidité relative       [%]
        'Twb'/'Tbh'  – bulbe humide            [°C]
        'Tdp'/'Tro'  – température de rosée   [°C]
        'W'          – teneur en eau           [g/kg]
    """
    _alias = {"Tdb": "Tbs", "Twb": "Tbh", "Tdp": "Tro", "RH": "HR"}
    param1 = _alias.get(param1, param1)
    param2 = _alias.get(param2, param2)
    ent = {param1: val1, param2: val2}

    if "HR" in ent:
        ent["HR"] /= 100.0
    if "W" in ent:
        ent["W"]  /= 1000.0

    # ---------------------------------------------------------------
    # Résoudre (Tbs, W) depuis les deux paramètres fournis.
    # 10 combinaisons possibles parmi {Tbs, HR, Tbh, Tro, W} :
    #   Tbs+HR, Tbs+Tbh, Tbs+Tro, Tbs+W,
    #   HR+W,   HR+Tbh,  HR+Tro,
    #   W+Tbh,  W+Tro,
    #   Tbh+Tro
    # ---------------------------------------------------------------

    def _bissection(f_lo_pos, lo, hi, n=80, tol=1e-7):
        """Bissection générique. f_lo_pos(x) > 0 pour x proche de lo."""
        for _ in range(n):
            mid = (lo + hi) / 2.0
            if f_lo_pos(mid) > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        return (lo + hi) / 2.0

    # ── Combinaisons avec Tbs (Tbs connu directement) ──────────────
    if "Tbs" in ent and "HR" in ent:
        Tbs = ent["Tbs"]
        W   = W_de_Tbs_HR(Tbs, ent["HR"], P)

    elif "Tbs" in ent and "Tbh" in ent:
        Tbs = ent["Tbs"]
        W   = W_de_Tbs_Tbh(Tbs, ent["Tbh"], P)

    elif "Tbs" in ent and "Tro" in ent:
        Tbs = ent["Tbs"]
        W   = W_de_Tbs_HR(Tbs, psat(ent["Tro"]) / psat(Tbs), P)

    elif "Tbs" in ent and "W" in ent:
        Tbs = ent["Tbs"]
        W   = ent["W"]

    # ── W + HR : Tbs par Newton-Raphson sur psat(Tbs) = pw/HR ──────
    elif "W" in ent and "HR" in ent:
        W  = ent["W"]
        HR = ent["HR"]
        # pw = W*P/(0.621945+W)  et  psat(Tbs) = pw/HR
        pw   = W * P / (0.621945 + W)
        ps_c = pw / HR
        # Démarrage : inversion de Buck autour de 20 °C
        Tbs  = 20.0
        for _ in range(60):
            f  = psat(Tbs) - ps_c
            df = (psat(Tbs + 0.005) - psat(Tbs - 0.005)) / 0.01
            if abs(df) < 1e-14:
                break
            Tbs -= f / df
            if abs(f) < 0.05:
                break

    # ── Tbh + HR : bissection sur Tbs ──────────────────────────────
    elif "Tbh" in ent and "HR" in ent:
        Tbh = ent["Tbh"]
        HR  = ent["HR"]
        # HR_de_Tbs_W(Tbs, W_de_Tbs_Tbh(Tbs,Tbh)) décroît avec Tbs
        Tbs = _bissection(
            lambda t: HR_de_Tbs_W(t, W_de_Tbs_Tbh(t, Tbh, P), P) - HR,
            Tbh, Tbh + 80.0,
        )
        W = W_de_Tbs_Tbh(Tbs, Tbh, P)

    # ── Tro + HR : Newton-Raphson, psat(Tbs) = psat(Tro)/HR ────────
    elif "Tro" in ent and "HR" in ent:
        Tro  = ent["Tro"]
        HR   = ent["HR"]
        ps_c = psat(Tro) / HR
        Tbs  = Tro + 5.0
        for _ in range(60):
            f  = psat(Tbs) - ps_c
            df = (psat(Tbs + 0.005) - psat(Tbs - 0.005)) / 0.01
            if abs(df) < 1e-14:
                break
            Tbs -= f / df
            if abs(f) < 0.05:
                break
        W = W_de_Tbs_HR(Tbs, HR, P)

    # ── Tbh + W : bissection sur Tbs ───────────────────────────────
    elif "Tbh" in ent and "W" in ent:
        Tbh = ent["Tbh"]
        W   = ent["W"]
        # W_de_Tbs_Tbh décroît quand Tbs ↑ → f > 0 pour lo
        Tbs = _bissection(
            lambda t: W_de_Tbs_Tbh(t, Tbh, P) - W,
            Tbh, Tbh + 60.0,
        )

    # ── Tro + W : W connu ; Tbs par inversion psat (Tbs ≥ Tro) ────
    elif "Tro" in ent and "W" in ent:
        W   = ent["W"]
        Tro = ent["Tro"]
        # psat(Tbs) = pw/1 (HR libre) — Tbs = Tro est le cas limite HR=100 %
        # On fixe Tbs = Tro : état à la rosée (cohérent avec les deux paramètres)
        Tbs = Tro

    # ── Tbh + Tro : bissection sur Tbs ─────────────────────────────
    elif "Tbh" in ent and "Tro" in ent:
        Tbh = ent["Tbh"]
        Tro = ent["Tro"]
        # Contrainte physique : Tro ≤ Tbh ≤ Tbs
        if Tro > Tbh:
            raise ValueError(
                f"Incohérence physique : Tro ({Tro}°C) > Tbh ({Tbh}°C)."
            )
        # Tro_de_W(W_de_Tbs_Tbh(Tbs,Tbh)) croît avec Tbs
        # → f(Tbs) = Tro_calculé - Tro_cible > 0 quand Tbs > solution
        Tbs = _bissection(
            lambda t: Tro_de_W(W_de_Tbs_Tbh(t, Tbh, P), P) - Tro,
            Tbh, Tbh + 60.0,
        )
        W = W_de_Tbs_Tbh(Tbs, Tbh, P)

    else:
        raise ValueError(
            f"Combinaison non supportée : «{param1}» + «{param2}»"
        )

    W   = max(0.0, min(W, W_sat(Tbs, P)))
    HR  = HR_de_Tbs_W(Tbs, W, P)
    Tbh = Tbh_de_Tbs_W(Tbs, W, P)
    Tro = Tro_de_W(W, P)
    h   = enthalpie(Tbs, W)
    v   = volume_specifique(Tbs, W, P)

    return EtatPsychro(Tbs=Tbs, W=W, HR=HR, Tbh=Tbh, Tro=Tro, h=h, v=v, P=P)

# Alias anglais
def compute_state(param1, val1, param2, val2, P=PATM_MER):
    return calculer_etat(param1, val1, param2, val2, P)


# ═══════════════════════════════════════════════════════════════════════
# Générateurs de courbes vectorisés
# ═══════════════════════════════════════════════════════════════════════

def courbe_saturation(T_min=-10, T_max=55, n=350, P=PATM_MER):
    """(T[°C], W[g/kg]) de la courbe de saturation 100 % HR."""
    T = np.linspace(T_min, T_max, n)
    W = np.array([W_sat(t, P) * 1000 for t in T])
    return T, W

saturation_curve = courbe_saturation


def courbe_HR(hr_frac: float, T_min=-10, T_max=55, n=300, P=PATM_MER):
    """(T, W[g/kg]) pour une courbe HR constante."""
    T  = np.linspace(T_min, T_max, n)
    W  = np.array([W_de_Tbs_HR(t, hr_frac, P) * 1000 for t in T])
    Ws = np.array([W_sat(t, P) * 1000 for t in T])
    W  = np.minimum(W, Ws)
    mask = W > 0.05
    return T[mask], W[mask]

rh_curve = courbe_HR


def ligne_enthalpie(h_kJ: float, T_min=-10, T_max=55, n=250, P=PATM_MER):
    """(T, W[g/kg]) pour h constante [kJ/kg as]."""
    T  = np.linspace(T_min, T_max, n)
    W  = (h_kJ - CPa * T) / (hfg0 + CPv * T)
    Ws = np.array([W_sat(t, P) for t in T])
    mask = (W >= 0) & (W <= Ws)
    return T[mask], W[mask] * 1000

enthalpy_line = ligne_enthalpie


def ligne_bulbe_humide(Tbh_C: float, T_min=-10, T_max=55, n=250, P=PATM_MER):
    """(T, W[g/kg]) pour Tbh constante."""
    T  = np.linspace(T_min, T_max, n)
    W  = np.array([W_de_Tbs_Tbh(t, Tbh_C, P) * 1000 for t in T])
    Ws = np.array([W_sat(t, P) * 1000 for t in T])
    mask = (W >= 0) & (W <= Ws) & (T >= Tbh_C - 0.01)
    return T[mask], W[mask]

wetbulb_line = ligne_bulbe_humide


def ligne_volume(v_m3: float, T_min=-10, T_max=55, n=250, P=PATM_MER):
    """(T, W[g/kg]) pour v constante [m³/kg as]."""
    T  = np.linspace(T_min, T_max, n)
    W  = 0.621945 * (v_m3 * P / (Ra * (T + 273.15)) - 1.0)
    Ws = np.array([W_sat(t, P) for t in T])
    mask = (W >= 0) & (W <= Ws)
    return T[mask], W[mask] * 1000

specific_volume_line = ligne_volume
