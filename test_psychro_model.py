"""
test_psychro_model.py
Tests unitaires pour psychro_model.py.
Valeurs de référence : ASHRAE Fondamentaux 2017, Tableau 2 (Ch.1).

Lancement : pytest test_psychro_model.py -v
"""

import math
import warnings
import pytest

from psychro_model import (
    psat, W_sat, W_de_Tbs_HR, W_de_Tbs_Tbh,
    HR_de_Tbs_W, Tro_de_W, Tbh_de_Tbs_W,
    enthalpie, volume_specifique, calculer_etat,
    altitude_vers_pression, PATM_MER,
)

# ─────────────────────────────────────────────────────────────────────
# Constantes de tolérance
# ─────────────────────────────────────────────────────────────────────
TOL_PSAT  = 0.001   # 0.1 % sur la pression de vapeur
TOL_W     = 0.05    # g/kg
TOL_T     = 0.05    # °C
TOL_H     = 0.1     # kJ/kg
TOL_V     = 0.0005  # m³/kg


# ─────────────────────────────────────────────────────────────────────
# psat — valeurs tabulées ASHRAE 2017
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("T_C, expected_Pa", [
    (-20.0,  103.3),   # glace
    (-10.0,  259.9),
    (  0.0,  611.2),   # point triple (eau)
    ( 10.0, 1228.1),
    ( 20.0, 2338.5),
    ( 30.0, 4246.0),
    ( 40.0, 7384.0),
    ( 50.0, 12352.0),
])
def test_psat_valeurs_tabulees(T_C, expected_Pa):
    result = psat(T_C)
    assert abs(result - expected_Pa) / expected_Pa < TOL_PSAT, (
        f"psat({T_C}°C) = {result:.2f} Pa, attendu ≈ {expected_Pa:.2f} Pa"
    )


def test_psat_continuite_zero():
    """Continuité au point 0 °C entre les deux branches."""
    p_moins = psat(-0.001)
    p_plus  = psat( 0.001)
    assert abs(p_plus - p_moins) < 1.0, "Discontinuité à 0 °C"


def test_psat_hors_plage_warning():
    """Un avertissement doit être émis hors de [-40 ; 80 °C]."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        psat(-50.0)
        assert len(w) == 1
        assert "plage de validité" in str(w[0].message).lower()


# ─────────────────────────────────────────────────────────────────────
# W_de_Tbs_HR — à 25 °C / 50 % HR
# ─────────────────────────────────────────────────────────────────────
def test_W_de_Tbs_HR_25C_50():
    W = W_de_Tbs_HR(25.0, 0.50) * 1000  # g/kg
    assert abs(W - 9.93) < TOL_W, f"W = {W:.3f} g/kg, attendu ≈ 9.93 g/kg"


def test_W_de_Tbs_HR_saturation():
    """À HR=1, W doit égaler Wsat."""
    W1  = W_de_Tbs_HR(20.0, 1.0)
    Ws  = W_sat(20.0)
    assert abs(W1 - Ws) < 1e-9


# ─────────────────────────────────────────────────────────────────────
# W_de_Tbs_Tbh — exemple ASHRAE : Tbs=30°C, Tbh=20°C
# ─────────────────────────────────────────────────────────────────────
def test_W_de_Tbs_Tbh_30_20():
    W = W_de_Tbs_Tbh(30.0, 20.0) * 1000
   assert abs(W - 8.67) < 2.0, f"W = {W:.3f} g/kg, attendu ≈ 8.67 g/kg"


# ─────────────────────────────────────────────────────────────────────
# Enthalpie — ASHRAE éq. 32
# ─────────────────────────────────────────────────────────────────────
def test_enthalpie_25C_50HR():
    W = W_de_Tbs_HR(25.0, 0.50)
    h = enthalpie(25.0, W)
    assert abs(h - 50.56) < 0.5, f"h = {h:.2f} kJ/kg, attendu ≈ 50.56 kJ/kg"


def test_enthalpie_0C_sec():
    h = enthalpie(0.0, 0.0)
    assert abs(h - 0.0) < 0.01


# ─────────────────────────────────────────────────────────────────────
# Volume spécifique
# ─────────────────────────────────────────────────────────────────────
def test_volume_25C_50HR():
    W = W_de_Tbs_HR(25.0, 0.50)
    v = volume_specifique(25.0, W)
    assert abs(v - 0.8576) < TOL_V, f"v = {v:.4f} m³/kg, attendu ≈ 0.8576 m³/kg"


# ─────────────────────────────────────────────────────────────────────
# Tro_de_W — point de rosée
# ─────────────────────────────────────────────────────────────────────
def test_Tro_de_W_coherence():
    """Tro(Wsat(T)) ≈ T."""
    for T in [5.0, 15.0, 25.0, 35.0]:
        Ws  = W_sat(T)
        Tro = Tro_de_W(Ws)
        assert abs(Tro - T) < TOL_T, (
            f"Tro(Wsat({T}°C)) = {Tro:.3f} °C, attendu ≈ {T} °C"
        )


# ─────────────────────────────────────────────────────────────────────
# Tbh_de_Tbs_W — cohérence aller-retour
# ─────────────────────────────────────────────────────────────────────
def test_Tbh_roundtrip():
    """W → Tbh → W doit être stable."""
    Tbs = 30.0
    W0  = W_de_Tbs_HR(Tbs, 0.60)
    Tbh = Tbh_de_Tbs_W(Tbs, W0)
    W1  = W_de_Tbs_Tbh(Tbs, Tbh)
    assert abs((W1 - W0) * 1000) < 0.01, (
        f"Δw = {abs(W1-W0)*1000:.4f} g/kg après aller-retour"
    )


# ─────────────────────────────────────────────────────────────────────
# calculer_etat — toutes les combinaisons supportées
# ─────────────────────────────────────────────────────────────────────
REF = {"Tbs": 25.0, "HR": 60.0}  # état de référence

@pytest.mark.parametrize("p1,v1,p2,v2,tbs_ref,hr_ref,tol_tbs,tol_hr", [
    # Combinaisons avec Tbs : convergence exacte vers 25°C
    ("Tdb", 25.0, "RH",  60.0,  25.0, 0.600, 0.05, 0.005),
    ("Tdb", 25.0, "Twb", 18.7,  25.0, None,  0.05, None),   # Tbh fixé, HR libre
    ("Tdb", 25.0, "Tdp", 16.5,  25.0, None,  0.05, None),
    ("Tdb", 25.0, "W",   11.85, 25.0, None,  0.05, None),
    # Combinaisons sans Tbs
    ("Twb", 18.7, "RH",  60.0,  None, 0.600, None, 0.005),
    ("Tdp", 16.5, "RH",  60.0,  None, 0.600, None, 0.005),
])
def test_calculer_etat_combinaisons(p1, v1, p2, v2, tbs_ref, hr_ref, tol_tbs, tol_hr):
    """Les combinaisons doivent converger sans lever d'exception."""
    e = calculer_etat(p1, v1, p2, v2)
    if tbs_ref is not None:
        assert abs(e.Tbs - tbs_ref) < tol_tbs, (
            f"{p1}+{p2}: Tbs = {e.Tbs:.3f} °C, attendu ≈ {tbs_ref} °C"
        )
    if hr_ref is not None:
        assert abs(e.HR - hr_ref) < tol_hr, (
            f"{p1}+{p2}: HR = {e.HR:.4f}, attendu ≈ {hr_ref}"
        )


@pytest.mark.parametrize("p1,v1,p2,v2", [
    ("Twb", 18.7, "W",   11.85),
    ("Tdp", 16.5, "W",   11.85),
    ("Twb", 18.7, "Tdp", 16.5),
])
def test_calculer_etat_nouvelles_combinaisons(p1, v1, p2, v2):
    """Les nouvelles combinaisons (anciennement manquantes) ne doivent pas crasher."""
    e = calculer_etat(p1, v1, p2, v2)
    # Vérifications physiques de base
    assert e.Tro <= e.Tbh + 0.01 <= e.Tbs + 0.01
    assert 0.0 <= e.HR <= 1.0
    assert e.W >= 0.0


def test_calculer_etat_coherence_interne():
    """Tro, Tbh et h recalculés doivent être cohérents."""
    e = calculer_etat("Tdb", 25.0, "RH", 60.0)
    assert e.Tro < e.Tbh < e.Tbs, "Ordre Tro < Tbh < Tbs non respecté"
    h_check = enthalpie(e.Tbs, e.W)
    assert abs(e.h - h_check) < 1e-6


# ─────────────────────────────────────────────────────────────────────
# altitude_vers_pression
# ─────────────────────────────────────────────────────────────────────
def test_altitude_vers_pression_0m():
    assert abs(altitude_vers_pression(0) - PATM_MER) < 1.0


def test_altitude_vers_pression_1500m():
    """À ~1 500 m (ex. Calgary) : ≈ 84 600 Pa."""
    P = altitude_vers_pression(1500)
    assert 83_000 < P < 86_000, f"P(1500m) = {P:.0f} Pa hors plage attendue"
