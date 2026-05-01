"""
chart.py  —  PsychroCalc Pro
Diagramme psychrométrique ASHRAE complet :
  • Axes : Tbs ∈ [-10 ; 55] °C,  W ∈ [0 ; 30] g/kg
  • Courbe de saturation qui borde le diagramme (bord supérieur puis droit)
  • Toutes les courbes traversent tout l'espace du diagramme
  • Étiquettes anti-chevauchement
  • Cadre complet (mirror=True)
  • Export PDF haute résolution
"""

import numpy as np
import plotly.graph_objects as go

from psychro_model import (
    PATM_MER, PATM_SEA,
    W_sat, psat,
    courbe_saturation, courbe_HR, ligne_enthalpie,
    ligne_bulbe_humide, ligne_volume,
    # aliases anglais
    saturation_curve, rh_curve, enthalpy_line,
    wetbulb_line, specific_volume_line,
)


# ═══════════════════════════════════════════════════════════════════════════
# LIMITES DU DIAGRAMME  (identiques à un diagramme ASHRAE imprimé standard)
# ═══════════════════════════════════════════════════════════════════════════
T_MIN  = -10.0   # °C
T_MAX  =  55.0   # °C
W_MIN  =   0.0   # g/kg
W_MAX  =  30.0   # g/kg

# ── Niveaux des familles de courbes ─────────────────────────────────────────
NIVEAUX_HR   = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
NIVEAUX_H    = list(range(-10, 130, 10))   # kJ/kg — trait fin
NIVEAUX_H_L  = list(range(-10, 130, 20))   # kJ/kg — avec label
NIVEAUX_BH   = list(range(-8,  54,  4))    # °C bulbe humide
NIVEAUX_VOL  = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]  # m³/kg as

# Zone confort ASHRAE 55-2020 (polygone Tbs/W)
CONFORT_TBS = [20.0, 26.0, 26.0, 20.0, 20.0]
CONFORT_W   = [ 4.0,  4.0, 12.0, 12.0,  4.0]

# ── Palette de couleurs ──────────────────────────────────────────────────────
COL_SAT  = "#000000"    # noir épais — saturation + bordure
COL_HR   = "#555555"    # gris — courbes HR
COL_H    = "#8800BB"    # violet — lignes enthalpie
COL_BH   = "#CC0044"    # magenta — lignes bulbe humide
COL_VOL  = "#006600"    # vert — lignes volume spécifique
COL_CF   = "#228B22"    # vert forêt — zone confort
COL_PT   = "#0055CC"    # bleu — point d'état

COULEURS_PROCESSUS = {
    "chauffage"         : "#E8593C",
    "refroidissement"   : "#3B8BD4",
    "humidification"    : "#1D9E75",
    "deshumidification" : "#BA7517",
    "melange"           : "#7F77DD",
    "heating"   : "#E8593C",
    "cooling"   : "#3B8BD4",
    "humidify"  : "#1D9E75",
    "dehumidify": "#BA7517",
    "mixing"    : "#7F77DD",
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers internes
# ═══════════════════════════════════════════════════════════════════════════

def _trace(x, y, couleur, ep=0.8, tiret="solid", nom="", legende=False):
    return go.Scatter(
        x=list(x), y=list(y),
        mode="lines",
        line=dict(color=couleur, width=ep, dash=tiret),
        name=nom, showlegend=legende,
        hoverinfo="skip",
    )


def _etiq(x, y, txt, couleur, taille=8, ancre="middle left"):
    return go.Scatter(
        x=[x], y=[y], mode="text",
        text=[txt],
        textfont=dict(size=taille, color=couleur, family="Arial"),
        textposition=ancre,
        hoverinfo="skip", showlegend=False,
    )


def _clipper(T_arr, W_arr, w_max=W_MAX, t_min=T_MIN, t_max=T_MAX, w_min=W_MIN):
    """Garde uniquement les points dans la fenêtre du diagramme."""
    mask = (
        (T_arr >= t_min) & (T_arr <= t_max) &
        (W_arr >= w_min) & (W_arr <= w_max)
    )
    return T_arr[mask], W_arr[mask]


class _AntiChev:
    """Grille d'exclusion 2D pour éviter le chevauchement des étiquettes."""
    def __init__(self):
        self._zones = []

    def libre(self, cx, cy, dx=3.5, dy=1.4):
        for (bx, by, bdx, bdy) in self._zones:
            if abs(cx - bx) < (dx + bdx) / 2 and abs(cy - by) < (dy + bdy) / 2:
                return False
        return True

    def reserver(self, cx, cy, dx=3.5, dy=1.4):
        self._zones.append((cx, cy, dx, dy))

    def placer(self, cx, cy, dx=3.5, dy=1.4):
        if self.libre(cx, cy, dx, dy):
            self.reserver(cx, cy, dx, dy)
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# BORDURE DU DIAGRAMME
# La bordure suit la courbe de saturation (en haut) puis les bords du rectangle.
# ═══════════════════════════════════════════════════════════════════════════

def _bordure_saturation(P):
    """
    Retourne (T, W) de la frontière complète du diagramme :
      bas-gauche (T_MIN, 0) → saturation jusqu'à W_MAX ou T_MAX → coin → bords.

    La courbe de saturation est la frontière supérieure-gauche.
    À partir du point où W_sat > W_MAX, la frontière devient W = W_MAX (bord supérieur).
    """
    # Saturation complète sur [-10 ; 55] à haute résolution
    T_full = np.linspace(T_MIN, T_MAX, 1000)
    W_full = np.array([W_sat(t, P) * 1000 for t in T_full])

    # Partie de la saturation DANS le diagramme (W ≤ W_MAX)
    mask_in = W_full <= W_MAX
    T_sat_in = T_full[mask_in]
    W_sat_in = W_full[mask_in]

    # Point où la saturation sort par le haut (W=W_MAX)
    # → continuer sur le bord supérieur jusqu'à T_MAX
    if W_sat_in[-1] < W_MAX - 0.1:
        # La saturation ne sort jamais par le haut → sort par le côté droit
        T_bord_h = np.array([])
        W_bord_h = np.array([])
        T_bord_d = np.array([T_sat_in[-1], T_MAX])
        W_bord_d = np.array([W_sat_in[-1], W_sat_in[-1]])
    else:
        # La saturation sort par W_MAX → bord supérieur jusqu'à T_MAX
        T_bord_h = np.array([T_sat_in[-1], T_MAX])
        W_bord_h = np.array([W_MAX,        W_MAX])
        T_bord_d = np.array([])
        W_bord_d = np.array([])

    # Ligne verticale gauche : (T_MIN, 0) → premier point saturation
    T_vert_g = np.array([T_MIN, T_MIN])
    W_vert_g = np.array([W_MIN, W_sat_in[0]])

    # Ligne horizontale basse : (T_MIN, 0) → (T_MAX, 0)
    T_bas = np.array([T_MIN, T_MAX])
    W_bas = np.array([W_MIN, W_MIN])

    # Ligne verticale droite : (T_MAX, 0) → (T_MAX, W_MAX)
    T_vert_d = np.array([T_MAX, T_MAX])
    W_vert_d = np.array([W_MIN, W_MAX])

    return (T_vert_g, W_vert_g,
            T_sat_in, W_sat_in,
            T_bord_h, W_bord_h)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCTION DU FOND (mis en cache par Streamlit)
# ═══════════════════════════════════════════════════════════════════════════

def construire_fond(P: float = PATM_MER) -> go.Figure:
    """
    Construit la figure de fond complète du diagramme psychrométrique.
    Tout le rectangle [-10 ; 55] × [0 ; 30] est rempli de courbes.
    """
    traces = []
    ac = _AntiChev()

    # ── 1. Remplissage zone de confort ASHRAE 55 ────────────────────────
    traces.append(go.Scatter(
        x=CONFORT_TBS, y=CONFORT_W,
        mode="lines", fill="toself",
        fillcolor="rgba(34,139,34,0.09)",
        line=dict(color=COL_CF, width=1.0, dash="dash"),
        name="Zone de confort ASHRAE 55",
        showlegend=True, hoverinfo="skip",
    ))
    ac.reserver(23.0, 8.0, dx=10, dy=6)

    # ── 2. Courbes HR constante ─────────────────────────────────────────
    # Chaque courbe va de T_MIN jusqu'à ce qu'elle sorte par W_MAX ou T_MAX.
    for hr in sorted(NIVEAUX_HR):
        T_raw, W_raw = courbe_HR(hr, T_MIN, T_MAX, n=400, P=P)
        if len(T_raw) < 2:
            continue

        # Clipper à W_MAX : couper la courbe dès qu'elle dépasse W_MAX
        mask = W_raw <= W_MAX
        T_c  = T_raw[mask]
        W_c  = W_raw[mask]
        if len(T_c) < 2:
            continue

        traces.append(_trace(T_c, W_c, COL_HR, ep=0.8))

        # Étiquette : au dernier point visible (bord droit ou bord supérieur)
        lx, ly = float(T_c[-1]), float(W_c[-1])
        etiq   = f"{int(hr * 100)} %"
        # Si la courbe sort par le haut (W ≈ W_MAX), étiqueter à droite
        if ly >= W_MAX - 1.0:
            # Étiquette sur bord supérieur
            for dx_off in [0.3, -0.3, 1.0, -1.0]:
                cx, cy = lx + dx_off, W_MAX + 0.3
                if ac.placer(cx, cy, dx=3.0, dy=0.8):
                    traces.append(_etiq(cx, cy, etiq, COL_HR, taille=8,
                                        ancre="bottom center"))
                    break
        else:
            # Étiquette sur bord droit
            for dx_off in [0.4, 1.2]:
                cx, cy = T_MAX + dx_off, ly
                if ac.placer(cx, cy, dx=3.0, dy=0.8):
                    traces.append(_etiq(cx, cy, etiq, COL_HR, taille=8,
                                        ancre="middle left"))
                    break

    # ── 3. Courbe de saturation ─────────────────────────────────────────
    # La saturation est la frontière gauche/supérieure du diagramme.
    T_vert_g, W_vert_g, T_sat_in, W_sat_in, T_bord_h, W_bord_h = \
        _bordure_saturation(P)

    # Ligne verticale gauche (W=0 → W_sat(-10°C))
    traces.append(_trace(T_vert_g, W_vert_g, COL_SAT, ep=2.5))

    # Courbe de saturation proprement dite
    traces.append(_trace(
        T_sat_in, W_sat_in, COL_SAT, ep=2.5,
        nom="Saturation (100 % HR)", legende=True,
    ))

    # Si la saturation sort par le haut → compléter avec bord supérieur
    if len(T_bord_h) > 0:
        traces.append(_trace(T_bord_h, W_bord_h, COL_SAT, ep=2.5))

    # Interpolateur W_sat(T) pour le placement des labels
    def w_sat_at(t):
        return float(np.interp(t, T_sat_in, W_sat_in,
                               left=W_sat_in[0], right=W_MAX))

    # ── 4. Lignes d'enthalpie ───────────────────────────────────────────
    for h_val in NIVEAUX_H:
        T_h, W_h = ligne_enthalpie(h_val, T_MIN, T_MAX, n=300, P=P)
        if len(T_h) < 2:
            continue
        # Clipper à la fenêtre
        T_h, W_h = _clipper(T_h, W_h)
        if len(T_h) < 2:
            continue

        ep = 1.0 if h_val % 20 == 0 else 0.45
        traces.append(_trace(T_h, W_h, COL_H, ep=ep, tiret="dot"))

        if h_val not in NIVEAUX_H_L:
            continue

        # Label : début de la ligne (côté saturation / gauche)
        # Chercher le premier point suffisamment loin de la saturation
        etiq = f"h={h_val}"
        place = False
        for i in range(len(T_h)):
            t_i, w_i = float(T_h[i]), float(W_h[i])
            dist_sat  = w_sat_at(t_i) - w_i
            if dist_sat > 2.0:
                for decal in [0.0, -0.8, 0.8, -1.5]:
                    cx, cy = t_i + decal, w_i
                    if ac.placer(cx, cy, dx=3.5, dy=1.0):
                        traces.append(_etiq(cx, cy, etiq, COL_H,
                                            taille=7, ancre="middle right"))
                        place = True
                        break
            if place:
                break

    # ── 5. Lignes de bulbe humide ───────────────────────────────────────
    for tbh in sorted(NIVEAUX_BH):
        T_b, W_b = ligne_bulbe_humide(tbh, T_MIN, T_MAX, n=300, P=P)
        if len(T_b) < 2:
            continue
        T_b, W_b = _clipper(T_b, W_b)
        if len(T_b) < 2:
            continue

        ep = 1.1 if tbh % 8 == 0 else 0.5
        traces.append(_trace(T_b, W_b, COL_BH, ep=ep))

        # Label au début de la ligne (point de saturation)
        etiq = f"{tbh}°"
        for i in range(min(5, len(T_b))):
            cx = float(T_b[i]) - 0.5
            cy = float(W_b[i]) + 0.2
            if cx < T_MIN:
                cx = T_MIN
            if ac.placer(cx, cy, dx=2.8, dy=0.9):
                traces.append(_etiq(cx, cy, etiq, COL_BH,
                                    taille=7, ancre="middle right"))
                break

    # ── 6. Lignes de volume spécifique ─────────────────────────────────
    for v_val in NIVEAUX_VOL:
        T_v, W_v = ligne_volume(v_val, T_MIN, T_MAX, n=300, P=P)
        if len(T_v) < 2:
            continue
        T_v, W_v = _clipper(T_v, W_v)
        if len(T_v) < 2:
            continue

        traces.append(_trace(T_v, W_v, COL_VOL, ep=0.8, tiret="longdash"))

        # Label au dernier point (bord droit ou bas)
        lx, ly = float(T_v[-1]), float(W_v[-1])
        etiq = f"v={v_val:.2f}"
        for dx_off in [0.3, 1.0, -0.5]:
            cx, cy = lx + dx_off, ly
            if ac.placer(cx, cy, dx=3.2, dy=0.9):
                traces.append(_etiq(cx, cy, etiq, COL_VOL,
                                    taille=7, ancre="middle left"))
                break

    # ── 7. Assemblage de la figure ──────────────────────────────────────
    fig = go.Figure(data=traces)

    _ax = dict(
        showgrid=True, gridcolor="#dddddd", gridwidth=0.5,
        zeroline=False,
        showline=True, linecolor="#000000", linewidth=2.0,
        mirror=True,            # ← cadre complet
        ticks="outside", tickwidth=1, ticklen=5, tickcolor="#000000",
        tickfont=dict(family="Arial", size=10),
        layer="above traces",   # axes dessinés PAR-DESSUS les courbes
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11, color="#111111"),
        margin=dict(l=90, r=80, t=70, b=85),
        height=680,
        uirevision="constant",

        title=dict(
            text=(
                "<b>Diagramme Psychrométrique</b>"
                "<span style='font-size:12px;color:#666'>"
                "  —  ASHRAE Fondamentaux 2017  |  P = "
                f"{round(P/1000,3)} kPa</span>"
            ),
            x=0.5, xanchor="center",
            font=dict(size=15, color="#111"),
        ),

        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#aaa", borderwidth=0.8,
            font=dict(size=10, family="Arial"),
        ),

        xaxis=dict(
            **_ax,
            title=dict(
                text="Température de bulbe sec  Tbs  (°C)",
                font=dict(size=12, family="Arial"),
            ),
            range=[T_MIN, T_MAX],
            dtick=5,
        ),

        yaxis=dict(
            **_ax,
            title=dict(
                text="Teneur en eau   W   (g / kg d'air sec)",
                font=dict(size=12, family="Arial"),
            ),
            range=[W_MIN, W_MAX],
            dtick=2,
            fixedrange=True,
        ),
    )

    return fig


# Alias anglais
build_background = construire_fond


# ═══════════════════════════════════════════════════════════════════════════
# POINT D'ÉTAT + RÉTICULE
# ═══════════════════════════════════════════════════════════════════════════

def ajouter_point_etat(fig: go.Figure,
                       Tbs: float, W_gkg: float,
                       etiquette: str = "État 1",
                       couleur: str = COL_PT,
                       indice: int = 1) -> go.Figure:
    """Ajoute un marqueur d'état avec réticule de lecture sur le diagramme."""
    # Décodage robuste : accepte "#RRGGBB", "#RGB" et noms CSS via plotly
    try:
        couleur_hex = couleur.strip()
        if not couleur_hex.startswith("#"):
            import plotly.colors as _pc
            couleur_hex = _pc.label_rgb(_pc.unconvert_from_RGB_255(
                _pc.unlabel_rgb(_pc.color_parser(couleur_hex, _pc.unlabel_rgb))
            ))
            r, g, b = [int(x) for x in couleur_hex.replace("rgb(","").replace(")","").split(",")]
        else:
            c = couleur_hex.lstrip("#")
            if len(c) == 3:
                c = "".join(ch * 2 for ch in c)
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except Exception:
        r, g, b = 0, 85, 204   # COL_PT par défaut

    # Réticule
    fig.add_trace(go.Scatter(
        x=[T_MIN, Tbs, Tbs],
        y=[W_gkg,  W_gkg, W_MIN],
        mode="lines",
        line=dict(color=couleur, width=0.9, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))
    # Halo
    fig.add_trace(go.Scatter(
        x=[Tbs], y=[W_gkg], mode="markers",
        marker=dict(
            size=18,
            color=f"rgba({r},{g},{b},0.15)",
            line=dict(color=couleur, width=1.8),
        ),
        hoverinfo="skip", showlegend=False,
    ))
    # Point + label
    num = "❶❷❸❹❺"[indice - 1] if indice <= 5 else str(indice)
    fig.add_trace(go.Scatter(
        x=[Tbs], y=[W_gkg],
        mode="markers+text",
        marker=dict(size=9, color=couleur),
        text=[f"  {num}"],
        textfont=dict(size=12, color=couleur, family="Arial"),
        textposition="middle right",
        name=etiquette,
        showlegend=True,
        hovertemplate=(
            f"<b>{etiquette}</b><br>"
            f"Tbs = {Tbs:.1f} °C<br>"
            f"W   = {W_gkg:.2f} g/kg<br>"
            "<extra></extra>"
        ),
    ))
    return fig


# Alias anglais
def add_state_point(fig, Tdb, W_gkg, label="État 1",
                    color=COL_PT, point_index=1):
    return ajouter_point_etat(fig, Tdb, W_gkg,
                               etiquette=label, couleur=color, indice=point_index)


# ═══════════════════════════════════════════════════════════════════════════
# FLÈCHE DE PROCESSUS HVAC
# ═══════════════════════════════════════════════════════════════════════════

def ajouter_fleche_processus(fig: go.Figure,
                              etat1, etat2,
                              nom_processus: str = "Processus",
                              couleur: str = "#E8593C") -> go.Figure:
    x0, y0 = etat1.Tbs, etat1.W * 1000
    x1, y1 = etat2.Tbs, etat2.W * 1000
    dh = etat2.h - etat1.h
    signe = "+" if dh >= 0 else ""
    fig.add_annotation(
        x=x1, y=y1, ax=x0, ay=y0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True,
        arrowhead=3, arrowsize=1.3,
        arrowwidth=2.2, arrowcolor=couleur,
        text=(f"<b>{nom_processus}</b><br>"
              f"Δh = {signe}{dh:.1f} kJ/kg"),
        font=dict(size=9, color=couleur, family="Arial"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=couleur, borderwidth=0.8, borderpad=3,
    )
    return fig


def add_process_arrow(fig, state1, state2,
                      process_name="Processus", color="#E8593C"):
    return ajouter_fleche_processus(fig, state1, state2,
                                     nom_processus=process_name, couleur=color)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT PDF
# ═══════════════════════════════════════════════════════════════════════════

def exporter_pdf(fig: go.Figure,
                 chemin: str = "diagramme_psychrometrique.pdf",
                 titre_rapport: str = "Rapport Psychrométrique",
                 projet: str = "",
                 ingenieur: str = "",
                 date_str: str = "") -> tuple:
    import copy
    from datetime import date as _date

    fig_pdf = copy.deepcopy(fig)
    date_aff = date_str or _date.today().strftime("%d/%m/%Y")
    meta = "  |  ".join(filter(None, [projet, ingenieur, date_aff]))

    fig_pdf.update_layout(
        title=dict(
            text=(f"<b>{titre_rapport}</b><br>"
                  f"<span style='font-size:11px;color:#555'>{meta}</span>"),
            x=0.5, xanchor="center",
            font=dict(size=17, family="Arial"),
        ),
        margin=dict(l=100, r=80, t=110, b=95),
    )

    # Pied de page
    annots = list(fig_pdf.layout.annotations or [])
    annots.append(dict(
        x=0.5, y=-0.12, xref="paper", yref="paper",
        text=("Équations : ASHRAE Fondamentaux 2017  |  "
              "Pression de référence : 101.325 kPa  |  PsychroCalc Pro"),
        showarrow=False,
        font=dict(size=8, color="#777", family="Arial"),
        align="center",
    ))
    fig_pdf.update_layout(annotations=annots)

    try:
        fig_pdf.write_image(chemin, format="pdf", width=1600, height=950, scale=1)
        return True, chemin
    except Exception as err_pdf:
        chemin_png = chemin.replace(".pdf", "_HR.png")
        try:
            fig_pdf.write_image(chemin_png, format="png",
                                width=1600, height=950, scale=2)
            return True, chemin_png
        except Exception as err_png:
            return False, (f"Erreur PDF : {err_pdf}\n"
                           f"Erreur PNG : {err_png}\n"
                           "Solution : pip install -U kaleido")


def export_pdf(fig, path="diagramme_psychrometrique.pdf"):
    return exporter_pdf(fig, chemin=path)
