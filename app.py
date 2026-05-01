"""
app.py  –  PsychroCalc Pro
Application Streamlit de diagramme psychrométrique.
Lancement :  streamlit run app.py
"""

import copy
import streamlit as st
import plotly.graph_objects as go

from psychro_model import (
    calculer_etat, altitude_vers_pression,
    PATM_MER, EtatPsychro,
    # aliases anglais (compatibilité)
    compute_state, altitude_to_pressure, PATM_SEA, PsychroState,
)
from chart import (
    construire_fond, ajouter_point_etat,
    ajouter_fleche_processus, exporter_pdf,
    COULEURS_PROCESSUS,
    # aliases anglais
    build_background, add_state_point,
    add_process_arrow, export_pdf,
)


# ═══════════════════════════════════════════════════════════════════════
# Configuration de la page
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PsychroCalc Pro",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# CSS — apparence professionnelle (logiciel HVAC industriel)
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Police générale */
html, body, [class*="css"] {
    font-family: "Segoe UI", Arial, sans-serif;
}

/* Largeur fixe de la barre latérale */
[data-testid="stSidebar"] { min-width: 285px; max-width: 295px; }

/* Entêtes de section */
.titre-section {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.07em;
    color: #777777;
    text-transform: uppercase;
    margin: 14px 0 5px;
}

/* Grille de cartes de résultats */
.grille-resultats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 7px;
    margin-bottom: 8px;
}
.carte-resultat {
    background: #f4f4f4;
    border-radius: 6px;
    padding: 7px 10px;
    border: 0.5px solid #dddddd;
}
.carte-large {
    background: #f4f4f4;
    border-radius: 6px;
    padding: 7px 10px;
    border: 0.5px solid #dddddd;
    margin-bottom: 7px;
}
.res-nom   { font-size: 10px; color: #888888; margin-bottom: 1px; }
.res-val   {
    font-size: 17px;
    font-weight: 500;
    font-family: "Courier New", monospace;
    color: #111111;
}
.res-unite { font-size: 11px; color: #888888; }

/* Badges confort */
.badge-ok {
    background: #e6f4ea; color: #1a7f37;
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 10px;
}
.badge-hors {
    background: #fff3cd; color: #856404;
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# État de session
# ═══════════════════════════════════════════════════════════════════════
for cle in ("etat1", "etat2", "processus_cle"):
    if cle not in st.session_state:
        st.session_state[cle] = None


# ═══════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════════

def dans_confort(e: EtatPsychro) -> bool:
    """
    Vérifie si l'état est dans la zone de confort ASHRAE 55 (simplifié).
    Note : implémentation simplifiée — seuils fixes sans prise en compte
    de la vitesse d'air, du métabolisme ou de l'habillement (PMV/PPD).
    """
    return (20.0 <= e.Tbs <= 26.0 and
            4.0  <= e.W * 1000 <= 12.0 and
            e.HR <= 0.80)


def fmt(valeur, decimales=2) -> str:
    return f"{valeur:.{decimales}f}"


def carte(nom, valeur, unite) -> str:
    return (
        f'<div class="carte-resultat">'
        f'<div class="res-nom">{nom}</div>'
        f'<div class="res-val">{valeur}'
        f'<span class="res-unite"> {unite}</span></div>'
        f'</div>'
    )


def carte_large(nom, valeur, unite) -> str:
    return (
        f'<div class="carte-large">'
        f'<div class="res-nom">{nom}</div>'
        f'<div class="res-val">{valeur}'
        f'<span class="res-unite"> {unite}</span></div>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════
# BARRE LATÉRALE
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("##  PsychroCalc Pro")
    st.caption("")
    st.divider()

    # ── Badge disponibilité export PDF ────────────────────────────────
    try:
        import kaleido  # noqa: F401
        st.success("", icon=None)
    except ImportError:
        st.warning("")

    # ── Conditions atmosphériques ──────────────────────────────────────
    st.markdown('<div class="titre-section">Conditions atmosphériques</div>',
                unsafe_allow_html=True)

    mode_pression = st.radio(
        "Entrée pression",
        ["Altitude (m)", "Pression directe (kPa)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode_pression == "Altitude (m)":
        altitude = st.number_input(
            "Altitude au-dessus du niveau de la mer (m)",
            min_value=0, max_value=5000, value=0, step=50,
        )
        P = altitude_vers_pression(altitude)
        st.caption(f"Pression atm. calculée : **{P/1000:.3f} kPa**")
    else:
        P_kPa = st.number_input(
            "Pression atmosphérique (kPa)",
            min_value=60.0, max_value=105.0,
            value=101.325, step=0.1, format="%.3f",
        )
        P = P_kPa * 1000

    # ── Paramètres d'entrée de l'état 1 ───────────────────────────────
    st.divider()
    st.markdown('<div class="titre-section">État 1 — paramètres d\'entrée</div>',
                unsafe_allow_html=True)

    LABELS_PARAMS = {
        "Tdb": "Bulbe sec  Tbs  (°C)",
        "RH" : "Humidité relative  HR  (%)",
        "Twb": "Bulbe humide  Tbh  (°C)",
        "Tdp": "Température de rosée  Tro  (°C)",
        "W"  : "Teneur en eau  W  (g/kg)",
    }
    CLES_PARAMS = list(LABELS_PARAMS.keys())

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        param1 = st.selectbox("1er paramètre", CLES_PARAMS,
                              format_func=lambda k: k, index=0)
    with col_p2:
        choix_p2 = [k for k in CLES_PARAMS if k != param1]
        param2   = st.selectbox("2e paramètre", choix_p2,
                                format_func=lambda k: k, index=0)

    VALEURS_DEF = {"Tdb": 25.0, "RH": 60.0, "Twb": 18.7,
                   "Tdp": 16.4, "W" : 11.9}

    val1 = st.number_input(
        LABELS_PARAMS[param1],
        value=VALEURS_DEF[param1], step=0.1, format="%.2f",
    )
    val2 = st.number_input(
        LABELS_PARAMS[param2],
        value=VALEURS_DEF[param2], step=0.1, format="%.2f",
    )

    # ── Processus HVAC ─────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="titre-section">Processus HVAC (optionnel)</div>',
                unsafe_allow_html=True)

    PROCESSUS = {
        "Aucun"              : None,
        "Chauffage sensible" : "chauffage",
        "Refroidissement + déshumid." : "refroidissement",
        "Humidification"     : "humidification",
        "Mélange (50/50)"    : "melange",
        "Déshumidification"  : "deshumidification",
    }
    label_proc = st.selectbox("Processus depuis l'État 1",
                              list(PROCESSUS.keys()))
    cle_proc   = PROCESSUS[label_proc]

    entrees_etat2 = {}
    if cle_proc:
        st.caption("État 2 — conditions finales")
        p2a      = st.selectbox("Param. A (État 2)", CLES_PARAMS, index=0, key="p2a")
        options_p2b = [k for k in CLES_PARAMS if k != p2a]
        p2b      = st.selectbox("Param. B (État 2)", options_p2b, index=0, key="p2b")
        v2a      = st.number_input(f"Valeur – {p2a}",
                                   value=VALEURS_DEF[p2a], step=0.1,
                                   format="%.2f", key="v2a")
        v2b      = st.number_input(f"Valeur – {p2b}",
                                   value=VALEURS_DEF[p2b], step=0.1,
                                   format="%.2f", key="v2b")
        entrees_etat2 = dict(p2a=p2a, p2b=p2b, v2a=v2a, v2b=v2b)

    # ── Bouton Calculer ─────────────────────────────────────────────────
    st.divider()
    calculer = st.button(" Calculer", use_container_width=True, type="primary")

    # ── Export PDF ──────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="titre-section">Export PDF</div>',
                unsafe_allow_html=True)

    titre_rapport = st.text_input("Titre du rapport",
                                  value="Rapport Psychrométrique")
    nom_projet    = st.text_input("Nom du projet", value="")
    ingenieur     = st.text_input("Bureau / Ingénieur", value="")
    btn_pdf       = st.button(" Générer le PDF", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# CALCUL
# ═══════════════════════════════════════════════════════════════════════
msg_erreur = None

if calculer:
    try:
        st.session_state.etat1       = calculer_etat(param1, val1, param2, val2, P)
        st.session_state.processus_cle = cle_proc

        if cle_proc and entrees_etat2:
            st.session_state.etat2 = calculer_etat(
                entrees_etat2["p2a"], entrees_etat2["v2a"],
                entrees_etat2["p2b"], entrees_etat2["v2b"], P,
            )
        else:
            st.session_state.etat2 = None

    except Exception as exc:
        msg_erreur = str(exc)

etat1 = st.session_state.etat1
etat2 = st.session_state.etat2


# ═══════════════════════════════════════════════════════════════════════
# MISE EN PAGE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════
col_titre, _, col_export_haut = st.columns([3, 2, 1])
with col_titre:
    st.markdown("### Diagramme Psychrométrique")

col_graphique, col_resultats = st.columns([3, 1])


# ── Panneau des résultats ─────────────────────────────────────────────
with col_resultats:
    if etat1:
        confort = dans_confort(etat1)
        badge   = "badge-ok" if confort else "badge-hors"
        texte_b = " Zone de confort ASHRAE 55" if confort else "Hors zone de confort"
        st.markdown(f'<div class="{badge}">{texte_b}</div>', unsafe_allow_html=True)

        st.markdown('<div class="titre-section">État thermodynamique</div>',
                    unsafe_allow_html=True)

        # Grille 2×2 — paramètres principaux
        st.markdown(
            '<div class="grille-resultats">'
            + carte("Tbs (bulbe sec)", fmt(etat1.Tbs, 1), "°C")
            + carte("HR",             fmt(etat1.HR * 100, 1), "%")
            + carte("Tbh (bulbe humide)", fmt(etat1.Tbh, 1), "°C")
            + carte("Tro (rosée)",    fmt(etat1.Tro, 1), "°C")
            + carte("W",              fmt(etat1.W * 1000, 2), "g/kg")
            + carte("h (enthalpie)",  fmt(etat1.h, 1), "kJ/kg")
            + '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            carte_large("Volume spécifique v",
                        fmt(etat1.v, 4), "m³/kg as"),
            unsafe_allow_html=True,
        )
        st.markdown(
            carte_large("Pression atm.",
                        fmt(etat1.P / 1000, 3), "kPa"),
            unsafe_allow_html=True,
        )

        if etat2:
            st.divider()
            st.markdown('<div class="titre-section">État 2</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="grille-resultats">'
                + carte("Tbs", fmt(etat2.Tbs, 1), "°C")
                + carte("HR",  fmt(etat2.HR * 100, 1), "%")
                + carte("W",   fmt(etat2.W * 1000, 2), "g/kg")
                + carte("h",   fmt(etat2.h, 1), "kJ/kg")
                + '</div>',
                unsafe_allow_html=True,
            )
            dh = etat2.h - etat1.h
            dW = (etat2.W - etat1.W) * 1000
            st.markdown(
                '<div class="grille-resultats">'
                + carte("Δh", ("+" if dh >= 0 else "") + fmt(dh, 1), "kJ/kg")
                + carte("ΔW", ("+" if dW >= 0 else "") + fmt(dW, 2), "g/kg")
                + '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info(
            "Sélectionnez deux paramètres et cliquez sur **▶ Calculer** "
            "pour afficher les résultats."
        )


# ── Zone graphique ────────────────────────────────────────────────────
with col_graphique:
    if msg_erreur:
        st.error(f"Erreur de calcul : {msg_erreur}")

    @st.cache_data(show_spinner="Génération du fond…")
    def obtenir_fond(pression: float) -> go.Figure:
        return construire_fond(pression)

    # Fond (mis en cache par pression)
    fig = copy.deepcopy(obtenir_fond(round(P, 1)))

    # Superposition des points d'état
    if etat1:
        fig = ajouter_point_etat(
            fig, etat1.Tbs, etat1.W * 1000,
            etiquette="État 1", indice=1,
        )

    if etat2:
        couleur_proc = COULEURS_PROCESSUS.get(
            st.session_state.processus_cle, "#555555"
        )
        fig = ajouter_point_etat(
            fig, etat2.Tbs, etat2.W * 1000,
            etiquette="État 2",
            couleur=couleur_proc,
            indice=2,
        )
        fig = ajouter_fleche_processus(
            fig, etat1, etat2,
            nom_processus=label_proc,
            couleur=couleur_proc,
        )

    # Affichage Plotly
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar" : True,
            "displaylogo"    : False,
            "scrollZoom"     : True,
            "modeBarButtonsToRemove": [
                "select2d", "lasso2d", "autoScale2d",
            ],
            "toImageButtonOptions": {
                "format"  : "png",
                "filename": "diagramme_psychrometrique",
                "scale"   : 2,
            },
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# EXPORT PDF
# ═══════════════════════════════════════════════════════════════════════
if btn_pdf:
    if not etat1:
        st.sidebar.warning("Calculez d'abord un état pour générer le PDF.")
    else:
        # Recomposer la figure complète
        fig_pdf_base = copy.deepcopy(obtenir_fond(round(P, 1)))
        fig_pdf_base = ajouter_point_etat(
            fig_pdf_base, etat1.Tbs, etat1.W * 1000,
            etiquette="État 1", indice=1,
        )
        if etat2:
            couleur_proc = COULEURS_PROCESSUS.get(
                st.session_state.processus_cle, "#555555"
            )
            fig_pdf_base = ajouter_point_etat(
                fig_pdf_base, etat2.Tbs, etat2.W * 1000,
                etiquette="État 2", couleur=couleur_proc, indice=2,
            )
            fig_pdf_base = ajouter_fleche_processus(
                fig_pdf_base, etat1, etat2,
                nom_processus=label_proc, couleur=couleur_proc,
            )

        chemin_sortie = "diagramme_psychrometrique.pdf"
        with st.sidebar:
            with st.spinner("Génération du PDF…"):
                succes, resultat = exporter_pdf(
                    fig_pdf_base,
                    chemin=chemin_sortie,
                    titre_rapport=titre_rapport or "Rapport Psychrométrique",
                    projet=nom_projet,
                    ingenieur=ingenieur,
                )

        if succes:
            # Détecter si le fichier de repli est PNG
            est_png = resultat.endswith(".png")
            mime    = "image/png" if est_png else "application/pdf"
            label_dl = "⬇ Télécharger le PNG" if est_png else "⬇ Télécharger le PDF"

            with open(resultat, "rb") as fich:
                st.sidebar.download_button(
                    label        = label_dl,
                    data         = fich,
                    file_name    = resultat.split("/")[-1],
                    mime         = mime,
                    use_container_width=True,
                )
            if est_png:
                st.sidebar.info(
                    "Le PDF n'a pas pu être généré (kaleido manquant). "
                    "Un PNG haute résolution a été produit à la place.\n\n"
                    "Pour activer l'export PDF : `pip install -U kaleido`"
                )
        else:
            st.sidebar.error(f"Échec de l'export :\n\n{resultat}")


# ═══════════════════════════════════════════════════════════════════════
# PIED DE PAGE
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "PsychroCalc Pro  ·   ·  "
    "Visualisation : Plotly  ·  Interface : Streamlit"
)
