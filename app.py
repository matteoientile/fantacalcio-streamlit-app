import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.modifier import simulate_modifier
from src.metrics import (
    calculate_role_vorp,
    compute_dynamic_target,
    compute_role_pool_sum,
    compute_role_inflation,
    REPLACEMENT_RANKS,
    ROSTER_SLOTS,
    ROSTER_LIMIT,
)

st.set_page_config(page_title="Fantacalcio Auction Engine", layout="wide", initial_sidebar_state="expanded")

# File paths
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
DRAFT_LOG_PATH = os.path.join(ROOT_DIR, "data", "draft_log.json")
DRAFT_LOG_BACKUP_PATH = os.path.join(ROOT_DIR, "data", "draft_log.backup.json")
LEAGUE_CONFIG_PATH = os.path.join(ROOT_DIR, "data", "league_config.json")
TARGETS_PATH = os.path.join(ROOT_DIR, "data", "player_targets.json")

# Default league parameters
DEFAULT_CONFIG = {
    "num_teams": 12,
    "starting_budget": 500,
    "managers": ["Me"] + [f"Team {i}" for i in range(2, 13)]
}

# ----------------- TRANSLATIONS DICTIONARY -----------------
TRANSLATIONS = {
    "EN": {
        "lang_label": "🌐 Language / Lingua",
        "sidebar_title": "💰 Live Auction State",
        "league_settings": "⚙️ League Settings & Managers",
        "teams": "Teams",
        "budget": "Budget (cr)",
        "paste_caption": "Paste all **{n}** manager names (one per line or comma-separated). **The first name is your team.**",
        "save_settings": "Save League Settings",
        "settings_saved": "✅ League settings saved!",
        "settings_error": "Provided {found} names, but selected {expected} teams.",
        "my_budget": "{name}'s Remaining Budget",
        "spent_delta": "-{spent} spent",
        "slots_filled": "Roster Slots Filled",
        "avg_cr_slot": "Avg Credits / Remaining Slot",
        "total_room_spent": "Total Room Spent",
        "inflation_idx": "Global Inflation Index",
        "still_need": "🧩 Still Need",
        "alloc_title": "🎛️ Target Budget Allocation",
        "alloc_mode": "Allocation Mode:",
        "alloc_pct": "Percentage (%)",
        "alloc_abs": "Absolute Credits (cr)",
        "role_p": "Portieri (P) %",
        "role_d": "Difensori (D) %",
        "role_c": "Centrocampisti (C) %",
        "role_a": "Attaccanti (A) %",
        "role_p_cr": "Portieri (P) cr",
        "role_d_cr": "Difensori (D) cr",
        "role_c_cr": "Centrocampisti (C) cr",
        "role_a_cr": "Attaccanti (A) cr",
        "alloc_sum_warn_pct": "⚠️ Total allocation: **{tot}%** (must sum to 100%)",
        "alloc_sum_warn_cr": "⚠️ Total: **{tot} / {start} cr** ({sign}{diff} cr)",
        "role_inflation": "📉 Inflation by Role",
        "reset_draft": "⚠️ Reset Entire Draft",
        "reset_confirm_msg": "This wipes the entire draft log. Are you sure?",
        "reset_yes": "Yes, reset",
        "reset_cancel": "Cancel",
        "tab1": "📋 Master Listone & Live Logger",
        "tab2": "🎯 Live Player HUD",
        "tab3": "🏆 League & Opponent Rosters",
        "tab4": "🛡️ Modificatore Sandbox",
        "tab5": "🎯 Personal Targets & Flags",
        "quick_logger": "⚡ Live Call & Quick Logger",
        "player_called": "Player Called:",
        "bought_by": "Bought By:",
        "winning_bid": "Winning Bid (cr):",
        "log_purchase": "Log Purchase",
        "over_walkaway_warn": "⚠️ {price} cr exceeds your walk-away price of {wa} cr (fair price: {fp} cr) for **{p}**.",
        "confirm_over": "Confirm over walk-away",
        "log_success": "Logged {player} to {buyer} for {price} cr",
        "recent_history": "🕒 Recent Auction History ({n} logged)",
        "undo_last": "↩️ Undo Last",
        "undrafted_listone": "📋 Complete Undrafted Listone",
        "role_filter": "Role:",
        "min_pres": "Min Pres. 25/26:",
        "quick_search": "Quick Search (Name or Team):",
        "visible_cols": "Select Visible Columns:",
        "tag_filter": "Filter by Tag:",
        "tag_all": "All Players",
        "tag_only_flagged": "⭐ Only Flagged Targets",
        "search_hud": "Search Target Player:",
        "drafted_by_warn": "⚠️ Drafted by **{buyer}** for **{price} cr**",
        "log_this_player": "Log This Player",
        "hud_role_arch": "Role & Archetype",
        "hud_mv_w": "Weighted Pure Voto",
        "hud_fm_w": "Weighted Fantamedia",
        "hud_target": "Fair Target Bid",
        "hud_max": "Walk-Away Max",
        "hud_p_ge_6": "P(Voto ≥ 6.0)",
        "hud_p_ge_6_5": "P(Voto ≥ 6.5)",
        "hud_tot_ga": "Total Goals / Assists",
        "hud_malus": "Discipline Malus / Game",
        "hud_chart_title": "📈 Pure Grade vs. Fantavoto Distribution",
        "hud_trace_pure": "Pure Voto (Modifier Floor)",
        "hud_trace_fanta": "Fantavoto (Bonus Upside)",
        "t3_title": "🏆 League Standing, Liquidity & Opponent Rosters",
        "t3_col_mgr": "Manager",
        "t3_col_rem": "Remaining Budget (cr)",
        "t3_col_spent": "Spent (cr)",
        "t3_col_slots": "Slots",
        "t3_col_max_bid": "Max Bid (cr)",
        "t3_col_avg_cr": "Avg cr/Slot",
        "t3_inspect": "🔍 Opponent Roster Inspector",
        "t3_inspect_select": "Select Team to Inspect:",
        "t3_spending_breakdown": "**{name} Spending Breakdown by Role**",
        "t3_no_players": "{name} has not drafted any players yet.",
        "t4_title": "🛡️ Live Defensive Modifier Engine",
        "t4_gk": "Portiere (GK)",
        "t4_d1": "Defender 1",
        "t4_d2": "Defender 2",
        "t4_d3": "Defender 3",
        "t4_d4": "Defender 4",
        "t4_exp_bonus": "Expected Modifier Bonus / Matchday",
        "t4_proj_rating": "Projected Mean Backline Pure Rating",
        "t4_sampling_mode": "Sampling Mode",
        "t4_joint_mode": "Joint matchday (correlated)",
        "t4_indep_mode": "Independent (fallback)",
        "t4_chart_title": "Modifier Tier Probability Breakdown",
        "t4_chart_tier": "Modifier Tier",
        "t4_chart_prob": "Probability (%)",
        "t4_insufficient": "Insufficient appearance records for one or more chosen players.",
        "t5_title": "🎯 Personal Targets & Custom Color Flags",
        "t5_subtitle": "Assign strategic tags and personal notes to priority targets.",
        "t5_select_player": "Select Player to Flag:",
        "t5_select_tag": "Assign Color Tag / Strategy:",
        "t5_notes": "Personal Scouting Note / Notes (Optional):",
        "t5_save_btn": "Save / Update Tag",
        "t5_saved_msg": "Flag updated for {p}: {tag}",
        "t5_flagged_title": "📑 Active Priority Targets Portfolio",
        "t5_no_targets": "No targets flagged yet. Add tags above to build your shortlist!"
    },
    "IT": {
        "lang_label": "🌐 Language / Lingua",
        "sidebar_title": "💰 Stato Asta Live",
        "league_settings": "⚙️ Impostazioni Lega & Fantallenatori",
        "teams": "Squadre",
        "budget": "Budget (cr)",
        "paste_caption": "Incolla tutti i **{n}** fantallenatori (uno per riga o separati da virgola). **Il primo sei tu.**",
        "save_settings": "Salva Impostazioni Lega",
        "settings_saved": "✅ Impostazioni lega salvate!",
        "settings_error": "Inseriti {found} nomi, ma sono state selezionate {expected} squadre.",
        "my_budget": "Budget Residuo di {name}",
        "spent_delta": "-{spent} spesi",
        "slots_filled": "Slot Roster Completati",
        "avg_cr_slot": "Media Crediti / Slot Rimanente",
        "total_room_spent": "Crediti Spesi Totali nella Lega",
        "inflation_idx": "Indice d'Inflazione Globale",
        "still_need": "🧩 Mancanti al Roster",
        "alloc_title": "🎛️ Allocazione Budget Obiettivo",
        "alloc_mode": "Modalità Allocazione:",
        "alloc_pct": "Percentuale (%)",
        "alloc_abs": "Crediti Assoluti (cr)",
        "role_p": "Portieri (P) %",
        "role_d": "Difensori (D) %",
        "role_c": "Centrocampisti (C) %",
        "role_a": "Attaccanti (A) %",
        "role_p_cr": "Portieri (P) cr",
        "role_d_cr": "Difensori (D) cr",
        "role_c_cr": "Centrocampisti (C) cr",
        "role_a_cr": "Attaccanti (A) cr",
        "alloc_sum_warn_pct": "⚠️ Totale allocazione: **{tot}%** (la somma deve fare 100%)",
        "alloc_sum_warn_cr": "⚠️ Totale: **{tot} / {start} cr** ({sign}{diff} cr)",
        "role_inflation": "📉 Inflazione per Ruolo",
        "reset_draft": "⚠️ Resetta Intera Asta",
        "reset_confirm_msg": "Questo cancellerà l'intero storico acquisti. Sei sicuro?",
        "reset_yes": "Sì, resetta",
        "reset_cancel": "Annulla",
        "tab1": "📋 Master Listone & Live Logger",
        "tab2": "🎯 HUD Giocatore Live",
        "tab3": "🏆 Rose Lega & Avversari",
        "tab4": "🛡️ Modificatore Sandbox",
        "tab5": "🎯 Obiettivi Personali & Flag",
        "quick_logger": "⚡ Chiamata Live & Inserimento Rapido",
        "player_called": "Giocatore Chiamato:",
        "bought_by": "Acquistato Da:",
        "winning_bid": "Prezzo Finale (cr):",
        "log_purchase": "Registra Acquisto",
        "over_walkaway_warn": "⚠️ {price} cr supera il prezzo limite di {wa} cr (prezzo target: {fp} cr) per **{p}**.",
        "confirm_over": "Conferma acquisto sopra il limite",
        "log_success": "Registrato {player} a {buyer} per {price} cr",
        "recent_history": "🕒 Storico Acquisti Recenti ({n} registrati)",
        "undo_last": "↩️ Annulla Ultimo",
        "undrafted_listone": "📋 Listone Completo Svincolati",
        "role_filter": "Ruolo:",
        "min_pres": "Presenze Minime 25/26:",
        "quick_search": "Cerca Rapida (Nome o Squadra):",
        "visible_cols": "Seleziona Colonne Visibili:",
        "tag_filter": "Filtro Flag / Tag:",
        "tag_all": "Tutti i Giocatori",
        "tag_only_flagged": "⭐ Solo Obiettivi Flagged",
        "search_hud": "Cerca Giocatore Bersaglio:",
        "drafted_by_warn": "⚠️ Acquistato da **{buyer}** a **{price} cr**",
        "log_this_player": "Registra Questo Giocatore",
        "hud_role_arch": "Ruolo & Archetipo",
        "hud_mv_w": "Media Voto Pura Ponderata",
        "hud_fm_w": "Fantamedia Ponderata",
        "hud_target": "Prezzo Target Equo",
        "hud_max": "Prezzo Limite (Walk-Away)",
        "hud_p_ge_6": "P(Voto ≥ 6.0)",
        "hud_p_ge_6_5": "P(Voto ≥ 6.5)",
        "hud_tot_ga": "Gol / Assist Totali",
        "hud_malus": "Malus Disciplina / Partita",
        "hud_chart_title": "📈 Distribuzione Voto Puro vs. Fantavoto",
        "hud_trace_pure": "Voto Puro (Floor Modificatore)",
        "hud_trace_fanta": "Fantavoto (Upside Bonus)",
        "t3_title": "🏆 Classifica Lega, Liquidità & Rose Avversari",
        "t3_col_mgr": "Fantallenatore",
        "t3_col_rem": "Budget Residuo (cr)",
        "t3_col_spent": "Spesi (cr)",
        "t3_col_slots": "Slot",
        "t3_col_max_bid": "Offerta Max (cr)",
        "t3_col_avg_cr": "Media cr/Slot",
        "t3_inspect": "🔍 Ispettore Rosa Avversario",
        "t3_inspect_select": "Seleziona Squadra da Analizzare:",
        "t3_spending_breakdown": "**Distribuzione Spesa per Ruolo di {name}**",
        "t3_no_players": "{name} non ha ancora acquistato giocatori.",
        "t4_title": "🛡️ Simulatore Modificatore di Difesa Live",
        "t4_gk": "Portiere (P)",
        "t4_d1": "Difensore 1",
        "t4_d2": "Difensore 2",
        "t4_d3": "Difensore 3",
        "t4_d4": "Difensore 4",
        "t4_exp_bonus": "Bonus Modificatore Atteso / Giornata",
        "t4_proj_rating": "Media Voto Pura Reparto Prevista",
        "t4_sampling_mode": "Modalità Campionamento",
        "t4_joint_mode": "Giornata Congiunta (correlata)",
        "t4_indep_mode": "Indipendente (fallback)",
        "t4_chart_title": "Distribuzione Probabilità Fasce Modificatore",
        "t4_chart_tier": "Fascia Modificatore",
        "t4_chart_prob": "Probabilità (%)",
        "t4_insufficient": "Storico presenze insufficiente per uno o più giocatori selezionati.",
        "t5_title": "🎯 Obiettivi Personali & Flag Colorati",
        "t5_subtitle": "Assegna tag strategici e note personali ai tuoi bersagli d'asta.",
        "t5_select_player": "Seleziona Giocatore da Contrassegnare:",
        "t5_select_tag": "Assegna Flag / Categoria Strategica:",
        "t5_notes": "Note Personali / Strategia (Opzionale):",
        "t5_save_btn": "Salva / Aggiorna Tag",
        "t5_saved_msg": "Tag aggiornato per {p}: {tag}",
        "t5_flagged_title": "📑 Portfolio Obiettivi Attivi",
        "t5_no_targets": "Nessun obiettivo contrassegnato. Aggiungi i tuoi primi tag in alto per costruire la shortlist!"
    }
}

TAG_OPTIONS = [
    "⚪ Nessun Tag / Clear",
    "🔴 Rigorista",
    "🟢 Titolare",
    "🟡 Punizioni / Corner",
    "🟣 Obiettivo primario",
    "🔵 Scommessa / Slot Scommessa"
]

# ----------------- CONFIG & PERSISTENCE -----------------
def load_league_config():
    if os.path.exists(LEAGUE_CONFIG_PATH):
        try:
            with open(LEAGUE_CONFIG_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "managers" in data:
                    return data
        except Exception:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

def save_league_config(config_data):
    with open(LEAGUE_CONFIG_PATH, "w") as f:
        json.dump(config_data, f, indent=2)

def load_draft_log():
    if os.path.exists(DRAFT_LOG_PATH):
        try:
            with open(DRAFT_LOG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_draft_log(log_data):
    if os.path.exists(DRAFT_LOG_PATH):
        try:
            with open(DRAFT_LOG_PATH, "r") as f:
                prev = f.read()
            with open(DRAFT_LOG_BACKUP_PATH, "w") as f:
                f.write(prev)
        except Exception:
            pass
    with open(DRAFT_LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=2)

def load_player_targets():
    if os.path.exists(TARGETS_PATH):
        try:
            with open(TARGETS_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            return {}
    return {}

def save_player_targets(targets_data):
    with open(TARGETS_PATH, "w") as f:
        json.dump(targets_data, f, indent=2)

# ----------------- LOAD DATA & SESSION STATE -----------------
@st.cache_data
def load_data():
    matches = pd.read_parquet(os.path.join(DATA_DIR, "master_matches.parquet"), engine='fastparquet')
    profiles = pd.read_parquet(os.path.join(DATA_DIR, "master_player_profiles.parquet"), engine='fastparquet')
    profiles = calculate_role_vorp(profiles)
    return matches, profiles

matches_df, profiles_df = load_data()

if "league_config" not in st.session_state:
    st.session_state.league_config = load_league_config()
if "draft_log" not in st.session_state:
    st.session_state.draft_log = load_draft_log()
if "player_targets" not in st.session_state:
    st.session_state.player_targets = load_player_targets()
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False
if "selected_player" not in st.session_state:
    st.session_state.selected_player = profiles_df['Nome'].sort_values().iloc[0]

TOTAL_TEAMS = st.session_state.league_config.get("num_teams", 12)
STARTING_BUDGET = st.session_state.league_config.get("starting_budget", 500)
MANAGERS = st.session_state.league_config.get("managers", DEFAULT_CONFIG["managers"])
MY_NAME = MANAGERS[0]
TOTAL_LEAGUE_CREDITS = TOTAL_TEAMS * STARTING_BUDGET

# ----------------- LANGUAGE SWITCHER & TRANSLATOR -----------------
lang_choice = st.sidebar.radio("Language / Lingua", ["🇮🇹 IT", "🇬🇧 EN"], horizontal=True, key="lang_radio")
current_lang = "IT" if "IT" in lang_choice else "EN"

def t(key, **kwargs):
    lang_dict = TRANSLATIONS.get(current_lang, TRANSLATIONS["IT"])
    text = lang_dict.get(key, TRANSLATIONS["EN"].get(key, key))
    return text.format(**kwargs) if kwargs else text

# ----------------- LIVE DRAFT CALCULATIONS -----------------
draft_df = pd.DataFrame(st.session_state.draft_log) if st.session_state.draft_log else pd.DataFrame(columns=["player", "role", "buyer", "price"])

if not draft_df.empty:
    drafted_names = set(draft_df['player'])
    total_spent_room = int(draft_df['price'].sum())

    my_purchases = draft_df[draft_df['buyer'] == MY_NAME]
    my_spent = int(my_purchases['price'].sum())
    my_budget = STARTING_BUDGET - my_spent
    my_slots_filled = len(my_purchases)
    my_role_slots_filled = my_purchases.groupby('role').size().to_dict()
else:
    drafted_names = set()
    total_spent_room = 0
    my_spent = 0
    my_budget = STARTING_BUDGET
    my_slots_filled = 0
    my_role_slots_filled = {}

remaining_league_pool = max(0, TOTAL_LEAGUE_CREDITS - total_spent_room)
global_inflation_index = max(0.2, remaining_league_pool / TOTAL_LEAGUE_CREDITS)

available_profiles_df = profiles_df[~profiles_df['Nome'].isin(drafted_names)].copy()
role_pool_sum = compute_role_pool_sum(available_profiles_df)

# ----------------- SIDEBAR: CONFIG & AUCTION STATE -----------------
st.sidebar.title(t("sidebar_title"))

with st.sidebar.expander(t("league_settings"), expanded=False):
    c_teams, c_bud = st.columns(2)
    cfg_teams = c_teams.selectbox(t("teams"), [6, 8, 10, 12, 14], index=[6, 8, 10, 12, 14].index(TOTAL_TEAMS) if TOTAL_TEAMS in [6, 8, 10, 12, 14] else 3)
    cfg_budget = c_bud.number_input(t("budget"), min_value=100, max_value=2000, value=STARTING_BUDGET, step=50)

    st.caption(t("paste_caption", n=cfg_teams))
    curr_names_str = "\n".join(MANAGERS[:cfg_teams])
    raw_names_input = st.text_area("Manager List", value=curr_names_str, height=160, label_visibility="collapsed")

    if st.button(t("save_settings"), type="primary"):
        parsed_names = [n.strip() for n in raw_names_input.replace(",", "\n").split("\n") if n.strip()]
        if len(parsed_names) != cfg_teams:
            st.error(t("settings_error", found=len(parsed_names), expected=cfg_teams))
        else:
            updated_cfg = {
                "num_teams": int(cfg_teams),
                "starting_budget": int(cfg_budget),
                "managers": parsed_names
            }
            st.session_state.league_config = updated_cfg
            save_league_config(updated_cfg)
            st.success(t("settings_saved"))
            st.rerun()

st.sidebar.metric(t("my_budget", name=MY_NAME), f"{my_budget} cr", delta=t("spent_delta", spent=my_spent) if my_spent > 0 else None)
st.sidebar.metric(t("slots_filled"), f"{my_slots_filled} / {ROSTER_LIMIT}")

slots_left = max(1, ROSTER_LIMIT - my_slots_filled)
avg_per_slot = my_budget / slots_left
st.sidebar.metric(t("avg_cr_slot"), f"{avg_per_slot:.1f} cr")

st.sidebar.metric(t("total_room_spent"), f"{total_spent_room} / {TOTAL_LEAGUE_CREDITS} cr")
st.sidebar.metric(t("inflation_idx"), f"{global_inflation_index:.2f}x")

# Role Needs
st.sidebar.markdown(f"##### {t('still_need')}")
need_cols = st.sidebar.columns(4)
for i, role in enumerate(['P', 'D', 'C', 'A']):
    have = my_role_slots_filled.get(role, 0)
    total = ROSTER_SLOTS[role]
    open_slots = max(0, total - have)
    need_cols[i].metric(role, f"{open_slots}", help=f"{have}/{total} filled")

st.sidebar.markdown("---")
st.sidebar.markdown(f"### {t('alloc_title')}")

alloc_mode = st.sidebar.radio(t("alloc_mode"), [t("alloc_pct"), t("alloc_abs")], horizontal=True)

if alloc_mode == t("alloc_pct"):
    b_gk = st.sidebar.slider(t("role_p"), 3, 20, 9, step=1)
    b_def = st.sidebar.slider(t("role_d"), 5, 35, 20, step=1)
    b_mid = st.sidebar.slider(t("role_c"), 10, 45, 25, step=1)
    b_fwd = st.sidebar.slider(t("role_a"), 25, 75, 46, step=1)

    total_alloc = b_gk + b_def + b_mid + b_fwd
    if total_alloc != 100:
        st.sidebar.warning(t("alloc_sum_warn_pct", tot=total_alloc))

    role_budget_pct = {
        'P': b_gk / 100.0,
        'D': b_def / 100.0,
        'C': b_mid / 100.0,
        'A': b_fwd / 100.0
    }
else:
    b_gk_cr = st.sidebar.number_input(t("role_p_cr"), min_value=3, max_value=STARTING_BUDGET, value=int(0.09 * STARTING_BUDGET), step=5)
    b_def_cr = st.sidebar.number_input(t("role_d_cr"), min_value=8, max_value=STARTING_BUDGET, value=int(0.2 * STARTING_BUDGET), step=5)
    b_mid_cr = st.sidebar.number_input(t("role_c_cr"), min_value=8, max_value=STARTING_BUDGET, value=int(0.25 * STARTING_BUDGET), step=5)
    b_fwd_cr = st.sidebar.number_input(t("role_a_cr"), min_value=6, max_value=STARTING_BUDGET, value=int(0.46 * STARTING_BUDGET), step=5)

    total_alloc = b_gk_cr + b_def_cr + b_mid_cr + b_fwd_cr
    if total_alloc != STARTING_BUDGET:
        diff = total_alloc - STARTING_BUDGET
        st.sidebar.warning(t("alloc_sum_warn_cr", tot=total_alloc, start=STARTING_BUDGET, sign='+' if diff > 0 else '', diff=diff))

    role_budget_pct = {
        'P': b_gk_cr / float(STARTING_BUDGET),
        'D': b_def_cr / float(STARTING_BUDGET),
        'C': b_mid_cr / float(STARTING_BUDGET),
        'A': b_fwd_cr / float(STARTING_BUDGET)
    }

role_inflation = compute_role_inflation(draft_df, role_budget_pct, TOTAL_LEAGUE_CREDITS)

st.sidebar.markdown("---")
st.sidebar.markdown(f"##### {t('role_inflation')}")
infl_cols = st.sidebar.columns(4)
for i, role in enumerate(['P', 'D', 'C', 'A']):
    infl_cols[i].metric(role, f"{role_inflation[role]:.2f}x")

st.sidebar.markdown("---")
if not st.session_state.confirm_reset:
    if st.sidebar.button(t("reset_draft")):
        st.session_state.confirm_reset = True
        st.rerun()
else:
    st.sidebar.error(t("reset_confirm_msg"))
    rc1, rc2 = st.sidebar.columns(2)
    if rc1.button(t("reset_yes"), type="primary"):
        st.session_state.draft_log = []
        save_draft_log([])
        st.session_state.confirm_reset = False
        st.rerun()
    if rc2.button(t("reset_cancel")):
        st.session_state.confirm_reset = False
        st.rerun()

def get_dynamic_prices(row):
    return compute_dynamic_target(
        row,
        role_budget_pct=role_budget_pct,
        total_budget=STARTING_BUDGET,
        role_inflation=role_inflation,
        my_remaining_budget=my_budget,
        role_pool_sum=role_pool_sum,
        my_role_slots_filled=my_role_slots_filled,
        my_total_slots_filled=my_slots_filled,
    )

def log_purchase(player_name, buyer, price):
    player_role = profiles_df[profiles_df['Nome'] == player_name]['Ruolo'].iloc[0]
    entry = {
        "player": player_name,
        "role": player_role,
        "buyer": buyer,
        "price": int(price)
    }
    st.session_state.draft_log.append(entry)
    save_draft_log(st.session_state.draft_log)

# ----------------- SYNC CALLBACKS -----------------
def sync_from_t1():
    st.session_state.selected_player = st.session_state.t1_player
    st.session_state.t2_player = st.session_state.t1_player

def sync_from_t2():
    st.session_state.selected_player = st.session_state.t2_player
    st.session_state.t1_player = st.session_state.t2_player

# ----------------- MAIN TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t("tab1"), 
    t("tab2"), 
    t("tab3"), 
    t("tab4"),
    t("tab5")
])

# ----------------- TAB 1: MASTER LISTONE & LIVE LOGGER -----------------
with tab1:
    st.markdown(f"### {t('quick_logger')}")
    
    available_players = available_profiles_df.sort_values('Nome')
    t1_options = available_players['Nome'].tolist()
    
    if "t1_player" not in st.session_state or st.session_state.t1_player not in t1_options:
        if len(t1_options) > 0:
            st.session_state.t1_player = t1_options[0]
            st.session_state.selected_player = t1_options[0]

    c_p, c_buyer, c_price, c_btn = st.columns([3, 2, 2, 2])
    with c_p:
        log_player = st.selectbox(
            t("player_called"),
            options=t1_options,
            key="t1_player",
            on_change=sync_from_t1
        )
    with c_buyer:
        log_buyer = st.selectbox(t("bought_by"), options=MANAGERS, key="t1_buyer")
    with c_price:
        max_bid_possible = my_budget if log_buyer == MY_NAME else STARTING_BUDGET
        log_price = st.number_input(t("winning_bid"), min_value=1, max_value=max_bid_possible, value=1, step=1, key="t1_price")
    with c_btn:
        st.write("")
        st.write("")
        log_btn_placeholder = st.container()

    over_walkaway = False
    if log_buyer == MY_NAME:
        row = profiles_df[profiles_df['Nome'] == log_player].iloc[0]
        fp, wa = get_dynamic_prices(row)
        if log_price > wa:
            over_walkaway = True
            st.warning(t("over_walkaway_warn", price=log_price, wa=wa, fp=fp, p=log_player))

    with log_btn_placeholder:
        if over_walkaway:
            confirm_over = st.checkbox(t("confirm_over"), key="t1_confirm_over")
            disabled = not confirm_over
        else:
            disabled = False
        if st.button(t("log_purchase"), use_container_width=True, type="primary", disabled=disabled, key="t1_log_btn"):
            log_purchase(log_player, log_buyer, log_price)
            st.success(t("log_success", player=log_player, buyer=log_buyer, price=log_price))
            st.rerun()

    if not draft_df.empty:
        with st.expander(t("recent_history", n=len(draft_df)), expanded=False):
            col_table, col_undo = st.columns([5, 1])
            with col_table:
                st.dataframe(draft_df.iloc[::-1], use_container_width=True, hide_index=True)
            with col_undo:
                if st.button(t("undo_last")):
                    st.session_state.draft_log.pop()
                    save_draft_log(st.session_state.draft_log)
                    st.rerun()

    st.markdown("---")

    # 3. Master Undrafted Listone
    st.markdown(f"##### {t('undrafted_listone')}")
    
    c_filter1, c_filter2, c_filter3, c_filter4 = st.columns([1, 1, 1.5, 2])
    with c_filter1:
        role_filter = st.selectbox(t("role_filter"), ['All', 'P', 'D', 'C', 'A'], key="t1_role")
    with c_filter2:
        min_presenze = st.number_input(t("min_pres"), min_value=0, max_value=38, value=0, step=1, key="t1_pres")
    with c_filter3:
        tag_filter = st.selectbox(t("tag_filter"), [t("tag_all"), t("tag_only_flagged")] + TAG_OPTIONS[1:], key="t1_tag_filter")
    with c_filter4:
        search_query = st.text_input(t("quick_search"), "", key="t1_search")
    
    avail_pool = available_profiles_df.copy()
    
    # Attach Tags from persistent storage
    targets_map = st.session_state.player_targets
    avail_pool['Tag'] = avail_pool['Nome'].map(lambda name: targets_map.get(name, {}).get("tag", ""))
    avail_pool['Note'] = avail_pool['Nome'].map(lambda name: targets_map.get(name, {}).get("note", ""))
    
    if role_filter != 'All':
        avail_pool = avail_pool[avail_pool['Ruolo'] == role_filter]
        
    if min_presenze > 0 and 'Presenze_Last_Season' in avail_pool.columns:
        avail_pool = avail_pool[avail_pool['Presenze_Last_Season'] >= min_presenze]
        
    if tag_filter == t("tag_only_flagged"):
        avail_pool = avail_pool[avail_pool['Tag'] != ""]
    elif tag_filter != t("tag_all") and tag_filter != "":
        avail_pool = avail_pool[avail_pool['Tag'] == tag_filter]

    if search_query:
        mask = (
            avail_pool['Nome'].str.contains(search_query, case=False, na=False) |
            avail_pool['Squadra'].str.contains(search_query, case=False, na=False)
        )
        avail_pool = avail_pool[mask]

    prices = avail_pool.apply(lambda r: get_dynamic_prices(r), axis=1)
    avail_pool['Target_cr'] = [p[0] for p in prices]
    avail_pool['Max_cr'] = [p[1] for p in prices]

    avail_pool['FM_W'] = avail_pool['Fanta_Media_Weighted'].round(2)
    avail_pool['FM_Raw'] = avail_pool.get('Fanta_Media_Raw', avail_pool['Fanta_Media_Weighted']).round(2)
    avail_pool['MV_W'] = avail_pool['Media_Voto_Weighted'].round(2)
    avail_pool['MV_Raw'] = avail_pool.get('Media_Voto_Raw', avail_pool['Media_Voto_Weighted']).round(2)
    avail_pool['P_ge_6'] = (avail_pool['P_Voto_ge_6'] * 100).round(1)
    avail_pool['P_ge_6_5'] = (avail_pool['P_Voto_ge_6_5'] * 100).round(1)
    if 'P_Voto_lt_6' in avail_pool.columns:
        avail_pool['P_lt_6'] = (avail_pool['P_Voto_lt_6'] * 100).round(1)
    else:
        avail_pool['P_lt_6'] = (100.0 - avail_pool['P_ge_6']).round(1)

    for int_col in ['Presenze_Last_Season', 'Presenze_Tot', 'Tot_Gol', 'Tot_Ass', 'Tot_Gs', 'Tot_Amm', 'Tot_Esp', 'Target_cr', 'Max_cr']:
        if int_col in avail_pool.columns:
            avail_pool[int_col] = avail_pool[int_col].fillna(0).astype(int)

    all_possible_cols = [
        'Tag', 'Nome', 'Squadra', 'Ruolo', 'Target_cr', 'Max_cr', 'Archetype',
        'Presenze_Last_Season', 'Presenze_Tot',
        'FM_W', 'FM_Raw', 'MV_W', 'MV_Raw',
        'P_ge_6', 'P_ge_6_5', 'P_lt_6',
        'Tot_Gol', 'Tot_Ass', 'Tot_Gs', 'Tot_Amm', 'Tot_Esp',
        'Bonus_per_Game', 'Malus_per_Game', 'Clean_Sheet_Rate', 'Note'
    ]
    
    default_selected = [
        'Nome', 'Squadra', 'Ruolo', 'Tag', 'Note', 
        'MV_Raw', 'FM_Raw', 'P_ge_6',
        'Target_cr', 'Max_cr',
        'Presenze_Last_Season', 'Presenze_Tot',
        'FM_W', 'MV_W', 'P_ge_6_5', 'P_lt_6', 
        'Bonus_per_Game', 'Malus_per_Game', 'Clean_Sheet_Rate'
    ]

    selected_columns = st.multiselect(
        t("visible_cols"),
        options=[c for c in all_possible_cols if c in avail_pool.columns],
        default=[c for c in default_selected if c in avail_pool.columns],
        key="t1_cols_picker"
    )

    column_configuration = {
        "Tag": st.column_config.TextColumn("Tag", width="medium"),
        "Target_cr": st.column_config.NumberColumn("Target (cr)", format="%d cr"),
        "Max_cr": st.column_config.NumberColumn("Max (cr)", format="%d cr"),
        "Presenze_Last_Season": st.column_config.NumberColumn("Pres 25/26", format="%d"),
        "Presenze_Tot": st.column_config.NumberColumn("Pres Tot", format="%d"),
        "FM_W": st.column_config.NumberColumn("FM (W)", format="%.2f"),
        "FM_Raw": st.column_config.NumberColumn("FM (Raw)", format="%.2f"),
        "MV_W": st.column_config.NumberColumn("MV (W)", format="%.2f"),
        "MV_Raw": st.column_config.NumberColumn("MV (Raw)", format="%.2f"),
        "P_ge_6": st.column_config.NumberColumn("P(Voto ≥ 6)", format="%.1f%%"),
        "P_ge_6_5": st.column_config.NumberColumn("P(Voto ≥ 6.5)", format="%.1f%%"),
        "P_lt_6": st.column_config.NumberColumn("P(Voto < 6)", format="%.1f%%"),
        "Tot_Gol": st.column_config.NumberColumn("Gol", format="%d"),
        "Tot_Ass": st.column_config.NumberColumn("Ass", format="%d"),
        "Bonus_per_Game": st.column_config.NumberColumn("Bonus/G", format="%.2f"),
        "Malus_per_Game": st.column_config.NumberColumn("Malus/G", format="%.2f"),
    }

    st.dataframe(
        avail_pool.sort_values(by='Target_cr', ascending=False)[selected_columns],
        use_container_width=True,
        hide_index=True,
        height=700,
        column_config=column_configuration
    )

# ----------------- TAB 2: PLAYER HUD -----------------
with tab2:
    all_player_names = profiles_df['Nome'].sort_values().unique().tolist()
    
    if "t2_player" not in st.session_state or st.session_state.t2_player not in all_player_names:
        st.session_state.t2_player = st.session_state.selected_player

    col_search, _ = st.columns([2, 1])
    with col_search:
        selected_player = st.selectbox(
            t("search_hud"),
            options=all_player_names,
            key="t2_player",
            on_change=sync_from_t2
        )

    p_info = profiles_df[profiles_df['Nome'] == selected_player].iloc[0]
    p_matches = matches_df[(matches_df['Cod.'] == p_info['Cod.']) & (matches_df['Voto_Puro'].notna())]
    fair_price, walk_away_price = get_dynamic_prices(p_info)

    # Display active flag if tagged
    p_tag = st.session_state.player_targets.get(selected_player, {}).get("tag", "")
    p_note = st.session_state.player_targets.get(selected_player, {}).get("note", "")
    if p_tag:
        st.info(f"**Target Flag:** {p_tag}" + (f" | *Note:* {p_note}" if p_note else ""))

    is_drafted = selected_player in drafted_names
    if is_drafted:
        bid_info = draft_df[draft_df['player'] == selected_player].iloc[0]
        st.warning(t("drafted_by_warn", buyer=bid_info['buyer'], price=bid_info['price']))
    else:
        hud_p, hud_price, hud_btn = st.columns([2, 2, 2])
        with hud_p:
            hud_buyer = st.selectbox(t("bought_by"), options=MANAGERS, key="t2_buyer")
        with hud_price:
            max_bid_possible = my_budget if hud_buyer == MY_NAME else STARTING_BUDGET
            hud_price_val = st.number_input(
                t("winning_bid"),
                min_value=1,
                max_value=max_bid_possible,
                value=max(1, fair_price) if hud_buyer == MY_NAME else 1,
                step=1,
                key="t2_price",
            )
        hud_over = hud_buyer == MY_NAME and hud_price_val > walk_away_price
        if hud_over:
            st.warning(t("over_walkaway_warn", price=hud_price_val, wa=walk_away_price, fp=fair_price, p=selected_player))
        with hud_btn:
            st.write("")
            hud_confirm = st.checkbox(t("confirm_over"), key="t2_confirm") if hud_over else True
            if st.button(t("log_this_player"), use_container_width=True, type="primary", disabled=not hud_confirm, key="t2_log_btn"):
                log_purchase(selected_player, hud_buyer, hud_price_val)
                st.success(t("log_success", player=selected_player, buyer=hud_buyer, price=hud_price_val))
                st.rerun()

    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(t("hud_role_arch"), f"{p_info['Ruolo']} - {p_info['Archetype']}")
    k2.metric(t("hud_mv_w"), f"{p_info['Media_Voto_Weighted']:.2f}")
    k3.metric(t("hud_fm_w"), f"{p_info['Fanta_Media_Weighted']:.2f}")
    k4.metric(t("hud_target"), f"{fair_price} cr")
    k5.metric(t("hud_max"), f"{walk_away_price} cr", delta=f"+{walk_away_price - fair_price}", delta_color="inverse")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(t("hud_p_ge_6"), f"{p_info['P_Voto_ge_6'] * 100:.1f}%")
    r2.metric(t("hud_p_ge_6_5"), f"{p_info['P_Voto_ge_6_5'] * 100:.1f}%")
    r3.metric(t("hud_tot_ga"), f"{int(p_info['Tot_Gol'])} G / {int(p_info['Tot_Ass'])} A")
    r4.metric(t("hud_malus"), f"-{p_info['Malus_per_Game']:.2f} pts")

    st.markdown(f"##### {t('hud_chart_title')}")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=p_matches['Voto_Puro'], name=t('hud_trace_pure'), opacity=0.75, xbins=dict(start=3.5, end=9.0, size=0.5)))
    fig.add_trace(go.Histogram(x=p_matches['Fantavoto'], name=t('hud_trace_fanta'), opacity=0.6, xbins=dict(start=3.5, end=15.0, size=0.5)))
    fig.update_layout(barmode='overlay', height=280, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 3: LEAGUE & OPPONENT ROSTERS -----------------
with tab3:
    st.markdown(f"### {t('t3_title')}")

    league_summary = []
    for mgr in MANAGERS:
        mgr_drafted = draft_df[draft_df['buyer'] == mgr] if not draft_df.empty else pd.DataFrame()
        spent = int(mgr_drafted['price'].sum()) if not mgr_drafted.empty else 0
        rem_budget = STARTING_BUDGET - spent
        filled_slots = len(mgr_drafted)
        open_slots = ROSTER_LIMIT - filled_slots
        
        max_bid = max(1, rem_budget - max(0, open_slots - 1)) if open_slots > 0 else 0
        avg_cr_slot = round(rem_budget / open_slots, 1) if open_slots > 0 else 0.0

        p_count = len(mgr_drafted[mgr_drafted['role'] == 'P']) if not mgr_drafted.empty else 0
        d_count = len(mgr_drafted[mgr_drafted['role'] == 'D']) if not mgr_drafted.empty else 0
        c_count = len(mgr_drafted[mgr_drafted['role'] == 'C']) if not mgr_drafted.empty else 0
        a_count = len(mgr_drafted[mgr_drafted['role'] == 'A']) if not mgr_drafted.empty else 0

        league_summary.append({
            t("t3_col_mgr"): mgr,
            t("t3_col_rem"): rem_budget,
            t("t3_col_spent"): spent,
            t("t3_col_slots"): f"{filled_slots}/{ROSTER_LIMIT}",
            t("t3_col_max_bid"): max_bid,
            t("t3_col_avg_cr"): avg_cr_slot,
            "P": f"{p_count}/3",
            "D": f"{d_count}/8",
            "C": f"{c_count}/8",
            "A": f"{a_count}/6",
        })

    summary_df = pd.DataFrame(league_summary)
    st.dataframe(summary_df.sort_values(by=t("t3_col_rem"), ascending=False), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"#### {t('t3_inspect')}")
    
    selected_mgr = st.selectbox(t("t3_inspect_select"), options=MANAGERS, index=1 if len(MANAGERS) > 1 else 0)
    
    mgr_squad = draft_df[draft_df['buyer'] == selected_mgr] if not draft_df.empty else pd.DataFrame()
    
    if not mgr_squad.empty:
        col_sq1, col_sq2 = st.columns([2, 1])
        with col_sq1:
            mgr_detailed = pd.merge(
                mgr_squad,
                profiles_df[['Nome', 'Squadra', 'Media_Voto_Weighted', 'Fanta_Media_Weighted', 'Tot_Gol', 'Tot_Ass']],
                left_on='player',
                right_on='Nome',
                how='left'
            )
            st.dataframe(
                mgr_detailed[['player', 'role', 'Squadra', 'price', 'Fanta_Media_Weighted', 'Media_Voto_Weighted', 'Tot_Gol', 'Tot_Ass']].rename(
                    columns={'player': 'Player', 'role': 'Role', 'price': 'Price (cr)', 'Fanta_Media_Weighted': 'FM (W)', 'Media_Voto_Weighted': 'MV (W)'}
                ),
                use_container_width=True,
                hide_index=True
            )
        with col_sq2:
            st.markdown(t("t3_spending_breakdown", name=selected_mgr))
            role_spend = mgr_squad.groupby('role')['price'].sum().reset_index()
            fig_pie = px.pie(role_spend, values='price', names='role', hole=0.4, color='role',
                             color_discrete_map={'P': '#1f77b4', 'D': '#2ca02c', 'C': '#ff7f0e', 'A': '#d62728'})
            fig_pie.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info(t("t3_no_players", name=selected_mgr))

# ----------------- TAB 4: MODIFICATORE SANDBOX -----------------
with tab4:
    st.markdown(f"### {t('t4_title')}")

    gk_options = profiles_df[profiles_df['Ruolo'] == 'P']['Nome'].sort_values().tolist()
    def_options = profiles_df[profiles_df['Ruolo'] == 'D']['Nome'].sort_values().tolist()

    col_gk, col_d1, col_d2, col_d3, col_d4 = st.columns(5)
    sel_gk = col_gk.selectbox(t("t4_gk"), options=gk_options, index=0)
    sel_d1 = col_d1.selectbox(t("t4_d1"), options=def_options, index=0)
    sel_d2 = col_d2.selectbox(t("t4_d2"), options=def_options, index=1 if len(def_options) > 1 else 0)
    sel_d3 = col_d3.selectbox(t("t4_d3"), options=def_options, index=2 if len(def_options) > 2 else 0)
    sel_d4 = col_d4.selectbox(t("t4_d4"), options=def_options, index=3 if len(def_options) > 3 else 0)

    gk_rows = matches_df[(matches_df['Nome'] == sel_gk) & (matches_df['Voto_Puro'].notna())]
    d1_rows = matches_df[(matches_df['Nome'] == sel_d1) & (matches_df['Voto_Puro'].notna())]
    d2_rows = matches_df[(matches_df['Nome'] == sel_d2) & (matches_df['Voto_Puro'].notna())]
    d3_rows = matches_df[(matches_df['Nome'] == sel_d3) & (matches_df['Voto_Puro'].notna())]
    d4_rows = matches_df[(matches_df['Nome'] == sel_d4) & (matches_df['Voto_Puro'].notna())]

    if all(len(r) > 0 for r in [gk_rows, d1_rows, d2_rows, d3_rows, d4_rows]):
        sim_res = simulate_modifier(gk_rows, [d1_rows, d2_rows, d3_rows, d4_rows])

        m1, m2, m3 = st.columns(3)
        m1.metric(t("t4_exp_bonus"), f"+{sim_res['expected_bonus']} pts")
        m2.metric(t("t4_proj_rating"), f"{sim_res['mean_rating']} / 10")
        mode_label = t("t4_joint_mode") if sim_res['sampling_mode'] == "joint_matchday" else t("t4_indep_mode")
        m3.metric(t("t4_sampling_mode"), mode_label)

        prob_df = pd.DataFrame({
            t("t4_chart_tier"): [f"+{k}" for k in sim_res['probs'].keys()],
            t("t4_chart_prob"): [v * 100 for v in sim_res['probs'].values()]
        })
        fig_prob = px.bar(prob_df, x=t("t4_chart_tier"), y=t("t4_chart_prob"), text_auto='.1f', title=t("t4_chart_title"))
        fig_prob.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.warning(t("t4_insufficient"))

# ----------------- TAB 5: PERSONAL TARGETS & FLAGS -----------------
with tab5:
    st.markdown(f"### {t('t5_title')}")
    st.caption(t("t5_subtitle"))

    all_players_sorted = profiles_df['Nome'].sort_values().unique().tolist()

    c_t5_p, c_t5_tag, c_t5_note, c_t5_btn = st.columns([2.5, 2, 2.5, 1.5])
    with c_t5_p:
        tag_target_player = st.selectbox(t("t5_select_player"), options=all_players_sorted, key="t5_target_player")
    with c_t5_tag:
        curr_saved_tag = st.session_state.player_targets.get(tag_target_player, {}).get("tag", TAG_OPTIONS[0])
        default_tag_idx = TAG_OPTIONS.index(curr_saved_tag) if curr_saved_tag in TAG_OPTIONS else 0
        tag_choice = st.selectbox(t("t5_select_tag"), options=TAG_OPTIONS, index=default_tag_idx, key="t5_tag_choice")
    with c_t5_note:
        curr_saved_note = st.session_state.player_targets.get(tag_target_player, {}).get("note", "")
        note_choice = st.text_input(t("t5_notes"), value=curr_saved_note, key="t5_note_choice")
    with c_t5_btn:
        st.write("")
        st.write("")
        if st.button(t("t5_save_btn"), type="primary", use_container_width=True, key="t5_save_tag_btn"):
            if tag_choice == TAG_OPTIONS[0]:
                st.session_state.player_targets.pop(tag_target_player, None)
            else:
                st.session_state.player_targets[tag_target_player] = {
                    "tag": tag_choice,
                    "note": note_choice.strip()
                }
            save_player_targets(st.session_state.player_targets)
            st.success(t("t5_saved_msg", p=tag_target_player, tag=tag_choice))
            st.rerun()

    st.markdown("---")
    st.markdown(f"#### {t('t5_flagged_title')}")

    active_targets = st.session_state.player_targets
    if active_targets:
        targets_table = []
        for p_name, p_meta in active_targets.items():
            if p_name in profiles_df['Nome'].values:
                row = profiles_df[profiles_df['Nome'] == p_name].iloc[0]
                is_taken = p_name in drafted_names
                buyer_str = draft_df[draft_df['player'] == p_name]['buyer'].iloc[0] if is_taken else "Available / Svincolato"
                fp, wa = get_dynamic_prices(row)
                
                targets_table.append({
                    "Tag": p_meta.get("tag", ""),
                    "Giocatore": p_name,
                    "Ruolo": row['Ruolo'],
                    "Squadra": row['Squadra'],
                    "Target (cr)": int(fp),
                    "Max (cr)": int(wa),
                    "FM (W)": round(row['Fanta_Media_Weighted'], 2),
                    "MV (W)": round(row['Media_Voto_Weighted'], 2),
                    "P(Voto ≥ 6)": round(row['P_Voto_ge_6'] * 100, 1),
                    "Status": buyer_str,
                    "Note": p_meta.get("note", "")
                })

        targets_df = pd.DataFrame(targets_table)
        
        target_col_config = {
            "Tag": st.column_config.TextColumn("Tag", width="medium"),
            "Target (cr)": st.column_config.NumberColumn("Target (cr)", format="%d cr"),
            "Max (cr)": st.column_config.NumberColumn("Max (cr)", format="%d cr"),
            "FM (W)": st.column_config.NumberColumn("FM (W)", format="%.2f"),
            "MV (W)": st.column_config.NumberColumn("MV (W)", format="%.2f"),
            "P(Voto ≥ 6)": st.column_config.NumberColumn("P(Voto ≥ 6)", format="%.1f%%"),
        }

        st.dataframe(
            targets_df.sort_values(by=["Tag", "Target (cr)"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
            column_config=target_col_config
        )
    else:
        st.info(t("t5_no_targets"))