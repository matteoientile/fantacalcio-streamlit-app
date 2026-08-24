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

# Default league parameters
DEFAULT_CONFIG = {
    "num_teams": 12,
    "starting_budget": 500,
    "managers": ["Me"] + [f"Team {i}" for i in range(2, 13)]
}

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
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False
if "selected_player" not in st.session_state:
    st.session_state.selected_player = profiles_df['Nome'].sort_values().iloc[0]

TOTAL_TEAMS = st.session_state.league_config.get("num_teams", 12)
STARTING_BUDGET = st.session_state.league_config.get("starting_budget", 500)
MANAGERS = st.session_state.league_config.get("managers", DEFAULT_CONFIG["managers"])
MY_NAME = MANAGERS[0]
TOTAL_LEAGUE_CREDITS = TOTAL_TEAMS * STARTING_BUDGET

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
st.sidebar.title("💰 Live Auction State")

with st.sidebar.expander("⚙️ League Settings & Managers", expanded=False):
    c_teams, c_bud = st.columns(2)
    cfg_teams = c_teams.selectbox("Teams", [6, 8, 10, 12, 14], index=[6, 8, 10, 12, 14].index(TOTAL_TEAMS) if TOTAL_TEAMS in [6, 8, 10, 12, 14] else 3)
    cfg_budget = c_bud.number_input("Budget (cr)", min_value=100, max_value=2000, value=STARTING_BUDGET, step=50)

    st.caption(f"Paste all **{cfg_teams}** manager names (one per line or comma-separated). **The first name is your team.**")
    curr_names_str = "\n".join(MANAGERS[:cfg_teams])
    raw_names_input = st.text_area("Manager List", value=curr_names_str, height=160)

    if st.button("Save League Settings", type="primary"):
        parsed_names = [n.strip() for n in raw_names_input.replace(",", "\n").split("\n") if n.strip()]
        if len(parsed_names) != cfg_teams:
            st.error(f"Provided {len(parsed_names)} names, but selected {cfg_teams} teams.")
        else:
            updated_cfg = {
                "num_teams": int(cfg_teams),
                "starting_budget": int(cfg_budget),
                "managers": parsed_names
            }
            st.session_state.league_config = updated_cfg
            save_league_config(updated_cfg)
            st.success("✅ League settings saved!")
            st.rerun()

st.sidebar.metric(f"{MY_NAME}'s Remaining Budget", f"{my_budget} cr", delta=f"-{my_spent} spent" if my_spent > 0 else None)
st.sidebar.metric("Roster Slots Filled", f"{my_slots_filled} / {ROSTER_LIMIT}")

slots_left = max(1, ROSTER_LIMIT - my_slots_filled)
avg_per_slot = my_budget / slots_left
st.sidebar.metric("Avg Credits / Remaining Slot", f"{avg_per_slot:.1f} cr")

st.sidebar.metric("Total Room Spent", f"{total_spent_room} / {TOTAL_LEAGUE_CREDITS} cr")
st.sidebar.metric("Global Inflation Index", f"{global_inflation_index:.2f}x")

# Role Needs
st.sidebar.markdown("##### 🧩 Still Need")
need_cols = st.sidebar.columns(4)
for i, role in enumerate(['P', 'D', 'C', 'A']):
    have = my_role_slots_filled.get(role, 0)
    total = ROSTER_SLOTS[role]
    open_slots = max(0, total - have)
    need_cols[i].metric(role, f"{open_slots}", help=f"{have}/{total} filled")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Target Budget Allocation")

alloc_mode = st.sidebar.radio("Allocation Mode:", ["Percentage (%)", "Absolute Credits (cr)"], horizontal=True)

if alloc_mode == "Percentage (%)":
    b_gk = st.sidebar.slider("Portieri (P) %", 3, 20, 8, step=1)
    b_def = st.sidebar.slider("Difensori (D) %", 5, 35, 18, step=1)
    b_mid = st.sidebar.slider("Centrocampisti (C)", 10, 45, 26, step=1)
    b_fwd = st.sidebar.slider("Attaccanti (A) %", 25, 75, 48, step=1)

    total_alloc = b_gk + b_def + b_mid + b_fwd
    if total_alloc != 100:
        st.sidebar.warning(f"⚠️ Total allocation: **{total_alloc}%** (must sum to 100%)")

    role_budget_pct = {
        'P': b_gk / 100.0,
        'D': b_def / 100.0,
        'C': b_mid / 100.0,
        'A': b_fwd / 100.0
    }
else:
    b_gk_cr = st.sidebar.number_input("Portieri (P) cr", min_value=3, max_value=STARTING_BUDGET, value=int(0.08 * STARTING_BUDGET), step=5)
    b_def_cr = st.sidebar.number_input("Difensori (D) cr", min_value=8, max_value=STARTING_BUDGET, value=int(0.18 * STARTING_BUDGET), step=5)
    b_mid_cr = st.sidebar.number_input("Centrocampisti (C) cr", min_value=8, max_value=STARTING_BUDGET, value=int(0.26 * STARTING_BUDGET), step=5)
    b_fwd_cr = st.sidebar.number_input("Attaccanti (A) cr", min_value=6, max_value=STARTING_BUDGET, value=int(0.48 * STARTING_BUDGET), step=5)

    total_alloc = b_gk_cr + b_def_cr + b_mid_cr + b_fwd_cr
    if total_alloc != STARTING_BUDGET:
        diff = total_alloc - STARTING_BUDGET
        st.sidebar.warning(f"⚠️ Total: **{total_alloc} / {STARTING_BUDGET} cr** ({'+' if diff > 0 else ''}{diff} cr)")

    role_budget_pct = {
        'P': b_gk_cr / float(STARTING_BUDGET),
        'D': b_def_cr / float(STARTING_BUDGET),
        'C': b_mid_cr / float(STARTING_BUDGET),
        'A': b_fwd_cr / float(STARTING_BUDGET)
    }

role_inflation = compute_role_inflation(draft_df, role_budget_pct, TOTAL_LEAGUE_CREDITS)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 📉 Inflation by Role")
infl_cols = st.sidebar.columns(4)
for i, role in enumerate(['P', 'D', 'C', 'A']):
    infl_cols[i].metric(role, f"{role_inflation[role]:.2f}x")

st.sidebar.markdown("---")
if not st.session_state.confirm_reset:
    if st.sidebar.button("⚠️ Reset Entire Draft"):
        st.session_state.confirm_reset = True
        st.rerun()
else:
    st.sidebar.error("This wipes the entire draft log. Are you sure?")
    rc1, rc2 = st.sidebar.columns(2)
    if rc1.button("Yes, reset", type="primary"):
        st.session_state.draft_log = []
        save_draft_log([])
        st.session_state.confirm_reset = False
        st.rerun()
    if rc2.button("Cancel"):
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

# ----------------- MAIN TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Master Listone & Live Logger", 
    "🎯 Live Player HUD", 
    "🏆 League & Opponent Rosters", 
    "🛡️ Modificatore Sandbox"
])

# ----------------- TAB 1: MASTER LISTONE & LIVE LOGGER -----------------
with tab1:
    st.markdown("### ⚡ Live Call & Quick Logger")
    
    def on_tab1_player_change():
        st.session_state.selected_player = st.session_state.t1_player

    available_players = available_profiles_df.sort_values('Nome')
    t1_options = available_players['Nome'].tolist()
    t1_idx = t1_options.index(st.session_state.selected_player) if st.session_state.selected_player in t1_options else 0

    c_p, c_buyer, c_price, c_btn = st.columns([3, 2, 2, 2])
    with c_p:
        log_player = st.selectbox(
            "Player Called:",
            options=t1_options,
            index=t1_idx,
            key="t1_player",
            on_change=on_tab1_player_change
        )
    with c_buyer:
        log_buyer = st.selectbox("Bought By:", options=MANAGERS, key="t1_buyer")
    with c_price:
        max_bid_possible = my_budget if log_buyer == MY_NAME else STARTING_BUDGET
        log_price = st.number_input("Winning Bid (cr):", min_value=1, max_value=max_bid_possible, value=1, step=1, key="t1_price")
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
            st.warning(f"⚠️ {log_price} cr exceeds your walk-away price of {wa} cr (fair price: {fp} cr) for **{log_player}**.")

    with log_btn_placeholder:
        if over_walkaway:
            confirm_over = st.checkbox("Confirm over walk-away", key="t1_confirm_over")
            disabled = not confirm_over
        else:
            disabled = False
        if st.button("Log Purchase", use_container_width=True, type="primary", disabled=disabled, key="t1_log_btn"):
            log_purchase(log_player, log_buyer, log_price)
            st.success(f"Logged {log_player} to {log_buyer} for {log_price} cr")
            st.rerun()

    if not draft_df.empty:
        with st.expander(f"🕒 Recent Auction History ({len(draft_df)} logged)", expanded=False):
            col_table, col_undo = st.columns([5, 1])
            with col_table:
                st.dataframe(draft_df.iloc[::-1], use_container_width=True, hide_index=True)
            with col_undo:
                if st.button("↩️ Undo Last"):
                    st.session_state.draft_log.pop()
                    save_draft_log(st.session_state.draft_log)
                    st.rerun()

    st.markdown("---")

    # 3. Master Undrafted Listone
    st.markdown("##### 📋 Complete Undrafted Listone")
    
    c_filter1, c_filter2, c_filter3 = st.columns([1, 1, 2])
    with c_filter1:
        role_filter = st.selectbox("Role:", ['All', 'P', 'D', 'C', 'A'], key="t1_role")
    with c_filter2:
        min_presenze = st.number_input("Min Pres. 25/26:", min_value=0, max_value=38, value=0, step=1, key="t1_pres")
    with c_filter3:
        search_query = st.text_input("Quick Search (Name or Team):", "", key="t1_search")
    
    avail_pool = available_profiles_df.copy()
    
    if role_filter != 'All':
        avail_pool = avail_pool[avail_pool['Ruolo'] == role_filter]
        
    if min_presenze > 0 and 'Presenze_Last_Season' in avail_pool.columns:
        avail_pool = avail_pool[avail_pool['Presenze_Last_Season'] >= min_presenze]
        
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
    avail_pool['P_ge_6'] = avail_pool['P_Voto_ge_6']
    avail_pool['P_ge_6_5'] = avail_pool['P_Voto_ge_6_5']
    avail_pool['P_lt_6'] = avail_pool['P_Voto_lt_6'] if 'P_Voto_lt_6' in avail_pool.columns else (1.0 - avail_pool['P_Voto_ge_6'])

    for int_col in ['Presenze_Last_Season', 'Presenze_Tot', 'Tot_Gol', 'Tot_Ass', 'Tot_Gs', 'Tot_Amm', 'Tot_Esp', 'Target_cr', 'Max_cr']:
        if int_col in avail_pool.columns:
            avail_pool[int_col] = avail_pool[int_col].fillna(0).astype(int)

    all_possible_cols = [
        'Nome', 'Squadra', 'Ruolo', 'Target_cr', 'Max_cr', 'Archetype',
        'Presenze_Last_Season', 'Presenze_Tot',
        'FM_W', 'FM_Raw', 'MV_W', 'MV_Raw',
        'P_ge_6', 'P_ge_6_5', 'P_lt_6',
        'Tot_Gol', 'Tot_Ass', 'Tot_Gs', 'Tot_Amm', 'Tot_Esp',
        'Bonus_per_Game', 'Malus_per_Game', 'Clean_Sheet_Rate'
    ]
    
    default_selected = [
        'Nome', 'Squadra', 'Ruolo', 'Target_cr', 'Max_cr',
        'Presenze_Last_Season', 'Presenze_Tot',
        'FM_W', 'FM_Raw', 'MV_W', 'MV_Raw',
        'P_ge_6', 'P_ge_6_5', 'P_lt_6', 
        'Bonus_per_Game', 'Malus_per_Game', 'Clean_Sheet_Rate'
    ]

    selected_columns = st.multiselect(
        "Select Visible Columns:",
        options=[c for c in all_possible_cols if c in avail_pool.columns],
        default=[c for c in default_selected if c in avail_pool.columns],
        key="t1_cols_picker"
    )

    column_configuration = {
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
        height=550,
        column_config=column_configuration
    )

# ----------------- TAB 2: PLAYER HUD -----------------
with tab2:
    def on_tab2_player_change():
        st.session_state.selected_player = st.session_state.t2_player

    all_player_names = profiles_df['Nome'].sort_values().unique().tolist()
    t2_idx = all_player_names.index(st.session_state.selected_player) if st.session_state.selected_player in all_player_names else 0

    col_search, _ = st.columns([2, 1])
    with col_search:
        selected_player = st.selectbox(
            "Search Target Player:",
            options=all_player_names,
            index=t2_idx,
            key="t2_player",
            on_change=on_tab2_player_change
        )

    p_info = profiles_df[profiles_df['Nome'] == selected_player].iloc[0]
    p_matches = matches_df[(matches_df['Cod.'] == p_info['Cod.']) & (matches_df['Voto_Puro'].notna())]
    fair_price, walk_away_price = get_dynamic_prices(p_info)

    is_drafted = selected_player in drafted_names
    if is_drafted:
        bid_info = draft_df[draft_df['player'] == selected_player].iloc[0]
        st.warning(f"⚠️ Drafted by **{bid_info['buyer']}** for **{bid_info['price']} cr**")
    else:
        hud_p, hud_price, hud_btn = st.columns([2, 2, 2])
        with hud_p:
            hud_buyer = st.selectbox("Bought By:", options=MANAGERS, key="t2_buyer")
        with hud_price:
            max_bid_possible = my_budget if hud_buyer == MY_NAME else STARTING_BUDGET
            hud_price_val = st.number_input(
                "Winning Bid (cr):",
                min_value=1,
                max_value=max_bid_possible,
                value=max(1, fair_price) if hud_buyer == MY_NAME else 1,
                step=1,
                key="t2_price",
            )
        hud_over = hud_buyer == MY_NAME and hud_price_val > walk_away_price
        if hud_over:
            st.warning(f"⚠️ Exceeds walk-away ({walk_away_price} cr).")
        with hud_btn:
            st.write("")
            hud_confirm = st.checkbox("Confirm over walk-away", key="t2_confirm") if hud_over else True
            if st.button("Log This Player", use_container_width=True, type="primary", disabled=not hud_confirm, key="t2_log_btn"):
                log_purchase(selected_player, hud_buyer, hud_price_val)
                st.success(f"Logged {selected_player} to {hud_buyer} for {hud_price_val} cr")
                st.rerun()

    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Role & Archetype", f"{p_info['Ruolo']} - {p_info['Archetype']}")
    k2.metric("Weighted Pure Voto", f"{p_info['Media_Voto_Weighted']:.2f}")
    k3.metric("Weighted Fantamedia", f"{p_info['Fanta_Media_Weighted']:.2f}")
    k4.metric("Fair Target Bid", f"{fair_price} cr")
    k5.metric("Walk-Away Max", f"{walk_away_price} cr", delta=f"+{walk_away_price - fair_price}", delta_color="inverse")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("P(Voto ≥ 6.0)", f"{p_info['P_Voto_ge_6'] * 100:.1f}%")
    r2.metric("P(Voto ≥ 6.5)", f"{p_info['P_Voto_ge_6_5'] * 100:.1f}%")
    r3.metric("Total Goals / Assists", f"{int(p_info['Tot_Gol'])} G / {int(p_info['Tot_Ass'])} A")
    r4.metric("Discipline Malus / Game", f"-{p_info['Malus_per_Game']:.2f} pts")

    st.markdown("##### 📈 Pure Grade vs. Fantavoto Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=p_matches['Voto_Puro'], name='Pure Voto (Modifier Floor)', opacity=0.75, xbins=dict(start=3.5, end=9.0, size=0.5)))
    fig.add_trace(go.Histogram(x=p_matches['Fantavoto'], name='Fantavoto (Bonus Upside)', opacity=0.6, xbins=dict(start=3.5, end=15.0, size=0.5)))
    fig.update_layout(barmode='overlay', height=280, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- TAB 3: LEAGUE & OPPONENT ROSTERS -----------------
with tab3:
    st.markdown("### 🏆 League Standing, Liquidity & Opponent Rosters")

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
            "Manager": mgr,
            "Remaining Budget (cr)": rem_budget,
            "Spent (cr)": spent,
            "Slots": f"{filled_slots}/{ROSTER_LIMIT}",
            "Max Bid (cr)": max_bid,
            "Avg cr/Slot": avg_cr_slot,
            "P": f"{p_count}/3",
            "D": f"{d_count}/8",
            "C": f"{c_count}/8",
            "A": f"{a_count}/6",
        })

    summary_df = pd.DataFrame(league_summary)
    st.dataframe(summary_df.sort_values(by="Remaining Budget (cr)", ascending=False), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔍 Opponent Roster Inspector")
    
    selected_mgr = st.selectbox("Select Team to Inspect:", options=MANAGERS, index=1 if len(MANAGERS) > 1 else 0)
    
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
            st.markdown(f"**{selected_mgr} Spending Breakdown by Role**")
            role_spend = mgr_squad.groupby('role')['price'].sum().reset_index()
            fig_pie = px.pie(role_spend, values='price', names='role', hole=0.4, color='role',
                             color_discrete_map={'P': '#1f77b4', 'D': '#2ca02c', 'C': '#ff7f0e', 'A': '#d62728'})
            fig_pie.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info(f"{selected_mgr} has not drafted any players yet.")

# ----------------- TAB 4: MODIFICATORE SANDBOX -----------------
with tab4:
    st.markdown("### 🛡️ Live Defensive Modifier Engine")

    gk_options = profiles_df[profiles_df['Ruolo'] == 'P']['Nome'].sort_values().tolist()
    def_options = profiles_df[profiles_df['Ruolo'] == 'D']['Nome'].sort_values().tolist()

    col_gk, col_d1, col_d2, col_d3, col_d4 = st.columns(5)
    sel_gk = col_gk.selectbox("Portiere (GK)", options=gk_options, index=0)
    sel_d1 = col_d1.selectbox("Defender 1", options=def_options, index=0)
    sel_d2 = col_d2.selectbox("Defender 2", options=def_options, index=1 if len(def_options) > 1 else 0)
    sel_d3 = col_d3.selectbox("Defender 3", options=def_options, index=2 if len(def_options) > 2 else 0)
    sel_d4 = col_d4.selectbox("Defender 4", options=def_options, index=3 if len(def_options) > 3 else 0)

    gk_rows = matches_df[(matches_df['Nome'] == sel_gk) & (matches_df['Voto_Puro'].notna())]
    d1_rows = matches_df[(matches_df['Nome'] == sel_d1) & (matches_df['Voto_Puro'].notna())]
    d2_rows = matches_df[(matches_df['Nome'] == sel_d2) & (matches_df['Voto_Puro'].notna())]
    d3_rows = matches_df[(matches_df['Nome'] == sel_d3) & (matches_df['Voto_Puro'].notna())]
    d4_rows = matches_df[(matches_df['Nome'] == sel_d4) & (matches_df['Voto_Puro'].notna())]

    if all(len(r) > 0 for r in [gk_rows, d1_rows, d2_rows, d3_rows, d4_rows]):
        sim_res = simulate_modifier(gk_rows, [d1_rows, d2_rows, d3_rows, d4_rows])

        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Modifier Bonus / Matchday", f"+{sim_res['expected_bonus']} pts")
        m2.metric("Projected Mean Backline Pure Rating", f"{sim_res['mean_rating']} / 10")
        mode_label = "Joint matchday (correlated)" if sim_res['sampling_mode'] == "joint_matchday" else "Independent (fallback)"
        m3.metric("Sampling Mode", mode_label)

        prob_df = pd.DataFrame({
            'Modifier Tier': [f"+{k}" for k in sim_res['probs'].keys()],
            'Probability (%)': [v * 100 for v in sim_res['probs'].values()]
        })
        fig_prob = px.bar(prob_df, x='Modifier Tier', y='Probability (%)', text_auto='.1f', title="Modifier Tier Probability Breakdown")
        fig_prob.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.warning("Insufficient appearance records for one or more chosen players.")