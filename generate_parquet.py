import os
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

ACTIVE_QUOTAZIONI_FILE = os.path.join(RAW_DIR, "Quotazioni_Fantacalcio_Stagione_2026_27.xlsx")

# Historical full season dumps location
HISTORY_DIR = "/Users/matteoientile/Desktop/Extra/Code/Fantacalcio/database_fantacalcio_voti_perGame"
SEASON_FILES = {
    "2023-24": os.path.join(HISTORY_DIR, "season24_23_full.xlsx"),
    "2024-25": os.path.join(HISTORY_DIR, "season25_24_full.xlsx"),
    "2025-26": os.path.join(HISTORY_DIR, "season25_26_full.xlsx")
}

SEASON_WEIGHTS = {
    "2023-24": 0.10,  # 2 seasons ago (10%)
    "2024-25": 0.30,  # 1 season ago (30%)
    "2025-26": 0.60   # Most recent season (60%)
}

def load_active_listone(filepath: str) -> pd.DataFrame:
    """Reads the active Serie A roster file and standardizes columns."""
    excel_file = pd.ExcelFile(filepath)
    sheet_name = 'Tutti' if 'Tutti' in excel_file.sheet_names else excel_file.sheet_names[0]
    
    # Auto-detect header row (row 0 or row 1)
    df_preview = pd.read_excel(filepath, sheet_name=sheet_name, nrows=3)
    header_row = 1 if 'Id' not in df_preview.columns and 'Cod.' not in df_preview.columns else 0
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=header_row)
    
    rename_map = {'Id': 'Cod.', 'R': 'Ruolo', 'Squadra': 'Squadra', 'Qt. A': 'Quotazione'}
    df = df.rename(columns=rename_map)
    
    cols = [c for c in ['Cod.', 'Nome', 'Ruolo', 'Squadra'] if c in df.columns]
    active_df = df[cols].dropna(subset=['Cod.', 'Nome']).drop_duplicates(subset=['Cod.'])
    active_df['Cod.'] = active_df['Cod.'].astype(int)
    return active_df

def main():
    # 1. Load Active Serie A Players (Ground Truth)
    print(f"Reading active listone: {ACTIVE_QUOTAZIONI_FILE}...")
    active_df = load_active_listone(ACTIVE_QUOTAZIONI_FILE)
    active_codes = set(active_df['Cod.'])
    print(f" Active players found: {len(active_df)}")

    # 2. Load historical matches ONLY for currently active players
    all_matches = []
    for season_name, filepath in SEASON_FILES.items():
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: {filepath} not found, skipping.")
            continue
            
        print(f"Processing history from {season_name}...")
        df = pd.read_excel(filepath, sheet_name='Per_Game_Data')
        df['Stagione'] = season_name
        df['Season_Weight'] = SEASON_WEIGHTS[season_name]
        
        if 'Rf' in df.columns and 'Gf' in df.columns:
            df['Tot_Gol'] = df['Gf'].fillna(0) + df['Rf'].fillna(0)
        elif 'Gf' in df.columns:
            df['Tot_Gol'] = df['Gf'].fillna(0)
        else:
            df['Tot_Gol'] = 0.0
            
        # Keep only match events for players currently active in Serie A
        df_active = df[df['Cod.'].isin(active_codes)].copy()
        all_matches.append(df_active)
        
    master_matches = pd.concat(all_matches, ignore_index=True)

    # 3. Calculate multi-year stats
    valid_votes = master_matches.dropna(subset=['Voto_Puro']).copy()
    
    # groupby aggregation:
    profiles = valid_votes.groupby('Cod.').apply(lambda g: pd.Series({
        'Presenze_Tot': len(g),
        'Presenze_Last_Season': len(g[g['Stagione'] == '2025-26']),
        'Media_Voto_Weighted': np.average(g['Voto_Puro'], weights=g['Season_Weight']),
        'Fanta_Media_Weighted': np.average(g['Fantavoto'], weights=g['Season_Weight']),
        'Media_Voto_Raw': float(g['Voto_Puro'].mean()),
        'Fanta_Media_Raw': float(g['Fantavoto'].mean()),
        'Std_Voto': float(g['Voto_Puro'].std()) if len(g) > 1 else 0.0,
        'P_Voto_ge_6': float((g['Voto_Puro'] >= 6.0).mean()),
        'P_Voto_ge_6_5': float((g['Voto_Puro'] >= 6.5).mean()),
        'P_Voto_lt_6': float((g['Voto_Puro'] < 6.0).mean()),  
        'Tot_Gol': float(g['Tot_Gol'].sum()),
        'Tot_Ass': float(g['Ass'].fillna(0).sum()),
        'Tot_Gs': float(g['Gs'].fillna(0).sum()),
        'Tot_Amm': float(g['Amm'].fillna(0).sum()),
        'Tot_Esp': float(g['Esp'].fillna(0).sum()),
        'Bonus_per_Game': float((g['Tot_Gol'] * 3.0 + g['Ass'].fillna(0) * 1.0).mean()),
        'Malus_per_Game': float((g['Amm'].fillna(0) * 0.5 + g['Esp'].fillna(0) * 1.0).mean()),
        'Clean_Sheet_Rate': float((g['Gs'].fillna(0) == 0).mean()) if (g['Ruolo'].iloc[0] == 'P') else np.nan,
    }), include_groups=False).reset_index()

    # 4. Left join to active list (ensures rookies/transfers with 0 past games are included)
    master_profiles = pd.merge(active_df, profiles, on='Cod.', how='left')
    
    # Fill defaults for rookies
    master_profiles['Presenze_Tot'] = master_profiles['Presenze_Tot'].fillna(0).astype(int)
    master_profiles['Presenze_Last_Season'] = master_profiles['Presenze_Last_Season'].fillna(0).astype(int)
    master_profiles['Media_Voto_Weighted'] = master_profiles['Media_Voto_Weighted'].fillna(5.85)
    master_profiles['Fanta_Media_Weighted'] = master_profiles['Fanta_Media_Weighted'].fillna(5.90)
    master_profiles['Media_Voto_Raw'] = master_profiles['Media_Voto_Raw'].fillna(5.85)
    master_profiles['Fanta_Media_Raw'] = master_profiles['Fanta_Media_Raw'].fillna(5.90)
    master_profiles['Std_Voto'] = master_profiles['Std_Voto'].fillna(0.0)
    master_profiles['P_Voto_ge_6'] = master_profiles['P_Voto_ge_6'].fillna(0.45)
    master_profiles['P_Voto_ge_6_5'] = master_profiles['P_Voto_ge_6_5'].fillna(0.15)
    master_profiles['P_Voto_lt_6'] = master_profiles['P_Voto_lt_6'].fillna(0.55)
    master_profiles['Tot_Gol'] = master_profiles['Tot_Gol'].fillna(0)
    master_profiles['Tot_Ass'] = master_profiles['Tot_Ass'].fillna(0)
    master_profiles['Tot_Gs'] = master_profiles['Tot_Gs'].fillna(0)
    master_profiles['Tot_Amm'] = master_profiles['Tot_Amm'].fillna(0)
    master_profiles['Tot_Esp'] = master_profiles['Tot_Esp'].fillna(0)
    master_profiles['Bonus_per_Game'] = master_profiles['Bonus_per_Game'].fillna(0.0)
    master_profiles['Malus_per_Game'] = master_profiles['Malus_per_Game'].fillna(0.0)

    # 5. Assign Archetypes
    def assign_archetype(row):
        role = row['Ruolo']
        if role == 'P': return 'Goalkeeper'
        if role == 'D':
            if row['P_Voto_ge_6'] >= 0.70 and row['Media_Voto_Weighted'] >= 6.05:
                return 'Modificatore Anchor'
            elif row['Bonus_per_Game'] >= 0.35:
                return 'Offensive Fullback'
            return 'Standard Defender'
        if role == 'C':
            if row['Bonus_per_Game'] >= 0.50: return 'Bonus Midfielder'
            elif row['P_Voto_ge_6'] >= 0.70: return 'Midfield Stabilizer'
            return 'Rotation Midfielder'
        if role == 'A':
            if row['Bonus_per_Game'] >= 1.0: return 'Top Striker'
            elif row['Bonus_per_Game'] >= 0.50: return 'Starting Forward'
            return 'Rotation Forward'
        return 'Rookie / Unclassified'

    master_profiles['Archetype'] = master_profiles.apply(assign_archetype, axis=1)

    # 6. Export to processed Parquet files
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    matches_parquet = os.path.join(PROCESSED_DIR, "master_matches.parquet")
    profiles_parquet = os.path.join(PROCESSED_DIR, "master_player_profiles.parquet")
    
    master_matches.to_parquet(matches_parquet, engine='fastparquet', index=False)
    master_profiles.to_parquet(profiles_parquet, engine='fastparquet', index=False)
    
    print(f"\nColumns written to profiles: {list(master_profiles.columns)}")
    print(f"✅ Saved: {profiles_parquet} ({len(master_profiles)} active Serie A players)")

if __name__ == "__main__":
    main()