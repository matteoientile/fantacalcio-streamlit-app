import numpy as np
import pandas as pd

MIN_JOINT_ROWS = 5  # Below this, fall back to independent bootstrap

def _prepare_match_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a unique identifier for every single matchday across multi-year data
    to prevent false cross-season cartesian merges.
    """
    df = df.copy()
    if 'Stagione' in df.columns and 'Giornata' in df.columns:
        df['Match_ID'] = df['Stagione'].astype(str) + "_" + df['Giornata'].astype(str)
    elif 'Giornata' in df.columns:
        df['Match_ID'] = df['Giornata'].astype(str)
    else:
        df['Match_ID'] = np.arange(len(df))
    return df

def _independent_bootstrap(gk_votes: np.ndarray, def_votes_list: list, n_sims: int) -> tuple:
    gk_samples = np.random.choice(gk_votes, size=n_sims, replace=True)
    def_samples = np.column_stack([
        np.random.choice(d, size=n_sims, replace=True) for d in def_votes_list
    ])
    return gk_samples, def_samples

def simulate_modifier(
    gk_df: pd.DataFrame,
    def_dfs: list[pd.DataFrame],
    n_sims: int = 10000,
) -> dict:
    """
    Simulates the weekly Modificatore Difesa bonus:
    - Formula: (V_P + Top 3 of 4 defenders) / 4
    - Thresholds:
        < 6.00       -> 0
        [6.00, 6.25) -> +1
        [6.25, 6.50) -> +2
        [6.50, 6.75) -> +3
        [6.75, 7.00) -> +4
        >= 7.00      -> +6
    """
    if len(def_dfs) < 4 or gk_df is None or len(gk_df) == 0:
        return {
            "expected_bonus": 0.0,
            "mean_rating": 0.0,
            "probs": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 6: 0.0},
            "sampling_mode": "none",
        }

    gk_prepared = _prepare_match_key(gk_df)
    def_prepared = [_prepare_match_key(d) for d in def_dfs]

    gk_votes = gk_prepared['Voto_Puro'].dropna().values
    def_votes_list = [d['Voto_Puro'].dropna().values for d in def_prepared]

    if any(len(v) == 0 for v in def_votes_list) or len(gk_votes) == 0:
        return {
            "expected_bonus": 0.0,
            "mean_rating": 0.0,
            "probs": {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 6: 0.0},
            "sampling_mode": "none",
        }

    sampling_mode = "independent"
    gk_samples = None
    def_samples = None

    # Joint matchday merge using unique Season + Matchday key
    merged = gk_prepared[['Match_ID', 'Voto_Puro']].dropna().rename(columns={'Voto_Puro': 'gk'})
    for i, d in enumerate(def_prepared):
        leg = d[['Match_ID', 'Voto_Puro']].dropna().rename(columns={'Voto_Puro': f'd{i}'})
        merged = merged.merge(leg, on='Match_ID', how='inner')

    if len(merged) >= MIN_JOINT_ROWS:
        idx = np.random.choice(len(merged), size=n_sims, replace=True)
        joint_rows = merged.iloc[idx]
        gk_samples = joint_rows['gk'].values
        def_samples = joint_rows[[f'd{i}' for i in range(4)]].values
        sampling_mode = "joint_matchday"
    else:
        # Fallback to independent bootstrap if joint history is sparse
        gk_samples, def_samples = _independent_bootstrap(gk_votes, def_votes_list, n_sims)

    # Sort defenders row-wise to isolate top 3 (drop lowest at index 0)
    sorted_defs = np.sort(def_samples, axis=1)
    top_3_defs = sorted_defs[:, 1:]

    # Average of GK + top 3 defenders
    def_means = (gk_samples + np.sum(top_3_defs, axis=1)) / 4.0

    bonuses = np.zeros(n_sims)
    bonuses[(def_means >= 6.00) & (def_means < 6.25)] = 1
    bonuses[(def_means >= 6.25) & (def_means < 6.50)] = 2
    bonuses[(def_means >= 6.50) & (def_means < 6.75)] = 3
    bonuses[(def_means >= 6.75) & (def_means < 7.00)] = 4
    bonuses[def_means >= 7.00] = 6

    expected_bonus = float(np.mean(bonuses))
    probs = {
        0: float(np.mean(bonuses == 0)),
        1: float(np.mean(bonuses == 1)),
        2: float(np.mean(bonuses == 2)),
        3: float(np.mean(bonuses == 3)),
        4: float(np.mean(bonuses == 4)),
        6: float(np.mean(bonuses == 6)),
    }

    return {
        "expected_bonus": round(expected_bonus, 2),
        "mean_rating": round(float(np.mean(def_means)), 2),
        "probs": probs,
        "sampling_mode": sampling_mode,
    }