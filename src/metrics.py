import numpy as np
import pandas as pd

ROSTER_SLOTS = {
    'P': 3,
    'D': 8,
    'C': 8,
    'A': 6
}
ROSTER_LIMIT = sum(ROSTER_SLOTS.values())  # 25

# Replacement rank cutoffs for a 12-team league
REPLACEMENT_RANKS = {
    'P': 36,
    'D': 96,
    'C': 96,
    'A': 72
}

# How much weight the small-sample shrinkage prior gets, and where it pulls toward.
# NOTE: this is a linear regularizer toward (role_mean - SHRINK_PENALTY), not a formal
# Bayesian posterior (no variance/precision model behind it). Treat SHRINK_PENALTY as a
# hand-tuned heuristic constant, not a derived quantity. Worth re-checking sensitivity
# if rankings near the replacement threshold look off.
SHRINK_PENALTY = 0.2
SHRINK_SAMPLE_SIZE = 25  # presenze below this get shrunk toward the (pessimistic) role mean


def calculate_role_vorp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes empirical VORP per player with shrinkage against the replacement level.
    Operates on whatever slice of `df` is passed in — call it on the FULL player pool
    at load time, and again on the UNDRAFTED pool during the live auction if you want
    VORP shares that reflect the shrinking talent pool (see compute role_pool_sum below).
    """
    df = df.copy()
    df['VORP'] = 0.0

    for role, group in df.groupby('Ruolo'):
        rep_rank = REPLACEMENT_RANKS.get(role, 50)

        # Bayesian-flavored shrinkage on small sample sizes (< SHRINK_SAMPLE_SIZE appearances)
        presenze = group['Presenze_Tot'].fillna(1).values
        prior_weight = np.clip((SHRINK_SAMPLE_SIZE - presenze) / SHRINK_SAMPLE_SIZE, 0, 1)

        if role == 'D':
            raw_metric = (
                group['Media_Voto_Weighted'].fillna(0.0) * 0.7
                + group['Fanta_Media_Weighted'].fillna(0.0) * 0.3
            )
        elif role == 'P':
            cs = group['Clean_Sheet_Rate'].fillna(0.0)
            raw_metric = group['Media_Voto_Weighted'].fillna(0.0) + cs * 0.5
        else:  # C and A
            raw_metric = group['Fanta_Media_Weighted'].fillna(0.0)

        # Guard against an all-NaN role slice producing a NaN role_mean, which would
        # silently poison every shrunk value (and therefore the sort/threshold below).
        role_mean = np.nanmean(raw_metric) if raw_metric.notna().any() else 0.0
        shrunk_metric = (1 - prior_weight) * raw_metric + prior_weight * (role_mean - SHRINK_PENALTY)
        shrunk_metric = shrunk_metric.fillna(0.0)

        sorted_vals = np.sort(shrunk_metric.values)[::-1]
        rep_thresh = sorted_vals[min(len(sorted_vals) - 1, rep_rank)] if len(sorted_vals) > 0 else 0.0

        vorp = np.maximum(0.0, shrunk_metric - rep_thresh)
        if role == 'A':
            vorp = vorp ** 1.2  # Convex scaling for elite strikers

        df.loc[group.index, 'VORP'] = vorp

    return df


def compute_role_pool_sum(pool_df: pd.DataFrame) -> dict:
    pool_sum = {}
    for role, group in pool_df.groupby('Ruolo'):
        quota = REPLACEMENT_RANKS.get(role, 48)
        top_vorps = group['VORP'].sort_values(ascending=False).head(quota)
        pool_sum[role] = max(1.0, float(top_vorps.sum()))
    return pool_sum


def compute_role_inflation(draft_df: pd.DataFrame, role_budget_pct: dict, total_league_credits: int) -> dict:
    """
    Per-role inflation index instead of one global scalar. Approximates each role's
    "expected total pool" as total_league_credits * role_budget_pct[role], under the
    simplifying assumption that other teams allocate similarly to your own sliders.
    That assumption is a real approximation, not a measured fact about opponents'
    budgets — treat this as a directional signal, not a precise multiplier.
    """
    inflation = {}
    if draft_df is None or draft_df.empty:
        spent_by_role = {}
    else:
        spent_by_role = draft_df.groupby('role')['price'].sum().to_dict()

    for role in ['P', 'D', 'C', 'A']:
        role_total_pool = total_league_credits * role_budget_pct.get(role, 0.25)
        spent = spent_by_role.get(role, 0)
        remaining = max(0.0, role_total_pool - spent)
        inflation[role] = max(0.2, remaining / role_total_pool) if role_total_pool > 0 else 1.0
    return inflation


def compute_dynamic_target(
    row: pd.Series,
    role_budget_pct: dict,
    total_budget: int,
    role_inflation: dict,
    my_remaining_budget: int,
    role_pool_sum: dict,
    my_role_slots_filled: dict,
    my_total_slots_filled: int,
) -> tuple:
    """
    Dynamically maps VORP into credit valuations based on the custom budget distribution.

    Key differences from the original version:
    - `role_inflation` is a per-role dict, not one global scalar.
    - `role_pool_sum` should be computed on the UNDRAFTED pool (see compute_role_pool_sum),
      so the player's VORP share reflects the current state of the auction, not draft day 1.
    - The budget reserve for "slots you still need to fill" now uses your ACTUAL remaining
      roster slots (ROSTER_LIMIT - my_total_slots_filled - 1 for this pick), not a hardcoded
      constant. This matters a lot late in the draft: a static reserve makes the tool tell you
      to underbid in the final rounds, exactly when you should have the most room to spend.
    - A scarcity multiplier collapses the price toward the floor once you've already filled
      a role's slots — the model previously had no concept of "I don't need another one of these."
    """
    role = row['Ruolo']
    vorp = row.get('VORP', 0.0)

    pct = role_budget_pct.get(role, 0.25)
    role_slots_total = ROSTER_SLOTS.get(role, 8)
    role_slots_filled = my_role_slots_filled.get(role, 0)
    role_slots_open = max(0, role_slots_total - role_slots_filled)

    role_budget = total_budget * pct
    surplus_pool = max(0.0, role_budget - role_slots_total)

    role_sum_vorp = role_pool_sum.get(role, 1.0)
    if role_sum_vorp <= 0 or vorp <= 0:
        base_price = 1.0
    else:
        share = vorp / (role_sum_vorp / 12.0)
        base_price = 1.0 + (share * surplus_pool)

    liquidity_ratio = min(1.0, my_remaining_budget / float(total_budget))
    inflation_index = role_inflation.get(role, 1.0)
    adj_price = base_price * inflation_index * (0.6 + 0.4 * liquidity_ratio)

    # Scarcity collapse: if you've already filled this role, the marginal value of
    # another one is near zero even if the player's raw VORP is high. Ramps down
    # smoothly rather than hard-cutting at exactly "slots full", since teams sometimes
    # upgrade a bench-tier player late.
    if role_slots_total > 0:
        scarcity_mult = max(0.05, role_slots_open / role_slots_total)
    else:
        scarcity_mult = 0.05
    adj_price *= scarcity_mult

    max_player_cap = role_budget * 0.75
    adj_price = min(adj_price, max_player_cap)

    # Dynamic reserve: leave enough credits for every roster slot you still need to
    # fill AFTER this pick (1 credit floor per remaining slot is the classic fantacalcio
    # minimum-bid convention). Replaces the old hardcoded "-24".
    slots_remaining_after_this_pick = max(0, ROSTER_LIMIT - my_total_slots_filled - 1)
    spendable_ceiling = max(1, my_remaining_budget - slots_remaining_after_this_pick)

    fair_price = int(np.clip(adj_price, 1, spendable_ceiling))
    walk_away = int(min(fair_price * 1.15 + 1, spendable_ceiling, max_player_cap * 1.2))
    walk_away = max(fair_price, walk_away)

    return fair_price, walk_away