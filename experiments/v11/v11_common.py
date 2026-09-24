"""Shared definitions for the v11 residual stacker (arms, parameters, nested target encodings)."""
import numpy as np

BASIC = ['apt', 'off_aobt3', 'off_sched', 'hour', 'aobt_missing']
CLOCKS = ['nm_delay', 'off_iobt', 'lobt_m_iobt', 'eobt_m_lobt', 'a3_m_eobt', 'a3_m_lobt', 'eobt_m_sched', 'iobt_m_sched']
STAMP = ['pa_stamped', 'pa_inb_m_sibt', 'pa_ldg_m_sibt']
IDENT = ['ades']
TE = ['te_op_st', 'te_stand_st', 'te_ades_r']          # nested / out-of-month encodings, built per fit
BACK = ['si_before_sched', 'si_same_typ', 'sd_prev_to']
LOOK = ['sl_to', 'sl_after_sched', 'sl_same_typ', 'sl_same_op', 'sn_minus_to', 'sn_same_op', 'sn_same_typ', 'sa_after_to', 'sd_next_to']
CORE = BASIC + CLOCKS + STAMP + IDENT
ARMS = {
    'null': (BASIC, []),                     # diagnostic: the stack with inputs the base already has
    'a_pb': (CORE + ['pb'], TE),             # diagnostic: rv_stack2 'core' exactly (base prediction as an input)
    'a': (CORE, TE),                          # (a) core, base-independent -> can be added to v10 on ranking
    'b': (CORE + BACK, TE),                   # (b) core + turnover BACKWARD
    'c': (CORE + BACK + LOOK, TE),            # (c) core + turnover FULL (backward + look-ahead)
}
CATS = ['apt', 'ades']
W_TE = 20.0
CLIP = 3000.0
ROUNDS = 200
PARAMS = dict(objective='regression', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=4, seed=0, deterministic=True,
              force_row_wise=True, cat_smooth=50, max_cat_to_onehot=4)


def key_month_sums(codes, month, mask, val, nkeys):
    S = np.zeros((nkeys, 13)); N = np.zeros((nkeys, 13))
    np.add.at(S, (codes[mask], month[mask]), val[mask]); np.add.at(N, (codes[mask], month[mask]), 1)
    return S, N


def te_oof(S, N, codes, month, m, prior):
    """Fold m: rows of month m exclude month m; rows of month k != m exclude k and m."""
    s = S[codes].sum(1) - S[codes, m]; n = N[codes].sum(1) - N[codes, m]
    other = month != m
    s = s - np.where(other, S[codes, month], 0); n = n - np.where(other, N[codes, month], 0)
    return (s + W_TE * prior) / (n + W_TE)


def te_final_train(S, N, codes, month, prior):
    """Final fit on all 12 months: each training row excludes its own month."""
    s = S[codes].sum(1) - S[codes, month]; n = N[codes].sum(1) - N[codes, month]
    return (s + W_TE * prior) / (n + W_TE)


def te_final_apply(S, N, codes, prior):
    """Ranking rows: encoded from all 12 training months."""
    s = S[codes].sum(1); n = N[codes].sum(1)
    return (s + W_TE * prior) / (n + W_TE)
