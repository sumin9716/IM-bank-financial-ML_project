import math

from fx_external_pipeline_full.pricing_option import garman_kohlhagen_call_put

def test_gk_basic_and_parity():
    S, K, T, rd, rf, vol = 1300, 1300, 30 / 365, 0.035, 0.045, 0.12
    c, p = garman_kohlhagen_call_put(S, K, T, rd, rf, vol)
    assert c>0 and p>0
    # Put-Call parity for GK: C - P = S*e^{-rfT} - K*e^{-rdT}
    lhs = c - p
    rhs = S*math.exp(-rf*T) - K*math.exp(-rd*T)
    assert abs(lhs - rhs) < 1e-6
