import math
from scipy.stats import norm

def garman_kohlhagen_call_put(S, K, T, rd, rf, vol):
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return 0.0, 0.0
    d1 = (math.log(S/K) + (rd - rf + 0.5*vol*vol)*T) / (vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    call = S*math.exp(-rf*T)*norm.cdf(d1) - K*math.exp(-rd*T)*norm.cdf(d2)
    put  = K*math.exp(-rd*T)*norm.cdf(-d2) - S*math.exp(-rf*T)*norm.cdf(-d1)
    return float(call), float(put)
