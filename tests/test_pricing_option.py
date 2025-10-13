import sys
from pathlib import Path
# add project src to path (../src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fx_external_pipeline_full.pricing_option import garman_kohlhagen_call_put
import math

def test_gk_basic_and_parity():
    S=1300; K=1300; T=30/365; rd=0.035; rf=0.045; vol=0.12
    c,p = garman_kohlhagen_call_put(S,K,T,rd,rf,vol)
    assert c>0 and p>0
    # Put-Call parity for GK: C - P = S*e^{-rfT} - K*e^{-rdT}
    lhs = c - p
    rhs = S*math.exp(-rf*T) - K*math.exp(-rd*T)
    assert abs(lhs - rhs) < 1e-6
