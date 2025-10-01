
# External Data Loaders (FRED / ECOS / OpenHolidays)

## Quick Start
export FRED_API_KEY=YOUR_FRED_KEY
# optional for KR rates / ECOS-based USDKRW:
export ECOS_API_KEY=YOUR_ECOS_KEY

python - <<'PY'
from src.data_loaders import load_us_rates, load_usdkrw_spot, load_holidays
print('Try calling load_us_rates(["SOFR","DGS2","DGS10"], start="2020-01-01")')
PY

## Functions
- load_us_rates(tenors=["SOFR","DGS1","DGS2","DGS10"], start="2000-01-01")
- load_usdkrw_spot(source="FRED", start="2000-01-01")  # FRED DEXKOUS
- load_kr_rates(["CD91","CALL_OVN"], start="2000-01-01")  # fill ECOS codes first
- load_holidays(country="KR", years=[2024,2025])  # OpenHolidays

## Configs
- src/configs/fred_series.yml
- src/configs/ecos_series.yml  # fill real ECOS codes

## Caching
- .cache/ namespace folders; override with DATA_LOADER_CACHE_DIR


---
## Step 2: CIP Forward Fair Value (USD/KRW)

**Module**: `src/pricing/forwards.py`  
**Key API**:
```python
from src.pricing.forwards import price_forward_cip, CIPConfig

cfg = CIPConfig(method="cont", dcc_dom="act/365", dcc_for="act/360")
out = price_forward_cip(spot_df, kr_rate_df, us_rate_df,
                        tenor="3M", cfg=cfg,
                        kr_rate_series="CD91", us_rate_series="SOFR")
```
- `spot_df`: DataFrame with columns `date`, `value` (KRW per USD)
- `*_rate_df`: DataFrame with `date`, `value`, optional `series` (pick via `*_rate_series`)
- `method`: `"cont"` → `F = S * exp((r_dom*T_dom) - (r_for*T_for))`  
             `"simple"` → `F = S * (1 + r_dom*T_dom) / (1 + r_for*T_for)`
- `tenor`: e.g., `"1M"`, `"3M"`, `"6M"`, `"1Y"`
- `biz_conv`: following/preceding/modified_following (holiday-aware if you pass a holidays df)

**Demo**: `examples/cip_demo.py` (uses FRED-only placeholders).


---
## Step 3: Options PoC (Garman–Kohlhagen)

**Module**: `src/pricing/options.py`  
**Key APIs**:
```python
from src.pricing.options import GKInputs, garman_kohlhagen, realized_vol_from_spot, implied_vol

x = GKInputs(S, K, T, sigma, r_dom, r_for, opt_type="call")
res = garman_kohlhagen(x)
# res: price, d1, d2, delta, gamma, vega, theta, rho_dom, rho_for

rv = realized_vol_from_spot(spot_df, window=63)
iv = implied_vol(target_price, x)  # solve σ from observed price
```
**Demo**: `examples/options_demo.py`  
- DEXKOUS(USD/KRW), DGS1 임시 금리 → 3M ATM-Forward 옵션가/그릭스 산출
- 실사용에선 KRW 금리를 ECOS 콜/CM/CD 등으로 교체


---
## Step 4: Risk Metrics — Historical VaR/ES

**Module**: `src/risk/metrics.py`  
**Key APIs**:
```python
from src.risk.metrics import hist_var_es, worst_events, rolling_var_series

res = hist_var_es(spot_df, horizon_days=1, alpha=0.99, notional_usd=1.0)
print(res["VaR"], res["ES"])           # KRW loss units for long USD 1
top = worst_events(res["loss_series"], top_n=10)
roll = rolling_var_series(spot_df, window=252, horizon_days=1, alpha=0.99)
```
- VaR/ES는 **손실(KRW, +)** 기준 (long USD)  
- PnL 정의: `ΔV = N * S_base * (exp(r_h) - 1)` (log 수익률 기준)  
- `horizon_days>1`이면 누적(log 합)로 집계

**Demo**: `examples/var_demo.py`


---
## Step 5: NDF PnL Mini-Simulator

**Module**: `src/ndf/simulator.py`  
**Core APIs**
```python
from src.ndf.simulator import simulate_ndf_pnl, make_trades_from_fair_values

# Build trades from CIP fair forwards (tenor='1M'/'3M'/...)
trades = make_trades_from_fair_values(fair_df, tenor="3M", notional_usd=1_000_000.0, direction="long_usd")

# Compute KRW PnL with fixing from spot series
pnl = simulate_ndf_pnl(trades, spot_df, fixing_pick="preceding")
```
- 서명 규약:
  - `direction=="long_usd"`  → PnL = `+ N * (fix - fwd)`  
  - `direction=="short_usd"` → PnL = `- N * (fix - fwd)`  
- `fixing_pick`: 지정일에 값이 없을 때 `preceding`(기본) 또는 `following` 선택  

**Demo**: `examples/ndf_demo.py`  
- DEXKOUS(USD/KRW) + DGS1(임시 금리)로 3M 공정가→거래 생성→정산 PnL 계산


---
## Step 6: Rule-based Hedge Ratio (0/50/80)

**Module**: `src/hedge/rules.py`  
**Key APIs**
```python
from src.hedge.rules import compute_market_features, recommend_hedge_ratio, RuleThresholds

feats = compute_market_features(spot_df, kr_rate_df, us_rate_df, kr_series="CD91", us_series="SOFR")
reco  = recommend_hedge_ratio(feats, risk="standard")
```

**Feature set (v0, 시장데이터 기반)**
- `rv_20d`, `rv_60d` (연환산 실현변동성)
- `trend_up` (MA20 > MA60)
- `carry` = (r_dom - r_for) in decimal

**규칙 요약**
1) 변동성 기반 기본 스텝:  
   high → 80%, medium → 50%, low → 0%  
2) 캐리 조정:  
   carry ≥ +1% → +1 스텝, carry ≤ -1% → -1 스텝  
3) 위험성향:  
   conservative: +1, aggressive: -1 (0/50/80 클램핑)

**Demo**: `examples/hedge_demo.py`
