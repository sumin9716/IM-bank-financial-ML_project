import os

import pandas as pd
import requests

from .utils import redact_secret, to_month_end

ECOS_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"


def _ecos_get(stat_code, cycle, start, end, item_code, api_key, lang="json", kr="kr"):
    url = f"{ECOS_BASE}/{api_key}/{lang}/{kr}/1/10000/{stat_code}/{cycle}/{start}/{end}/{item_code}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        # 키가 요청 URL 경로에 들어가 예외 메시지와 경고 출력에 그대로 남으므로 가린다
        raise redact_secret(e, api_key) from None
    rows = (r.json().get("StatisticSearch") or {}).get("row") or []
    if not rows:
        return pd.DataFrame(columns=["date","value"])
    df = pd.DataFrame(rows)
    time_col = "TIME" if "TIME" in df.columns else next((c for c in df.columns if c.lower()=="time"), None)
    val_col = "DATA_VALUE" if "DATA_VALUE" in df.columns else next((c for c in df.columns if c.lower()=="data_value"), None)
    # 월 주기 TIME(예: 202401)에서 숫자만 남겨 앞 6자리(연월)의 1일로 바꾼다
    df["date"] = pd.to_datetime(df[time_col].astype(str).str.replace(r"[^0-9]","", regex=True).apply(lambda t: t[:6] + "01"))
    df["value"] = pd.to_numeric(df[val_col].replace({"": None, ".": None}), errors="coerce")
    return df[["date","value"]].dropna()


def load_ecos_series(stat_code, cycle, start, end, item_code=""):
    """한국은행 ECOS 통계를 월말 인덱스 시계열로 읽는다(단위는 원자료 그대로)."""
    api_key = os.getenv("ECOS_API_KEY", "")
    if not api_key:
        raise EnvironmentError("ECOS_API_KEY is not set")
    df = _ecos_get(stat_code, cycle, start, end, item_code, api_key)
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["value"].sort_index()
    return to_month_end(s).groupby(level=0).last()
