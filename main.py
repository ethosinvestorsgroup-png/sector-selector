import os
import csv
import io
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

import httpx
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_NAME = "Sector Selector — by Marc-Anthony Richardson, MBA, CPM"

# Massive (Polygon rebrand) base
MASSIVE_BASE = "https://api.massive.com"
MASSIVE_KEY = os.getenv("MASSIVE_API_KEY", "").strip()

app = FastAPI(title=APP_NAME)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def _require_key() -> Optional[str]:
    if not MASSIVE_KEY:
        return "Missing API key. Set environment variable MASSIVE_API_KEY before running."
    return None


async def massive_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Massive supports apiKey in query string.
    Docs: https://massive.com/docs/rest/quickstart
    """
    params = dict(params or {})
    params["apiKey"] = MASSIVE_KEY
    url = f"{MASSIVE_BASE}{path}"
    async with httpx.AsyncClient(timeout=40) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "app_name": APP_NAME})


@app.get("/api/health")
async def health():
    err = _require_key()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    return {
        "ok": True,
        "provider": "Massive.com",
        "auth": "apiKey query param or Authorization header supported by Massive",
        "base": MASSIVE_BASE,
    }


@app.get("/api/sectors")
async def sectors():
    # Keep your 11-sector UI list (we can refine classification later)
    return {
        "sectors": [
            "Energy","Materials","Industrials","Consumer Discretionary","Consumer Staples",
            "Health Care","Financials","Information Technology","Communication Services",
            "Utilities","Real Estate",
        ]
    }


def _approx_sector_from_sic(sic_desc: str) -> str:
    """
    Massive reference endpoints often return SIC descriptions.
    This is a lightweight heuristic mapping so your UI can still show a 'sector'.
    """
    s = (sic_desc or "").lower()
    if any(k in s for k in ["oil", "gas", "petroleum", "drilling", "pipeline", "energy"]):
        return "Energy"
    if any(k in s for k in ["bank", "insurance", "broker", "lending", "financial"]):
        return "Financials"
    if any(k in s for k in ["software", "semiconductor", "computer", "technology", "it "]):
        return "Information Technology"
    if any(k in s for k in ["telecom", "media", "broadcast", "interactive", "communications"]):
        return "Communication Services"
    if any(k in s for k in ["pharma", "medical", "hospital", "biotech", "health"]):
        return "Health Care"
    if any(k in s for k in ["retail", "apparel", "consumer", "restaurant", "auto", "travel"]):
        return "Consumer Discretionary"
    if any(k in s for k in ["food", "beverage", "household", "tobacco", "staples"]):
        return "Consumer Staples"
    if any(k in s for k in ["utility", "electric", "water", "gas utility"]):
        return "Utilities"
    if any(k in s for k in ["reit", "real estate", "property", "mortgage"]):
        return "Real Estate"
    if any(k in s for k in ["chemical", "steel", "mining", "materials", "paper", "lumber"]):
        return "Materials"
    return "Industrials"


async def _aggs_daily(symbol: str, days: int) -> List[Tuple[str, float]]:
    """
    Fetch daily close prices for the past N calendar days and return list of (date, close).
    Uses v2 aggs endpoint.
    """
    end = datetime.now(timezone.utc).date()
    start = (datetime.now(timezone.utc) - timedelta(days=int(days * 1.6))).date()  # buffer for weekends/holidays
    path = f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    js = await massive_get(path, {"adjusted": "true", "sort": "asc", "limit": 50000})
    results = js.get("results") or []
    out = []
    for r in results[-days:]:
        # t is ms epoch
        ts = r.get("t")
        c = r.get("c")
        if ts is None or c is None:
            continue
        d = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).date().isoformat()
        out.append((d, float(c)))
    return out


def _beta_from_prices(prices: List[Tuple[str, float]], spy_prices: List[Tuple[str, float]]) -> Optional[float]:
    """
    Compute beta from aligned daily returns.
    """
    if len(prices) < 40 or len(spy_prices) < 40:
        return None

    p_map = {d: c for d, c in prices}
    s_map = {d: c for d, c in spy_prices}
    dates = sorted(set(p_map.keys()) & set(s_map.keys()))
    if len(dates) < 40:
        return None

    # daily returns
    pr = []
    sr = []
    for i in range(1, len(dates)):
        d0, d1 = dates[i - 1], dates[i]
        p0, p1 = p_map[d0], p_map[d1]
        s0, s1 = s_map[d0], s_map[d1]
        if p0 <= 0 or s0 <= 0:
            continue
        pr.append((p1 / p0) - 1.0)
        sr.append((s1 / s0) - 1.0)

    n = min(len(pr), len(sr))
    if n < 30:
        return None
    pr = pr[-n:]
    sr = sr[-n:]

    mean_p = sum(pr) / n
    mean_s = sum(sr) / n
    cov = sum((pr[i] - mean_p) * (sr[i] - mean_s) for i in range(n)) / (n - 1)
    var_s = sum((sr[i] - mean_s) ** 2 for i in range(n)) / (n - 1)
    if var_s <= 0:
        return None
    return cov / var_s


@app.get("/api/screener")
async def screener(
    sector: Optional[str] = None,
    exchange: Optional[str] = Query(default=None, description="Optional. Used only when available via ref
