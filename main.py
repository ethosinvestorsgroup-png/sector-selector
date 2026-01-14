import os
import io
import csv
import time
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# ----------------------------
# Config
# ----------------------------
APP_TITLE = "Sector Selector — by Marc-Anthony Richardson, MBA, CPM"

MASSIVE_BASE = "https://api.massive.com"
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY", "").strip()

# In-memory caching to reduce API calls on free/low tiers
CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours
_universe_cache: Dict[str, Any] = {"ts": 0, "data": []}
_spy_cache: Dict[str, Any] = {"ts": 0, "data": []}

# Keep per-symbol snapshot/agg cached briefly
_symbol_cache: Dict[str, Any] = {}  # sym -> {ts, snapshot, aggs}
SYMBOL_CACHE_SECONDS = 60 * 15  # 15 minutes


# ----------------------------
# App
# ----------------------------
app = FastAPI(title=APP_TITLE)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ----------------------------
# Helpers
# ----------------------------
def _missing_key_error() -> Optional[str]:
    if not MASSIVE_API_KEY:
        return "Missing API key. Set environment variable MASSIVE_API_KEY in Render (Environment tab)."
    return None


async def massive_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Massive supports apiKey in the query string.
    """
    params = dict(params or {})
    params["apiKey"] = MASSIVE_API_KEY
    url = f"{MASSIVE_BASE}{path}"

    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.get(url, params=params)
        # Raise for non-2xx to surface 401/403 clearly
        r.raise_for_status()
        return r.json()


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _passes_minmax(val: Optional[float], min_v: Optional[float], max_v: Optional[float]) -> bool:
    if val is None:
        return False if (min_v is not None or max_v is not None) else True
    if min_v is not None and val < min_v:
        return False
    if max_v is not None and val > max_v:
        return False
    return True


# ----------------------------
# Massive data fetchers
# ----------------------------
async def get_ticker_universe(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Pull a universe of US stock tickers. We cache it because it’s large.
    """
    now = int(time.time())
    if _universe_cache["data"] and (now - _universe_cache["ts"]) < CACHE_TTL_SECONDS:
        return _universe_cache["data"]

    # Reference tickers endpoint (paged). Many plans allow reference data.
    # We start simple with one page. Later we can paginate/crawl to “all”.
    js = await massive_get(
        "/v3/reference/tickers",
        {
            "market": "stocks",
            "locale": "us",
            "active": "true",
            "limit": int(limit),
        },
    )
    rows = js.get("results") or []
    universe = []
    for r in rows:
        sym = (r.get("ticker") or "").upper().strip()
        if not sym:
            continue
        universe.append(
            {
                "symbol": sym,
                "name": r.get("name") or "",
                "type": r.get("type") or "",
                "primary_exchange": r.get("primary_exchange") or "",
                "sic_description": r.get("sic_description") or "",
            }
        )

    _universe_cache["ts"] = now
    _universe_cache["data"] = universe
    return universe


async def get_snapshot(symbol: str) -> Dict[str, Any]:
    """
    Snapshot gives a lot of quick fields (price, volume, sometimes fundamentals including market cap/dividends).
    """
    symbol = symbol.upper().strip()
    js = await massive_get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}", {})
    return js


async def get_aggs_daily(symbol: str, days: int = 260) -> List[Tuple[int, float, float]]:
    """
    Daily aggregates: returns list of (t_ms, close, volume) tuples.
    """
    symbol = symbol.upper().strip()
    end = datetime.now(timezone.utc).date()
    # buffer for weekends/holidays
    start = (datetime.now(timezone.utc) - timedelta(days=int(days * 1.7))).date()

    js = await massive_get(
        f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
        {"adjusted": "true", "sort": "asc", "limit": 50000},
    )
    results = js.get("results") or []

    out: List[Tuple[int, float, float]] = []
    for r in results:
        t = r.get("t")
        c = r.get("c")
        v = r.get("v")
        if t is None or c is None:
            continue
        out.append((int(t), float(c), float(v) if v is not None else 0.0))

    # keep last ~days points
    if len(out) > days:
        out = out[-days:]
    return out


async def get_spy_prices(days: int = 260) -> List[Tuple[int, float]]:
    """
    Cached SPY closes for beta calculation.
    """
    now = int(time.time())
    if _spy_cache["data"] and (now - _spy_cache["ts"]) < CACHE_TTL_SECONDS:
        return _spy_cache["data"]

    aggs = await get_aggs_daily("SPY", days=days)
    spy = [(t, c) for (t, c, _v) in aggs]
    _spy_cache["ts"] = now
    _spy_cache["data"] = spy
    return spy


def compute_beta(symbol_aggs: List[Tuple[int, float, float]], spy: List[Tuple[int, float]]) -> Optional[float]:
    """
    Compute beta from aligned daily returns vs SPY.
    """
    if len(symbol_aggs) < 40 or len(spy) < 40:
        return None

    p_map = {t: c for (t, c, _v) in symbol_aggs}
    s_map = {t: c for (t, c) in spy}
    ts = sorted(set(p_map.keys()) & set(s_map.keys()))
    if len(ts) < 40:
        return None

    pr: List[float] = []
    sr: List[float] = []
    for i in range(1, len(ts)):
        t0, t1 = ts[i - 1], ts[i]
        p0, p1 = p_map[t0], p_map[t1]
        s0, s1 = s_map[t0], s_map[t1]
        if p0 <= 0 or s0 <= 0:
            continue
        pr.append((p1 / p0) - 1.0)
        sr.append((s1 / s0) - 1.0)

    n = min(len(pr), len(sr))
    if n < 30:
        return None
    pr = pr[-n:]
    sr = sr[-n:]

    mp = sum(pr) / n
    ms = sum(sr) / n
    cov = sum((pr[i] - mp) * (sr[i] - ms) for i in range(n)) / (n - 1)
    var = sum((sr[i] - ms) ** 2 for i in range(n)) / (n - 1)
    if var <= 0:
        return None
    return cov / var


def compute_momentum_3m(symbol_aggs: List[Tuple[int, float, float]]) -> Optional[float]:
    """
    3M momentum approx = last close / close 63 trading days ago - 1 (percent).
    """
    if len(symbol_aggs) < 70:
        return None
    p_now = symbol_aggs[-1][1]
    p_then = symbol_aggs[-64][1]
    if p_then <= 0:
        return None
    return (p_now / p_then - 1.0) * 100.0


def extract_snapshot_fields(symbol: str, snapshot_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Snapshot schema can vary by plan; we defensively pull what’s available.
    """
    tk = snapshot_json.get("ticker") or snapshot_json.get("results") or snapshot_json
    day = tk.get("day") or {}
    prev = tk.get("prevDay") or {}
    last_trade = tk.get("lastTrade") or {}
    fundamentals = tk.get("fundamentals") or {}

    price = day.get("c") or last_trade.get("p") or prev.get("c")
    volume = day.get("v") or prev.get("v")

    market_cap = fundamentals.get("marketCap") or fundamentals.get("market_cap")
    dividend_yield = fundamentals.get("dividendYield") or fundamentals.get("dividend_yield")
    dividend_cash = fundamentals.get("dividendCashAmount") or fundamentals.get("dividend_cash_amount")

    return {
        "symbol": symbol,
        "price": _to_float(price),
        "volume": _to_float(volume),
        "marketCap": _to_float(market_cap),
        "dividendYield": _to_float(dividend_yield),
        "dividendCash": _to_float(dividend_cash),
        "rawFundamentals": fundamentals,
    }


async def get_symbol_bundle(symbol: str, days: int = 260) -> Dict[str, Any]:
    """
    Cached snapshot + aggs bundle per symbol (reduces repeated calls).
    """
    now = int(time.time())
    sym = symbol.upper().strip()
    cached = _symbol_cache.get(sym)
    if cached and (now - cached["ts"]) < SYMBOL_CACHE_SECONDS:
        return cached["data"]

    snap = await get_snapshot(sym)
    aggs = await get_aggs_daily(sym, days=days)

    bundle = {"snapshot": snap, "aggs": aggs}
    _symbol_cache[sym] = {"ts": now, "data": bundle}
    return bundle


# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "app_name": APP_TITLE})


@app.get("/api/health")
async def health():
    err = _missing_key_error()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    return {"ok": True, "provider": "Massive.com", "base": MASSIVE_BASE}


@app.get("/api/stock/{symbol}")
async def stock_detail(symbol: str):
    """
    Detail endpoint for your ticker modal:
    returns snapshot + candles-like arrays for charting
    """
    err = _missing_key_error()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)

    sym = symbol.upper().strip()
    try:
        bundle = await get_symbol_bundle(sym, days=520)
        snap_fields = extract_snapshot_fields(sym, bundle["snapshot"])

        # Convert aggs to candle-like arrays for chart.js
        t_arr = [t for (t, _c, _v) in bundle["aggs"]]
        c_arr = [_c for (_t, _c, _v) in bundle["aggs"]]
        v_arr = [_v for (_t, _c, _v) in bundle["aggs"]]

        return {
            "ok": True,
            "symbol": sym,
            "snapshot": snap_fields,
            "candles": {"t": t_arr, "c": c_arr, "v": v_arr},
        }
    except httpx.HTTPStatusError as e:
        return JSONResponse(status_code=502, content={"ok": False, "error": f"Upstream Massive error: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/api/screener")
async def screener(
    # Core screen inputs
    market_cap_min: Optional[float] = Query(None),
    market_cap_max: Optional[float] = Query(None),
    beta_min: Optional[float] = Query(None),
    beta_max: Optional[float] = Query(None),
    pe_min: Optional[float] = Query(None),              # may not be available on your tier; kept for UI compatibility
    pe_max: Optional[float] = Query(None),              # (we won’t filter if we can’t compute it)
    dividend_min: Optional[float] = Query(None),
    dividend_max: Optional[float] = Query(None),
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
    volume_min: Optional[float] = Query(None),
    momentum_3m_min: Optional[float] = Query(None, description="3-month momentum minimum (percent)."),
    limit: int = Query(100, ge=1, le=200),
):
    """
    Screener without Massive paid ratios endpoint:
      - Universe (first 1000 active US stock tickers)
      - Snapshot + aggs for a bounded subset
      - Compute beta vs SPY
      - Compute 3M momentum
      - Filter locally
    """
    err = _missing_key_error()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)

    try:
        universe = await get_ticker_universe(limit=1000)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Failed to load ticker universe: {str(e)}"}, status_code=500)

    # Fetch SPY for beta only if user is using beta filters
    spy = None
    if beta_min is not None or beta_max is not None:
        try:
            spy = await get_spy_prices(days=260)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Failed to fetch SPY for beta: {str(e)}"}, status_code=500)

    # Process a subset to stay within low-tier limits; tighten filters for best results.
    # You can increase this later after adding DB caching.
    subset = universe[:250]

    sem = asyncio.Semaphore(10)

    async def process_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sym = row["symbol"]
        async with sem:
            try:
                bundle = await get_symbol_bundle(sym, days=260)
                snap_fields = extract_snapshot_fields(sym, bundle["snapshot"])
                aggs = bundle["aggs"]

                mom3 = compute_momentum_3m(aggs)
                b = compute_beta(aggs, spy) if spy is not None else None

                out = {
                    "symbol": sym,
                    "name": row.get("name", ""),
                    "exchange": row.get("primary_exchange", ""),
                    "industry": row.get("sic_description", ""),
                    "price": snap_fields.get("price"),
                    "volume": snap_fields.get("volume"),
                    "marketCap": snap_fields.get("marketCap"),
                    "dividendYield": snap_fields.get("dividendYield"),
                    "dividendCash": snap_fields.get("dividendCash"),
                    "beta": b,
                    "momentum3m": mom3,
                    # Placeholders (kept so UI doesn’t break if it references them)
                    "pe": None,
                }
                return out
            except Exception:
                return None

    processed = await asyncio.gather(*(process_row(r) for r in subset))

    results: List[Dict[str, Any]] = []
    for r in processed:
        if not r:
            continue

        # Apply filters (skip a filter if value isn’t available)
        if market_cap_min is not None or market_cap_max is not None:
            if not _passes_minmax(r.get("marketCap"), market_cap_min, market_cap_max):
                continue

        if price_min is not None or price_max is not None:
            if not _passes_minmax(r.get("price"), price_min, price_max):
                continue

        if volume_min is not None:
            v = r.get("volume")
            if v is None or v < volume_min:
                continue

        if dividend_min is not None or dividend_max is not None:
            dy = r.get("dividendYield")
            # If dividend data missing, it fails dividend filter by design
            if not _passes_minmax(dy, dividend_min, dividend_max):
                continue

        if beta_min is not None or beta_max is not None:
            b = r.get("beta")
            if not _passes_minmax(b, beta_min, beta_max):
                continue

        if momentum_3m_min is not None:
            m = r.get("momentum3m")
            if m is None or m < momentum_3m_min:
                continue

        results.append(r)

    # Truncate to limit after filtering
    results = results[:limit]

    return {
        "ok": True,
        "count": len(results),
        "results": results,   # common frontend key
        "data": results,      # also provided for flexibility
        "note": "Using Massive reference + snapshot + aggregates (ratios endpoint not enabled on this plan). Tighten filters for best results.",
        "coverage": "First 250 active US tickers (expandable with caching/pagination).",
    }


@app.get("/api/export")
async def export_csv(
    market_cap_min: Optional[float] = Query(None),
    market_cap_max: Optional[float] = Query(None),
    beta_min: Optional[float] = Query(None),
    beta_max: Optional[float] = Query(None),
    dividend_min: Optional[float] = Query(None),
    dividend_max: Optional[float] = Query(None),
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
    volume_min: Optional[float] = Query(None),
    momentum_3m_min: Optional[float] = Query(None),
    limit: int = Query(200, ge=1, le=200),
):
    # Reuse screener, then export results
    resp = await screener(
        market_cap_min=market_cap_min,
        market_cap_max=market_cap_max,
        beta_min=beta_min,
        beta_max=beta_max,
        dividend_min=dividend_min,
        dividend_max=dividend_max,
        price_min=price_min,
        price_max=price_max,
        volume_min=volume_min,
        momentum_3m_min=momentum_3m_min,
        limit=limit,
    )

    if isinstance(resp, JSONResponse):
        return resp

    rows = resp.get("results", []) if isinstance(resp, dict) else []

    output = io.StringIO()
    fieldnames = [
        "symbol",
        "name",
        "exchange",
        "industry",
        "price",
        "volume",
        "marketCap",
        "dividendYield",
        "dividendCash",
        "beta",
        "momentum3m",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in fieldnames})
    output.seek(0)

    filename = f"SectorSelector_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
