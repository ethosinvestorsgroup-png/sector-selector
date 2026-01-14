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
    exchange: Optional[str] = Query(default=None, description="Optional. Used only when available via reference endpoints."),
    market_cap_min: Optional[float] = None,
    market_cap_max: Optional[float] = None,
    beta_min: Optional[float] = None,
    beta_max: Optional[float] = None,
    pe_min: Optional[float] = None,
    pe_max: Optional[float] = None,
    dividend_min: Optional[float] = None,
    dividend_max: Optional[float] = None,
    volume_min: Optional[float] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    limit: int = 100,
    enrich: bool = False,
    net_margin_min: Optional[float] = None,          # not used in this MVP unless you enable ratios expansion
    current_ratio_min: Optional[float] = None,
    roe_min: Optional[float] = None,
    debt_to_equity_max: Optional[float] = None,
):
    """
    MVP approach (Massive):
    - Primary dataset: /stocks/financials/v1/ratios (if your plan includes it)
    - Optional: compute beta on-the-fly for the *returned* tickers using daily aggregates vs SPY
    """
    err = _require_key()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)

    limit = max(1, min(int(limit), 200))

    # ---- 1) Pull ratios table (best for screening) ----
    # NOTE: If your Massive plan does NOT include this endpoint, you'll get 403/401 here.
    ratios_params: Dict[str, Any] = {"limit": limit}

    if market_cap_min is not None: ratios_params["market_cap.gte"] = market_cap_min
    if market_cap_max is not None: ratios_params["market_cap.lte"] = market_cap_max

    if pe_min is not None: ratios_params["price_to_earnings.gte"] = pe_min
    if pe_max is not None: ratios_params["price_to_earnings.lte"] = pe_max

    if dividend_min is not None: ratios_params["dividend_yield.gte"] = dividend_min
    if dividend_max is not None: ratios_params["dividend_yield.lte"] = dividend_max

    if price_min is not None: ratios_params["price.gte"] = price_min
    if price_max is not None: ratios_params["price.lte"] = price_max

    if volume_min is not None: ratios_params["average_volume.gte"] = volume_min

    if current_ratio_min is not None: ratios_params["current.gte"] = current_ratio_min
    if debt_to_equity_max is not None: ratios_params["debt_to_equity.lte"] = debt_to_equity_max

    try:
        ratios = await massive_get("/stocks/financials/v1/ratios", ratios_params)
    except Exception as e:
        return JSONResponse(
            {
                "ok": False,
                "error": f"Massive ratios screener failed. This usually means your plan doesn’t include /stocks/financials/v1/ratios yet. Details: {str(e)}"
            },
            status_code=500
        )

    rows = ratios.get("results") or []
    data = []
    tickers = []
    for r in rows:
        sym = (r.get("ticker") or "").upper().strip()
        if not sym:
            continue
        tickers.append(sym)
        data.append({
            "symbol": sym,
            "companyName": "",  # filled below via reference endpoint
            "sector": "",
            "industry": "",
            "exchange": exchange or "",
            "price": r.get("price"),
            "marketCap": r.get("market_cap"),
            "beta": None,  # computed below if requested
            "pe": r.get("price_to_earnings"),
            "dividendYield": r.get("dividend_yield"),
            "volumeAvg": r.get("average_volume"),
            "currentRatioTTM": r.get("current"),
            "debtToEquityTTM": r.get("debt_to_equity"),
            "roeTTM": r.get("return_on_equity"),
        })

    # ---- 2) Enrich names/industry via reference endpoint (fast) ----
    async def enrich_one(sym: str) -> Dict[str, Any]:
        try:
            ref = await massive_get(f"/v3/reference/tickers/{sym}", {})
            res = (ref.get("results") or {})
            name = res.get("name") or ""
            sic_desc = res.get("sic_description") or ""
            sector_guess = _approx_sector_from_sic(sic_desc)
            return {"symbol": sym, "companyName": name, "industry": sic_desc, "sector": sector_guess}
        except Exception:
            return {"symbol": sym, "companyName": "", "industry": "", "sector": ""}

    # keep this bounded
    sem = asyncio.Semaphore(12)

    async def bounded(sym: str):
        async with sem:
            return await enrich_one(sym)

    enrich_results = await asyncio.gather(*(bounded(s) for s in tickers))
    enrich_map = {e["symbol"]: e for e in enrich_results}

    for item in data:
        e = enrich_map.get(item["symbol"], {})
        item["companyName"] = e.get("companyName", "")
        item["industry"] = e.get("industry", "")
        item["sector"] = e.get("sector", "")

    # optional sector filter (heuristic)
    if sector:
        data = [d for d in data if (d.get("sector") == sector)]

    # ---- 3) Compute beta only if user is filtering by beta ----
    if beta_min is not None or beta_max is not None:
        try:
            spy_prices = await _aggs_daily("SPY", days=252)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Failed to fetch SPY data for beta calc: {str(e)}"}, status_code=500)

        async def beta_one(sym: str) -> Tuple[str, Optional[float]]:
            try:
                px = await _aggs_daily(sym, days=252)
                b = _beta_from_prices(px, spy_prices)
                return sym, b
            except Exception:
                return sym, None

        sem2 = asyncio.Semaphore(6)

        async def bounded_beta(sym: str):
            async with sem2:
                return await beta_one(sym)

        beta_results = await asyncio.gather(*(bounded_beta(d["symbol"]) for d in data[:limit]))
        beta_map = dict(beta_results)

        for d in data:
            d["beta"] = beta_map.get(d["symbol"])

        def pass_min(val, mn): return mn is None or (val is not None and val >= mn)
        def pass_max(val, mx): return mx is None or (val is not None and val <= mx)

        data = [d for d in data if pass_min(d.get("beta"), beta_min) and pass_max(d.get("beta"), beta_max)]

    return {"ok": True, "count": len(data), "data": data, "enriched": True}


@app.get("/api/stock/{symbol}")
async def stock_detail(symbol: str):
    err = _require_key()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    symbol = symbol.upper().strip()

    try:
        profile = await massive_get(f"/v3/reference/tickers/{symbol}", {})
        snap = await massive_get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}", {})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Detail request failed: {str(e)}"}, status_code=500)

    return {"ok": True, "profile": profile.get("results") or {}, "quote": snap.get("ticker") or snap, "symbol": symbol}


@app.get("/api/history/{symbol}")
async def history(symbol: str, days: int = 220):
    err = _require_key()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    symbol = symbol.upper().strip()
    days = max(30, min(int(days), 1000))

    try:
        px = await _aggs_daily(symbol, days=days)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"History request failed: {str(e)}"}, status_code=500)

    historical = [{"date": d, "close": c} for d, c in px]
    return {"ok": True, "symbol": symbol, "historical": historical}


@app.get("/api/export")
async def export_csv(
    sector: Optional[str] = None,
    exchange: Optional[str] = None,
    market_cap_min: Optional[float] = None,
    market_cap_max: Optional[float] = None,
    beta_min: Optional[float] = None,
    beta_max: Optional[float] = None,
    pe_min: Optional[float] = None,
    pe_max: Optional[float] = None,
    dividend_min: Optional[float] = None,
    dividend_max: Optional[float] = None,
    volume_min: Optional[float] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    limit: int = 200
):
    resp = await screener(
        sector=sector, exchange=exchange,
        market_cap_min=market_cap_min, market_cap_max=market_cap_max,
        beta_min=beta_min, beta_max=beta_max,
        pe_min=pe_min, pe_max=pe_max,
        dividend_min=dividend_min, dividend_max=dividend_max,
        volume_min=volume_min,
        price_min=price_min, price_max=price_max,
        limit=limit,
        enrich=True
    )
    if isinstance(resp, JSONResponse):
        return resp
    data = resp.get("data", []) if isinstance(resp, dict) else []

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "symbol","companyName","sector","industry","exchange","price","marketCap","beta","pe","dividendYield","volumeAvg"
    ])
    writer.writeheader()
    for row in data:
        writer.writerow(row)
    output.seek(0)

    filename = f"SectorSelector_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'}
    )
