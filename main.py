import os
import csv
import io
from datetime import datetime
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_NAME = "Sector Selector — by Marc-Anthony Richardson, MBA, CPM"

FMP_BASE = "https://financialmodelingprep.com/api/v3"
FMP_KEY = os.getenv("FMP_API_KEY", "").strip()

app = FastAPI(title=APP_NAME)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def _require_key() -> Optional[str]:
    if not FMP_KEY:
        return "Missing API key. Set environment variable FMP_API_KEY before running."
    return None


async def fmp_get(path: str, params: Dict[str, Any]) -> Any:
    params = dict(params or {})
    params["apikey"] = FMP_KEY
    url = f"{FMP_BASE}{path}"
    async with httpx.AsyncClient(timeout=30) as client:
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
    return {"ok": True, "provider": "Financial Modeling Prep", "delayed_quotes": "Typically ~15 min (free tiers vary)"}


@app.get("/api/sectors")
async def sectors():
    return {
        "sectors": [
            "Energy","Materials","Industrials","Consumer Discretionary","Consumer Staples",
            "Health Care","Financials","Information Technology","Communication Services",
            "Utilities","Real Estate",
        ]
    }


@app.get("/api/screener")
async def screener(
    sector: Optional[str] = None,
    exchange: Optional[str] = Query(default=None, description="NYSE, NASDAQ, AMEX, etc. Leave blank for all."),
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
    net_margin_min: Optional[float] = None,
    current_ratio_min: Optional[float] = None,
    roe_min: Optional[float] = None,
    debt_to_equity_max: Optional[float] = None,
):
    err = _require_key()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)

    limit = max(1, min(int(limit), 250))
    params: Dict[str, Any] = {"limit": limit}

    if sector: params["sector"] = sector
    if exchange: params["exchange"] = exchange

    if market_cap_min is not None: params["marketCapMoreThan"] = int(market_cap_min)
    if market_cap_max is not None: params["marketCapLowerThan"] = int(market_cap_max)

    if beta_min is not None: params["betaMoreThan"] = beta_min
    if beta_max is not None: params["betaLowerThan"] = beta_max

    if pe_min is not None: params["peMoreThan"] = pe_min
    if pe_max is not None: params["peLowerThan"] = pe_max

    if dividend_min is not None: params["dividendMoreThan"] = dividend_min
    if dividend_max is not None: params["dividendLowerThan"] = dividend_max

    if volume_min is not None: params["volumeMoreThan"] = volume_min
    if price_min is not None: params["priceMoreThan"] = price_min
    if price_max is not None: params["priceLowerThan"] = price_max

    try:
        results = await fmp_get("/stock-screener", params)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Screener request failed: {str(e)}"}, status_code=500)

    if not isinstance(results, list):
        return {"ok": True, "count": 0, "data": [], "note": "No results"}

    data = []
    for r in results:
        data.append({
            "symbol": r.get("symbol"),
            "companyName": r.get("companyName") or r.get("company") or "",
            "sector": r.get("sector") or "",
            "industry": r.get("industry") or "",
            "exchange": r.get("exchangeShortName") or r.get("exchange") or "",
            "price": r.get("price"),
            "marketCap": r.get("marketCap"),
            "beta": r.get("beta"),
            "pe": r.get("pe"),
            "dividendYield": r.get("dividendYield") or r.get("lastAnnualDividend"),
            "volumeAvg": r.get("volumeAvg") or r.get("volAvg") or r.get("volume"),
        })

    if not enrich:
        return {"ok": True, "count": len(data), "data": data, "enriched": False}

    enriched = []
    for item in data:
        sym = (item.get("symbol") or "").upper().strip()
        if not sym:
            continue
        try:
            profile = await fmp_get(f"/profile/{sym}", {})
            ratios_ttm = await fmp_get(f"/ratios-ttm/{sym}", {})
        except Exception:
            profile, ratios_ttm = [], []

        p0 = profile[0] if isinstance(profile, list) and profile else {}
        r0 = ratios_ttm[0] if isinstance(ratios_ttm, list) and ratios_ttm else {}

        net_margin = r0.get("netProfitMarginTTM")
        if net_margin is not None and net_margin <= 1.5:
            net_margin *= 100.0
        roe = r0.get("returnOnEquityTTM")
        if roe is not None and roe <= 1.5:
            roe *= 100.0
        current_ratio = r0.get("currentRatioTTM")
        debt_to_equity = r0.get("debtEquityRatioTTM")

        item2 = dict(item)
        item2.update({
            "country": p0.get("country"),
            "currency": p0.get("currency"),
            "website": p0.get("website"),
            "description": p0.get("description"),
            "netMarginTTM": net_margin,
            "roeTTM": roe,
            "currentRatioTTM": current_ratio,
            "debtToEquityTTM": debt_to_equity,
        })

        def pass_min(val, mn):
            return mn is None or (val is not None and val >= mn)
        def pass_max(val, mx):
            return mx is None or (val is not None and val <= mx)

        ok = True
        ok = ok and pass_min(item2.get("netMarginTTM"), net_margin_min)
        ok = ok and pass_min(item2.get("currentRatioTTM"), current_ratio_min)
        ok = ok and pass_min(item2.get("roeTTM"), roe_min)
        ok = ok and pass_max(item2.get("debtToEquityTTM"), debt_to_equity_max)

        if ok:
            enriched.append(item2)

    return {"ok": True, "count": len(enriched), "data": enriched, "enriched": True, "note": "Enrichment uses extra API calls; keep limit small on free tiers."}


@app.get("/api/stock/{symbol}")
async def stock_detail(symbol: str):
    err = _require_key()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    symbol = symbol.upper().strip()

    try:
        profile = await fmp_get(f"/profile/{symbol}", {})
        quote = await fmp_get(f"/quote/{symbol}", {})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Detail request failed: {str(e)}"}, status_code=500)

    p0 = profile[0] if isinstance(profile, list) and profile else {}
    q0 = quote[0] if isinstance(quote, list) and quote else {}
    return {"ok": True, "profile": p0, "quote": q0, "symbol": symbol}


@app.get("/api/history/{symbol}")
async def history(symbol: str, days: int = 220):
    err = _require_key()
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    symbol = symbol.upper().strip()
    days = max(30, min(int(days), 1000))

    try:
        hist = await fmp_get(f"/historical-price-full/{symbol}", {"timeseries": days, "serietype": "line"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"History request failed: {str(e)}"}, status_code=500)

    historical = hist.get("historical", []) if isinstance(hist, dict) else []
    historical = sorted(historical, key=lambda x: x.get("date", ""))
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
        enrich=False
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
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
