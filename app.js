const $ = (id) => document.getElementById(id);

const fmtMoney = (n) => {
  if (n === null || n === undefined || isNaN(n)) return "";
  const abs = Math.abs(n);
  if (abs >= 1e12) return `$${(n/1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `$${(n/1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `$${(n/1e6).toFixed(2)}M`;
  return `$${n.toLocaleString()}`;
};

const fmt = (n, d=2) => {
  if (n === null || n === undefined || isNaN(n)) return "";
  return Number(n).toFixed(d);
};

let chartInstance = null;

async function loadSectors(){
  const r = await fetch("/api/sectors");
  const j = await r.json();
  const sel = $("sector");
  j.sectors.forEach(s => {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    sel.appendChild(opt);
  });
}

function readFilters(){
  const getNum = (id) => {
    const v = $(id).value.trim();
    if (!v) return null;
    const n = Number(v);
    return isNaN(n) ? null : n;
  };
  return {
    sector: $("sector").value || null,
    exchange: $("exchange").value || null,
    limit: getNum("limit") || 100,

    market_cap_min: getNum("mcapMin"),
    market_cap_max: getNum("mcapMax"),
    beta_min: getNum("betaMin"),
    beta_max: getNum("betaMax"),
    pe_min: getNum("peMin"),
    pe_max: getNum("peMax"),
    dividend_min: getNum("divMin"),
    dividend_max: getNum("divMax"),
    volume_min: getNum("volMin"),
    price_min: getNum("priceMin"),
    price_max: getNum("priceMax"),

    enrich: $("enrich").checked,
    net_margin_min: getNum("netMarginMin"),
    roe_min: getNum("roeMin"),
    current_ratio_min: getNum("currentRatioMin"),
    debt_to_equity_max: getNum("deMax"),
  };
}

function buildQuery(params){
  const qp = new URLSearchParams();
  Object.entries(params).forEach(([k,v]) => {
    if (v === null || v === undefined || v === "") return;
    qp.set(k, String(v));
  });
  return qp.toString();
}

function setStatus(msg){ $("status").textContent = msg || ""; }
function setCount(msg){ $("count").textContent = msg || ""; }

function resetFilters(){
  ["mcapMin","mcapMax","betaMin","betaMax","peMin","peMax","divMin","divMax","volMin","priceMin","priceMax",
   "netMarginMin","roeMin","currentRatioMin","deMax"].forEach(id => $(id).value = "");
  $("sector").value = "";
  $("exchange").value = "";
  $("limit").value = 100;
  $("enrich").checked = false;
}

async function runScreen(){
  const filters = readFilters();
  setStatus("Running screen…");
  setCount("");
  $("tbody").innerHTML = "";

  const qs = buildQuery(filters);
  const r = await fetch(`/api/screener?${qs}`);
  const j = await r.json();

  if (!j.ok){
    setStatus(`Error: ${j.error || "Unknown error"}`);
    return;
  }

  setStatus(j.enriched ? (j.note || "Enriched screen complete.") : "Screen complete.");
  setCount(`${j.count} results`);

  const exportParams = {...filters};
  delete exportParams.enrich;
  delete exportParams.net_margin_min;
  delete exportParams.roe_min;
  delete exportParams.current_ratio_min;
  delete exportParams.debt_to_equity_max;
  $("downloadBtn").href = `/api/export?${buildQuery(exportParams)}`;

  j.data.forEach(row => {
    const tr = document.createElement("tr");
    tr.className = "border-t border-[rgba(11,31,59,0.06)] hover:bg-[rgba(11,31,59,0.03)] transition";

    const sym = row.symbol || "";
    const company = row.companyName || "";
    const sector = row.sector || "";
    const mcap = fmtMoney(row.marketCap);
    const price = (row.price !== null && row.price !== undefined) ? `$${fmt(row.price, 2)}` : "";
    const beta = fmt(row.beta, 2);
    const pe = fmt(row.pe, 1);
    const div = (row.dividendYield !== null && row.dividendYield !== undefined) ? `${fmt(row.dividendYield, 2)}` : "";

    tr.innerHTML = `
      <td class="px-3 py-2 whitespace-nowrap">
        <button class="mm-link font-semibold" data-sym="${sym}">${sym}</button>
      </td>
      <td class="px-3 py-2 min-w-[240px]">${company}</td>
      <td class="px-3 py-2 whitespace-nowrap">${sector}</td>
      <td class="px-3 py-2 whitespace-nowrap">${mcap}</td>
      <td class="px-3 py-2 whitespace-nowrap">${price}</td>
      <td class="px-3 py-2 whitespace-nowrap">${beta}</td>
      <td class="px-3 py-2 whitespace-nowrap">${pe}</td>
      <td class="px-3 py-2 whitespace-nowrap">${div}</td>
    `;
    tr.querySelector("button").addEventListener("click", () => openModal(sym));
    $("tbody").appendChild(tr);
  });
}

function openModal(symbol){
  $("modal").classList.remove("hidden");
  $("mTitle").textContent = `${symbol} — loading…`;
  $("mSub").textContent = "";
  $("mStats").innerHTML = "";
  $("mDesc").textContent = "";
  loadStock(symbol);
}

function closeModal(){
  $("modal").classList.add("hidden");
  if (chartInstance){
    chartInstance.destroy();
    chartInstance = null;
  }
}

async function loadStock(symbol){
  try{
    const [detailR, histR] = await Promise.all([
      fetch(`/api/stock/${encodeURIComponent(symbol)}`),
      fetch(`/api/history/${encodeURIComponent(symbol)}?days=220`)
    ]);
    const detailJ = await detailR.json();
    const histJ = await histR.json();

    if (!detailJ.ok){
      $("mTitle").textContent = `${symbol}`;
      $("mSub").textContent = `Error loading details: ${detailJ.error || "Unknown error"}`;
      return;
    }

    const p = detailJ.profile || {};
    const q = detailJ.quote || {};

    const company = p.companyName || p.name || "";
    const sector = p.sector || "";
    const industry = p.industry || "";
    const exch = p.exchangeShortName || p.exchange || "";
    const price = q.price ?? p.price;
    const chg = q.change;
    const chgPct = q.changesPercentage;

    $("mTitle").textContent = `${symbol} — ${company}`;
    $("mSub").textContent = `${sector}${industry ? " • " + industry : ""}${exch ? " • " + exch : ""}`;

    const stats = [
      ["Price", price !== undefined && price !== null ? `$${fmt(price,2)}` : ""],
      ["Change", (chg !== undefined && chg !== null) ? `${fmt(chg,2)} (${fmt(chgPct,2)}%)` : ""],
      ["Market cap", p.mktCap ? fmtMoney(p.mktCap) : (p.marketCap ? fmtMoney(p.marketCap) : "")],
      ["Beta", p.beta !== undefined ? fmt(p.beta,2) : ""],
      ["P/E", p.pe !== undefined ? fmt(p.pe,1) : ""],
      ["Dividend", p.lastDiv !== undefined ? fmt(p.lastDiv,2) : ""],
      ["Vol (avg)", p.volAvg ? p.volAvg.toLocaleString() : ""],
      ["Website", p.website ? `<a class="mm-link" target="_blank" rel="noreferrer" href="${p.website}">Open</a>` : ""],
    ];

    $("mStats").innerHTML = stats.map(([k,v]) => `
      <div class="flex items-start justify-between gap-3">
        <div class="text-[var(--mm-navy)]/70">${k}</div>
        <div class="text-right font-semibold">${v || ""}</div>
      </div>
    `).join("");

    $("mDesc").textContent = p.description || "No description available.";

    if (!histJ.ok) return;

    const pts = histJ.historical || [];
    const labels = pts.map(x => x.date);
    const values = pts.map(x => x.close);

    const ctx = $("chart").getContext("2d");
    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: `${symbol} Close`,
          data: values,
          borderColor: "#1FA46A",
          backgroundColor: "rgba(31,164,106,0.12)",
          tension: 0.25,
          pointRadius: 0,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { intersect: false, mode: "index" } },
        scales: { x: { ticks: { maxTicksLimit: 8 } }, y: { ticks: { maxTicksLimit: 6 } } }
      }
    });

  }catch(e){
    $("mTitle").textContent = `${symbol}`;
    $("mSub").textContent = `Error: ${String(e)}`;
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  await loadSectors();
  $("runBtn").addEventListener("click", runScreen);
  $("resetBtn").addEventListener("click", resetFilters);
  $("closeModal").addEventListener("click", closeModal);
  $("modal").addEventListener("click", (e) => { if (e.target === $("modal")) closeModal(); });

  const hr = await fetch("/api/health");
  const hj = await hr.json();
  $("status").textContent = hj.ok ? "Ready. Set filters and run your screen." : `Setup needed: ${hj.error}`;
});
