"""CapitalMatch core logic: the same steps as the n8n workflow (shortlist -> AI analyst -> guardrail),
rewritten in Python so it can run in a public web app."""
import json, re, time
from datetime import date, timedelta
import pandas as pd
import requests

GEMINI_MODEL = "gemini-flash-lite-latest"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

SECTORS = {
    "Biotech & life sciences": "biotechnology, biotech, life sciences, pharma, genomics, diagnostics",
    "Health care & digital health": "health care, healthcare, health and wellness, mobile health, health care information technology, telehealth",
    "Medical devices": "medical devices, health care, hardware",
    "Fintech & payments": "fintech, payments, financial services, finance, banking, accounting",
    "SaaS & business software": "saas, software, enterprise software, productivity software, b2b, crm",
    "AI, data & analytics": "artificial intelligence, ai, analytics, big data, data visualization, business intelligence",
    "E-commerce & marketplaces": "e-commerce, marketplaces, online shopping, retail",
    "Consumer & mobile apps": "mobile, apps, consumers, social media",
    "Education": "education, edtech, corporate training",
    "Games & entertainment": "games, online gaming, entertainment, media, sports",
    "Real estate & property": "real estate, commercial real estate, property, home & garden",
    "Clean energy & climate": "clean technology, energy, climate",
    "Security": "security, mobile security, cybersecurity, defense",
    "Travel & hospitality": "travel, hospitality, tourism, restaurants",
    "Logistics & transportation": "logistics, transportation, supply chain, automotive",
    "Agriculture & food": "agriculture, agtech, food",
    "Marketing & advertising": "advertising, advertising platforms, social media marketing",
    "HR & recruiting": "recruiting, freelancers, professional services",
}
STAGES = ["Pre-seed", "Seed", "Series A", "Series B"]
COUNTRIES = {"United States": "USA", "India": "IND", "United Kingdom": "GBR", "Canada": "CAN", "Germany": "DEU",
             "Singapore": "SGP", "France": "FRA", "Israel": "ISR", "China": "CHN", "Australia": "AUS",
             "Netherlands": "NLD", "Spain": "ESP", "Japan": "JPN", "United Arab Emirates": "ARE",
             "Brazil": "BRA", "Other": ""}


class InputProblem(Exception):
    """A problem with what the user typed, explained in plain English."""


# ---------------------------------------------------------------- data
def load_records(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    df = df[df["company_name"].str.strip() != ""].drop_duplicates(["company_name", "date", "source"])
    recs = df.to_dict("records")
    for i, r in enumerate(recs):
        r["record_id"] = r.get("record_id") or f"R-{i + 1:04d}"
    return recs


# ---------------------------------------------------------------- input checks
def parse_amount(text):
    t = str(text or "").strip().lower().replace(",", "").replace("$", "").replace("usd", "").strip()
    if not t:
        raise InputProblem("Please enter how much you are raising, for example 1.5M or 750K.")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k|thousand|m|mn|million|b|bn|billion)?", t)
    if not m:
        raise InputProblem(f"We couldn't read \"{text}\" as an amount. Type a number like 1500000, 1.5M, or 750K.")
    n = float(m.group(1)); u = m.group(2) or ""
    n *= 1e3 if u.startswith(("k", "th")) else 1e6 if u.startswith("m") else 1e9 if u.startswith("b") else 1
    if n < 10_000 or n > 500_000_000:
        raise InputProblem("That amount looks unusual for a startup round. Please enter an amount between $10K and $500M.")
    return int(n)


def validate(form):
    problems = []
    if len(form["company_name"].strip()) < 2:
        problems.append("Please enter your company's name.")
    if not form["sector"]:
        problems.append("Please choose the sector that best describes your company.")
    if form["sector"] == "Other" and len(form["keywords"].strip()) < 3:
        problems.append("You chose \"Other\" as your sector, so please add a few keywords that describe what you do (for example: drones, insurance).")
    try:
        form["raise_usd"] = parse_amount(form["amount"])
    except InputProblem as e:
        problems.append(str(e))
    if len(form["pitch"].strip()) < 15:
        problems.append("Please add a one-line pitch of at least a few words. The AI uses it to judge which deals are truly similar to yours.")
    if len(form["pitch"]) > 300:
        problems.append("Please keep your pitch to one or two sentences (under 300 characters).")
    if problems:
        raise InputProblem("\n".join(problems))
    return form


# ---------------------------------------------------------------- shortlist (rules)
def stage_level(s):
    s = (s or "").lower()
    if re.search(r"pre-seed|angel|seed", s): return 1
    if re.search(r"series a|venture", s): return 2
    if re.search(r"series [b-e]", s): return 3
    return 0


def is_fund(name):
    return bool(re.search(r"\b(fund|capital|ventures|partners|investors?)\b|\bL\.?P\.?$", name or "", re.I))


def shortlist(founder, records, n=12):
    words = [founder.get("sector_keywords", ""), founder.get("keywords", "")]
    keys = [k.strip().lower() for w in words for k in re.split(r"[,|]", w) if k.strip()]
    pats = [re.compile(r"\b" + re.escape(k) + r"\b") for k in keys]
    f_stage = stage_level(founder["stage"]); f_raise = founder["raise_usd"]
    recent = (date.today() - timedelta(days=365)).isoformat()
    scored = []
    for r in records:
        text = f"{r.get('sector', '')} {r.get('company_name', '')}".lower()
        score, why = 0, []
        if any(p.search(text) for p in pats): score += 40; why.append("sector")
        rs = stage_level(r.get("stage"))
        if rs and rs == f_stage: score += 20; why.append("same stage")
        elif rs and abs(rs - f_stage) == 1: score += 10; why.append("nearby stage")
        try: amt = float(r.get("amount_usd") or 0)
        except ValueError: amt = 0
        if amt and f_raise:
            ratio = min(amt, f_raise) / max(amt, f_raise)
            if ratio >= 0.2: score += round(20 * ratio); why.append("similar check size")
        if founder["country_code"] and (r.get("geography") or "").endswith(founder["country_code"]):
            score += 10; why.append("same country")
        if (r.get("date") or "") >= recent: score += 10; why.append("recent")
        if r.get("record_type") == "funding_filing" and is_fund(r.get("company_name")):
            score += 5; why.append("active fund")
        if score > 0:
            scored.append({**r, "rule_score": score, "rule_reasons": ", ".join(why)})
    scored.sort(key=lambda x: -x["rule_score"])
    return scored[:n]


# ---------------------------------------------------------------- AI analyst
def build_prompt(f, cands):
    deals = "\n".join(
        f"{c['record_id']} | {c['company_name']} | type={c.get('record_type','')} | sector={c.get('sector') or 'n/a'} | "
        f"stage={c.get('stage') or 'n/a'} | amount={c.get('amount_usd') or 'n/a'} | where={c.get('geography') or 'n/a'} | "
        f"date={c.get('date','')} | investors={c.get('investor_names') or 'n/a'} | rule_score={c['rule_score']}"
        for c in cands) or "none found"
    return f"""You are a venture capital analyst helping an early-stage founder find the right investors.

FOUNDER
Company: {f['company_name']}
Sector: {f['sector']} ({f.get('sector_keywords','')}{', ' + f['keywords'] if f.get('keywords') else ''})
Stage: {f['stage']}
Raising: ${f['raise_usd']:,}
Location: {f['city']}, {f['country']}
Pitch: {f['pitch']}

COMPARABLE FUNDING RECORDS (pre-filtered by simple rules; some may be poor fits)
{deals}

TASK
1. Judge each record as evidence of investor appetite for this founder. Reject records that matched on keywords but are not truly comparable.
2. Pick up to 5 of the strongest matches and give each a fit_score from 0 to 100.
3. Describe the type of investor this founder should target, based only on the evidence above.
4. Do not invent investor or fund names. Only name investors that appear in the records.
5. Write for a first-time founder: plain English, no jargon.

Return ONLY valid JSON in exactly this shape:
{{
  "match_quality": "strong | moderate | weak",
  "summary": "2 sentences on overall investor appetite for this founder",
  "top_matches": [{{"record_id": "", "company_name": "", "fit_score": 0, "why": "", "investor_angle": ""}}],
  "rejected": [{{"record_id": "", "reason": ""}}],
  "target_investor_profile": "",
  "outreach_tip": "",
  "data_gaps": ""
}}"""


def call_gemini(prompt, api_key, tries=3):
    """Returns (response_json, error_message). Retries on rate limits, overload, and timeouts."""
    if not api_key:
        return None, "the AI service is not configured"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}
    last = "unknown error"
    for attempt in range(tries):
        try:
            r = requests.post(GEMINI_URL, params={"key": api_key}, json=body, timeout=60)
            if r.status_code == 200:
                return r.json(), None
            last = {429: "the AI service is getting too many requests right now",
                    503: "the AI service is busy right now"}.get(r.status_code, f"the AI service returned error {r.status_code}")
            if r.status_code not in (429, 500, 503):
                break
        except requests.Timeout:
            last = "the AI service took too long to answer"
        except requests.RequestException:
            last = "the AI service could not be reached"
        time.sleep([2, 5, 10][attempt])
    return None, last


def check_answer(res, err, cands):
    """Guardrail: parse and validate the AI answer; fall back to rules if anything is wrong."""
    status, problem = "ok", ""
    try:
        if err:
            raise ValueError(err)
        cand = (res or {}).get("candidates", [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
        if not text:
            raise ValueError("the AI returned an empty answer")
        ai = json.loads(re.sub(r"```json|```", "", text).strip())
        if not isinstance(ai.get("top_matches"), list) or not ai.get("summary"):
            raise ValueError("the AI answer was incomplete")
        ids = {c["record_id"] for c in cands}
        before = len(ai["top_matches"])
        ai["top_matches"] = [{**m, "fit_score": max(0, min(100, int(float(m.get("fit_score") or 0))))}
                             for m in ai["top_matches"] if m.get("record_id") in ids][:5]
        if before and not ai["top_matches"]:
            raise ValueError("the AI answer did not point to real records")
        if len(ai["top_matches"]) < before:
            status = "ok_with_corrections"
        if ai.get("match_quality") not in ("strong", "moderate", "weak"):
            ai["match_quality"] = "moderate"
        ai["rejected"] = [x for x in ai.get("rejected", []) if x.get("record_id") in ids and x.get("reason")]
    except Exception as e:  # any failure -> rule-based result, clearly labeled
        status, problem = "fallback", str(e) if isinstance(e, ValueError) else "the AI answer could not be read"
        best = cands[0]["rule_score"] if cands else 0
        ai = {"match_quality": "strong" if best >= 70 else "moderate" if best >= 45 else "weak",
              "summary": "The AI analyst was unavailable, so this ranking uses our matching rules only. "
                         "Run it again in a minute for the full written analysis.",
              "top_matches": [{"record_id": c["record_id"], "fit_score": c["rule_score"],
                               "why": "Rule match: " + c["rule_reasons"], "investor_angle": ""} for c in cands[:5]],
              "rejected": [], "target_investor_profile": "", "outreach_tip": "", "data_gaps": ""}
    by_id = {c["record_id"]: c for c in cands}
    ai["top_matches"] = [{**by_id[m["record_id"]], **{k: m.get(k, "") for k in ("why", "investor_angle")},
                          "fit_score": m["fit_score"]} for m in ai["top_matches"]]
    names = {c["record_id"]: c["company_name"] for c in cands}
    for x in ai["rejected"]:
        x["company_name"] = names.get(x["record_id"], x["record_id"])
    usage = (res or {}).get("usageMetadata", {}) if not err else {}
    return {"analysis": ai, "status": status, "problem": problem, "tokens": usage.get("totalTokenCount", 0)}


# ---------------------------------------------------------------- formatting
def money(v):
    try: v = float(v)
    except (TypeError, ValueError): return "amount not reported"
    if not v: return "amount not reported"
    if v >= 1e9: return f"${v/1e9:.1f}B".replace(".0B", "B")
    if v >= 1e6: return f"${v/1e6:.1f}M".replace(".0M", "M")
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def report_html(f, result, n_records):
    """A standalone HTML copy of the report the founder can download and share."""
    esc = lambda s: (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    a = result["analysis"]; q = a["match_quality"]
    col = {"strong": "#059669", "moderate": "#d97706", "weak": "#dc2626"}[q]
    rows = "".join(
        f"<div class='m'><div class='h'><b>{i+1}. {esc(m['company_name'])}</b><span class='s'>fit {m['fit_score']}</span></div>"
        f"<div class='meta'>{esc(m.get('sector') or 'sector not reported')} · {esc(m.get('stage') or 'stage not reported')} · "
        f"{money(m.get('amount_usd'))} · {esc(m.get('geography') or '')} · {esc(m.get('date'))}</div>"
        f"<p>{esc(m['why'])}</p>{'<p class=a><b>Investor angle:</b> ' + esc(m['investor_angle']) + '</p>' if m.get('investor_angle') else ''}"
        f"<a href='{esc(m.get('source_url'))}'>Source: {esc(m.get('source'))}</a></div>" for i, m in enumerate(a["top_matches"]))
    rej = "".join(f"<li><b>{esc(x['company_name'])}</b>: {esc(x['reason'])}</li>" for x in a["rejected"])
    warn = (f"<p class='w'>The AI analyst was unavailable ({esc(result['problem'])}), so this report uses rule-based scores only.</p>"
            if result["status"] == "fallback" else "")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>CapitalMatch report: {esc(f['company_name'])}</title>
<style>body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:820px;margin:32px auto;padding:0 20px;color:#0f172a;line-height:1.55}}
.b{{display:inline-block;background:{col};color:#fff;padding:5px 12px;border-radius:99px;font-weight:700;font-size:13px;text-transform:uppercase}}
.m{{border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;margin:10px 0}}.h{{display:flex;justify-content:space-between}}
.s{{font-weight:800;color:#0f766e}}.meta{{color:#64748b;font-size:13px}}.a{{color:#115e59;font-size:14px}}
.box{{background:#f1f5f9;border-radius:12px;padding:12px 16px;margin:10px 0}}.w{{background:#fff7ed;border:1px solid #fed7aa;padding:10px;border-radius:10px}}
a{{color:#0f766e;font-size:13px}}small{{color:#64748b}}</style></head><body>
<small>CapitalMatch · investor match report · {date.today().isoformat()}</small>
<h1>{esc(f['company_name'])}</h1><p>{esc(f['pitch'])}</p>
<p>{esc(f['sector'])} · {esc(f['stage'])} · raising {money(f['raise_usd'])} · {esc(f['city'])}, {esc(f['country'])}</p>
<span class="b">{q} match</span>{warn}<h3>Summary</h3><p>{esc(a['summary'])}</p>
<h3>Funding deals most like yours</h3>{rows or '<p>No comparable deals found in the current data.</p>'}
<h3>Who to target</h3><div class="box">{esc(a.get('target_investor_profile')) or 'Not available for this run.'}</div>
<h3>How to open the conversation</h3><div class="box">{esc(a.get('outreach_tip')) or 'Not available for this run.'}</div>
{'<h3>Deals the AI ruled out</h3><ul>' + rej + '</ul>' if rej else ''}
<p><small>Compared against {n_records} real funding records: Crunchbase history (2014), SEC Form D filings, and recent funding news.
Every deal links to its source so you can check it before you pitch.</small></p></body></html>"""
