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
    "Clean energy & climate": "clean technology, energy, climate, solar, carbon",
    "Hardware, robotics & manufacturing": "hardware, robotics, manufacturing, wearables, industrial, devices",
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
    # GIGO fix: SEC's "type=D" feed also returns proxy filings (DEF 14C, DEFA14A, DFAN14A). They are not raises.
    not_form_d = (df.get("record_type", "") == "funding_filing") & df["company_name"].str.match(r"^[A-Z0-9]{3,8}( [A-Z0-9]+)? - ")
    df = df[~not_form_d]
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


def shortlist(founder, records, n=12, recent_first=False):
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
        if (r.get("date") or "") >= recent:
            score += 35 if recent_first else 10; why.append("recent")
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
What this round will pay for: {f.get('use_of_funds') or 'not provided'}

COMPARABLE FUNDING RECORDS (pre-filtered by simple rules; some may be poor fits)
{deals}

TASK
1. Judge each record as evidence of investor appetite for this founder. Reject records that matched on keywords but are not truly comparable.
2. Pick up to {f.get('max_matches', 5)} of the strongest matches and give each a fit_score from 0 to 100.
3. Describe the type of investor this founder should target, based only on the evidence above.
4. Do not invent investor or fund names. Only name investors that appear in the records.
5. Write for a first-time founder: plain English, no jargon. If the founder says what the round will pay for, use it in the outreach tip.
6. If the founder asked for recent deals first, prefer records from the last 12 months when they are comparable.

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


def check_answer(res, err, cands, max_matches=5):
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
                             for m in ai["top_matches"] if m.get("record_id") in ids][:max_matches]
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
                               "why": "Rule match: " + c["rule_reasons"], "investor_angle": ""} for c in cands[:max_matches]],
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
def freshness(records):
    """Newest record date and how many records are from the last 12 months."""
    dates = sorted(r.get("date", "") for r in records if r.get("date"))
    cutoff = (date.today() - timedelta(days=365)).isoformat()
    return (dates[-1] if dates else "unknown"), sum(1 for d in dates if d >= cutoff)


def money(v):
    try: v = float(v)
    except (TypeError, ValueError): return "amount not reported"
    if not v: return "amount not reported"
    if v >= 1e9: return f"${v/1e9:.1f}B".replace(".0B", "B")
    if v >= 1e6: return f"${v/1e6:.1f}M".replace(".0M", "M")
    if v >= 1e3: return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def report_html(f, result, n_records, apply=None, recent=None, active=None):
    """A standalone HTML copy of the report the founder can download and share."""
    esc = lambda s: (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    a = result["analysis"]; q = a["match_quality"]
    col = {"strong": "#059669", "moderate": "#d97706", "weak": "#dc2626"}[q]
    rows = "".join(
        f"<div class='m'><div class='h'><b>{i+1}. {esc(m['company_name'])}</b><span class='s'>fit {m['fit_score']}</span></div>"
        f"<div class='meta'>{esc(m.get('sector') or 'sector not reported')} · {esc(m.get('stage') or 'stage not reported')} · "
        f"{money(m.get('amount_usd'))} · {esc(m.get('geography') or '')} · {esc(m.get('date'))}</div>"
        f"<p>{esc(m['why'])}</p>{'<p class=a><b>Investor angle:</b> ' + esc(m['investor_angle']) + '</p>' if m.get('investor_angle') else ''}"
        f"<a href='{esc(m.get('source_url'))}'>Source: {esc(m.get('source'))}</a> · "
        f"<a href='{esc(deal_investors_url(m['company_name'], m.get('date')))}'>find who invested</a></div>" for i, m in enumerate(a["top_matches"]))
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
{'<h3>Investment funds active right now (SEC Form D, last 12 months)</h3><ul>' + "".join(f"<li><b>{esc(i['fund'])}</b> · filed {esc(i['date'])} · <a href='{esc(i['filing_url'])}'>SEC filing</a> · <a href='{esc(i['search_url'])}'>find website &amp; pitch page</a></li>" for i in (active or [])) + '</ul>' if active else ''}
{'<h3>Recent funding activity in your sector</h3><ul>' + ''.join(f"<li><b>{esc(x['company_name'])}</b> · {esc(x.get('stage') or 'stage not reported')} · {money(x.get('amount_usd'))} · {esc(x['date'])} · <a href='{esc(x.get('source_url'))}'>source</a></li>" for x in recent) + '</ul>' if recent else ''}
{'<h3>Where to apply (official pages)</h3><ul>' + ''.join(f"<li><b>{esc(i['name'])}</b> ({esc(i['type'])}): {esc(i['focus'])}. <a href='{esc(i['url'])}'>{esc(i['link_label'])}</a></li>" for i in apply) + '</ul>' if apply else ''}
<p><small>Compared against {n_records} real funding records: Crunchbase history (2014), SEC Form D filings, and recent funding news.
Every deal links to its source so you can check it before you pitch.</small></p></body></html>"""


# ---------------------------------------------------------------- live funding news (same quality gate as A3)
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

NEWS_FEEDS = [("TechCrunch", "https://techcrunch.com/category/venture/feed/"),
              ("Crunchbase News", "https://news.crunchbase.com/feed/")]
FUNDING_VERB = re.compile(r"\b(raise|raises|raised|secures|lands|closes|bags|nabs)\b", re.I)
COMPANY_SPLIT = re.compile(r" (raise|raises|raised|secures|lands|bags|closes|nabs|gets|snags)\b", re.I)


def _headline_amount(title):
    m = re.search(r"\$(\d+(?:\.\d+)?)\s*(m|million|b|billion|k)?\b", title, re.I)
    if not m: return ""
    n = float(m.group(1)); u = (m.group(2) or "").lower()
    return str(int(n * (1e9 if u.startswith("b") else 1e6 if u.startswith("m") else 1e3 if u == "k" else 1)))


def parse_news(xml_text, source):
    """Turn one RSS feed into funding records. Returns (records, rejected_count)."""
    root = ET.fromstring(xml_text)
    out, rejected = [], 0
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
        try:
            day = parsedate_to_datetime(item.findtext("pubDate") or "").date().isoformat()
        except (TypeError, ValueError):
            day = ""
        if not FUNDING_VERB.search(title):
            continue  # not a funding announcement (general news or roundup)
        company = COMPANY_SPLIT.split(title)[0]
        company = re.sub(r"^Exclusive:\s*", "", company, flags=re.I).split(",")[-1]
        company = re.sub(r"^[\w-]+-backed\s+", "", company, flags=re.I).strip()
        if not (company and _valid_date(day) and link.startswith("http")):
            rejected += 1  # fails the quality gate: missing name, date, or link
            continue
        text = f"{title} {desc}"
        stage = re.search(r"pre-seed|seed|series [a-e]", text, re.I)
        investor = re.search(r"led by ([A-Z][\w&.'-]*(?: [A-Z][\w&.'-]*)*)", text)
        cats = [c.text.strip() for c in item.findall("category") if c.text]
        out.append({"record_type": "funding_news", "company_name": company, "sector": ", ".join(cats),
                    "stage": stage.group(0) if stage else "", "amount_usd": _headline_amount(title),
                    "geography": "", "date": day, "investor_names": investor.group(1) if investor else "",
                    "source": f"{source} (live)", "source_url": link})
    return out, rejected


def live_news():
    """Fetch today's funding news. Returns (records, status_message). Never raises."""
    recs, notes = [], []
    for source, url in NEWS_FEEDS:
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "CapitalMatch/1.2 (student project)"})
            r.raise_for_status()
            got, rej = parse_news(r.text, source)
            recs += got
            notes.append(f"{source}: {len(got)} funding stories" + (f", {rej} rejected" if rej else ""))
        except Exception:
            notes.append(f"{source}: unavailable right now")
    unique, seen = [], set()
    for r in recs:  # the same story can appear in both feeds
        key = (r["source_url"], r["company_name"].lower())
        if key not in seen:
            seen.add(key); unique.append(r)
    for i, r in enumerate(unique):
        r["record_id"] = f"LIVE-{i + 1:03d}"
    return unique, "; ".join(notes)


def merge_records(verified, live):
    seen = {(r.get("source_url") or "") for r in verified} | {(r["company_name"].lower(), r["date"]) for r in verified}
    fresh = [r for r in live if r["source_url"] not in seen and (r["company_name"].lower(), r["date"]) not in seen]
    return verified + fresh, len(fresh)


# ---------------------------------------------------------------- where to apply (official public pages only)
def load_investors(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return df.to_dict("records")


def match_investors(founder, investors, n=5):
    scored = []
    for inv in investors:
        sectors = [s.strip() for s in inv["sectors"].split(";")]
        stages = [s.strip() for s in inv["stages"].split(";")]
        regions = [s.strip() for s in inv["regions"].split(";")]
        score = (3 if founder["sector"] in sectors else 1 if "All" in sectors else 0)
        if score == 0:
            continue
        score += 2 if founder["stage"] in stages else 0
        score += 2 if founder["country_code"] and founder["country_code"] in regions else 1 if "Global" in regions else 0
        if founder["stage"] in stages:
            scored.append({**inv, "score": score})
    scored.sort(key=lambda x: -x["score"])
    return scored[:n]


def check_link(url):
    """Live check that an official page is up. Returns 'ok', 'unchecked', or 'down'."""
    try:
        r = requests.head(url, allow_redirects=True, timeout=6, headers={"User-Agent": "Mozilla/5.0 CapitalMatch"})
        if r.status_code in (403, 405):
            r = requests.get(url, allow_redirects=True, timeout=6, stream=True, headers={"User-Agent": "Mozilla/5.0 CapitalMatch"})
        if r.status_code < 400: return "ok"
        if r.status_code in (403, 429): return "unchecked"
        return "down"
    except requests.RequestException:
        return "unchecked"


# ---------------------------------------------------------------- live SEC Form D filings (same quality gate as A3)
SEC_FEED = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=D&company=&dateb=&owner=include&count=100&output=atom"
SEC_HEADERS = {"User-Agent": "CapitalMatch student project (Northeastern INFO 7375) goke.r@northeastern.edu"}
ATOM = "{http://www.w3.org/2005/Atom}"


def _valid_date(d):
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d or "")) and d <= date.today().isoformat()


def parse_sec_atom(xml_text):
    """Form D filings -> records. Gate: clean name, valid date not in the future, sec.gov link."""
    kept, rejected = [], 0
    for e in ET.fromstring(xml_text).findall(f"{ATOM}entry"):
        title = (e.findtext(f"{ATOM}title") or "").strip()
        if not re.match(r"^D(/A)? - ", title):
            rejected += 1; continue  # only Form D / D/A filings (the feed also returns DEF 14C and similar)
        name = re.sub(r" \(\d+\).*$", "", re.sub(r"^D(/A)? - ", "", title)).strip()
        d = (e.findtext(f"{ATOM}updated") or "")[:10]
        link_el = e.find(f"{ATOM}link")
        link = link_el.get("href", "") if link_el is not None else ""
        if len(name) < 2 or not _valid_date(d) or "sec.gov" not in link:
            rejected += 1; continue
        kept.append({"record_type": "funding_filing", "company_name": name, "sector": "", "stage": "", "amount_usd": "",
                     "geography": "", "date": d, "investor_names": "", "source": "SEC EDGAR – Form D (live)", "source_url": link})
    return kept, rejected


def live_sec():
    """Fetch today's Form D filings. Returns (records, status_message). Never raises."""
    try:
        r = requests.get(SEC_FEED, headers=SEC_HEADERS, timeout=12)
        r.raise_for_status()
        recs, rej = parse_sec_atom(r.text)
    except Exception:
        return [], "SEC filings: unavailable right now"
    seen, unique = set(), []
    for x in recs:
        key = (x["company_name"].lower(), x["date"])
        if key not in seen:
            seen.add(key); unique.append(x)
    for i, x in enumerate(unique):
        x["record_id"] = f"LIVE-SEC-{i + 1:03d}"
    return unique, f"SEC filings: {len(unique)} passed the quality gate" + (f", {rej} rejected" if rej else "")


# ---------------------------------------------------------------- active investors (from SEC fund filings)
NOT_VC = re.compile(r"timberland|income|real estate|special situations|offshore|value add|credit|mortgage|reit|oil|gas|"
                    r"insurance|municipal|bond|treasury|"
                    r"platform funds|mv funds|cd venture funds|roll up vehicles", re.I)  # last line: SPV platforms that don't name the investor


def fund_parent(name):
    m = re.search(r"a series of (.+?)(?:,|$)", name, re.I)
    base = m.group(1) if m else name
    return re.sub(r",?\s*(L\.?P\.?|LLC|Ltd\.?|SCSp)\.?$", "", base, flags=re.I).strip(" ,")


def active_investors(founder, records, n=6, directory=None):
    """Investment funds that filed a Form D in the last 12 months, ranked by fit with the founder's sector words."""
    keys = [k.strip().lower() for w in (founder.get("sector_keywords", ""), founder.get("keywords", ""))
            for k in re.split(r"[,|]", w) if len(k.strip()) > 2]
    cutoff = (date.today() - timedelta(days=365)).isoformat()
    best = {}
    for r in records:
        nm = r.get("company_name", "")
        if r.get("record_type") != "funding_filing" or not is_fund(nm) or NOT_VC.search(nm) or (r.get("date") or "") < cutoff:
            continue
        parent = fund_parent(nm)
        score = 10 * sum(1 for k in keys if k in nm.lower()) + (3 if re.search(r"venture|angel|seed|vc|innovat|climate", nm, re.I) else 0)
        known = next((d for d in (directory or []) if d.get("sec_filer_name") and d["sec_filer_name"].lower() in nm.lower()), None)
        if known:
            fits = founder.get("sector") in known["sectors"].split(";") or "All" in known["sectors"].split(";")
            if not fits:
                continue  # a known fund whose published focus doesn't match this founder's sector
            score += 8 if "All" not in known["sectors"].split(";") else 2
        item = {"fund": known["name"] if known else parent, "filing_name": nm, "date": r.get("date"),
                "filing_url": r.get("source_url"), "score": score,
                "website": known["url"] if known else "", "focus": known["focus"] if known else "",
                "search_url": "https://www.google.com/search?q=" + quote_plus(f'"{parent}" venture fund official website apply pitch')}
        if parent not in best or (score, item["date"]) > (best[parent]["score"], best[parent]["date"]):
            best[parent] = item
    return sorted(best.values(), key=lambda x: (x["score"], x["date"]), reverse=True)[:n]


def investor_search_url(name):
    return "https://www.google.com/search?q=" + quote_plus(f'"{name}" investor official website apply pitch')


def deal_investors_url(company, d):
    return "https://www.google.com/search?q=" + quote_plus(f'"{company}" funding round investors {str(d)[:4]}')
