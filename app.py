import os
from html import escape

def e(s):  # for HTML blocks
    return escape(str(s or ""))

def md(s):  # for markdown text: keep quotes readable
    return escape(str(s or ""), quote=False)
import streamlit as st
from core import (SECTORS, STAGES, COUNTRIES, InputProblem, load_records, validate, shortlist,
                  build_prompt, call_gemini, check_answer, report_html, money, freshness,
                  live_news, merge_records, load_investors, match_investors, check_link,
                  live_sec, active_investors, investor_search_url, deal_investors_url)
from datetime import date, timedelta

VERSION = "CapitalMatch v1.5 · updated Oct 22, 2026"

st.set_page_config(page_title="CapitalMatch", page_icon="🎯", layout="wide")
try:
    st.set_option("client.showErrorDetails", False)  # never show stack traces to users
except Exception:
    pass

st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1150px}
.hero{background:linear-gradient(135deg,#0b1f3a,#1e3a8a 55%,#0f766e);color:#fff;border-radius:18px;padding:28px 32px;margin-bottom:18px}
.hero h1{color:#fff;margin:0 0 6px;font-size:34px}.hero p{margin:0;opacity:.92;font-size:17px}
.step{background:#f1f5f9;border-radius:14px;padding:14px 16px;height:100%}.step b{color:#0f766e}
.badge{display:inline-block;color:#fff;font-weight:700;font-size:13px;letter-spacing:.05em;padding:6px 14px;border-radius:99px;text-transform:uppercase}
.card{border:1px solid #e2e8f0;border-radius:14px;padding:16px 18px;margin-bottom:12px;background:#fff}
.card .name{font-size:18px;font-weight:700}.card .meta{color:#64748b;font-size:13.5px;margin:2px 0 8px}
.card .angle{color:#115e59;font-size:14px}.score{font-size:30px;font-weight:800;text-align:center;line-height:1}
.score small{display:block;font-size:11px;color:#64748b;font-weight:600;letter-spacing:.06em}
.panel{background:#f1f5f9;border-radius:14px;padding:14px 18px;min-height:90px}
/* Always dark text on the light cards, even when the viewer's browser is in dark mode */
.step,.card,.panel{color:#0f172a !important}.step *,.card *,.panel *{color:inherit}
.card .meta,.score small{color:#64748b !important}.card .angle{color:#115e59 !important}
.card a,.panel a{color:#0f766e !important}
</style>""", unsafe_allow_html=True)

COLORS = {"strong": "#059669", "moderate": "#d97706", "weak": "#dc2626"}
EXAMPLE = {"company_name": "CareBridge", "sector": "Health care & digital health", "keywords": "remote patient follow-up",
           "stage": "Seed", "amount": "1.5M", "city": "Austin", "country": "United States",
           "pitch": "Remote follow-up care that keeps discharged patients from being readmitted.",
           "use_of_funds": "Hire two nurses and launch with three hospital partners", "recent_first": True, "max_matches": 5}


HERE = os.path.dirname(__file__)


@st.cache_data
def get_verified():
    return load_records(os.path.join(HERE, "data", "capitalmatch_data.csv"))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_live_news():
    return live_news()  # never raises; returns ([], message) if feeds are down


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_live_sec():
    return live_sec()  # never raises; returns ([], message) if the SEC feed is down


@st.cache_data
def get_investors():
    return load_investors(os.path.join(HERE, "data", "investor_links", "investor_links.csv"))


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def link_status(url):
    return check_link(url)


def api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")


def fill_example():
    for k, v in EXAMPLE.items():
        st.session_state[k] = v
    st.session_state.pop("result", None)


def clear_form():
    for k in EXAMPLE:
        st.session_state[k] = (None if k in ("sector", "stage", "country") else
                               True if k == "recent_first" else 5 if k == "max_matches" else "")
    st.session_state.pop("result", None)


try:
    verified = get_verified()
except Exception:
    st.error("The funding data could not be loaded right now. Please refresh the page in a minute.")
    st.stop()
live, live_note = get_live_news()
sec_live, sec_note = get_live_sec()
records, n_live = merge_records(verified, live + sec_live)
try:
    investors = get_investors()
except Exception:
    investors = []
st.session_state.setdefault("recent_first", True)
st.session_state.setdefault("max_matches", 5)

# ---------------------------------------------------------------- header (empty state explains itself)
st.markdown("""<div class="hero"><h1>🎯 CapitalMatch</h1>
<p>Find the real funding deals most like yours, so you can target the kind of investors who back companies like yours
instead of wasting months pitching the wrong ones.</p></div>""", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown("<div class='step'><b>1. Tell us about your raise</b><br>Your sector, stage, how much you're raising, and a one-line pitch.</div>", unsafe_allow_html=True)
c2.markdown(f"<div class='step'><b>2. We compare it</b><br>against {len(records)} real funding deals from Crunchbase, SEC filings, and funding news.</div>", unsafe_allow_html=True)
c3.markdown("<div class='step'><b>3. Get your matches</b><br>The deals most like yours, why they fit, who to target, and how to open the conversation.</div>", unsafe_allow_html=True)

newest, n_recent = freshness(records)
with st.sidebar:
    st.header("About the data")
    st.info(f"**Data freshness:** newest record {newest} · {n_recent} records from the last 12 months")
    st.caption(f"**Live update** (refreshed every 6 hours): {n_live} new records added. {live_note}; {sec_note}. "
               "If a live source is down, the app uses the verified snapshot.")
    st.markdown(f"""**{len(records)} funding records** from three public sources:
- **Crunchbase history:** sector, stage, and amount for startups funded up to Dec 2014
- **SEC Form D filings:** companies and funds legally raising money right now (live)
- **Funding news:** recent rounds from TechCrunch and Crunchbase News (live; each story must name a company, a date, and a link to be used)
- **Where to apply:** official application pages of early-stage investors, checked live. CapitalMatch never shows personal contact details.

**Good to know**
- Most comparable deals are historical, so CapitalMatch tells you the *kind* of investor to target rather than naming individual funds.
- Every match links to its source, so you can check it before you pitch.
- Your answers are sent to Google's Gemini to write the analysis. Don't enter confidential information.""")

st.write("")
b1, b2, _ = st.columns([1.3, 1, 4])
b1.button("✨ Try an example", on_click=fill_example, help="Fills the form with a sample health-care founder so you can see a full report.")
b2.button("Clear form", on_click=clear_form)

# ---------------------------------------------------------------- input form (Data Contract: founder profile)
with st.form("founder"):
    st.subheader("Tell us about your raise")
    a, b = st.columns(2)
    company = a.text_input("Company name", key="company_name", placeholder="e.g. CareBridge")
    sector = b.selectbox("Sector", list(SECTORS) + ["Other"], index=None, key="sector",
                         placeholder="Choose the sector that fits best",
                         help="We use this to find deals in the same market as yours.")
    keywords = a.text_input("A few words about what you do (optional)", key="keywords",
                            placeholder="e.g. remote patient follow-up, insurance claims",
                            help="Extra words help us find deals in a niche the sector list doesn't cover.")
    stage = b.selectbox("Stage of this round", STAGES, index=None, key="stage", placeholder="Choose your stage",
                        help="Pre-seed: idea or first product. Seed: early users or revenue. Series A/B: growing and scaling.")
    amount = a.text_input("How much are you raising (USD)?", key="amount", placeholder="e.g. 1.5M or 750K",
                          help="Type a number like 1500000, 1.5M, or 750K.")
    country = b.selectbox("Country", list(COUNTRIES), index=None, key="country", placeholder="Where is your company based?")
    city = a.text_input("City (optional)", key="city", placeholder="e.g. Austin")
    pitch = st.text_area("Your one-line pitch", key="pitch", max_chars=300,
                         placeholder="e.g. Remote follow-up care that keeps discharged patients from being readmitted.",
                         help="One or two sentences. The AI uses this to tell truly similar deals from look-alikes.")
    use_of_funds = st.text_input("What will this round pay for? (optional)", key="use_of_funds",
                                 placeholder="e.g. hire 2 engineers and launch in 3 new cities",
                                 help="Helps the AI suggest how to open the conversation with investors.")
    o1, o2 = st.columns(2)
    recent_first = o1.checkbox("Prioritize recent deals (last 12 months)", key="recent_first",
                               help="Ranks the newest SEC filings and funding news above older historical deals.")
    max_matches = o2.selectbox("How many matches to show", [5, 8], key="max_matches")
    go = st.form_submit_button("Find my matches", type="primary", use_container_width=True)

if go:
    form = {"company_name": company or "", "sector": sector, "keywords": keywords or "", "stage": stage,
            "amount": amount or "", "city": (city or "").strip(), "country": country, "pitch": pitch or "",
            "use_of_funds": (use_of_funds or "").strip(), "max_matches": max_matches or 5,
            "recent_first": bool(recent_first)}
    try:
        problems = []
        if not stage: problems.append("Please choose the stage of this round.")
        if not country: problems.append("Please choose the country your company is based in.")
        try:
            validate(form)
        except InputProblem as e:
            problems = str(e).split("\n") + problems
        if problems:
            raise InputProblem("\n".join(problems))
        form["sector_keywords"] = SECTORS.get(sector, "")
        form["country_code"] = COUNTRIES.get(country, "")
        with st.spinner(f"Comparing your raise with {len(records)} real funding deals…"):
            cands = shortlist(form, records, n=12 if (max_matches or 5) <= 5 else 16, recent_first=bool(recent_first))
        if not cands:
            st.session_state.pop("result", None)
            st.info("We couldn't find any deals similar to yours in the current data. Try a broader sector, or add a few "
                    "keywords that describe what you do, then press **Find my matches** again.")
        else:
            with st.spinner("Our AI analyst is reviewing the 12 closest deals. This usually takes 5 to 15 seconds…"):
                res, err = call_gemini(build_prompt(form, cands), api_key())
                result = check_answer(res, err, cands, max_matches=form["max_matches"])
            st.session_state["result"] = {"form": form, "result": result, "n": len(cands)}
    except InputProblem as e:
        st.session_state.pop("result", None)
        st.error("**A few things need fixing before we can search:**\n\n" + "\n".join(f"- {p}" for p in str(e).split("\n")))
    except Exception:
        st.session_state.pop("result", None)
        st.error("Something went wrong on our side while building your report. Please press **Find my matches** again. "
                 "If it keeps happening, wait a minute and retry.")

# ---------------------------------------------------------------- output (Conductor Brief Part 4, step 4)
if "result" in st.session_state:
    f = st.session_state["result"]["form"]; r = st.session_state["result"]["result"]
    n = st.session_state["result"]["n"]; a = r["analysis"]; q = a["match_quality"]
    st.divider()
    st.subheader(f"Your matches: {f['company_name']}")
    if r["status"] == "fallback":
        st.warning(f"Our AI analyst wasn't available ({r['problem']}), so these results use our matching rules only. "
                   "They're still real deals, but without the written analysis. Press **Find my matches** again in a minute for the full report.")
    st.markdown(f"<span class='badge' style='background:{COLORS[q]}'>{q} match</span>", unsafe_allow_html=True)
    st.caption(f"Data freshness: newest record {newest} · {n_recent} records from the last 12 months"
               + (" · recent deals prioritized" if f.get("recent_first") else ""))
    st.markdown(f"#### {a['summary']}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Closest deals reviewed", n)
    m2.metric("Kept as real matches", len(a["top_matches"]))
    m3.metric("Ruled out by the AI", len(a["rejected"]))

    st.markdown("### Funding deals most like yours")
    if not a["top_matches"]:
        st.info("None of the closest deals were a real match. See *Deals the AI ruled out* below for the reasons.")
    for i, m in enumerate(a["top_matches"], 1):
        s = m["fit_score"]; colr = "#059669" if s >= 75 else "#d97706" if s >= 50 else "#dc2626"
        angle = f"<div class='angle'><b>Investor angle:</b> {e(m['investor_angle'])}</div>" if m.get("investor_angle") else ""
        named = f"<div class='angle'><b>Named investor:</b> {e(m['investor_names'])}</div>" if m.get("investor_names") else ""
        left, right = st.columns([6, 1])
        left.markdown(f"""<div class='card'><div class='name'>{i}. {e(m['company_name'])}</div>
<div class='meta'>{e(m.get('sector') or 'Sector not reported')} · {e(m.get('stage') or 'Stage not reported')} · {money(m.get('amount_usd'))}
{' · ' + e(m['geography']) if m.get('geography') else ''} · {e(m.get('date',''))} · {e(m.get('source',''))}</div>
<div>{e(m['why'])}</div>{angle}{named}<div style='margin-top:6px'><a href='{e(m.get('source_url',''))}' target='_blank'>Check the source ↗</a>
&nbsp;·&nbsp; <a href='{e(deal_investors_url(m['company_name'], m.get('date')))}' target='_blank'>Find who invested ↗</a>
{"&nbsp;·&nbsp; <a href='" + e(investor_search_url(m['investor_names'])) + "' target='_blank'>Investor's website ↗</a>" if m.get('investor_names') else ""}</div></div>""",
                      unsafe_allow_html=True)
        right.markdown(f"<div class='score' style='color:{colr};margin-top:22px'>{s}<small>FIT / 100</small></div>", unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    p1.markdown(f"<div class='panel'><b>Who to target</b><br>{e(a.get('target_investor_profile') or 'Not available for this run.')}</div>", unsafe_allow_html=True)
    p2.markdown(f"<div class='panel'><b>How to open the conversation</b><br>{e(a.get('outreach_tip') or 'Not available for this run.')}</div>", unsafe_allow_html=True)
    active = active_investors(f, records, directory=investors)
    st.markdown("### Investment funds active right now")
    if active:
        st.caption("These funds filed a Form D with the SEC in the last 12 months, which means they are raising or "
                   "deploying money now. CapitalMatch does not know each fund's focus, so check its website before you pitch.")
        for inv in active:
            st.markdown(f"<div class='card'><div class='name'>{e(inv['fund'])}</div>"
                        f"<div class='meta'>Filed {e(inv['date'])} · {e(inv['filing_name'])}</div>"
                        + (f"<div class='angle'>{e(inv['focus'])}</div>" if inv.get('focus') else "")
                        + f"<a href='{e(inv['filing_url'])}' target='_blank'>View SEC filing ↗</a> &nbsp;·&nbsp; "
                        + (f"<a href='{e(inv['website'])}' target='_blank'>Official website ↗</a></div>" if inv.get('website')
                           else f"<a href='{e(inv['search_url'])}' target='_blank'>Find website &amp; pitch page ↗</a></div>"),
                        unsafe_allow_html=True)
    else:
        st.caption("No recently active funds found in the current data. Try again later; this list refreshes every 6 hours.")
    if a["rejected"]:
        with st.expander(f"Deals the AI ruled out ({len(a['rejected'])}) and why"):
            for x in a["rejected"]:
                st.markdown(f"- **{md(x['company_name'])}**: {md(x['reason'])}")
    if a.get("data_gaps"):
        st.caption(f"What's missing from the data: {a['data_gaps']}")

    st.markdown("### Recent funding activity in your sector (last 90 days)")
    cutoff = (date.today() - timedelta(days=90)).isoformat()
    recent_recs = [x for x in records if (x.get("date") or "") >= cutoff]
    recent = [x for x in shortlist(f, recent_recs, n=20, recent_first=True) if "sector" in x["rule_reasons"]][:5]
    if recent:
        for x in recent:
            inv_txt = f" · led by **{md(x['investor_names'])}**" if x.get("investor_names") else ""
            st.markdown(f"- **{md(x['company_name'])}** · {md(x.get('stage') or 'stage not reported')} · {money(x.get('amount_usd'))} · "
                        f"{md(x['date'])}{inv_txt} · [read the source]({x.get('source_url','')})")
    else:
        st.caption("No funding announcements in your sector in the last 90 days yet. Live news is refreshed every 6 hours.")

    st.markdown("### Where to apply")
    st.caption("Official public application pages from early-stage investors who fund your stage, sector, and region. "
               "Links are checked live. CapitalMatch does not share personal contact details.")
    apply_list = match_investors(f, investors) if investors else []
    shown = 0
    for inv in apply_list:
        status = link_status(inv["url"])
        if status == "down":
            continue
        badge = "✅ link checked today" if status == "ok" else "ℹ️ couldn't auto-check, opens the official site"
        st.markdown(f"- **[{md(inv['name'])}]({inv['url']})** ({md(inv['type'])}): {md(inv['focus'])}. "
                    f"[{md(inv['link_label'])} ↗]({inv['url']}) · {badge}")
        shown += 1
    if not shown:
        st.caption("No matching application pages right now. Try a different stage or sector.")
    st.session_state["result"]["apply"] = [i for i in apply_list if link_status(i["url"]) != "down"]
    st.session_state["result"]["recent"] = recent
    st.download_button("⬇ Download this report",
                       report_html(f, r, len(records), st.session_state["result"].get("apply", []),
                                   st.session_state["result"].get("recent", []), active),
                       file_name=f"CapitalMatch_{f['company_name'].replace(' ', '_')}.html", mime="text/html")
    st.caption("Fit scores compare your raise with each deal's sector, stage, size, location, and timing, then the AI adjusts them "
               "after reading your pitch. Always check the source before you pitch.")

st.caption(VERSION)
