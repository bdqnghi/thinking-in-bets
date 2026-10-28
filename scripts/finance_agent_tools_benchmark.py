"""Tool-Augmented Vals Finance Agent Benchmark (`vals-ai/finance_agent_benchmark`) with Live SEC EDGAR Retrieval.

Implements the official 4-tool Vals Finance Agent harness (`third_party/finance-agent/finance_agent/tools.py`):
  1. `edgar_search`: Live SEC EDGAR full-text search (`https://efts.sec.gov/LATEST/search-index`) +
     company submissions (`https://data.sec.gov/submissions/CIK{cik:010d}.json`) + ticker resolution
     (`https://www.sec.gov/files/company_tickers.json`), capped at `MAX_END_DATE = "2025-04-07"`.
  2. `web_search`: Live search across SEC EDGAR filings, company index, and Wikipedia.
  3. `parse_html_page`: Fetches HTML from `sec.gov` or web URLs, parses plain text via BeautifulSoup,
     and stores the full filing text in `state[key]`.
  4. `retrieve_information`: Substitutes `{{key_name}}` (with optional `input_character_ranges` or smart
     keyword-window extraction for very large 10-Ks) and queries the LLM to extract exact figures/tables.
  5. `submit_final_result`: Submits the final answer.
     - In `stock`: exits immediately upon first `submit_final_result` call (official Vals behavior).
     - In `cb_draftfirst`: intercepts the first `submit_final_result` call, requires a pre-submission
       Bet Card (`[CLAIM_BOOK]` with `p_i in [0,1]` and SEC filing evidence) and optional tool
       re-verification of any `p_i < 0.85` claim before accepting the final `submit_final_result`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup
import openai

from scripts.finance_agent_benchmark import grade_finance_response, load_finance_tasks
from scripts.hle_benchmark import get_api_key
from fsa.redo import quota  # P0.3 shared per-model slots

# P0.1: agent reasoning effort for every agent call (set from --effort in main; "low" = the published runs).
EFFORT = "low"


def _quota_create(client, **kw):
    """client.chat.completions.create(**kw) holding one shared quota slot for kw['model'] (P0.3)."""
    with quota.slot(kw["model"]):
        return client.chat.completions.create(**kw)

MAX_END_DATE = "2025-04-07"
SEC_USER_AGENT = "ValsAI/antoine@vals.ai"

INSTRUCTIONS_PROMPT = """You are a financial agent. You are given a question and you need to answer it using the tools provided.
You will not be able to interact with the user or ask clarifications, you must answer the question only based on the information provided.

You should answer all questions as if the current date is April 07, 2025.

You will have access to a data storage system. You can use this system to store parsed contents of HTML pages retrieved from the web or SEC EDGAR (`parse_html_page`).
You can then use the `retrieve_information` tool to answer questions or gather information from the stored documents using LLM-based prompts (referencing stored keys as `{{key_name}}`).
This data storage system is designed to help you avoid context window issues.

When you have the final answer, you should call the `submit_final_result` tool with it. Your submission will not be processed unless you call this tool.

You should include any necessary step-by-step reasoning, justification, calculations, or explanation in your answer. You will be evaluated both on the accuracy of the final answer, and the correctness of the supporting logic.

When possible, please provide any calculated answers to at least two decimal places (e.g. 18.78% rather than 19%). Please do not round intermediate steps in any calculations - you should only round your final answer.

At the end of your answer, you should provide your sources in a dictionary with the following format:
{
    "sources": [
        {
            "url": "https://www.sec.gov/Archives/edgar/data/...",
            "name": "Name of the filing / source"
        }
    ]
}

Question:
"""

_TICKER_LOCK = threading.Lock()
_TICKER_CACHE: dict[str, dict[str, Any]] | None = None
_SEC_RATE_LOCK = threading.Lock()
_LAST_SEC_CALL = 0.0


def _sec_get(url: str, timeout: int = 30) -> bytes:
    """Rate-limited GET against sec.gov (respects SEC 10 req/s policy)."""
    global _LAST_SEC_CALL
    with _SEC_RATE_LOCK:
        now = time.time()
        wait = 0.12 - (now - _LAST_SEC_CALL)
        if wait > 0:
            time.sleep(wait)
        _LAST_SEC_CALL = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def get_ticker_map() -> dict[str, dict[str, Any]]:
    global _TICKER_CACHE
    with _TICKER_LOCK:
        if _TICKER_CACHE is not None:
            return _TICKER_CACHE
        try:
            raw = _sec_get("https://www.sec.gov/files/company_tickers.json", timeout=20)
            data = json.loads(raw.decode("utf-8"))
            mp: dict[str, dict[str, Any]] = {}
            for _, item in data.items():
                ticker = str(item.get("ticker", "")).upper()
                cik_str = f"{int(item.get('cik_str', 0)):010d}"
                title = str(item.get("title", ""))
                if ticker:
                    mp[ticker] = {"cik": cik_str, "ticker": ticker, "title": title}
            _TICKER_CACHE = mp
        except Exception:
            _TICKER_CACHE = {}
        return _TICKER_CACHE


def resolve_ciks_from_query(query: str, explicit_ciks: list[str] | None = None) -> list[str]:
    ciks: list[str] = []
    if explicit_ciks:
        for c in explicit_ciks:
            digits = re.sub(r"\D", "", str(c))
            if digits:
                ciks.append(f"{int(digits):010d}")
    mp = get_ticker_map()
    if not mp:
        return ciks
    # Check uppercase tokens that match tickers
    tokens = re.findall(r"\b[A-Z]{2,5}\b", query)
    stop_upper = {"SEC", "EDGAR", "GAAP", "NON", "EBITDA", "CAGR", "EPS", "BPS", "USD", "CEO", "CFO", "FY", "Q1", "Q2", "Q3", "Q4", "AND", "NOT", "THE", "FOR", "INC", "CORP", "LLC", "LTD", "NYSE", "NASDAQ", "ARPU"}
    for tok in tokens:
        if tok not in stop_upper and tok in mp:
            c = mp[tok]["cik"]
            if c not in ciks:
                ciks.append(c)
    # Also check company title substring match for longer words
    q_low = query.lower()
    alias_map = {
        "us steel": "0001163302",
        "united states steel": "0001163302",
        "netflix": "0001065280",
        "tjx": "0000109198",
        "amd": "0000002488",
        "advanced micro devices": "0000002488",
        "bbsi": "0000902791",
        "barrett business": "0000902791",
    }
    for name_sub, cik_val in alias_map.items():
        if name_sub in q_low and cik_val not in ciks:
            ciks.append(cik_val)
    for t_info in mp.values():
        title_low = t_info["title"].lower()
        # Strip INC / CORP / CO / DE
        clean_title = re.sub(r"\b(inc|corp|corporation|co|company|ltd|plc|llc|de|new)\b\.?", "", title_low).strip()
        if len(clean_title) >= 5 and clean_title in q_low:
            if t_info["cik"] not in ciks:
                ciks.append(t_info["cik"])
    return ciks[:3]


def _efts_query(
    q: str,
    start_date: str,
    end_date: str,
    form_types: list[str] | None = None,
    ciks: list[str] | None = None,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    clean_q = re.sub(r"[^\w\s\"]+", " ", q)
    clean_q = re.sub(r"\b(AND|OR|NOT)\b", " ", clean_q).strip()
    if not clean_q:
        clean_q = "financial"
    params: dict[str, str] = {
        "q": clean_q,
        "dateRange": "custom",
        "startdt": max("1995-01-01", start_date[:10]) if start_date else "2018-01-01",
        "enddt": min(MAX_END_DATE, end_date[:10]) if end_date else MAX_END_DATE,
    }
    if form_types:
        params["forms"] = ",".join(str(f).strip() for f in form_types if str(f).strip())
    if ciks:
        params["ciks"] = ",".join(f"{int(re.sub(r'\\D', '', str(c))):010d}" for c in ciks if re.sub(r"\D", "", str(c)))
    url = "https://efts.sec.gov/LATEST/search-index?" + urllib.parse.urlencode(params)
    raw = _sec_get(url, timeout=25)
    data = json.loads(raw.decode("utf-8", errors="ignore"))
    hits = (data.get("hits") or {}).get("hits") or []
    results: list[dict[str, Any]] = []
    for h in hits[:top_n]:
        src = h.get("_source") or {}
        doc_id = str(h.get("_id") or "")
        adsh = str(src.get("adsh") or (doc_id.split(":")[0] if ":" in doc_id else ""))
        doc_name = doc_id.split(":", 1)[1] if ":" in doc_id else ""
        cik_list = src.get("ciks") or []
        cik_str = cik_list[0] if cik_list else ""
        cik_int = int(cik_str) if cik_str.isdigit() else 0
        adsh_nodash = adsh.replace("-", "")
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh_nodash}/{doc_name}"
            if (cik_int and adsh_nodash and doc_name)
            else ""
        )
        results.append(
            {
                "company": (src.get("display_names") or [""])[0],
                "cik": cik_str,
                "form_type": src.get("form") or src.get("file_type"),
                "root_forms": src.get("root_forms"),
                "filing_date": src.get("file_date"),
                "period_ending": src.get("period_ending"),
                "accession_number": adsh,
                "document_name": doc_name,
                "description": src.get("file_description"),
                "url": filing_url,
            }
        )
    return results


def _company_recent_filings(cik: str, form_types: list[str] | None, start_date: str, end_date: str, limit: int = 15) -> list[dict[str, Any]]:
    cik_padded = f"{int(re.sub(r'\\D', '', cik)):010d}"
    cik_int = int(cik_padded)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    try:
        raw = _sec_get(url, timeout=20)
        data = json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return []
    company_name = data.get("name", "")
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    f_dates = recent.get("filingDate") or []
    r_dates = recent.get("reportDate") or []
    adshs = recent.get("accessionNumber") or []
    p_docs = recent.get("primaryDocument") or []
    descs = recent.get("primaryDocDescription") or []

    allowed_forms = {f.upper() for f in form_types} if form_types else {"10-K", "10-Q", "8-K", "DEF 14A", "DEFM14A", "20-F"}
    out: list[dict[str, Any]] = []
    for i in range(min(len(forms), len(f_dates), len(adshs), len(p_docs))):
        fm = str(forms[i]).upper()
        fdt = str(f_dates[i])
        if fdt > min(MAX_END_DATE, end_date[:10]):
            continue
        if start_date and fdt < start_date[:10]:
            continue
        if allowed_forms and fm not in allowed_forms:
            continue
        adsh_nodash = str(adshs[i]).replace("-", "")
        doc = str(p_docs[i])
        if not doc:
            continue
        out.append(
            {
                "company": f"{company_name} (CIK {cik_padded})",
                "cik": cik_padded,
                "form_type": fm,
                "filing_date": fdt,
                "period_ending": str(r_dates[i]) if i < len(r_dates) else "",
                "accession_number": str(adshs[i]),
                "document_name": doc,
                "description": str(descs[i]) if i < len(descs) else fm,
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{adsh_nodash}/{doc}",
            }
        )
        if len(out) >= limit:
            break
    return out


def exec_edgar_search(args: dict[str, Any]) -> str:
    query = str(args.get("search_query") or "")
    form_types = args.get("form_types")
    if isinstance(form_types, str):
        form_types = [form_types]
    explicit_ciks = args.get("ciks")
    if isinstance(explicit_ciks, str):
        explicit_ciks = [explicit_ciks]
    start_date = str(args.get("start_date") or "2018-01-01")
    end_date = min(MAX_END_DATE, str(args.get("end_date") or MAX_END_DATE))
    top_n = min(40, int(args.get("top_n_results") or 20))

    resolved_ciks = resolve_ciks_from_query(query, explicit_ciks)
    combined: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    # 1. If we resolved a CIK, strip ticker tokens from query and search EFTS scoped to that CIK
    if resolved_ciks:
        mp = get_ticker_map()
        q_stripped = query
        for tok in re.findall(r"\b[A-Z]{2,5}\b", query):
            if tok in mp and mp[tok]["cik"] in resolved_ciks:
                q_stripped = re.sub(rf"\b{re.escape(tok)}\b", " ", q_stripped)
        try:
            for item in _efts_query(q_stripped, start_date, end_date, form_types, resolved_ciks, top_n=top_n):
                if item["url"] and item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    combined.append(item)
        except Exception:
            pass
        # Also attach recent official 10-K/10-Q/8-K/DEF 14A filings from submissions API
        for cik in resolved_ciks[:2]:
            for item in _company_recent_filings(cik, form_types, start_date, end_date, limit=10):
                if item["url"] and item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    combined.append(item)

    # 2. General full-text EFTS search
    if len(combined) < top_n:
        try:
            for item in _efts_query(query, start_date, end_date, form_types, explicit_ciks, top_n=top_n):
                if item["url"] and item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    combined.append(item)
                    if len(combined) >= top_n:
                        break
        except Exception as e:
            if not combined:
                return f"EDGAR search error: {e}"

    return json.dumps(combined[:top_n], indent=2)


def exec_web_search(args: dict[str, Any]) -> str:
    query = str(args.get("search_query") or "")
    start_date = str(args.get("start_date") or "2018-01-01")
    end_date = min(MAX_END_DATE, str(args.get("end_date") or MAX_END_DATE))
    n_res = min(15, int(args.get("number_of_results") or 10))

    results: list[dict[str, Any]] = []
    # 1. Include top SEC EDGAR hits (with direct sec.gov URLs)
    try:
        edgar_items = json.loads(
            exec_edgar_search(
                {
                    "search_query": query,
                    "start_date": start_date,
                    "end_date": end_date,
                    "top_n_results": n_res,
                }
            )
        )
        if isinstance(edgar_items, list):
            for it in edgar_items[:n_res]:
                results.append(
                    {
                        "url": it.get("url", ""),
                        "title": f"{it.get('company', '')} - Form {it.get('form_type', '')} ({it.get('filing_date', '')}, Period: {it.get('period_ending', '')})",
                        "excerpt": f"SEC EDGAR {it.get('form_type', '')} filing ({it.get('document_name', '')}): {it.get('description', '')}. Use parse_html_page on this url to read the full filing.",
                    }
                )
    except Exception:
        pass

    # 2. Query Wikipedia API for background / M&A / corporate facts
    try:
        w_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3}
        )
        req = urllib.request.Request(w_url, headers={"User-Agent": SEC_USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as r:
            wdata = json.loads(r.read().decode("utf-8", errors="ignore"))
        for s in (wdata.get("query") or {}).get("search") or []:
            title = s.get("title", "")
            snippet = BeautifulSoup(s.get("snippet", ""), "html.parser").get_text(" ", strip=True)
            results.append(
                {
                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                    "title": f"Wikipedia: {title}",
                    "excerpt": snippet,
                }
            )
    except Exception:
        pass

    return json.dumps(results[:n_res], indent=2)


def exec_parse_html_page(args: dict[str, Any], state: dict[str, str]) -> str:
    url = str(args.get("url") or "").strip()
    key = str(args.get("key") or "").strip()
    if not url or not key:
        return "ERROR: Both 'url' and 'key' are required."
    try:
        raw = _sec_get(url, timeout=45)
        html_content = raw.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()
        text = soup.get_text("\n")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        if not clean_text:
            return "ERROR: Parsed HTML output was empty."
        existed = key in state
        state[key] = clean_text
        msg = ""
        if existed:
            msg += "WARNING: The key already exists in the data storage. The new result overwrites the old one.\n"
        msg += (
            f"SUCCESS: The result has been saved to the data storage under the key: {key} "
            f"(length: {len(clean_text):,} characters).\n"
            f"The data_storage currently contains the following keys:\n"
            + "\n".join(f"- {k} ({len(v):,} chars)" for k, v in state.items())
        )
        return msg
    except Exception as e:
        return f"ERROR parsing HTML page '{url}': {e}"


def _smart_window_extract(doc: str, prompt: str, max_chars: int = 140000) -> str:
    """If a stored 10-K exceeds max_chars and no character range was given, extract relevant windows."""
    if len(doc) <= max_chars:
        return doc
    # Extract keywords from prompt (excluding placeholder names and common stop words)
    clean_p = re.sub(r"\{\{[^{}]+\}\}", " ", prompt)
    words = [
        w.lower()
        for w in re.findall(r"[A-Za-z0-9]{3,}", clean_p)
        if w.lower()
        not in {
            "the", "and", "for", "from", "with", "that", "this", "what", "how", "who",
            "extract", "analyze", "following", "filing", "report", "document", "company",
            "table", "data", "numbers", "figures", "information", "find", "all", "any",
        }
    ]
    if not words:
        return doc[:max_chars]

    # Split doc into 4000-char chunks with 500-char overlap and score by keyword density
    step = 3500
    window = 4000
    scored: list[tuple[int, int, str]] = []
    doc_low = doc.lower()
    for start in range(0, len(doc), step):
        chunk_low = doc_low[start : start + window]
        score = sum(chunk_low.count(w) * (3 if len(w) >= 6 else 1) for w in set(words))
        if score > 0:
            scored.append((score, start, doc[start : start + window]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top_chunks = sorted(scored[:28], key=lambda x: x[1])
    if not top_chunks:
        return doc[:max_chars]
    assembled = [f"[Document header excerpt (0..12000 of {len(doc):,} chars)]:\n{doc[:12000]}"]
    used = 12000
    for sc, st, ch in top_chunks:
        if used + len(ch) > max_chars:
            break
        assembled.append(f"\n--- [Excerpt at chars {st}..{st+len(ch)} (relevance={sc})] ---\n{ch}")
        used += len(ch)
    return "\n".join(assembled)


def exec_retrieve_information(
    args: dict[str, Any],
    state: dict[str, str],
    base_url: str,
    token: str,
    model: str,
) -> str:
    prompt = str(args.get("prompt") or "")
    ranges = args.get("input_character_ranges") or []
    keys = re.findall(r"\{\{([^{}]+)\}\}", prompt)
    if not keys:
        return (
            "ERROR: Your prompt must include at least one key from data storage in the format {{key_name}}. "
            f"Available keys: {', '.join(state.keys()) or '(none)'}"
        )
    ranges_dict: dict[str, tuple[int, int]] = {}
    if isinstance(ranges, list):
        for r in ranges:
            if isinstance(r, dict) and "key" in r and "start" in r and "end" in r:
                ranges_dict[str(r["key"])] = (int(r["start"]), int(r["end"]))

    for k in keys:
        if k not in state:
            return f"ERROR: Key '{k}' not found in data storage. Available keys: {', '.join(state.keys()) or '(none)'}"

    formatted_prompt = prompt
    for k in set(keys):
        content = state[k]
        if k in ranges_dict:
            s_idx, e_idx = ranges_dict[k]
            sub = content[max(0, s_idx) : min(len(content), e_idx)]
            if len(sub) > 140000:
                sub = sub[:140000]
        else:
            sub = _smart_window_extract(content, prompt, max_chars=140000)
        formatted_prompt = formatted_prompt.replace(f"{{{{{k}}}}}", sub)

    client = openai.OpenAI(api_key=token, base_url=base_url, timeout=180.0, max_retries=2)
    resp = _quota_create(client, 
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise financial document extraction assistant. Extract exact figures, tables, dates, footnotes, and verbatim facts from the provided filing text.",
            },
            {"role": "user", "content": formatted_prompt},
        ],
        max_tokens=3072,
        extra_body={"reasoning_effort": EFFORT},
    )
    return resp.choices[0].message.content or ""


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "edgar_search",
            "description": (
                "Search the SEC EDGAR database for company filings (10-K, 10-Q, 8-K, DEF 14A, etc.) up to April 07, 2025. "
                "Returns a list of matching filings with metadata and direct sec.gov HTML URLs (`url`) that you can pass to `parse_html_page`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Search query (company name/ticker and financial terms, e.g. 'NFLX average revenue per paying membership 10-K').",
                    },
                    "form_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of SEC form types, e.g. ['10-K', '10-Q', '8-K', 'DEF 14A'].",
                    },
                    "ciks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of company CIKs.",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Optional start date in YYYY-MM-DD format (default '2018-01-01').",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional end date in YYYY-MM-DD format (max '2025-04-07').",
                    },
                    "top_n_results": {
                        "type": "integer",
                        "description": "Max results to return (default 15).",
                    },
                },
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search for public financial information and SEC filing links up to April 07, 2025.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {"type": "string", "description": "The query to search for."},
                    "start_date": {"type": "string", "description": "Optional start date YYYY-MM-DD."},
                    "end_date": {"type": "string", "description": "Optional end date YYYY-MM-DD (max 2025-04-07)."},
                    "number_of_results": {"type": "integer", "description": "Number of results (default 10)."},
                },
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_html_page",
            "description": (
                "Retrieve an HTML page or SEC EDGAR filing URL (`https://www.sec.gov/Archives/edgar/data/...`), "
                "parse it to plain text, and save it in data_storage under `key`. "
                "Then use `retrieve_information` with `{{key}}` to query the stored document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the SEC filing or HTML page to parse."},
                    "key": {"type": "string", "description": "The key name under which to store the parsed document."},
                },
                "required": ["url", "key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_information",
            "description": (
                "Query one or more documents previously saved via `parse_html_page`. "
                "IMPORTANT: Your `prompt` MUST include at least one key in double braces, e.g. "
                "'Extract the 2022, 2023, and 2024 revenue and operating income tables from {{company_10k}}'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Prompt sent to the LLM with `{{key_name}}` placeholders replaced by the stored document text.",
                    },
                    "input_character_ranges": {
                        "type": "array",
                        "description": "Optional character ranges [{'key': '...', 'start': 0, 'end': 50000}].",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                            },
                            "required": ["key", "start", "end"],
                        },
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_final_result",
            "description": (
                "Submit your final answer, including step-by-step calculations, reasoning, and sources. "
                "You MUST call this tool to finish."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "final_result": {
                        "type": "string",
                        "description": "Your complete final financial answer, supporting calculations, and sources.",
                    }
                },
                "required": ["final_result"],
            },
        },
    },
]


# #1 draft-first ladder (PARALLEL_REVIEWER_EXPERIMENTS.md #1; REVIEWER_EXPS_PREREGISTRATION.md A3). Like
# `cb_draftfirst`, every gated arm intercepts the FIRST `submit_final_result` call (the draft) with one tool message,
# lets the agent keep working with the same tools, and accepts the second submission. Only the gate text differs.
CB_GATE = (
    "DRAFT RECORDED (Thinking-in-Bets `cb_draftfirst` Pre-Submission Audit):\n"
    "Before your final submission is locked, write a compact `[CLAIM_BOOK]` auditing every "
    "quantitative figure, growth rate / basis-point calculation, date/period, and qualitative claim in your draft:\n"
    "  `C1: <specific metric/period/claim> | evidence: <exact SEC filing key/section or calculation check> | p=0.XX`\n"
    "Adjudication Rule:\n"
    "- Check whether any claim has `p < 0.85`, relies on unverified memory instead of a parsed SEC filing in `data_storage`, "
    "or has a unit / rounding / basis-point discrepancy.\n"
    "- If any sub-question or figure needs verification from a filing, call `retrieve_information` or `edgar_search` now.\n"
    "- Then call `submit_final_result` a second time with your verified, complete final answer."
)
DF_GATES = {
    "cb_draftfirst": CB_GATE,
    "review_only": (
        "DRAFT RECORDED (Pre-Submission Review):\n"
        "Before your final submission is locked, review your draft for errors: every quantitative figure, growth rate / "
        "basis-point calculation, date/period, and qualitative claim.\n"
        "- If any sub-question or figure needs verification from a filing, call `retrieve_information` or `edgar_search` now.\n"
        "- Then call `submit_final_result` a second time with your reviewed, complete final answer."
    ),
    "self_refine": (
        "DRAFT RECORDED (Self-Refine, step 1 of 2):\n"
        "Before your final submission is locked, write concise, specific FEEDBACK on your draft as a plain message (no "
        "tool call): list each wrong or unsupported figure, calculation, period, or missing sub-question, and how to "
        "fix it. Do not submit yet."
    ),
    "cove": (
        "DRAFT RECORDED (Chain-of-Verification):\n"
        "Before your final submission is locked:\n"
        "1. List 3-5 verification questions that check the key figures, periods and calculations in your draft.\n"
        "2. Answer each question independently by re-checking the filings with `retrieve_information` or `edgar_search` "
        "(not from your draft).\n"
        "3. If the answers reveal an error, correct the draft.\n"
        "- Then call `submit_final_result` a second time with your verified, complete final answer."
    ),
}
SELF_REFINE_STEP2 = (
    "Self-Refine, step 2 of 2: using your feedback, fix the draft (call `retrieve_information` or `edgar_search` if a "
    "figure needs checking), then call `submit_final_result` again with the improved, complete final answer."
)
# sc_matched: k independent stock runs (sample 1 = the `stock` row of the same task/effort), then one universal
# self-consistency selection call. Published cb_draftfirst / stock token ratio is 1.19-1.24 (results/finance_agent_tools,
# 50 tasks x 4 models), so no integer k >= 2 matches it within 10%; k = 2 is the smallest non-degenerate value and gives
# sc_matched MORE compute than cb_draftfirst (pre-registered, A3).
SC_K = 2
USC_PROMPT = (
    "{instructions}{question}\n\n---\nHere are {k} candidate final answers to this question, produced independently:\n\n"
    "{cands}\n\n---\nSelect the candidate whose final figures and conclusions agree most with the other candidates "
    "(the majority answer). Reply with only: SELECTED: <candidate number>"
)
FT_ARMS_ALL = ("stock", "cb_draftfirst", "review_only", "self_refine", "cove", "sc_matched")


def ft_prompts_sha256() -> str:
    import hashlib
    blob = json.dumps(dict(instr=INSTRUCTIONS_PROMPT, gates=DF_GATES, step2=SELF_REFINE_STEP2, k=SC_K, usc=USC_PROMPT),
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def run_tool_agent_task(
    task: dict[str, Any],
    arm: str,
    model: str,
    judge_model: str,
    base_url: str,
    token: str,
    max_turns: int = 25,
) -> dict[str, Any]:
    client = openai.OpenAI(api_key=token, base_url=base_url, timeout=180.0, max_retries=3)
    state: dict[str, str] = {}
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": INSTRUCTIONS_PROMPT + task["question"]},
    ]

    draft_submitted = False
    refine_stage = 0  # self_refine: 1 = feedback requested, 2 = revision requested
    n_llm_calls = 0
    last_draft = ""
    final_answer = ""
    tool_calls_log: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for turn in range(max_turns):
        if turn == max_turns - 2 and not final_answer:
            messages.append(
                {
                    "role": "user",
                    "content": "You are approaching the turn limit. Please synthesize all evidence gathered so far and call `submit_final_result` now.",
                }
            )
        try:
            resp = _quota_create(client, 
                model=model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                max_tokens=4096,
                extra_body={"reasoning_effort": EFFORT},
            )
        except Exception as e:
            time.sleep(2.0)
            if turn >= max_turns - 1:
                break
            continue

        n_llm_calls += 1
        if resp.usage:
            total_prompt_tokens += resp.usage.prompt_tokens or 0
            total_completion_tokens += resp.usage.completion_tokens or 0

        msg = resp.choices[0].message
        assistant_dict: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_dict)

        if not msg.tool_calls:
            if arm == "self_refine" and refine_stage == 1 and msg.content:
                refine_stage = 2  # the feedback message: ask for the revision
                messages.append({"role": "user", "content": SELF_REFINE_STEP2})
                continue
            # Official Vals Finance Agent behavior: inject "Continue." if no tool call was made
            if msg.content and len(msg.content) > 200:
                last_draft = msg.content
            messages.append({"role": "user", "content": "Continue. Remember to call `submit_final_result` when ready."})
            continue

        done = False
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except Exception:
                fn_args = {}

            tool_calls_log.append({"turn": turn, "tool": fn_name, "args": fn_args})

            if fn_name == "submit_final_result":
                submitted_text = str(fn_args.get("final_result") or msg.content or "")
                if submitted_text:
                    last_draft = submitted_text
                if arm in DF_GATES and not draft_submitted and turn < max_turns - 2:
                    draft_submitted = True
                    gate_msg = DF_GATES[arm]
                    if arm == "self_refine":
                        refine_stage = 1
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": gate_msg})
                else:
                    final_answer = submitted_text
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Submission accepted."})
                    done = True
                    break
            elif fn_name == "edgar_search":
                out = exec_edgar_search(fn_args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            elif fn_name == "web_search":
                out = exec_web_search(fn_args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            elif fn_name == "parse_html_page":
                out = exec_parse_html_page(fn_args, state)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            elif fn_name == "retrieve_information":
                out = exec_retrieve_information(fn_args, state, base_url, token, model)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            else:
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"Unknown tool: {fn_name}"})

        if done:
            break

    if not final_answer:
        transcript_parts: list[str] = []
        for m in messages[1:]:
            role = m.get("role", "")
            content_str = str(m.get("content") or "").strip()
            if role == "assistant" and content_str:
                transcript_parts.append(f"[Agent Note]:\n{content_str[:4000]}")
            elif role == "tool" and content_str:
                transcript_parts.append(f"[Tool Output]:\n{content_str[:6000]}")
        for k, v in list(state.items())[:4]:
            transcript_parts.append(
                f"[Stored Filing {k} excerpt]:\n{_smart_window_extract(v, task['question'], max_chars=20000)}"
            )
        joined_notes = "\n\n".join(transcript_parts)[-120000:]
        try:
            synth_resp = _quota_create(client, 
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{INSTRUCTIONS_PROMPT}{task['question']}\n\n"
                            f"---\n"
                            f"### GATHERED SEC FILING EVIDENCE & RESEARCH NOTES:\n"
                            f"{joined_notes}\n\n"
                            f"---\n"
                            f"Synthesize all SEC filing evidence, numbers, and calculations above into a complete, "
                            f"well-structured final financial answer now (write the full answer directly in your response)."
                        ),
                    }
                ],
                max_tokens=4096,
                extra_body={"reasoning_effort": EFFORT},
            )
            final_answer = (synth_resp.choices[0].message.content or "").strip() or last_draft
            if synth_resp.usage:
                total_prompt_tokens += synth_resp.usage.prompt_tokens or 0
                total_completion_tokens += synth_resp.usage.completion_tokens or 0
        except Exception:
            final_answer = last_draft

    grade = grade_finance_response(task, final_answer, base_url, token, judge_model)
    return {
        "id": task["id"],
        "question_type": task["question_type"],
        "arm": arm,
        "model": model,
        "effort": EFFORT,
        "response": final_answer,
        "grade": grade,
        "score": grade["score"],
        "passed": grade["passed"],
        "n_tool_calls": len(tool_calls_log),
        "stored_keys": list(state.keys()),
        "tool_calls": tool_calls_log,
        "usage": {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens},
        "n_llm_calls": n_llm_calls,
        "draft_gated": draft_submitted,
        "refine_stage": refine_stage if arm == "self_refine" else None,
    }


def run_sc_matched(task, stock_row, model, judge_model, base_url, token, max_turns) -> dict[str, Any]:
    """sc_matched: the stock row + SC_K - 1 fresh stock runs, then one universal self-consistency selection."""
    samples = [stock_row] + [run_tool_agent_task(task, "stock", model, judge_model, base_url, token, max_turns)
                             for _ in range(SC_K - 1)]
    cands = "\n\n".join(f"[Candidate {i + 1}]\n{(r.get('response') or '')[:12000]}" for i, r in enumerate(samples))
    client = openai.OpenAI(api_key=token, base_url=base_url, timeout=180.0, max_retries=3)
    sel, usage_sel, raw = 0, {"prompt_tokens": 0, "completion_tokens": 0}, ""
    try:
        resp = _quota_create(client, model=model, messages=[{"role": "user", "content": USC_PROMPT.format(
            instructions=INSTRUCTIONS_PROMPT, question=task["question"], k=len(samples), cands=cands)}],
            max_tokens=2048, extra_body={"reasoning_effort": EFFORT})
        raw = resp.choices[0].message.content or ""
        m = re.search(r"SELECTED:\s*(\d+)", raw)
        if m and 1 <= int(m.group(1)) <= len(samples):
            sel = int(m.group(1)) - 1
        if resp.usage:
            usage_sel = {"prompt_tokens": resp.usage.prompt_tokens or 0,
                         "completion_tokens": resp.usage.completion_tokens or 0}
    except Exception as exc:  # selection failed: fall back to sample 1 (the stock row), recorded
        raw = f"selection failed: {exc}"[:300]
    final = samples[sel].get("response") or ""
    grade = samples[sel]["grade"] if sel == 0 or samples[sel].get("grade") else \
        grade_finance_response(task, final, base_url, token, judge_model)
    usage = {k: sum((r.get("usage") or {}).get(k, 0) for r in samples) + usage_sel[k]
             for k in ("prompt_tokens", "completion_tokens")}
    return {
        "id": task["id"], "question_type": task["question_type"], "arm": "sc_matched", "model": model,
        "effort": EFFORT, "response": final, "grade": grade, "score": grade["score"], "passed": grade["passed"],
        "n_tool_calls": sum(r.get("n_tool_calls", 0) for r in samples), "sc_k": SC_K, "sc_selected": sel,
        "sc_raw": raw[:500], "sc_sample_scores": [r.get("score") for r in samples], "usage": usage,
        "n_llm_calls": sum(r.get("n_llm_calls", 0) for r in samples) + 1,
    }


def synthesize_stock_from_row(
    task: dict[str, Any],
    row: dict[str, Any],
    model: str,
    judge_model: str,
    base_url: str,
    token: str,
) -> dict[str, Any]:
    """Synthesize a non-empty final answer from the SEC filings already parsed in `row['tool_calls']`."""
    client = openai.OpenAI(api_key=token, base_url=base_url, timeout=180.0, max_retries=3)
    state: dict[str, str] = {}
    tool_calls_log: list[dict[str, Any]] = list(row.get("tool_calls", []))
    for tc in tool_calls_log:
        if tc.get("tool") == "parse_html_page":
            args = tc.get("args") or {}
            if args.get("url") and args.get("key") and args["key"] not in state:
                exec_parse_html_page(args, state)

    if not state:
        # Fallback: run one EDGAR search for the question and parse the top 2 filings
        try:
            hits = json.loads(exec_edgar_search({"search_query": task["question"][:200], "top_n_results": 5}))
            if isinstance(hits, list):
                for idx, h in enumerate(hits[:2]):
                    if h.get("url"):
                        exec_parse_html_page({"url": h["url"], "key": f"filing_{idx+1}"}, state)
        except Exception:
            pass

    excerpts: list[str] = []
    used_queries: set[tuple[str, str]] = set()
    for tc in tool_calls_log:
        if tc.get("tool") == "retrieve_information":
            p_str = str((tc.get("args") or {}).get("prompt") or "")
            for k in set(re.findall(r"\{\{([^{}]+)\}\}", p_str)):
                if k in state and (k, p_str[:80]) not in used_queries:
                    used_queries.add((k, p_str[:80]))
                    excerpts.append(
                        f"#### Retrieved from `{k}` (query: {p_str[:120]}):\n{_smart_window_extract(state[k], p_str, max_chars=22000)}"
                    )
    for k, v in list(state.items())[:5]:
        excerpts.append(
            f"#### Filing `{k}` excerpt for question:\n{_smart_window_extract(v, task['question'], max_chars=25000)}"
        )

    evidence_text = "\n\n".join(excerpts[:8])[-130000:] if excerpts else "(No filing excerpts parsed.)"
    resp = _quota_create(client, 
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{INSTRUCTIONS_PROMPT}{task['question']}\n\n"
                    f"---\n"
                    f"### RETRIEVED SEC EDGAR FILING EXCERPTS:\n"
                    f"{evidence_text}\n\n"
                    f"---\n"
                    f"Write your complete, final financial answer now (including all step-by-step calculations to at least "
                    f"two decimal places and the required JSON sources block at the end)."
                ),
            }
        ],
        max_tokens=4096,
        extra_body={"reasoning_effort": EFFORT},
    )
    final_answer = (resp.choices[0].message.content or "").strip()
    grade = grade_finance_response(task, final_answer, base_url, token, judge_model)
    usage = dict(row.get("usage") or {})
    if resp.usage:
        usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + (resp.usage.prompt_tokens or 0)
        usage["completion_tokens"] = usage.get("completion_tokens", 0) + (resp.usage.completion_tokens or 0)
    return {
        "id": task["id"],
        "question_type": task["question_type"],
        "arm": "stock",
        "model": model,
        "response": final_answer,
        "grade": grade,
        "score": grade["score"],
        "passed": grade["passed"],
        "n_tool_calls": len(tool_calls_log),
        "stored_keys": list(state.keys()),
        "tool_calls": tool_calls_log,
        "usage": usage,
    }


def run_paired_cb_audit(
    task: dict[str, Any],
    stock_row: dict[str, Any],
    cb_row: dict[str, Any] | None,
    model: str,
    judge_model: str,
    base_url: str,
    token: str,
) -> dict[str, Any]:
    """Single-pass Thinking-in-Bets (`cb_draftfirst`) Claim Book audit over the gathered SEC filings and initial draft."""
    client = openai.OpenAI(api_key=token, base_url=base_url, timeout=180.0, max_retries=3)
    state: dict[str, str] = {}
    combined_tcs = list(stock_row.get("tool_calls", [])) + list((cb_row or {}).get("tool_calls", []))
    seen_urls: set[str] = set()
    for tc in combined_tcs:
        if tc.get("tool") == "parse_html_page":
            args = tc.get("args") or {}
            u = str(args.get("url") or "").strip()
            k = str(args.get("key") or "").strip()
            if u and k and u not in seen_urls:
                seen_urls.add(u)
                if k in state:
                    k = f"{k}_{len(state)+1}"
                exec_parse_html_page({"url": u, "key": k}, state)

    draft_text = (stock_row.get("response") or "").strip()
    excerpts: list[str] = []
    used_queries: set[tuple[str, str]] = set()
    for tc in combined_tcs:
        if tc.get("tool") == "retrieve_information":
            p_str = str((tc.get("args") or {}).get("prompt") or "")
            for k in set(re.findall(r"\{\{([^{}]+)\}\}", p_str)):
                if k in state and (k, p_str[:80]) not in used_queries:
                    used_queries.add((k, p_str[:80]))
                    excerpts.append(
                        f"#### Excerpt from `{k}` (query: {p_str[:120]}):\n{_smart_window_extract(state[k], p_str, max_chars=20000)}"
                    )
    for k, v in list(state.items())[:5]:
        excerpts.append(
            f"#### Filing `{k}` excerpt for question:\n{_smart_window_extract(v, task['question'], max_chars=22000)}"
        )
    excerpts_block = "\n\n".join(excerpts[:8])[-125000:] if excerpts else "(No additional excerpts.)"

    audit_prompt = (
        f"{INSTRUCTIONS_PROMPT}{task['question']}\n\n"
        f"---\n"
        f"### RETRIEVED SEC EDGAR FILING EXCERPTS:\n"
        f"{excerpts_block}\n\n"
        f"---\n"
        f"### INITIAL DRAFT SUBMISSION:\n"
        f"{draft_text}\n\n"
        f"---\n"
        f"### Thinking-in-Bets (`cb_draftfirst`) Pre-Submission Audit Protocol:\n"
        f"1. `[CLAIM_BOOK]`: List each quantitative figure, period, growth rate / margin / basis-point calculation, "
        f"and sub-question in the draft (`C_i: <claim> | evidence: <exact filing excerpt & digit-by-digit arithmetic check> | p=0.XX`).\n"
        f"   - Verify every sum, difference, ratio, percentage, and basis-point calculation digit-by-digit (`A + B = C`, `(A - B)/B = X.XX%`).\n"
        f"   - Check whether the SEC filing excerpts above contain any missing sub-question, missing year/quarter, or standard alternative accounting convention "
        f"(e.g., both Ending AND Average balance-sheet denominator for turnover/return ratios; both Undiscounted AND Present-Value lease/debt denominators; both GAAP AND Non-GAAP).\n"
        f"2. `[VERDICT]`: Output either `KEEP_DRAFT` or `REVISE_DRAFT`:\n"
        f"   - Output `[VERDICT]: KEEP_DRAFT` if all claims in `INITIAL DRAFT SUBMISSION` have `p >= 0.85`, all arithmetic is exact, and no sub-question or alternative convention is missing.\n"
        f"   - Output `[VERDICT]: REVISE_DRAFT` followed by `[FINAL_ANSWER]` ONLY if you found a verified arithmetic mistake (`p < 0.50`), a contradiction with the SEC filing excerpts, or a missing sub-question / alternative accounting convention that you can add from the filing excerpts.\n"
        f"   - **STRICT NON-DELETION RULE:** When writing `[FINAL_ANSWER]`, retain 100% of the verified figures, tables, calculations, and sources from `INITIAL DRAFT SUBMISSION` (`p >= 0.90`) and append/correct the missing convention or arithmetic fix."
    )

    resp = _quota_create(client, 
        model=model,
        messages=[{"role": "user", "content": audit_prompt}],
        max_tokens=8192,
        extra_body={"reasoning_effort": EFFORT},
    )
    raw_out = (resp.choices[0].message.content or "").strip()
    if "KEEP_DRAFT" in raw_out and "[FINAL_ANSWER]" not in raw_out:
        final_answer = draft_text
    elif "[FINAL_ANSWER]" in raw_out:
        final_answer = raw_out.split("[FINAL_ANSWER]", 1)[1].strip()
    else:
        m_ans = re.split(r"(?:###|\*\*)\s*(?:2\.\s*)?\[?FINAL[ _]ANSWER\]?:?\s*(?:\*\*)?", raw_out, maxsplit=1, flags=re.I)
        final_answer = m_ans[1].strip() if len(m_ans) > 1 else draft_text

    # Guard against accidental truncation when draft was long
    if len(final_answer) < max(200, int(0.75 * len(draft_text))):
        final_answer = draft_text + "\n\n---\n### Supplemental Verification Notes:\n" + final_answer

    if final_answer == draft_text and len(draft_text) <= 6000 and stock_row.get("grade"):
        grade = stock_row["grade"]
    else:
        grade = grade_finance_response(task, final_answer, base_url, token, judge_model)

    usage = dict(stock_row.get("usage") or {})
    if resp.usage:
        usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + (resp.usage.prompt_tokens or 0)
        usage["completion_tokens"] = usage.get("completion_tokens", 0) + (resp.usage.completion_tokens or 0)
    return {
        "id": task["id"],
        "question_type": task["question_type"],
        "arm": "cb_draftfirst",
        "model": model,
        "response": final_answer,
        "grade": grade,
        "score": grade["score"],
        "passed": grade["passed"],
        "n_tool_calls": int(stock_row.get("n_tool_calls", 0)) + 1,
        "stored_keys": list(state.keys()),
        "tool_calls": list((cb_row or stock_row).get("tool_calls", [])),
        "usage": usage,
    }


def print_summary(out_path: Path) -> None:
    print("=== Tool-Augmented Vals Finance Agent Benchmark (Live SEC EDGAR + HTML Parser + Retrieve Info) ===")
    print("Model | Arm | N | Mean Rubric Score (%) | Pass Rate (>=75%, no contra) | Avg Tool Calls | Paired W/T/L")
    for f in sorted(out_path.glob("tools_*.jsonl")):
        by_arm: dict[str, dict[str, dict[str, Any]]] = {}
        model_name = f.stem.replace("tools_", "")
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            model_name = r.get("model", model_name)
            by_arm.setdefault(r["arm"], {})[r["id"]] = r
        for arm in [a for a in FT_ARMS_ALL if a in by_arm]:
            rows = list(by_arm.get(arm, {}).values())
            if not rows:
                continue
            mean_sc = 100.0 * sum(r["score"] for r in rows) / len(rows)
            pass_rt = 100.0 * sum(1 for r in rows if r["passed"]) / len(rows)
            avg_tc = sum(r.get("n_tool_calls", 0) for r in rows) / len(rows)
            print(f"{model_name:21s} | {arm:13s} | {len(rows):2d} | {mean_sc:6.2f}% | {pass_rt:6.2f}% | {avg_tc:5.1f}")
        if "stock" in by_arm and "cb_draftfirst" in by_arm:
            common = sorted(set(by_arm["stock"].keys()) & set(by_arm["cb_draftfirst"].keys()))
            if common:
                s_sc = 100.0 * sum(by_arm["stock"][k]["score"] for k in common) / len(common)
                c_sc = 100.0 * sum(by_arm["cb_draftfirst"][k]["score"] for k in common) / len(common)
                s_pr = 100.0 * sum(1 for k in common if by_arm["stock"][k]["passed"]) / len(common)
                c_pr = 100.0 * sum(1 for k in common if by_arm["cb_draftfirst"][k]["passed"]) / len(common)
                w = sum(1 for k in common if by_arm["cb_draftfirst"][k]["score"] > by_arm["stock"][k]["score"] + 1e-6)
                l = sum(1 for k in common if by_arm["cb_draftfirst"][k]["score"] < by_arm["stock"][k]["score"] - 1e-6)
                t = len(common) - w - l
                print(
                    f"  -> PAIRED ({len(common)} tasks): Score {s_sc:.2f}% -> {c_sc:.2f}% ({c_sc - s_sc:+.2f}%) | "
                    f"Pass {s_pr:.2f}% -> {c_pr:.2f}% ({c_pr - s_pr:+.2f}%) | W/T/L = {w}/{t}/{l}"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemini-3.8-flash,gemini-3.5-flash,gemini-3.5-flash-lite")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=35)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--judge", default="gemini-3.5-flash")
    ap.add_argument("--out-dir", required=True,
                    help="P0.8: fresh directory per wave (results/finance_agent_tools holds the published rows)")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--repair-paired", action="store_true",
                    help="DISABLED (PARALLEL_REVIEWER_EXPERIMENTS §0/§7, #13b): outcome-dependent re-runs")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--arms", default="stock,cb_draftfirst", help=f"P0.2; all: {','.join(FT_ARMS_ALL)}")
    ap.add_argument("--prereg-commit", default=None)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in FT_ARMS_ALL]
    if bad:
        raise SystemExit(f"unknown arms {bad}")
    if args.repair_paired:
        raise SystemExit("--repair-paired is disabled: it re-ran cb_draftfirst only where it lost (see #13b). "
                         "Primary analysis is first-attempt.")
    global EFFORT
    EFFORT = args.effort

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        print_summary(out_path)
        return

    if not args.summary_only and Path(args.out_dir).resolve() == (
            Path(__file__).resolve().parents[1] / "results" / "finance_agent_tools").resolve():
        raise SystemExit("P0.8: use a fresh --out-dir per wave")
    token = get_api_key()
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai"
    tasks = load_finance_tasks()[: args.limit]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    lock = threading.Lock()
    import subprocess
    import sys
    mp = out_path / "manifest.json"
    man = json.loads(mp.read_text()) if mp.exists() else {"launches": []}
    man["launches"].append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                code_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                           text=True).stdout.strip(),
                                preregistration_commit=args.prereg_commit, prompts_sha256=ft_prompts_sha256(),
                                models=models, arms=arms, effort=EFFORT, limit=args.limit, argv=sys.argv))
    mp.write_text(json.dumps(man, indent=1))

    for model in models:
        mtag = model.replace(".", "_").replace("-", "_")
        out_file = out_path / f"tools_{mtag}.jsonl"
        done: dict[tuple[str, str], dict[str, Any]] = {}
        if out_file.exists():
            for line in out_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[(r["id"], r["arm"])] = r

        if args.repair_paired:
            empty_stock = [
                t for t in tasks if (t["id"], "stock") not in done or not done[(t["id"], "stock")].get("response", "").strip()
            ]
            if empty_stock:
                print(f"[Finance-Tools {model}] Synthesizing {len(empty_stock)} empty stock tasks from parsed filings...", flush=True)
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {
                        ex.submit(
                            synthesize_stock_from_row,
                            t,
                            done.get((t["id"], "stock"), {"tool_calls": []}),
                            model,
                            args.judge,
                            base_url,
                            token,
                        ): t["id"]
                        for t in empty_stock
                    }
                    for fut in as_completed(futs):
                        tid = futs[fut]
                        res = fut.result()
                        done[(tid, "stock")] = res
                        print(f"  [stock synthesized] {model} {tid} score={res['score']:.2f} len={len(res['response'])}", flush=True)

            repaired_stock_ids = {t["id"] for t in empty_stock}
            audit_tasks = []
            for t in tasks:
                tid = t["id"]
                s_row = done.get((tid, "stock"))
                c_row = done.get((tid, "cb_draftfirst"))
                if not s_row or not s_row.get("response", "").strip():
                    continue
                c_resp = (c_row or {}).get("response", "").strip()
                s_resp = s_row.get("response", "").strip()
                c_contra = bool((c_row or {}).get("grade", {}).get("has_contradiction"))
                s_contra = bool(s_row.get("grade", {}).get("has_contradiction"))
                if c_row and c_resp and (c_row.get("score", 0.0) >= 0.999 or c_row.get("score", 0.0) > s_row.get("score", 0.0) + 1e-6):
                    continue
                if (
                    not c_row
                    or not c_resp
                    or tid in repaired_stock_ids
                    or "CLAIM_BOOK" in c_resp
                    or len(c_resp) < int(0.75 * len(s_resp))
                    or (c_contra and not s_contra)
                    or (c_row.get("score", 0.0) < s_row.get("score", 0.0) - 1e-6)
                ):
                    audit_tasks.append(t)

            print(f"[Finance-Tools {model}] Running single-pass paired cb_draftfirst audit on {len(audit_tasks)} tasks...", flush=True)
            if audit_tasks:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = {
                        ex.submit(
                            run_paired_cb_audit,
                            t,
                            done[(t["id"], "stock")],
                            done.get((t["id"], "cb_draftfirst")),
                            model,
                            args.judge,
                            base_url,
                            token,
                        ): t["id"]
                        for t in audit_tasks
                    }
                    for fut in as_completed(futs):
                        tid = futs[fut]
                        res = fut.result()
                        done[(tid, "cb_draftfirst")] = res
                        s_sc = done[(tid, "stock")]["score"]
                        print(
                            f"  [cb_draftfirst paired] {model} {tid}: stock={s_sc:.2f} -> cb={res['score']:.2f}",
                            flush=True,
                        )

            with open(out_file, "w", encoding="utf-8") as f:
                for t in tasks:
                    for arm in ["stock", "cb_draftfirst"]:
                        if (t["id"], arm) in done:
                            f.write(json.dumps(done[(t["id"], arm)]) + "\n")
            continue

        import random as _random
        done = {k: v for k, v in done.items() if v.get("effort", "low") == EFFORT}  # P0.8: effort in the key
        jobs = [(t, arm) for t in tasks for arm in arms if (t["id"], arm) not in done]
        _random.Random(f"fintools:{model}:{EFFORT}").shuffle(jobs)  # P0.7
        print(f"[Finance-Tools {model}] {len(done)} done, {len(jobs)} remaining, arms={arms}", flush=True)
        stock_locks: dict[str, threading.Lock] = {}

        def get_stock(t, model=model, done=done, out_file=out_file):
            with lock:
                lk = stock_locks.setdefault(t["id"], threading.Lock())
            with lk:
                if (t["id"], "stock") not in done:
                    r = run_tool_agent_task(t, "stock", model, args.judge, base_url, token, args.max_turns)
                    with lock:
                        done[(t["id"], "stock")] = r
                        with open(out_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(r) + "\n")
                return done[(t["id"], "stock")]

        def ft_job(t, arm, model=model):
            if arm == "stock":
                return get_stock(t)
            if arm == "sc_matched":  # sample 1 = the stock row of the same task (shared, made once)
                return run_sc_matched(t, get_stock(t), model, args.judge, base_url, token, args.max_turns)
            return run_tool_agent_task(t, arm, model, args.judge, base_url, token, args.max_turns)

        if jobs:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(ft_job, t, arm): (t["id"], arm) for t, arm in jobs}
                for fut in as_completed(futs):
                    tid, arm = futs[fut]
                    try:
                        res = fut.result()
                        with lock:
                            if arm != "stock":  # stock rows are written by get_stock (exactly once)
                                done[(tid, arm)] = res
                                with open(out_file, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(res) + "\n")
                            arm_rows = [r for (_, a), r in done.items() if a == arm]
                            mean_sc = 100.0 * sum(r["score"] for r in arm_rows) / len(arm_rows)
                            print(
                                f"  [{model} | {arm:13s}] {len(arm_rows)}/{len(tasks)} | "
                                f"last={tid} score={res['score']:.2f} tools={res['n_tool_calls']} | running_mean={mean_sc:.1f}%",
                                flush=True,
                            )
                    except Exception as exc:
                        print(f"[Finance-Tools ERROR] {model} {arm} {tid}: {exc}", flush=True)

    print_summary(out_path)


if __name__ == "__main__":
    main()

