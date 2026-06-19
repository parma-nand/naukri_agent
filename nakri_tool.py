"""
Multi-tool single-agent LangGraph example for Naukri.com automation.

Tools:
    0. check_session         — checks if saved auth state exists on disk
    1. naukri_login          — opens browser, logs in, saves auth state to disk
    2. naukri_search         — loads saved state, searches jobs with filters
    3. naukri_update_resume  — loads saved state, goes to profile, uploads resume PDF

Project layout expected:
    naukri_agent/
    ├── nakri_tool.py          ← this file
    ├── .env
    ├── browser_state/
    │   └── auth.json          ← written by naukri_login (auto-created)
    └── resumes/
        └── Parmanand_Resume.pdf   ← your resume PDF

Install deps:
    pip install langgraph langchain-core langchain-openai playwright python-dotenv --break-system-packages
    playwright install chromium

Set env vars in .env:
    NAUKRI_EMAIL=<your_email>
    NAUKRI_PASSWORD=<your_password>
    OPENAI_API_KEY=<your_key>
    RESUME_FILE=Parmanand_Resume.pdf   # filename inside resumes/ folder
"""

import os
import asyncio
import glob
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from playwright.async_api import async_playwright

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
BROWSER_STATE_DIR = BASE_DIR / "browser_state"
AUTH_FILE = BROWSER_STATE_DIR / "auth.json"
RESUMES_DIR = BASE_DIR / "resumes"

BROWSER_STATE_DIR.mkdir(exist_ok=True)
RESUMES_DIR.mkdir(exist_ok=True)


# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = SystemMessage(content="""ou are a Naukri.com automation agent.

STRICT RULES — follow exactly:
1. Call tools ONE AT A TIME. Never batch or emit multiple tool calls in a single response.
2. ALWAYS call check_session FIRST before anything else — no exceptions.
3. If check_session says session exists → skip naukri_login, go directly to the requested tool.
4. If check_session says no session → call naukri_login first, then proceed.
5. Wait for each tool result before deciding the next step.
6. If login fails, stop and report the error.
7. If user asks to search AND apply — first call naukri_search, extract the URLs from results,
   then call naukri_apply_jobs with those URLs as a comma-separated string.
8. Never stop after search if the user also asked to apply.
""")


# ═════════════════════════════════════════════════════════════════════════════
# TOOL 0 — Check if saved session exists
# ═════════════════════════════════════════════════════════════════════════════
@tool
def check_session() -> str:
    """
    Check whether a saved Naukri login session (auth.json) exists on disk.
    Always call this FIRST before deciding whether to login.
    Returns confirmation if session exists, or instructs to call naukri_login.
    """
    if AUTH_FILE.exists():
        return (
            f"✅ Session found at {AUTH_FILE}. "
            "No need to login again. "
            "Proceed directly to naukri_search or naukri_update_resume."
        )
    return (
        "❌ No session found. "
        "You must call naukri_login before proceeding."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Login & save browser auth state
# ═════════════════════════════════════════════════════════════════════════════
@tool
async def naukri_login() -> str:
    """
    Open Naukri.com in a browser, log in with saved credentials,
    and persist the authenticated browser state to disk for reuse by
    other tools (so you don't have to log in again each time).
    Only call this if check_session reports no session exists.
    """
    email = os.getenv("NAUKRI_EMAIL")
    password = os.getenv("NAUKRI_PASSWORD")

    if not email or not password:
        return "❌ Missing NAUKRI_EMAIL / NAUKRI_PASSWORD in .env"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await page.goto("https://www.naukri.com/nlogin/login", timeout=60_000)
            await page.fill("#usernameField", email)
            await page.fill("#passwordField", password)
            await page.click("button[type='submit']")

            # Wait until redirected away from login page
            await page.wait_for_url(lambda url: "nlogin" not in url, timeout=30_000)
            print("✅ Logged in successfully")

            # ── Save auth state ───────────────────────────────────────────────
            await context.storage_state(path=str(AUTH_FILE))
            print(f"✅ Browser state saved → {AUTH_FILE}")

            return (
                f"Login successful. Auth state saved to {AUTH_FILE}.\n"
                f"Landed on: {page.url}\n"
                "You can now call naukri_search or naukri_update_resume."
            )

        except Exception as e:
            await page.screenshot(path=str(BASE_DIR / "debug_login_error.png"))
            return f"❌ Login error: {e}"

        finally:
            await context.close()
            await browser.close()


# ═════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Search jobs (reuses saved auth state)
# ═════════════════════════════════════════════════════════════════════════════
@tool
async def naukri_search(
    keyword: str = "Gen ai engineer or ml engineer or ai engineer",
    location: str = "bengaluru",
    experience_years: str = "4 years",
    freshness: str = "Last 1 day",
) -> str:
    """
    Search for jobs on Naukri using a previously saved login session.
    Parameters:
        keyword          — Job title / skills to search for (default: 'AI ML Engineer')
        location         — Preferred city (default: '')
        experience_years — Experience filter label e.g. '4 years' (default: '4 years')
        freshness        — Freshness filter e.g. 'Last 1 day' (default: 'Last 1 day')
    Returns a summary of matching job listings.
    Requires check_session to confirm session exists (or naukri_login to have been called).
    """
    if not AUTH_FILE.exists():
        return "❌ No saved login session found. Please call naukri_login first."

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state=str(AUTH_FILE))
        page = await context.new_page()

        try:
            print("Searching step 1: navigating to Naukri home")
            await page.goto("https://www.naukri.com/", timeout=60_000)
            print("Searching step 2: page loaded")
            await asyncio.sleep(1)
            await asyncio.sleep(1)

            # ── Search bar ────────────────────────────────────────────────────
            print("Searching step 3: clicking search bar")
            await page.click("div#ni-gnb-searchbar")
            print("Searching step 4: locating keyword input")
            keyword_input = page.locator(
                "input.suggestor-input[placeholder='Enter keyword / designation / companies']"
            )
            print("Searching step 5: filling keyword")
            await asyncio.sleep(1)
            await keyword_input.click()
            await keyword_input.fill(keyword)
            await asyncio.sleep(1)

            # ── Experience ────────────────────────────────────────────────────
            await page.click("input#experienceDD")
            await asyncio.sleep(1)
            await page.locator(f"ul.dropdown li[title='{experience_years}']").click()
            await asyncio.sleep(1)

            # ── Location ──────────────────────────────────────────────────────
            location_input = page.locator(
                "input.suggestor-input[placeholder='Enter location']"
            )
            await location_input.click()
            await location_input.fill(location)
            await asyncio.sleep(1)

            # ── Submit ────────────────────────────────────────────────────────
            await page.click("button.nI-gNb-sb__icon-wrapper")
            print("✅ Search triggered")

            # ── Wait for SRP ──────────────────────────────────────────────────
            await page.wait_for_selector("button#filter-freshness", timeout=20_000)
            print("✅ SRP loaded")

            # ── Freshness filter ──────────────────────────────────────────────
            await asyncio.sleep(1)
            freshness_btn = page.locator("button#filter-freshness")
            await freshness_btn.scroll_into_view_if_needed()
            await asyncio.sleep(1)
            await freshness_btn.click()
            await asyncio.sleep(1)

            await page.wait_for_selector(f"//li[@title='{freshness}']", timeout=8_000)
            await page.locator(f"//li[@title='{freshness}']").click()
            await asyncio.sleep(1)
            print(f"✅ Freshness filter set: {freshness}")
            await asyncio.sleep(1)

            # ── Scrape listings ───────────────────────────────────────────────
            job_cards = await page.eval_on_selector_all(
                "div.srp-jobtuple-wrapper, div.jobTuple, div[class*='jobTuple']",
                """els => els.map(el => ({
                    title:   el.querySelector('a.title, a[title]')?.innerText?.trim() || '',
                    company: el.querySelector('a.comp-name, a[href*="company"]')?.innerText?.trim() || '',
                    link:    el.querySelector('a.title, a[title]')?.href || ''
                }))"""
            )
            print(f"✅ Scraped {len(job_cards)} jobs")

            top10 = job_cards[:10]
            listing_text = "\n".join(
                f"  {i+1}. [{c['title']}] @ {c['company']} → {c['link']}"
                for i, c in enumerate(top10)
            )

            return (
                f"Search complete.\n"
                f"Query: '{keyword}' | Location: {location} | Freshness: {freshness}\n"
                f"Total jobs found: {len(job_cards)}\n\n"
                f"Top 10 results:\n{listing_text}"
            )

        except Exception as e:
            await page.screenshot(path=str(BASE_DIR / "debug_search_error.png"))
            return f"❌ Search error: {e}"

        finally:
            await context.close()
            await browser.close()


# ═════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Update / upload resume on Naukri profile
# ═════════════════════════════════════════════════════════════════════════════
@tool
async def naukri_update_resume(resume_filename: str = "") -> str:
    """
    Navigate to the Naukri profile page, click the Resume 'Update' link,
    and upload a PDF resume from the project's resumes/ folder.
    Parameters:
        resume_filename — Name of the PDF file inside the resumes/ folder.
                          If empty, the tool auto-picks the first PDF found.
    Requires check_session to confirm session exists (or naukri_login to have been called).
    """
    if not AUTH_FILE.exists():
        return "❌ No saved login session. Please call naukri_login first."

    # ── Resolve resume file ───────────────────────────────────────────────────
    if resume_filename:
        resume_path = RESUMES_DIR / resume_filename
    else:
        pdfs = sorted(glob.glob(str(RESUMES_DIR / "*.pdf")))
        if not pdfs:
            return (
                f"❌ No PDF found in {RESUMES_DIR}. "
                "Add your resume PDF there and retry."
            )
        resume_path = Path(pdfs[0])

    if not resume_path.exists():
        return f"❌ Resume file not found: {resume_path}"

    print(f"📄 Will upload: {resume_path}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state=str(AUTH_FILE))
        page = await context.new_page()

        try:
            # ── Go to profile ─────────────────────────────────────────────────
            await page.goto("https://www.naukri.com/mnjuser/profile", timeout=60_000)
            await asyncio.sleep(1.5)
            print("✅ Profile page loaded")

            # ── Click "Update" link next to Resume in Quick links ─────────────
            update_link = page.locator(
                "ul.collection li.collection-item:has(span.text:text('Resume')) "
                "a.secondary-content"
            )
            await asyncio.sleep(1.5)
            await update_link.wait_for(timeout=10_000)
            await update_link.scroll_into_view_if_needed()
            await asyncio.sleep(1.5)
            await update_link.click()
            print("✅ Clicked Resume → Update")

            await asyncio.sleep(1.5)

            # ── Handle the file upload input ──────────────────────────────────
            file_input = page.locator("input[type='file']").first
            await asyncio.sleep(1.5)
            await file_input.set_input_files(str(resume_path))
            print("✅ File selected in input")

            await asyncio.sleep(2)

            # ── Confirm / Save if a save button appears ───────────────────────
            save_btn = page.locator(
                "button:has-text('Save'), button:has-text('Upload'), "
                "button:has-text('Submit')"
            ).first
            if await save_btn.count() > 0:
                await save_btn.click()
                print("✅ Save/Upload button clicked")
                await asyncio.sleep(2)

            await page.screenshot(path=str(BASE_DIR / "resume_upload_success.png"))
            print("✅ Screenshot saved → resume_upload_success.png")

            return (
                f"Resume upload complete.\n"
                f"File uploaded: {resume_path.name}\n"
                f"Profile URL: {page.url}\n"
                "Screenshot saved to resume_upload_success.png"
            )

        except Exception as e:
            await page.screenshot(path=str(BASE_DIR / "debug_resume_error.png"))
            return f"❌ Resume update error: {e}"

        finally:
            await context.close()
            await browser.close()
# ═════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Apply to jobs with match score filtering
# ═════════════════════════════════════════════════════════════════════════════
@tool
async def naukri_apply_jobs(
    job_urls: str = "",
    require_keyskills: bool = True,
    require_work_experience: bool = True,
) -> str:
    """
    Visit each Naukri job URL, check the Job Match Score badges,
    and apply if criteria are met. Skips jobs with 'Apply on company site'.
    Parameters:
        job_urls              — Comma-separated list of Naukri job URLs to process
        require_keyskills     — Skip job if Keyskills badge missing (default: True)
        require_work_experience — Skip job if Work Experience badge missing (default: True)
    Returns a summary of applied / skipped jobs.
    Requires a valid session (call check_session first).
    """
    if not AUTH_FILE.exists():
        return "❌ No saved login session. Call naukri_login first."

    if not job_urls.strip():
        return "❌ No job_urls provided. Pass comma-separated Naukri job URLs."

    urls = [u.strip() for u in job_urls.split(",") if u.strip()]

    ANSWER_MAP = {
        "education": "B.Tech/B.E.",
        "experience": "4",
        "notice period": "Immediately",
        "current ctc": "8",
        "expected ctc": "12",
        "location": "Pune",
        "relocate": "Yes",
        "python": "Yes",
        "machine learning": "Yes",
        "llm": "Yes",
        "langchain": "Yes",
        "rag": "Yes",
        "generative ai": "Yes",
        "genai": "Yes",
        "sql": "Yes",
    }

    results = {"applied": [], "skipped": [], "errors": []}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state=str(AUTH_FILE))
        page = await context.new_page()

        for url in urls:
            print(f"\n{'='*60}")
            print(f"Processing: {url}")
            try:
                await page.goto(url, timeout=60_000)
                # await page.wait_for_load_state("networkidle")
                await asyncio.sleep(1.5)

                # ── Step 1: Check for "Apply on company site" ─────────────────
                company_site = await page.query_selector(
                    "a:has-text('Apply on company site'), "
                    "button:has-text('Apply on company site')"
                )
                if company_site:
                    reason = "Apply on company site"
                    print(f"  ⏭ SKIP — {reason}")
                    results["skipped"].append({"url": url, "reason": reason})
                    continue

                # ── Step 2: Read match score badges ───────────────────────────
                badge_texts = []
                badge_els = await page.query_selector_all(
                    "[class*='matchScore'] span, "
                    "[class*='match-score'] span, "
                    "[class*='jobMatch'] span, "
                    "[class*='job-match'] span"
                )
                for el in badge_els:
                    t = (await el.inner_text()).strip()
                    if t:
                        badge_texts.append(t)

                print(f"  Badges found: {badge_texts}")

                has_keyskills = any("keyskill" in b.lower() for b in badge_texts)
                has_work_exp = any("work experience" in b.lower() or "experience" in b.lower() for b in badge_texts)
                has_early = any("early" in b.lower() for b in badge_texts)

                # ── Step 3: Apply decision ────────────────────────────────────
                skip_reason = None
                if require_keyskills and not has_keyskills:
                    skip_reason = "Missing Keyskills badge"
                elif require_work_experience and not has_work_exp:
                    skip_reason = "Missing Work Experience badge"

                if skip_reason:
                    print(f"  ⏭ SKIP — {skip_reason}")
                    results["skipped"].append({"url": url, "reason": skip_reason})
                    continue

                # ── Step 4: Click Apply ───────────────────────────────────────
                apply_btn = await page.query_selector(
                    "button:has-text('Apply'), "
                    "a.apply-button, "
                    "[class*='applyBtn']:not([class*='company'])"
                )
                if not apply_btn:
                    print("  ⏭ SKIP — Apply button not found")
                    results["skipped"].append({"url": url, "reason": "Apply button not found"})
                    continue

                await apply_btn.click()
                print("  ✅ Clicked Apply")
                await asyncio.sleep(2)

                # ── Step 5: Handle chatbot if it opens ────────────────────────
                chatbot_opened = False
                try:
                    await page.wait_for_selector(
                        "#_4kyut3askChatbotContainer",
                        state="visible", timeout=6000
                    )
                    chatbot_opened = True
                    print("  💬 Chatbot opened — answering questions")
                except:
                    pass  # No chatbot = direct apply, that's fine

                if chatbot_opened:
                    await _answer_chatbot(page, ANSWER_MAP)

                results["applied"].append({
                    "url": url,
                    "badges": badge_texts,
                    "chatbot": chatbot_opened
                })
                print(f"  ✅ APPLIED")
                await asyncio.sleep(2)

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results["errors"].append({"url": url, "error": str(e)})
                await page.screenshot(path=str(BASE_DIR / f"debug_apply_error.png"))

        await context.close()
        await browser.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = (
        f"Apply run complete.\n"
        f"✅ Applied : {len(results['applied'])}\n"
        f"⏭ Skipped : {len(results['skipped'])}\n"
        f"❌ Errors  : {len(results['errors'])}\n\n"
    )
    if results["applied"]:
        summary += "Applied to:\n" + "\n".join(
            f"  • {r['url']} (chatbot={r['chatbot']})" for r in results["applied"]
        ) + "\n\n"
    if results["skipped"]:
        summary += "Skipped:\n" + "\n".join(
            f"  • {r['url']} — {r['reason']}" for r in results["skipped"]
        ) + "\n\n"
    if results["errors"]:
        summary += "Errors:\n" + "\n".join(
            f"  • {r['url']} — {r['error']}" for r in results["errors"]
        )
    return summary


async def _answer_chatbot(page, ANSWER_MAP: dict):
    """Internal helper — walks through Naukri chatbot Q&A"""
    for i in range(20):
        await asyncio.sleep(1.5)

        q_els = await page.query_selector_all(
            "#_4kyut3askMessages [class*='question'], "
            "#_4kyut3askMessages [class*='Question']"
        )
        if not q_els:
            break

        q_text = (await q_els[-1].inner_text()).lower().strip()
        print(f"    Q{i+1}: {q_text[:80]}")

        select_el = await page.query_selector(
            "#sendMsgbtn_container__4kyut3askInputBox select"
        )
        option_els = await page.query_selector_all(
            "#_4kyut3askMessages [class*='chatbot_option'], "
            "#_4kyut3askMessages [class*='Option']:not([class*='container'])"
        )
        input_el = await page.query_selector(
            "#sendMsgbtn_container__4kyut3askInputBox input[type='text'], "
            "#sendMsgbtn_container__4kyut3askInputBox textarea"
        )

        answered = False

        if select_el:
            options = await select_el.query_selector_all("option")
            chosen = None
            for key, val in ANSWER_MAP.items():
                if key in q_text:
                    for opt in options:
                        if val.lower() in (await opt.inner_text()).lower():
                            chosen = await opt.get_attribute("value")
                            break
                    break
            if not chosen:
                vals = [await o.get_attribute("value") for o in options if await o.get_attribute("value")]
                chosen = vals[0] if vals else None
            if chosen:
                await select_el.select_option(value=chosen)
            answered = True

        elif option_els:
            chosen_el = None
            for key, val in ANSWER_MAP.items():
                if key in q_text:
                    for opt_el in option_els:
                        if val.lower() in (await opt_el.inner_text()).lower():
                            chosen_el = opt_el
                            break
                    break
            await (chosen_el or option_els[0]).click()
            answered = True

        elif input_el:
            answer = "4"
            for key, val in ANSWER_MAP.items():
                if key in q_text:
                    answer = val
                    break
            await input_el.fill(answer)
            answered = True

        if not answered:
            print(f"    ⚠ Could not answer: {q_text[:60]}")
            break

        send_btn = await page.query_selector(
            "#sendMsgbtn_container__4kyut3askInputBox button[type='submit'], "
            "#sendMsgbtn_container__4kyut3askInputBox button:has-text('Send'), "
            "#_4kyut3askChatbotContainer button:has-text('Save')"
        )
        if send_btn:
            await send_btn.click()
            print(f"    ✅ Answered Q{i+1}")

        # Chatbot closed = submitted
        drawer = await page.query_selector("#_4kyut3askChatbotContainer")
        if not drawer or not await drawer.is_visible():
            print("    ✅ Chatbot closed — application submitted")
            break


# ═════════════════════════════════════════════════════════════════════════════
# Agent wiring
# ═════════════════════════════════════════════════════════════════════════════
tools = [check_session, naukri_login, naukri_search, naukri_update_resume, naukri_apply_jobs]

# parallel_tool_calls=False forces the LLM to emit ONE tool call per turn
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


def agent_node(state: AgentState) -> dict:
    # Prepend system prompt on every invocation
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()


async def run_query(query: str) -> str:
    result = await graph.ainvoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Search jobs for AI ML Engineer or Genai engineer last 1 day, get 10 results, then apply to all of them using naukri_apply_jobs with the URLs"
    )
    print(asyncio.run(run_query(query)))