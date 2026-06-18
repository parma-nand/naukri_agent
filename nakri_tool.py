"""
Minimal single-agent LangGraph example with an async Playwright tool
to log into Naukri.com.

Install deps:
    pip install langgraph langchain-core langchain-openai playwright --break-system-packages
    playwright install chromium

Set env vars (or hardcode for local testing only):
    NAUKRI_EMAIL=<your_email>
    NAUKRI_PASSWORD=<your_password>
    OPENAI_API_KEY=<your_key>
"""

import os
import asyncio
from typing import Annotated, TypedDict
from dotenv import load_dotenv


from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from playwright.async_api import async_playwright

load_dotenv()


# ---------------------------------------------------------------------
# 1. State
# ---------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------------
# 2. Tool — async Playwright login to Naukri
# ---------------------------------------------------------------------
@tool
async def naukri_login() -> str:
    """Open Naukri.com in a browser and log in using saved credentials."""
    email = os.getenv("NAUKRI_EMAIL")
    password = os.getenv("NAUKRI_PASSWORD")

    if not email or not password:
        return "Missing NAUKRI_EMAIL / NAUKRI_PASSWORD environment variables."

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        try:
            # ── Login ────────────────────────────────────────────────────────
            await page.goto("https://www.naukri.com/nlogin/login", timeout=60000)
            await page.fill("#usernameField", email)
            await page.fill("#passwordField", password)
            await page.click("button[type='submit']")

            # ── Search ───────────────────────────────────────────────────────
            await page.click("div#ni-gnb-searchbar")
            keyword_input = page.locator(
                "input.suggestor-input[placeholder='Enter keyword / designation / companies']"
            )
            await keyword_input.click()
            await keyword_input.fill("AI ML Engineer")
            await asyncio.sleep(1)

            await page.click("input#experienceDD")
            await asyncio.sleep(1)
            await page.locator("ul.dropdown li[title='4 years']").click()
            await asyncio.sleep(1)

            location_input = page.locator(
                "input.suggestor-input[placeholder='Enter location']"
            )
            await location_input.click()
            await location_input.fill("Bengaluru")
            await asyncio.sleep(1)

            await page.click("button.nI-gNb-sb__icon-wrapper")
            print("✅ Search triggered")
            await asyncio.sleep(1)

            # ── Wait for SRP (Search Result Page) ────────────────────────────
            # We confirmed button#filter-freshness EXISTS — just wait for it directly
            await page.wait_for_selector("button#filter-freshness", timeout=20000)
            print("✅ Filter button found")

            # ── Freshness Filter ─────────────────────────────────────────────
            await asyncio.sleep(1)
            freshness_btn = page.locator("button#filter-freshness")
            await freshness_btn.scroll_into_view_if_needed()
            await asyncio.sleep(1)
            await asyncio.sleep(0.5)
            await freshness_btn.click()
            await asyncio.sleep(1)
            print("✅ Freshness button clicked")

            # Wait for dropdown
            freshness = "Last 1 day"
            await page.wait_for_selector(
                f"//li[@title='{freshness}']", timeout=8000
            )
            await asyncio.sleep(1)
            await page.locator(f"//li[@title='{freshness}']").click()
            await asyncio.sleep(1)
            print(f"✅ Selected freshness: {freshness}")

            await page.wait_for_load_state("networkidle", timeout=15000)
            print("✅ Page settled after filter")

            # ── Scrape job listings ──────────────────────────────────────────
            job_cards = await page.eval_on_selector_all(
                "div.srp-jobtuple-wrapper, div.jobTuple, div[class*='jobTuple']",
                """els => els.map(el => ({
                    title:   el.querySelector('a.title, a[title]')?.innerText?.trim() || '',
                    company: el.querySelector('a.comp-name, a[href*="company"]')?.innerText?.trim() || '',
                    link:    el.querySelector('a.title, a[title]')?.href || ''
                }))"""
            )
            print(f"✅ Jobs scraped: {len(job_cards)}")

            return (
                f"Login + search + filter successful.\n"
                f"URL: {page.url}\n"
                f"Jobs found: {len(job_cards)}\n"
                f"Sample: {job_cards[:3]}"
            )
            print("Searched Completed")

        except Exception as e:
            # Capture screenshot on failure for easier debugging
            await page.screenshot(path="debug_screenshot.png")
            return f"Error: {e}"

        finally:
            await browser.close()




tools = [naukri_login]


# ---------------------------------------------------------------------
# 3. LLM bound with tools
# ---------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------
# 4. Node — agent reasoning step
# ---------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------
# 5. Graph — nodes, edges, compile
# ---------------------------------------------------------------------
graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)  # agent -> tools or END
graph_builder.add_edge("tools", "agent")  # loop back after tool result

graph = graph_builder.compile()


# ---------------------------------------------------------------------
# 6. Single entry point
# ---------------------------------------------------------------------
async def run_query(query: str) -> str:
    result = await graph.ainvoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    answer = asyncio.run(run_query("Can you please login to my naukri "))
    print(answer)
