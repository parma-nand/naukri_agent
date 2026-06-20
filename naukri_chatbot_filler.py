"""
naukri_chatbot_filler.py

Standalone module for answering Naukri's post-Apply chatbot questions.

WHY THIS IS SEPARATE:
    Keyword matching (`if key in q_text`) breaks the moment Naukri phrases a
    question differently than your map, or asks something you never anticipated
    (rotational shifts? willing to relocate to a specific city? notice period
    negotiable?). Splitting this out lets you:
        1. Debug form-filling against a single job URL without re-running
           search/apply.
        2. Swap keyword-matching for an LLM call without touching apply logic.
        3. Log every Q + extracted options + chosen answer to a file for review.

USAGE (standalone debug — tests ONLY the form filler, against one job URL):
    python naukri_chatbot_filler.py <job_url>

USAGE (from naukri_apply_jobs in nakri_tool.py):
    from naukri_chatbot_filler import answer_chatbot
    chat_summary = await answer_chatbot(page)
"""

import asyncio
import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from playwright.async_api import async_playwright, Page
from openai import AsyncOpenAI

load_dotenv()

# ── Logging — every run writes a transcript so you can see exactly what was
#    asked, what options existed, and what was chosen / why it failed ─────────
LOG_PATH = Path(__file__).parent / "chatbot_debug.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("naukri_chatbot")

# Also echo to console when run standalone
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Your profile facts — the LLM picks/composes answers grounded in this,
#    instead of you having to predict every possible question phrasing ───────
PROFILE = {
    "education": "B.Tech, NIT Jalandhar, 2022",
    "total_experience_years": "4",
    "relevant_genai_experience_years": "1.5",
    "notice_period": "Immediate",
    "current_ctc_lpa": "8",
    "expected_ctc_lpa": "12",
    "current_location": "Pune",
    "willing_to_relocate": "Yes",
    "skills_yes": [
        "python", "machine learning", "llm", "langchain", "rag",
        "generative ai", "genai", "sql", "fastapi", "docker",
    ],
    "notes": (
        "Associate Consultant / GenAI Engineer at Capgemini. "
        "Builds production RAG pipelines and fine-tunes open-source LLMs."
    ),
}

CHATBOT_CONTAINER = "#_4kyut3askChatbotContainer"
MESSAGES_SEL = "#_4kyut3askMessages"
INPUT_BOX = "#sendMsgbtn_container__4kyut3askInputBox"


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Extract the current question + whatever answer widget is present
# ═════════════════════════════════════════════════════════════════════════════
async def _extract_question_and_widget(page: Page) -> dict | None:
    """Reads the latest chatbot message and figures out what kind of input
    is expected: 'select' | 'options' | 'text' | None (no question pending)."""

    q_els = await page.query_selector_all(
        f"{MESSAGES_SEL} [class*='question'], {MESSAGES_SEL} [class*='Question']"
    )
    if not q_els:
        return None

    question_text = (await q_els[-1].inner_text()).strip()

    select_el = await page.query_selector(f"{INPUT_BOX} select")
    if select_el:
        options = await select_el.query_selector_all("option")
        labels = [(await o.inner_text()).strip() for o in options]
        values = [await o.get_attribute("value") for o in options]
        return {
            "type": "select",
            "question": question_text,
            "choices": labels,
            "values": values,
            "element": select_el,
        }

    option_els = await page.query_selector_all(
        f"{MESSAGES_SEL} [class*='chatbot_option'], "
        f"{MESSAGES_SEL} [class*='Option']:not([class*='container'])"
    )
    if option_els:
        labels = [(await el.inner_text()).strip() for el in option_els]
        return {
            "type": "options",
            "question": question_text,
            "choices": labels,
            "elements": option_els,
        }

    input_el = await page.query_selector(
        f"{INPUT_BOX} input[type='text'], {INPUT_BOX} textarea"
    )
    if input_el:
        return {
            "type": "text",
            "question": question_text,
            "element": input_el,
        }

    return None


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — Ask the LLM to choose/compose the answer, grounded in PROFILE
# ═════════════════════════════════════════════════════════════════════════════
async def _llm_answer(question: str, qtype: str, choices: list[str] | None) -> str:
    """
    Sends the question (+ available choices, if any) to the LLM along with
    the candidate profile, and asks for ONLY the answer text/choice back.
    This replaces the old keyword `if key in q_text` matching.
    """
    if qtype in ("select", "options"):
        choice_block = "\n".join(f"- {c}" for c in choices)
        instruction = (
            "Pick exactly ONE of the choices below that best answers the question. "
            "Reply with the choice text EXACTLY as written, nothing else.\n\n"
            f"Choices:\n{choice_block}"
        )
    else:
        instruction = (
            "Answer the question in a few words, suitable for pasting directly "
            "into a job application chat input. No extra commentary."
        )

    system = (
        "You are filling out a job application chatbot on behalf of a candidate. "
        "Use ONLY the candidate profile facts given below. If the question asks "
        "something not covered by the profile, give the most reasonable, honest, "
        "professional answer a 4-year experienced GenAI/ML engineer would give. "
        "Never invent qualifications the candidate doesn't have.\n\n"
        f"Candidate profile:\n{json.dumps(PROFILE, indent=2)}"
    )

    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}\n\n{instruction}"},
        ],
    )
    answer = resp.choices[0].message.content.strip()
    logger.info(f"LLM answer | Q: {question!r} -> A: {answer!r}")
    return answer


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — Apply the answer to whichever widget is present
# ═════════════════════════════════════════════════════════════════════════════
async def _apply_answer(page: Page, widget: dict, answer: str) -> bool:
    qtype = widget["type"]

    if qtype == "select":
        select_el = widget["element"]
        labels = widget["choices"]
        values = widget["values"]

        # exact, then case-insensitive, then substring match
        idx = None
        if answer in labels:
            idx = labels.index(answer)
        else:
            lower_labels = [l.lower() for l in labels]
            if answer.lower() in lower_labels:
                idx = lower_labels.index(answer.lower())
            else:
                for i, l in enumerate(lower_labels):
                    if answer.lower() in l or l in answer.lower():
                        idx = i
                        break

        if idx is None:
            logger.warning(f"No matching option for LLM answer {answer!r} in {labels}")
            return False

        await select_el.select_option(value=values[idx])
        return True

    if qtype == "options":
        elements = widget["elements"]
        choices = widget["choices"]
        idx = None
        lower_choices = [c.lower() for c in choices]
        if answer.lower() in lower_choices:
            idx = lower_choices.index(answer.lower())
        else:
            for i, c in enumerate(lower_choices):
                if answer.lower() in c or c in answer.lower():
                    idx = i
                    break
        if idx is None:
            logger.warning(f"No matching option element for {answer!r} in {choices}")
            return False
        await elements[idx].click()
        return True

    if qtype == "text":
        await widget["element"].fill(answer)
        return True

    return False


async def _submit(page: Page):
    send_btn = await page.query_selector(
        f"{INPUT_BOX} button[type='submit'], "
        f"{INPUT_BOX} button:has-text('Send'), "
        f"{CHATBOT_CONTAINER} button:has-text('Save')"
    )
    if send_btn:
        await send_btn.click()


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════
async def answer_chatbot(page: Page, max_questions: int = 20) -> dict:
    """
    Drives the full chatbot Q&A loop using LLM-grounded answers.
    Returns a summary dict: {"answered": [...], "failed": [...]}
    """
    summary = {"answered": [], "failed": []}

    for i in range(max_questions):
        await asyncio.sleep(1.2)

        widget = await _extract_question_and_widget(page)
        if widget is None:
            logger.info("No more pending questions — chatbot likely closed.")
            break

        logger.info(f"Q{i+1} ({widget['type']}): {widget['question']}")

        try:
            answer = await _llm_answer(
                widget["question"], widget["type"], widget.get("choices")
            )
            applied = await _apply_answer(page, widget, answer)

            if not applied:
                summary["failed"].append({"question": widget["question"], "reason": "no widget match"})
                break

            await _submit(page)
            summary["answered"].append({"question": widget["question"], "answer": answer})
            logger.info(f"Q{i+1} answered OK")

        except Exception as e:
            logger.exception(f"Error answering Q{i+1}: {e}")
            summary["failed"].append({"question": widget["question"], "reason": str(e)})
            break

        # chatbot closed = application submitted
        drawer = await page.query_selector(CHATBOT_CONTAINER)
        if not drawer or not await drawer.is_visible():
            logger.info("Chatbot container closed — application submitted.")
            break

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Standalone debug runner — test ONLY the form-filling, against one job URL,
# without touching naukri_search / naukri_apply_jobs / the LangGraph agent.
# ═════════════════════════════════════════════════════════════════════════════
async def debug_single_job(job_url: str, auth_file: str = "browser_state/auth.json"):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state=auth_file)
        page = await context.new_page()

        await page.goto(job_url, timeout=60_000)
        await asyncio.sleep(1.5)

        apply_btn = await page.query_selector(
            "button:has-text('Apply'), a.apply-button, [class*='applyBtn']:not([class*='company'])"
        )
        if not apply_btn:
            print("No Apply button found on this page.")
            await browser.close()
            return

        await apply_btn.click()
        print("Clicked Apply, waiting for chatbot...")

        try:
            await page.wait_for_selector(CHATBOT_CONTAINER, state="visible", timeout=8000)
        except Exception:
            print("No chatbot appeared — likely direct-applied.")
            await browser.close()
            return

        result = await answer_chatbot(page)
        print(json.dumps(result, indent=2))
        print(f"Full transcript: {LOG_PATH}")

        await asyncio.sleep(3)  # let you eyeball the final state
        await context.close()
        await browser.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python naukri_chatbot_filler.py <job_url>")
        sys.exit(1)
    asyncio.run(debug_single_job(sys.argv[1]))