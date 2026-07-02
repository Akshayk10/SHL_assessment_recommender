"""
agent.py
--------
Intent-Aware Conversational SHL Assessment Recommender.

Novel approach — three-layer architecture:
  Layer 1: Rule-based Intent State Machine (zero LLM calls for routing)
           States: CLARIFY | RECOMMEND | REFINE | COMPARE | REFUSE
  Layer 2: Structured search parameter extraction (regex + keyword mapping)
           Extracts: preferred test types, remote flag, boost keywords
  Layer 3: Hybrid retrieval → grounded LLM response (anti-hallucination guard)
           Every recommendation is validated against the retrieved catalog set.

Design philosophy (from SHL document):
  "Decide when the agent should ask, when it should retrieve, when it should
   answer, and when it should refuse."
  → The state machine answers exactly these questions deterministically.
"""

import json
import re
import logging
from enum import Enum
from typing import Optional, List, Tuple

from dotenv import load_dotenv
from groq import Groq
import os
from tenacity import retry, stop_after_attempt, wait_exponential

from models import ChatResponse, Recommendation, Message

load_dotenv()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"  # Fast & free: ~200ms median latency


# ─────────────────────────────────────────────────────────────────────────────
# Intent State Machine
# ─────────────────────────────────────────────────────────────────────────────

class IntentState(Enum):
    CLARIFY   = "clarify"    # Need job role — ask, return []
    RECOMMEND = "recommend"  # Enough context — return 1-10
    REFINE    = "refine"     # Update existing list — return 1-10
    COMPARE   = "compare"    # Comparison question — answer, return []
    REFUSE    = "refuse"     # Off-topic / injection — decline, return []


# Patterns that trigger REFUSE (checked as substrings of lowercased user msg)
_REFUSE_PATTERNS: List[str] = [
    # General hiring advice (not SHL scope)
    "interview question", "what questions should", "salary", "compensation",
    "background check", "reference check", "onboarding",
    # Legal / compliance
    "is it legal", "gdpr", "legal ", "employment law", "discrimination",
    "regulation", "compliance",
    # HR policy
    "hr policy", "hiring policy", "diversity policy",
    # Prompt injection / jailbreak
    "ignore previous", "ignore all", "disregard", "forget your instructions",
    "act as", "system prompt", "jailbreak", "new persona",
    "pretend you are", "you are now",
    # Non-SHL brand names
    "hogan ", "talentq", "talent q", "korn ferry", "wonderlic",
    "criteria corp", "psychometrics", "saville", "thomas international",
    "predictive index", "caliper ",
]

# Patterns that trigger COMPARE (regex searched on lowercased user msg)
_COMPARE_PATTERNS: List[str] = [
    r"difference between",
    r"what(?:'s| is) the difference",
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"better than",
    r"which (?:one|is|assessment|test) (?:is |are )?(?:better|best|more)",
    r"how does .+ differ",
    r"contrast .+ (?:with|and)",
]

# Tokens that indicate a specific job role / hiring context
_ROLE_INDICATORS: List[str] = [
    # Job titles / role words
    "developer", "engineer", "manager", "analyst", "designer", "scientist",
    "director", "executive", "specialist", "consultant", "coordinator",
    "administrator", "lead", "architect", "officer", "programmer", "coder",
    "accountant", "finance", "sales", "marketing", "operations", "logistic",
    "nurse", "doctor", "teacher", "trainer", "technician", "officer",
    "customer service", "support", "data", "devops", "security", "cloud",
    # Hiring intent phrases
    "hiring a", "hiring for", "we are hiring", "i am hiring",
    "looking for a", "looking for an", "need to hire", "recruit",
    "seeking a", "seeking an", "filling a", "opening for",
    "job description", "jd:", "role:", "position:",
    # Technology keywords that imply a role
    "java", "python", "javascript", "typescript", "react", "angular", "vue",
    "frontend", "backend", "fullstack", "full-stack", "sql", "aws", "azure",
    "gcp", "machine learning", "ml ", "ai ", "llm", "nlp", "tableau",
    ".net", "c#", "c++", "go ", "rust", "swift", "kotlin", "flutter",
    "spring", "django", "fastapi", "node",
]

# Map user natural language → SHL test type codes (for metadata boosting)
_TYPE_KEYWORD_MAP: dict = {
    "cognitive": ["A"],
    "aptitude": ["A"],
    "ability": ["A"],
    "numerical": ["A"],
    "verbal": ["A"],
    "logical": ["A"],
    "reasoning": ["A"],
    "inductive": ["A"],
    "deductive": ["A"],
    "situational": ["B"],
    "sjt": ["B"],
    "judgement": ["B"],
    "judgment": ["B"],
    "biodata": ["B"],
    "competenc": ["C"],
    "360": ["D"],
    "development": ["D"],
    "feedback report": ["D"],
    "exercise": ["E"],
    "in-tray": ["E"],
    "in tray": ["E"],
    "simulation": ["S"],
    "knowledge": ["K"],
    "skills test": ["K"],
    "technical test": ["K"],
    "coding test": ["K"],
    "programming test": ["K"],
    "software test": ["K"],
    "motivation": ["M"],
    "values": ["M"],
    "engagement": ["M"],
    "personality": ["P"],
    "behaviour": ["P"],
    "behavior": ["P"],
    "psychometric": ["P", "A"],
    "opq": ["P"],
    "gsma": ["P"],
}


def _lower_all_user(messages: List[Message]) -> str:
    """Aggregate all user messages, lowercased."""
    return " ".join(m.content.lower() for m in messages if m.role == "user")


def _last_user_msg(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.content.lower()
    return ""


def _has_prior_recommendations(messages: List[Message]) -> bool:
    """True if any prior assistant message contained ≥1 recommendation."""
    for m in messages:
        if m.role == "assistant":
            try:
                data = json.loads(m.content)
                if isinstance(data, dict) and len(data.get("recommendations", [])) > 0:
                    return True
            except (json.JSONDecodeError, TypeError):
                pass  # Plain-text assistant message — skip
    return False


def detect_intent(messages: List[Message], turn_number: int) -> IntentState:
    """
    Pure rule-based intent detection — no LLM, no latency.

    Priority order (first match wins):
      1. REFUSE   — off-topic, injection, competitor brands
      2. COMPARE  — comparison / versus questions
      3. REFINE   — prior recommendations exist
      4. RECOMMEND/CLARIFY — based on role presence & turn number
    """
    user_msg = _last_user_msg(messages)

    # ── 1. REFUSE ─────────────────────────────────────────────────────────────
    for pat in _REFUSE_PATTERNS:
        if pat in user_msg:
            logger.info(f"[REFUSE] triggered by: {pat!r}")
            return IntentState.REFUSE

    # ── 2. COMPARE ────────────────────────────────────────────────────────────
    for pat in _COMPARE_PATTERNS:
        if re.search(pat, user_msg):
            logger.info(f"[COMPARE] triggered by: {pat!r}")
            return IntentState.COMPARE

    # ── 3. REFINE — if prior recs exist, any follow-up is a refinement ────────
    if _has_prior_recommendations(messages):
        logger.info("[REFINE] prior recommendations detected")
        return IntentState.REFINE

    # ── 4. Turn-cap: MUST recommend by turn 3 (evaluator checks this) ─────────
    if turn_number >= 3:
        logger.info(f"[RECOMMEND] forced at turn {turn_number}")
        return IntentState.RECOMMEND

    # ── 5. RECOMMEND if role/context detected, else CLARIFY ───────────────────
    has_role = any(ind in user_msg for ind in _ROLE_INDICATORS)
    # Long messages (>20 words) almost always contain job context
    if not has_role and len(user_msg.split()) > 20:
        has_role = True

    state = IntentState.RECOMMEND if has_role else IntentState.CLARIFY
    logger.info(f"[{state.value.upper()}] role_detected={has_role}")
    return state


def extract_search_params(
    messages: List[Message],
) -> Tuple[str, List[str], Optional[bool], List[str]]:
    """
    Extract structured search parameters from full conversation.
    Returns: (composite_query, preferred_type_codes, remote_flag, boost_keywords)

    No LLM call — purely regex + keyword mapping.
    """
    all_text = _lower_all_user(messages)
    original_text = " ".join(m.content for m in messages if m.role == "user")

    # ── Preferred test types ──────────────────────────────────────────────────
    preferred_types: List[str] = []
    for keyword, codes in _TYPE_KEYWORD_MAP.items():
        if keyword in all_text:
            preferred_types.extend(codes)
    preferred_types = list(dict.fromkeys(preferred_types))  # dedup, preserve order

    # ── Remote preference ─────────────────────────────────────────────────────
    remote: Optional[bool] = None
    if any(w in all_text for w in ["remote", "work from home", "wfh", "virtual", "online"]):
        remote = True

    # ── Boost keywords: preserve casing from original for exact-match boosting ─
    # Extract: CamelCase words, ALL-CAPS acronyms, known tech terms
    boost_keywords: List[str] = []
    # Capitalised tech terms (Java, Python, Azure, OPQ32r, etc.)
    caps_terms = re.findall(r"\b[A-Z][A-Za-z0-9#.+]*\b", original_text)
    boost_keywords = list(dict.fromkeys(caps_terms))[:10]  # Top 10, deduped

    # ── Composite query: full conversation context for dense retrieval ─────────
    composite_query = all_text

    return composite_query, preferred_types, remote, boost_keywords


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_STATE_INSTRUCTIONS: dict = {
    IntentState.CLARIFY: (
        "The user hasn't specified a job role yet. "
        "Ask exactly ONE focused question to learn the role. "
        "Example: 'What role are you hiring for?' "
        "Do NOT recommend anything. Return recommendations=[]."
    ),
    IntentState.RECOMMEND: (
        "The user has provided enough context. "
        "Select 1–10 relevant assessments from the AVAILABLE ASSESSMENTS list below. "
        "Copy every name and URL character-for-character — do not paraphrase or invent. "
        "Explain in 1–2 sentences why the bundle fits this role. "
        "Where possible include a mix of cognitive (A), technical/knowledge (K), "
        "and personality (P) assessments if relevant."
    ),
    IntentState.REFINE: (
        "The user is updating or constraining a prior recommendation. "
        "Adjust the shortlist based on the new constraint. "
        "Only use assessments from the AVAILABLE ASSESSMENTS list. "
        "Return 1–10 items. Do not start over; keep relevant prior picks."
    ),
    IntentState.COMPARE: (
        "The user wants a comparison between SHL assessments. "
        "Use the AVAILABLE ASSESSMENTS descriptions to produce a factual, grounded answer. "
        "Do NOT use your training knowledge to fill gaps — only catalog facts. "
        "Return recommendations=[]."
    ),
    IntentState.REFUSE: (
        "This query is outside scope (off-topic, legal, competitor, or prompt injection). "
        "Respond politely: explain you only help with selecting SHL assessments. "
        "Return recommendations=[]."
    ),
}

_SYSTEM_PROMPT = """You are an SHL Assessment Recommender. Your sole purpose is to help hiring managers select assessments from the SHL catalog.

CONVERSATION STATE: {state}
INSTRUCTION: {instruction}

AVAILABLE ASSESSMENTS — use ONLY these. Copy names and URLs EXACTLY (no edits, no inventions):
{retrieved_assessments}

STRICT OUTPUT RULES:
1. Output ONLY a single JSON object. No markdown fences. No extra text.
2. NEVER include assessment names or URLs not listed above.
3. recommendations=[] for CLARIFY / COMPARE / REFUSE states.
4. recommendations=[1–10 items] for RECOMMEND / REFINE states.
5. Each item: {{"name": "exact name", "url": "exact url", "test_type": "letter"}}
6. end_of_conversation=true only when shortlist is delivered and conversation is complete.

{{"reply": "...", "recommendations": [], "end_of_conversation": false}}"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Call
# ─────────────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
def _call_llm(messages: list) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.05,   # Near-deterministic; we want grounded outputs
        max_tokens=1200,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_retrieved(assessments: List[dict]) -> str:
    """Format catalog items for injection into system prompt."""
    if not assessments:
        return "No relevant assessments found."
    lines = []
    for i, a in enumerate(assessments[:12], 1):
        lines.append(f"{i}. NAME: {a['name']}")
        lines.append(f"   URL:  {a['url']}")
        lines.append(f"   TYPE: {a.get('test_type','?')} — {a.get('test_type_label','')}")
        if a.get("description"):
            lines.append(f"   DESC: {a['description'][:180].strip()}")
        flags = []
        if a.get("remote_testing"):
            flags.append("Remote")
        if a.get("adaptive"):
            flags.append("Adaptive/IRT")
        if a.get("duration"):
            flags.append(f"Duration: {a['duration']}")
        if flags:
            lines.append(f"   {' | '.join(flags)}")
        lines.append("")
    return "\n".join(lines)


def _parse_llm_json(raw: str) -> dict:
    """Robustly parse JSON from LLM output, stripping markdown fences."""
    # Strip ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"```\s*$", "", cleaned)
    # Find first {...} block
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Fallback: wrap raw text
    return {"reply": raw[:300], "recommendations": [], "end_of_conversation": False}


def _validate_recommendations(
    data: dict,
    retrieved: List[dict],
) -> List[Recommendation]:
    """
    Anti-hallucination guard:
    Only accept recommendations whose URL or name exactly matches a retrieved item.
    Items that don't match are silently dropped.
    """
    url_map  = {a["url"]: a for a in retrieved}
    name_map = {a["name"].lower(): a for a in retrieved}

    valid: List[Recommendation] = []
    for rec in data.get("recommendations", []):
        url  = rec.get("url", "")
        name = rec.get("name", "")

        if url in url_map:
            entry = url_map[url]
        elif name.lower() in name_map:
            entry = name_map[name.lower()]
        else:
            logger.warning(f"Hallucination blocked: {name!r} / {url!r}")
            continue

        valid.append(Recommendation(
            name=entry["name"],
            url=entry["url"],
            test_type=entry["test_type"],
        ))

    return valid[:10]


def _direct_fallback_recs(retrieved: List[dict], n: int = 5) -> List[Recommendation]:
    """Use top retrieved items directly when LLM produces nothing valid."""
    return [
        Recommendation(name=a["name"], url=a["url"], test_type=a["test_type"])
        for a in retrieved[:n]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Responses (no LLM needed)
# ─────────────────────────────────────────────────────────────────────────────

_FAST_RESPONSES: dict = {
    IntentState.CLARIFY: ChatResponse(
        reply=(
            "I'd be happy to recommend SHL assessments! "
            "What role are you hiring for? "
            "(e.g., Java Developer, Sales Manager, Data Analyst)"
        ),
        recommendations=[],
        end_of_conversation=False,
    ),
    IntentState.REFUSE: ChatResponse(
        reply=(
            "I can only help you select SHL assessments for specific job roles. "
            "Could you describe the role you're hiring for?"
        ),
        recommendations=[],
        end_of_conversation=False,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def get_agent_reply(messages: List[Message], catalog_text: str = None) -> ChatResponse:
    """
    Core agent function.
    catalog_text kept for API compatibility with main.py (not used; we use hybrid retrieval).

    Flow:
      detect_intent → extract_search_params → hybrid_retrieve → LLM → validate → return
    """
    turn_number = sum(1 for m in messages if m.role == "assistant") + 1
    logger.info(f"=== Turn {turn_number} ===")

    # ── Step 1: Determine conversation state ──────────────────────────────────
    state = detect_intent(messages, turn_number)
    logger.info(f"State: {state.value}")

    # ── Step 2: Fast-path for CLARIFY and REFUSE (no retrieval or LLM needed) ─
    # Still use LLM for better reply quality, but have safe fallbacks ready
    if state == IntentState.REFUSE:
        return _FAST_RESPONSES[IntentState.REFUSE]

    # ── Step 3: Extract structured search parameters ──────────────────────────
    composite_query, preferred_types, remote, boost_keywords = extract_search_params(messages)

    # ── Step 4: Hybrid retrieval ───────────────────────────────────────────────
    retrieved: List[dict] = []
    try:
        from retrieval_hybrid import hybrid_retrieve_assessments
        # Retrieve more for RECOMMEND/REFINE (need diverse coverage for Recall@10)
        k = 12 if state in (IntentState.RECOMMEND, IntentState.REFINE) else 8
        retrieved = hybrid_retrieve_assessments(
            query=composite_query,
            k=k,
            preferred_types=preferred_types or None,
            remote=remote,
            boost_keywords=boost_keywords or None,
        )
        logger.info(f"Retrieved {len(retrieved)} assessments via hybrid retrieval")
    except Exception as exc:
        logger.error(f"Hybrid retrieval failed: {exc}", exc_info=True)
        try:
            from retrieval_rag import retrieve_assessments
            retrieved = retrieve_assessments(composite_query, k=8)
            logger.info(f"Fallback dense retrieval: {len(retrieved)} items")
        except Exception as exc2:
            logger.error(f"All retrieval failed: {exc2}")

    # ── CLARIFY fast-path (no good retrieval results) ──────────────────────────
    if state == IntentState.CLARIFY and not retrieved:
        return _FAST_RESPONSES[IntentState.CLARIFY]

    # ── Step 5: Build state-aware prompt and call LLM ─────────────────────────
    try:
        system_prompt = _SYSTEM_PROMPT.format(
            state=state.value.upper(),
            instruction=_STATE_INSTRUCTIONS[state],
            retrieved_assessments=_format_retrieved(retrieved),
        )

        llm_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            llm_messages.append({"role": m.role, "content": m.content})

        raw = _call_llm(llm_messages)
        data = _parse_llm_json(raw)

    except Exception as exc:
        logger.error(f"LLM call failed: {exc}", exc_info=True)
        # Safe fallback
        if state == IntentState.CLARIFY:
            return _FAST_RESPONSES[IntentState.CLARIFY]
        data = {
            "reply": "Here are the most relevant SHL assessments for your role.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ── Step 6: Validate & anti-hallucination guard ───────────────────────────
    if state in (IntentState.RECOMMEND, IntentState.REFINE):
        valid_recs = _validate_recommendations(data, retrieved)

        # Guarantee: RECOMMEND/REFINE always return ≥1 recommendation
        if not valid_recs and retrieved:
            logger.warning("LLM produced no valid recs; falling back to direct retrieval")
            valid_recs = _direct_fallback_recs(retrieved, n=5)
    else:
        # CLARIFY / COMPARE / REFUSE → always empty
        valid_recs = []

    # ── Step 7: end_of_conversation logic ─────────────────────────────────────
    end_conv = bool(data.get("end_of_conversation", False)) or (turn_number >= 7)

    return ChatResponse(
        reply=data.get("reply", "Here are the recommended SHL assessments:"),
        recommendations=valid_recs,
        end_of_conversation=end_conv,
    )