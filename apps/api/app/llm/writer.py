"""The Bangla narrative writer.

Implementations behind the `Writer` protocol:

* `GeminiWriter`   - primary Google Gemini Flash writer (with model fallback) + hard validator.
* `LlmWriter`      - optional OpenAI-compatible writer + hard validator.
* `TemplateWriter` - composes from the same facts with no network at all.

The template path is not a stub. It is what runs automatically whenever no LLM
key is configured, the provider is unreachable/rate-limited, or the model
returns unusable text, so the product never depends on an external API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from typing import Any, Protocol

from pydantic import BaseModel

from ..config import settings

log = logging.getLogger(__name__)

#: Bengali block plus whitespace, the danda, and the usual Indic punctuation.
_BANGLA = re.compile(r"[\u0980-\u09FF]")
_LATIN = re.compile(r"[A-Za-z]")
_MARKDOWN = re.compile(r"[*_`#>\[\]{}|]")
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\ufe0f\u2b00-\u2bff]"
)
_WHITESPACE = re.compile(r"\s+")

MIN_WORDS = 25
MAX_WORDS = 90


class Writer(Protocol):
    async def write(self, facts: dict[str, Any]) -> tuple[str, str]: ...


def normalise(text: str) -> str:
    """NFC-normalise, collapse whitespace.

    Bengali text has more than one valid encoding for the same glyphs (`য়` as one
    code point or two). Normalising before comparing or caching keeps equality
    checks honest.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


class InvalidNarrative(ValueError):
    """Raised when model output fails a safety or quality check."""


def validate_narrative(text: str, allowed_chemicals: list[str]) -> str:
    """Reject anything unsafe or unusable. Returns the cleaned text.

    The Latin-script check is the most important one here: the TTS normaliser
    *silently drops* Latin spans, so a Latin chemical name would vanish from the
    audio while leaving a fluent, complete-sounding sentence behind.
    """
    cleaned = normalise(text)

    if not cleaned:
        raise InvalidNarrative("empty")
    if not _BANGLA.search(cleaned):
        raise InvalidNarrative("no_bangla_script")
    if _LATIN.search(cleaned):
        raise InvalidNarrative(f"latin_script:{_LATIN.search(cleaned).group(0)}")  # type: ignore[union-attr]
    if _MARKDOWN.search(cleaned):
        raise InvalidNarrative("markdown")
    if _EMOJI.search(cleaned):
        raise InvalidNarrative("emoji")

    words = cleaned.split()
    if not (MIN_WORDS <= len(words) <= MAX_WORDS):
        raise InvalidNarrative(f"word_count:{len(words)}")

    # Every chemical the model names must be one the knowledge base supplied for
    # *this* crop. Latin is already rejected above, so the realistic failure is a
    # plausible Bangla name for a product that was never in the facts - which
    # reads perfectly and would be wrong.
    for candidate in KNOWN_CHEMICALS_BN:
        if candidate in cleaned and not any(candidate in allowed for allowed in allowed_chemicals):
            raise InvalidNarrative(f"unknown_medicine:{candidate}")

    return cleaned


#: Bangla names of active ingredients common in Bangladesh, used only to *detect*
#: a chemical in the text. Presence in this list grants nothing; a detected name
#: still has to appear in the facts handed to the model for that crop. Grown from
#: the knowledge base plus products a model plausibly reaches for.
KNOWN_CHEMICALS_BN: tuple[str, ...] = (
    # -- in the knowledge base ------------------------------------------
    "ট্রাইসাইক্লাজল",
    "আইসোপ্রোথায়োলেন",
    "কপার হাইড্রোক্সাইড",
    "ম্যানকোজেব",
    "মেটালাক্সিল",
    "ক্লোরোথ্যালোনিল",
    "ইমিডাক্লোপ্রিড",
    "কার্বেন্ডাজিম",
    # -- plausible but not supplied for every crop ---------------------
    "প্রোপিকোনাজল",
    "হেক্সাকোনাজল",
    "টেবুকোনাজল",
    "অ্যাজোক্সিস্ট্রবিন",
    "ডাইফেনোকোনাজল",
    "কার্বারিল",
    "ক্লোরপাইরিফস",
    "সাইপারমেথ্রিন",
    "ডাইমেথোয়েট",
    "থায়ামেথোক্সাম",
    "ফিপ্রোনিল",
    "অ্যাবামেক্টিন",
    "বুপ্রোফেজিন",
    "পাইমেট্রোজিন",
    "অ্যাসিটামিপ্রিড",
    "থায়োফানেট মিথাইল",
    "কপার অক্সিক্লোরাইড",
    "সালফার",
    "জিংক থায়োফসফেট",
)

SYSTEM_PROMPT = """\
তুমি বাংলাদেশের একজন কৃষি পরামর্শক। কৃষকের জন্য ছোট, সহজ, মুখে বলার মতো একটি পরামর্শ লেখো।

নিয়ম:
- সবসময় চলিত বাংলায়, কেবল বাংলা অক্ষরে লিখবে। কোনো ইংরেজি অক্ষর, রোমান হরফ বা ইংরেজি শব্দ ব্যবহার করবে না।
- ২ থেকে ৪ বাক্য। ২৫ থেকে ৯০ শব্দের মধ্যে রাখবে।
- একজন কৃষক আরেকজন কৃষককে যেভাবে বলেন, সেভাবে সরল ভাষায় লিখবে। সম্মানসূচক 'আপনি' ব্যবহার করবে।
- কোনো মার্কডাউন, বুলেট, ইমোজি, লিংক বা তালিকা নয়। শুধু টানা বাক্য।
- ওষুধ ও মাত্রা কেবল নিচে দেওয়া তথ্য থেকে নেবে। নিজের থেকে কোনো নতুন ওষুধ, মাত্রা বা রোগের নাম বানাবে না।
- ওষুধের নাম বাংলা অক্ষরে লিখবে, যেমনটি তথ্যে দেওয়া আছে।
- রাসায়নিক স্প্রে উল্লেখ থাকলে সংক্ষেপে গ্লাভস ও মাস্ক পরা এবং সাবান দিয়ে হাত ধোয়ার কথা মনে করিয়ে দেবে।
- সংখ্যা বাংলা অঙ্কে লিখবে (০-৯)।
- আবহাওয়ার তথ্য না থাকলে আবহাওয়ার কথা একেবারেই বলবে না।
- রোগ নিশ্চিত না হলে সেটি স্পষ্ট বলবে এবং উপজেলা কৃষি কর্মকর্তার পরামর্শ নিতে বলবে।
- শেষে সবসময় মনে করিয়ে দেবে যে এটি একটি আনুমানিক পরামর্শ।
"""

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
FALLBACK_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
)
_RETRYABLE_GEMINI_CODES = frozenset({404, 429, 503})


class _SpokenAdvice(BaseModel):
    spoken: str


def _candidate_gemini_models() -> list[str]:
    primary = settings.gemini_model.strip() or DEFAULT_GEMINI_MODEL
    candidates = [primary]
    for model in FALLBACK_GEMINI_MODELS:
        if model not in candidates:
            candidates.append(model)
    return candidates


class GeminiWriter:
    """Primary Bangla advisory writer backed by Google Gemini Flash."""

    def __init__(self) -> None:
        from google import genai
        from google.genai import types

        api_key = settings.effective_gemini_api_key
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or LLM_API_KEY) is not configured for GeminiWriter."
            )

        self._client: Any = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=settings.llm_timeout_ms),
        )
        self._semaphore = asyncio.Semaphore(settings.llm_max_concurrency)

    async def write(self, facts: dict[str, Any]) -> tuple[str, str]:
        allowed = [chemical["bnName"] for chemical in facts["treatment"]["chemical"]]
        last_error: Exception | None = None

        for attempt in (1, 2):
            try:
                text = await self._attempt(facts, allowed, is_retry=attempt == 2)
                return text, "llm"
            except InvalidNarrative as exc:
                last_error = exc
                log.warning("Gemini output rejected (attempt %d): %s", attempt, exc)
            except Exception as exc:
                log.warning("Gemini call failed: %s", exc.__class__.__name__)
                raise

        raise InvalidNarrative(f"unusable_after_retry:{last_error}")

    async def _attempt(self, facts: dict[str, Any], allowed: list[str], *, is_retry: bool) -> str:
        from google.genai import types

        payload = json.dumps(facts, ensure_ascii=False, indent=2)
        if is_retry:
            user_prompt = (
                "আগের উত্তরটি নিয়ম মানেনি। এবার কেবল বাংলা অক্ষরে, "
                "২ থেকে ৪ বাক্যে, কোনো ইংরেজি অক্ষর ছাড়া লেখো।\n\n"
                f"তথ্য:\n{payload}"
            )
        else:
            user_prompt = f"তথ্য:\n{payload}"

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_SpokenAdvice,
            temperature=0.35,
        )

        async with self._semaphore:
            response = await asyncio.to_thread(
                self._generate_with_fallback, user_prompt, config
            )

        raw_text = (getattr(response, "text", None) or "").strip()
        if not raw_text:
            raise InvalidNarrative("empty")

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise InvalidNarrative("invalid_json") from exc

        if not isinstance(parsed, dict) or not isinstance(parsed.get("spoken"), str):
            raise InvalidNarrative("missing_spoken")

        return validate_narrative(parsed["spoken"], allowed)

    def _generate_with_fallback(self, user_prompt: str, config: Any) -> Any:
        from google.genai import errors as genai_errors

        last_exc: Exception | None = None
        for model_name in _candidate_gemini_models():
            try:
                return self._client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=config,
                )
            except genai_errors.APIError as exc:
                last_exc = exc
                if exc.code in _RETRYABLE_GEMINI_CODES:
                    logger_msg = "Gemini model %s returned HTTP %s; trying fallback model."
                    log.warning(logger_msg, model_name, exc.code)
                    continue
                raise

        raise RuntimeError(f"All Gemini Flash models failed: {last_exc}") from last_exc

_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "speak_advice",
        "description": "কৃষকের জন্য মুখে বলার মতো সংক্ষিপ্ত বাংলা পরামর্শ",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"spoken": {"type": "string", "description": "বাংলা ভাষায় ২-৪ বাক্যের পরামর্শ"}},
            "required": ["spoken"],
            "additionalProperties": False,
        },
    },
}


class LlmWriter:
    def __init__(self) -> None:
        # Lazy import keeps the package importable without the SDK installed.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_ms / 1000,
            # The SDK's default retry would re-send on 429 window_exhausted,
            # which resets in hours. Retrying that burns quota for nothing.
            max_retries=0,
        )
        self._semaphore = asyncio.Semaphore(settings.llm_max_concurrency)

    async def write(self, facts: dict[str, Any]) -> tuple[str, str]:
        allowed = [chemical["bnName"] for chemical in facts["treatment"]["chemical"]]
        last_error: Exception | None = None

        for attempt in (1, 2):
            try:
                text = await self._attempt(facts, allowed, is_retry=attempt == 2)
                return text, "llm"
            except InvalidNarrative as exc:
                last_error = exc
                log.warning("LLM output rejected (attempt %d): %s", attempt, exc)
            except Exception as exc:  # network, auth, rate limit, spend cap
                log.warning("LLM call failed: %s", exc.__class__.__name__)
                raise

        raise InvalidNarrative(f"unusable_after_retry:{last_error}")

    async def _attempt(self, facts: dict[str, Any], allowed: list[str], *, is_retry: bool) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        payload = json.dumps(facts, ensure_ascii=False, indent=2)
        if is_retry:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "আগের উত্তরটি নিয়ম মানেনি। এবার কেবল বাংলা অক্ষরে, "
                        "২ থেকে ৪ বাক্যে, কোনো ইংরেজি অক্ষর ছাড়া লেখো।\n\n"
                        f"তথ্য:\n{payload}"
                    ),
                }
            )
        else:
            messages.append({"role": "user", "content": f"তথ্য:\n{payload}"})

        async with self._semaphore:
            response = await self._client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,  # type: ignore[arg-type]
                tools=[_TOOL],  # type: ignore[list-item]
                tool_choice={"type": "function", "function": {"name": "speak_advice"}},
                temperature=0.4,
                # Bengali needs more tokens than English for the same content;
                # a limit tuned in English truncates the JSON.
                max_tokens=600,
            )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise InvalidNarrative("truncated")

        calls = choice.message.tool_calls or []
        if not calls:
            raise InvalidNarrative("no_tool_call")

        arguments = calls[0].function.arguments
        spoken = json.loads(arguments)["spoken"]
        return validate_narrative(spoken, allowed)


class TemplateWriter:
    """Composes the narrative from facts alone. No network, no model, no risk."""

    async def write(self, facts: dict[str, Any]) -> tuple[str, str]:
        return self.compose(facts), "template"

    def compose(self, facts: dict[str, Any]) -> str:
        diagnosis = facts["diagnosis"]
        treatment = facts["treatment"]
        weather = facts.get("weather")
        uncertain: bool = facts.get("uncertain", False)

        parts: list[str] = []

        if uncertain:
            parts.append(
                f"আপনার {diagnosis['crop']} গাছের ছবি ভালোভাবে শনাক্ত করা যায়নি।"
                " অনুগ্রহ করে আরও পরিষ্কার ছবি দিন, অথবা উপজেলা কৃষি কর্মকর্তাকে দেখান।"
            )
        else:
            parts.append(f"আপনার {diagnosis['crop']} গাছে সম্ভবত {diagnosis['disease']} হয়েছে।")

            symptoms = facts.get("symptoms_observed") or []
            if symptoms:
                parts.append(f"লক্ষণ হিসেবে দেখা যাচ্ছে: {symptoms[0]}।")

            chemicals = treatment.get("chemical") or []
            if chemicals:
                first = chemicals[0]
                parts.append(
                    f"প্রাথমিকভাবে {first['bnName']} ব্যবহার করতে পারেন, "
                    f"মাত্রা {first['dose']}। প্রয়োগের আগে কৃষি কর্মকর্তার পরামর্শ নিন।"
                )
            elif treatment.get("organic"):
                parts.append(f"{treatment['organic'][0]}।")

        if weather:
            if weather["spray_window_ok"]:
                parts.append("আবহাওয়া এখন স্প্রে করার অনুকূল।")
            else:
                parts.append(f"তবে এখনই স্প্রে করবেন না, কারণ {weather['reason']}।")

        parts.append("মনে রাখবেন, এটি একটি আনুমানিক পরামর্শ।")

        return normalise(" ".join(parts))


def build_writer() -> Writer:
    if not settings.llm_configured:
        return TemplateWriter()

    if settings.llm_provider == "openai":
        try:
            return LlmWriter()
        except Exception as exc:  # noqa: BLE001 - any construction failure degrades
            log.error("OpenAI-compatible writer unavailable (%s); falling back to template", exc)
            return TemplateWriter()

    try:
        return GeminiWriter()
    except Exception as exc:  # noqa: BLE001 - any construction failure degrades
        log.error("Gemini writer unavailable (%s); falling back to template", exc)
        return TemplateWriter()


_writer: Writer | None = None


def get_writer() -> Writer:
    global _writer
    if _writer is None:
        _writer = build_writer()
    return _writer


def reset_writer() -> None:
    """Drop the cached writer. Used by tests after changing settings."""
    global _writer
    _writer = None
