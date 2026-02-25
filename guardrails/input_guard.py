# guardrails/input_guard.py
from typing import Tuple
from loguru import logger


LEGAL_FINANCIAL_KEYWORDS = [
    "contract", "clause", "agreement", "liability", "indemnif",
    "payment", "penalty", "term", "condition", "party", "parties",
    "revenue", "profit", "loss", "tax", "asset", "equity", "debt",
    "compliance", "regulation", "breach", "damages", "settlement",
    "document", "section", "article", "provision", "obligation",
    "what", "when", "how", "who", "where", "which", "summarize",
    "explain", "define", "list", "compare", "analyze"
]

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "forget your instructions",
    "you are now",
    "act as",
    "pretend you are",
    "disregard",
    "override",
    "jailbreak",
    "your new instructions",
]


class InputGuard:
    """
    WHAT ARE INPUT GUARDRAILS?
    They're the bouncer at the door.
    They check: is this query safe and relevant before we spend
    compute, tokens, and money processing it?
    
    TYPES WE CHECK:
    1. Prompt injection: attacker tries to override your system prompt
    2. Off-topic queries: user asks about unrelated topics
    3. Empty/malformed input: garbage in, garbage out
    4. Query length: too long = likely abuse or token stuffing
    
    WHY THIS MATTERS IN PRODUCTION:
    Without input guards, users can:
    - Make your LLM behave as a different chatbot
    - Waste your API budget on off-topic requests
    - Probe your system prompt through clever phrasing
    """
    
    def validate(self, query: str) -> Tuple[bool, str]:
        """
        Returns (is_valid, reason_if_invalid)
        """
        # Check 1: Empty query
        if not query or not query.strip():
            return False, "Query cannot be empty."
        
        # Check 2: Length limits
        if len(query) > 2000:
            return False, "Query too long. Please keep questions under 2000 characters."
        
        if len(query.strip()) < 3:
            return False, "Query too short to process meaningfully."
        
        # Check 3: Prompt injection detection
        query_lower = query.lower()
        for pattern in INJECTION_PATTERNS:
            if pattern in query_lower:
                logger.warning(f"Prompt injection attempt detected: '{pattern}'")
                return False, "Invalid query format detected."
        
        # Check 4: Domain relevance (soft check — warns but allows)
        # For strict systems, change this to return False
        has_relevant_term = any(kw in query_lower for kw in LEGAL_FINANCIAL_KEYWORDS)
        if not has_relevant_term:
            logger.warning(f"Potentially off-topic query: {query[:100]}")
            # We still allow it but log it — could be a general question about docs
        
        return True, ""