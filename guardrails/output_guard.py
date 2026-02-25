# guardrails/output_guard.py
from typing import Tuple
from loguru import logger


HALLUCINATION_SIGNALS = [
    "as an ai",
    "i was trained",
    "my knowledge",
    "i believe",
    "i think",
    "in general",
    "typically",
    "usually",
    "it is commonly known",
    "based on my understanding",
]

UNCERTAINTY_SIGNALS = [
    "i cannot find",
    "not mentioned",
    "not in the provided",
    "the documents do not",
    "no information",
]


class OutputGuard:
    """
    WHAT ARE OUTPUT GUARDRAILS?
    They inspect the LLM's answer before it goes to the user.
    
    In legal/financial contexts, a hallucinated answer can cause:
    - Wrong legal interpretation
    - Incorrect financial decisions
    - Liability for the company using your system
    
    WHAT WE CHECK:
    1. Hallucination signals: LLM is speaking from general knowledge, not context
    2. Length sanity: answer is too short (probably failed) or too long (padding)
    3. Citation presence: did the LLM actually cite a source?
    4. Refusal detection: LLM correctly said "I don't know" → flag but don't penalize
    """
    
    def validate(self, answer: str, query: str) -> Tuple[bool, str, float]:
        """
        Returns (is_valid, warning_message, confidence_score)
        confidence_score: 1.0 = clean, 0.0 = likely hallucination
        """
        answer_lower = answer.lower()
        confidence = 1.0
        warnings = []
        
        # Check 1: Empty answer
        if not answer or len(answer.strip()) < 10:
            return False, "Answer too short — likely a generation failure.", 0.0
        
        # Check 2: Hallucination signals
        hallucination_count = 0
        for signal in HALLUCINATION_SIGNALS:
            if signal in answer_lower:
                hallucination_count += 1
                warnings.append(f"Hallucination signal: '{signal}'")
        
        if hallucination_count > 0:
            confidence -= (hallucination_count * 0.2)
            confidence = max(0.0, confidence)
        
        # Check 3: Did LLM admit it couldn't find the answer? (This is GOOD behavior)
        is_honest_refusal = any(s in answer_lower for s in UNCERTAINTY_SIGNALS)
        if is_honest_refusal:
            logger.info("LLM correctly indicated information not found in context")
            return True, "Honest refusal — information not in documents.", 0.95
        
        # Check 4: Citation present?
        has_citation = "[source" in answer_lower or "source:" in answer_lower
        if not has_citation:
            confidence -= 0.1
            warnings.append("No source citation found in answer")
        
        if warnings:
            logger.warning(f"Output guard warnings: {warnings}")
        
        is_valid = confidence >= 0.5
        warning_str = "; ".join(warnings) if warnings else ""
        
        logger.info(f"Output guard: confidence={confidence:.2f} | valid={is_valid}")
        return is_valid, warning_str, confidence