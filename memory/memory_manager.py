# memory/memory_manager.py
from typing import List, Dict
from loguru import logger
from config import config


class ConversationMemory:
    """
    WHAT IS RAG MEMORY?
    Without memory, every question is asked by a stranger.
    With memory, the system knows what was discussed.
    
    Example without memory:
    Q1: "What is the penalty clause in the contract?"
    Q2: "What happens if it's violated?"
    → System has no idea what "it" refers to.
    
    Example with memory:
    Q1: "What is the penalty clause in the contract?"
    → System answers, stores Q1+A1
    Q2: "What happens if it's violated?"
    → System injects: "Previous context: we discussed the penalty clause..."
    → Now retrieval finds relevant chunks about penalty clause violations
    
    TWO TYPES WE IMPLEMENT:
    1. Short-term: last N Q&A pairs (sliding window)
    2. Entity: tracks important terms mentioned (contract names, parties, dates)
    """
    
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.entities: Dict[str, str] = {}  # entity name → context
    
    def add_turn(self, question: str, answer: str):
        """Store one Q&A exchange."""
        self.history.append({"question": question, "answer": answer})
        
        # Keep only last N turns (sliding window)
        max_turns = config.max_history_turns
        if len(self.history) > max_turns:
            self.history = self.history[-max_turns:]
        
        logger.debug(f"Memory: {len(self.history)} turns stored")
    
    def get_context_string(self) -> str:
        """
        Format history for injection into the prompt.
        This becomes part of the LLM's context window.
        """
        if not self.history:
            return ""
        
        lines = ["=== CONVERSATION HISTORY ==="]
        for turn in self.history:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer'][:300]}...")  # truncate long answers
        lines.append("=== END HISTORY ===")
        return "\n".join(lines)
    
    def build_contextualized_query(self, new_query: str) -> str:
        """
        Enhance the query with conversation context before retrieval.
        
        WHY: If user asks "what about the termination clause?" after discussing
        a specific contract, retrieval needs the full context to find 
        the right chunks. Without this, "termination clause" alone might
        retrieve chunks from the wrong document.
        """
        if not self.history:
            return new_query
        
        # Take last 2 turns for query contextualization
        recent = self.history[-2:]
        context_parts = [new_query]
        for turn in recent:
            context_parts.append(turn["question"])
        
        contextualized = " ".join(context_parts)
        logger.debug(f"Contextualized query: {contextualized[:150]}")
        return contextualized
    
    def clear(self):
        self.history = []
        self.entities = {}
        logger.info("Memory cleared")