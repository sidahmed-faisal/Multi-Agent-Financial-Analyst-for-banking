# agents/agent_definitions.py
from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.schema import Document
import operator
import json
import re

class AgentState(TypedDict):
    """State for the multi-agent workflow"""
    query: str
    plan: List[str]
    context: List[Dict[str, Any]]
    current_step: int
    final_answer: Optional[str]
    retrieval_history: List[Dict[str, Any]]
    calculation_results: Dict[str, Any]
    metadata: Dict[str, Any]

class BaseAgent:
    """Base class for all agents with common functionality"""
    
    def __init__(self, llm, name: str, description: str):
        self.llm = llm
        self.name = name
        self.description = description
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for LLM consumption"""
        formatted = []
        for item in context:
            if isinstance(item, dict):
                if 'content' in item and 'metadata' in item:
                    source = item['metadata'].get('source_document', 'Unknown')
                    section = item['metadata'].get('section_name', 'Unknown')
                    page = item['metadata'].get('page_number', 'Unknown')
                    formatted.append(f"Source: {source} | Section: {section} | Page: {page}\nContent: {item['content']}")
                else:
                    formatted.append(str(item))
            else:
                formatted.append(str(item))
        return "\n\n".join(formatted)