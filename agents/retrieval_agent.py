# agents/retrieval_agent.py
from .agent_definitions import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import re, json
from langchain_core.output_parsers import StrOutputParser
import logging

logging.basicConfig(level=logging.INFO)


class RetrievalAgent(BaseAgent):
    """Specialized agent for retrieving financial information"""
    
    def __init__(self, llm, chunker):
        super().__init__(
            llm=llm,
            name="Retrieval Agent",
            description="Finds and extracts financial data from documents"
        )
        self.chunker = chunker
        
        self.retrieval_prompt = ChatPromptTemplate.from_template("""
You are a Financial Data Retrieval Specialist for First Abu Dhabi Bank.

QUERY: {query}
CURRENT STEP: {current_step}
CONTEXT SO FAR: {context}

Your task is to formulate optimized search queries to find the specific financial information needed.

FINANCIAL DOCUMENT STRUCTURE:
- Financial Statements: Income Statement, Balance Sheet, Cash Flow Statement, Notes
- Earnings Presentations: Key metrics, management commentary, charts
- Results Calls: Management discussions, Q&A sessions, Operator statements, Q&A transcripts

COMMON FINANCIAL TERMS TO SEARCH FOR:
- Net Profit: "Net Profit After Tax", "NPAT", "Net Income", "Profit for the period"
- Revenue: "Total Revenue", "Operating Income", "Net Operating Income"  
- Assets: "Total Assets", "Assets", "Statement of Financial Position"
- Loans: "Total Loans", "Loans and Advances", "Financing Assets"
- Deposits: "Total Deposits", "Customer Deposits", "Deposits from Customers"

QUARTER/YEAR IDENTIFICATION:
Look for patterns like "Q1 2023", "Q1'23", "2023 Q1" in the query and context.

IMPORTANT FILTERING GUIDANCE:
- Only include filters that are ESSENTIAL and likely to exist in the document store
- Avoid filtering on section_name - this is unreliable and specific metadata
- For results calls, do NOT filter by section - instead, enhance search queries to find speakers/content
- If filtering would result in ZERO results, omit those filters and rely on semantic search
- Prefer broad semantic search over strict filtering - let embeddings find relevant content

Based on the current step and context, create specific search queries to find the required information.

Return your response as a JSON object with:
{{
    "search_queries": ["query1", "query2", ...],
    "filters": {{
        "quarter": "quarter wether it's Q1, Q2, Q3, or Q4",
        "year": "Year",
        "document_type": "wether it's a financial statement, earnings presentation, or results call"
    }},
    "reasoning": "Explanation of why these queries were chosen and why specific filters are applied"
}}

RESPONSE:
""")

    def execute_retrieval(self, query: str, current_step: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute retrieval based on the current step"""
        chain = self.retrieval_prompt | self.llm | StrOutputParser()
        
        try:
            response_text = chain.invoke({
                "query": query,
                "current_step": current_step,
                "context": self._format_context(context)
            })
            
            # Parse the JSON response
            retrieval_plan = json.loads(response_text)
            search_queries = retrieval_plan.get("search_queries", [query])
            filters = retrieval_plan.get("filters", {})

            logging.info(f"Retrieval search queries: {search_queries}")
            logging.info(f"filters applied: {filters}")
            
            # Execute searches with filters
            all_results = []
            for search_query in search_queries:
                results = self.chunker.search_chunks(
                    query=search_query,
                    filters=filters if filters else None,
                    n_results=5
                )
                all_results.extend(results)
            
            # If no results found with filters, retry without filters for broader search
            if len(all_results) == 0 and filters:
                logging.info(f"No results with filters {filters}, retrying without filters...")
                for search_query in search_queries:
                    results = self.chunker.search_chunks(
                        query=search_query,
                        filters=None,  # Try without filters
                        n_results=5
                    )
                    all_results.extend(results)
            
            # Remove duplicates based on content
            unique_results = []
            seen_content = set()
            for result in all_results:
                content_hash = hash(result['content'][:100])  # Hash first 100 chars
                if content_hash not in seen_content:
                    unique_results.append(result)
                    seen_content.add(content_hash)
            
            logging.info(f"Retrieved {len(unique_results)} unique results after deduplication ")
            return unique_results
            
        except Exception as e:
            print(f"Error in retrieval: {e}")
            # Fallback: simple search
            return self.chunker.search_chunks(query=query, n_results=5)
    
    def extract_financial_value(self, content: str, metric: str) -> float:
        """Extract numerical values from financial text"""
        # Patterns for financial numbers (AED millions/billions)
        patterns = [
            rf"{metric}.*?AED\s*([\d,]+\.?\d*)\s*(?:million|bn|billion)",
            rf"{metric}.*?([\d,]+\.?\d*)\s*(?:million|bn|billion)\s*AED",
            rf"{metric}.*?([\d,]+\.?\d*)",  # Just the number
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Take the first match and clean it
                value_str = matches[0].replace(',', '')
                try:
                    return float(value_str)
                except ValueError:
                    continue
        
        return None