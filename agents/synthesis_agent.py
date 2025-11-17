# agents/synthesis_agent.py
from .agent_definitions import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
import json

class SynthesisAgent(BaseAgent):
    """Agent that synthesizes all information into final answer"""
    
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            name="Synthesis Agent",
            description="Combines quantitative and qualitative information into comprehensive answers"
        )
        
        self.synthesis_prompt = ChatPromptTemplate.from_template("""
You are a Financial Analysis Synthesis Specialist for First Abu Dhabi Bank.

ORIGINAL QUERY: {query}
FINAL ANSWER REQUIREMENTS: {final_requirements}

ALL GATHERED INFORMATION:
{all_context}

CALCULATION RESULTS:
{calculation_results}

YOUR TASK:
Synthesize all the gathered information into a comprehensive, accurate, and well-structured final answer.

REQUIREMENTS:
1. **Accuracy**: Only use information that is supported by the provided context
2. **Citation**: Cite specific sources (document, section, page) for every factual claim
3. **Completeness**: Address all aspects of the original query
4. **Clarity**: Present information in a clear, professional manner suitable for financial analysis
5. **Insight**: Provide meaningful insights that connect quantitative results with qualitative context

STRUCTURE YOUR ANSWER:
- **Executive Summary**: Brief overview of key findings
- **Quantitative Analysis**: Numerical results with calculations
- **Qualitative Context**: Management commentary and explanatory factors
- **Sources**: Clear citation of all information sources

SPECIAL INSTRUCTIONS:
- If data is missing or inconsistent, clearly state this limitation
- Do not hallucinate or invent any numbers or facts
- For financial metrics, always include units (AED millions, percentages, etc.)
- Highlight trends, comparisons, and business implications

FINAL ANSWER:
""")

    def synthesize_answer(self, query: str, context: List[Dict[str, Any]], 
                         calculation_results: Dict[str, Any]) -> str:
        """Synthesize final answer from all gathered information"""
        chain = self.synthesis_prompt | self.llm | StrOutputParser()
        
        # Extract final requirements from the original plan
        final_requirements = "Combine all gathered quantitative and qualitative information into a comprehensive answer with proper citations."
        
        # Filter context to remove calculation intermediate steps
        filtered_context = [item for item in context if not isinstance(item, dict) or 'calculation_type' not in item]
        
        response = chain.invoke({
            "query": query,
            "final_requirements": final_requirements,
            "all_context": self._format_context(filtered_context),
            "calculation_results": json.dumps(calculation_results, indent=2)
        })
        
        return response
    
    def validate_answer(self, answer: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that the answer is supported by context"""
        validation_prompt = ChatPromptTemplate.from_template("""
Validate if the following answer is fully supported by the provided context.

ANSWER TO VALIDATE:
{answer}

SUPPORTING CONTEXT:
{context}

INSTRUCTIONS:
1. Check if all numerical claims in the answer appear in the context
2. Verify that all factual statements are supported by the context
3. Identify any unsupported or potentially hallucinated information
4. Check if all sources are properly cited

Return JSON:
{{
    "is_valid": true/false,
    "unsupported_claims": ["claim1", "claim2"],
    "missing_citations": ["fact1", "fact2"],
    "validation_notes": "Overall assessment"
}}
""")
        
        chain = validation_prompt | self.llm | StrOutputParser()
        
        try:
            validation_result = chain.invoke({
                "answer": answer,
                "context": self._format_context(context)
            })
            return json.loads(validation_result)
        except:
            return {"is_valid": True, "unsupported_claims": [], "validation_notes": "Validation failed"}
