# agents/orchestrator_agent.py
from .agent_definitions import BaseAgent, AgentState
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any
import json, re
from langchain_core.output_parsers import StrOutputParser


class OrchestratorAgent(BaseAgent):
    """Main agent that plans and coordinates the workflow"""
    
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            name="Orchestrator",
            description="Plans and coordinates multi-step financial analysis"
        )
        
        self.planning_prompt = ChatPromptTemplate.from_template("""
You are a Financial Analysis Orchestrator for First Abu Dhabi Bank (FAB). 
Your role is to break down complex financial queries into executable steps.

USER QUERY: {query}

AVAILABLE TOOLS:
1. RETRIEVAL - Find specific financial data from documents
2. CALCULATION - Perform mathematical operations and financial ratios
3. SYNTHESIS - Combine information into comprehensive answers

AVAILABLE DOCUMENT TYPES:
- Financial Statements (Balance Sheet, Income Statement, Cash Flow)
- Earnings Presentations (Management commentary, charts, metrics)
- Results Call Transcripts (Management and analyst discussions)

COMMON FINANCIAL CONCEPTS:
- Net Profit, Revenue, Assets, Liabilities, Equity
- ROE (Return on Equity) = Net Income / Shareholder's Equity
- Loan-to-Deposit Ratio = Total Loans / Total Deposits
- YoY (Year-over-Year) change = (Current Year - Previous Year) / Previous Year * 100
- QoQ (Quarter-over-Quarter) change

BREAKDOWN REQUIREMENTS:
1. Identify what specific data points are needed
2. Determine which quarters/years to compare
3. Plan retrieval steps in logical order
4. Include calculation steps where needed
5. Plan synthesis of quantitative and qualitative information

OUTPUT FORMAT:
Return a JSON array of steps. Each step should be a string describing the action.

Example for "YoY Net Profit change Q3 2023 vs Q3 2024":
[
    "RETRIEVE: Net Profit After Tax for Q3 2023 from financial statements",
    "RETRIEVE: Net Profit After Tax for Q3 2024 from financial statements", 
    "CALCULATE: Percentage change between Q3 2023 and Q3 2024 Net Profit",
    "RETRIEVE: Management commentary on profit drivers from Q3 2024 earnings presentation and results call",
    "SYNTHESIZE: Combine numerical results with qualitative factors into comprehensive answer"
]

Now, analyze this query and create a step-by-step plan:

QUERY: {query}

PLAN:
""")

    def create_plan(self, query: str) -> List[str]:
        """Create execution plan for the query"""
        chain = self.planning_prompt | self.llm | StrOutputParser()
        
        try:
            plan_text = chain.invoke({"query": query})
            # Extract JSON array from the response
            plan_match = re.search(r'\[.*\]', plan_text, re.DOTALL)
            if plan_match:
                plan = json.loads(plan_match.group())
            else:
                # Fallback: split by newlines and clean
                plan = [step.strip() for step in plan_text.split('\n') if step.strip()]
            
            return plan if plan else ["RETRIEVE: General information about the query"]
            
        except Exception as e:
            print(f"Error creating plan: {e}")
            return ["RETRIEVE: General information about the query"]
    
    def should_continue(self, state: AgentState) -> str:
        """Determine if we should continue or finish"""
        plan = state.get('plan', [])
        current_step = state.get('current_step', 0)
        
        if current_step >= len(plan):
            return "finish"
        
        next_step = plan[current_step]
        if next_step.upper().startswith('SYNTHESIZE'):
            return "synthesis"
        elif next_step.upper().startswith('CALCULATE'):
            return "calculation"
        else:
            return "retrieval"