# agents/calculation_agent.py
from .agent_definitions import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any

import numexpr
import re, json

class CalculationAgent(BaseAgent):
    """Specialized agent for financial calculations"""
    
    def __init__(self, llm):
        super().__init__(
            llm=llm,
            name="Calculation Agent", 
            description="Performs financial calculations and ratio analysis"
        )
        
        self.calculation_prompt = ChatPromptTemplate.from_template("""
You are a Financial Calculation Specialist. Your role is to perform accurate mathematical operations for financial analysis.

CALCULATION REQUEST: {calculation_request}
DATA CONTEXT: {data_context}

AVAILABLE DATA POINTS:
{available_data}

COMMON FINANCIAL FORMULAS:
- Percentage Change: (New Value - Old Value) / Old Value * 100
- ROE (Return on Equity): Net Income / Shareholder's Equity * 100  
- Loan-to-Deposit Ratio: Total Loans / Total Deposits
- Growth Rate: (Current Period - Previous Period) / Previous Period * 100
- Ratio: Numerator / Denominator

INSTRUCTIONS:
1. Identify which formula to use based on the calculation request
2. Extract the necessary numerical values from the data context
3. Perform the calculation safely using proper error handling
4. Return the result with proper formatting and units

Return your response as JSON:
{{
    "calculation_type": "percentage_change|ratio|growth_rate|custom",
    "formula_used": "mathematical formula",
    "input_values": {{
        "value1": 123.45,
        "value2": 67.89
    }},
    "result": 81.82,
    "units": "percentage|ratio|AED_millions",
    "explanation": "Step-by-step explanation",
    "validation": "Data sources and validation notes"
}}

RESPONSE:
""")

    def perform_calculation(self, calculation_request: str, data_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform financial calculation based on request and data"""
        chain = self.calculation_prompt | self.llm | StrOutputParser()
        
        try:
            # Extract numerical values from context
            available_data = self._extract_numerical_data(data_context)
            
            response_text = chain.invoke({
                "calculation_request": calculation_request,
                "data_context": self._format_context(data_context),
                "available_data": json.dumps(available_data, indent=2)
            })
            
            calculation_result = json.loads(response_text)
            
            # Validate and execute the calculation
            validated_result = self._validate_and_execute_calculation(calculation_result)
            return validated_result
            
        except Exception as e:
            print(f"Error in calculation: {e}")
            return {
                "calculation_type": "error",
                "result": None,
                "error": str(e),
                "units": "unknown"
            }
    
    def _extract_numerical_data(self, context: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract numerical data from context"""
        numerical_data = {}
        
        for item in context:
            if isinstance(item, dict) and 'content' in item:
                content = item['content']
                # Look for common financial metrics
                financial_metrics = {
                    'net_profit': self._extract_metric(content, ['net profit', 'npat', 'net income']),
                    'total_assets': self._extract_metric(content, ['total assets', 'assets']),
                    'total_liabilities': self._extract_metric(content, ['total liabilities', 'liabilities']),
                    'shareholder_equity': self._extract_metric(content, ['shareholder equity', 'total equity']),
                    'total_loans': self._extract_metric(content, ['total loans', 'loans and advances']),
                    'total_deposits': self._extract_metric(content, ['total deposits', 'customer deposits']),
                    'revenue': self._extract_metric(content, ['total revenue', 'operating income']),
                }
                
                # Add non-None values
                for metric, value in financial_metrics.items():
                    if value is not None:
                        numerical_data[metric] = value
        
        return numerical_data
    
    def _extract_metric(self, content: str, keywords: List[str]) -> float:
        """Extract a financial metric using multiple patterns"""
        content_lower = content.lower()
        
        for keyword in keywords:
            # Pattern for AED millions/billions
            patterns = [
                rf"{keyword}.*?AED\s*([\d,]+\.?\d*)\s*(?:million|bn|billion)",
                rf"{keyword}.*?([\d,]+\.?\d*)\s*(?:million|bn|billion)\s*AED",
                rf"{keyword}.*?([\d,]+\.?\d*)",  # Just the number
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content_lower)
                if matches:
                    try:
                        value_str = matches[0].replace(',', '')
                        return float(value_str)
                    except ValueError:
                        continue
        return None
    
    def _validate_and_execute_calculation(self, calculation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and safely execute the calculation"""
        try:
            formula = calculation_result.get('formula_used', '')
            input_values = calculation_result.get('input_values', {})
            
            # Replace variable names with actual values in formula
            expression = formula
            for var_name, value in input_values.items():
                expression = expression.replace(var_name, str(value))
            
            # Safe evaluation using numexpr
            result = numexpr.evaluate(expression).item()
            
            calculation_result['validated_result'] = result
            calculation_result['execution_success'] = True
            
        except Exception as e:
            calculation_result['validated_result'] = None
            calculation_result['execution_success'] = False
            calculation_result['execution_error'] = str(e)
        
        return calculation_result