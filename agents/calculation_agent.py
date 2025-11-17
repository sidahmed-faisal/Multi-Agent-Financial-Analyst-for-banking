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
        
        self.metrics_extraction_prompt = ChatPromptTemplate.from_template("""
You are a Financial Data Extraction Specialist. Your task is to identify and extract all financial metrics and their numerical values from the provided content.

CONTENT TO ANALYZE:
{content}

INSTRUCTIONS:
1. Scan the entire content for any financial metrics (e.g., revenues, profits, assets, deposits, loans, ratios, etc.)
2. For each metric found, identify:
   - The metric name as it appears in the content
   - The numerical value
   - The unit (AED millions, AED billions, percentage, ratio, etc.)
3. Be flexible and identify ANY metric mentioned, not just predefined ones
4. Return the results as a JSON object where keys are metric names (normalized/simplified) and values are objects with "value", "unit", and "original_name"

Return ONLY valid JSON (no markdown, no extra text):
{{
    "metric_name_1": {{"value": 12345.67, "unit": "AED millions", "original_name": "Total Revenue"}},
    "metric_name_2": {{"value": 9876.54, "unit": "AED millions", "original_name": "Net Profit"}},
    "metric_name_3": {{"value": 45.67, "unit": "percentage", "original_name": "ROE %"}}
}}

RESPONSE:
""")

    def perform_calculation(self, calculation_request: str, data_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform financial calculation based on request and data"""
        chain = self.calculation_prompt | self.llm | StrOutputParser()
        
        try:
            # Extract numerical values dynamically from context using LLM
            available_data = self._extract_numerical_data(data_context)
            
            # Format available data for display in prompt
            formatted_data = self._format_available_data(available_data)
            
            response_text = chain.invoke({
                "calculation_request": calculation_request,
                "data_context": self._format_context(data_context),
                "available_data": formatted_data
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
    
    def _extract_numerical_data(self, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract all financial metrics mentioned in the content using LLM"""
        numerical_data = {}
        
        # Combine all content from context
        combined_content = ""
        for item in context:
            if isinstance(item, dict) and 'content' in item:
                combined_content += item['content'] + "\n"
        
        if not combined_content.strip():
            return numerical_data
        
        try:
            # Use LLM to extract all metrics dynamically
            extraction_chain = self.metrics_extraction_prompt | self.llm | StrOutputParser()
            
            response_text = extraction_chain.invoke({"content": combined_content})
            
            # Parse the JSON response
            metrics_data = json.loads(response_text)
            
            # Process extracted metrics into numerical_data format
            for metric_key, metric_info in metrics_data.items():
                if isinstance(metric_info, dict) and 'value' in metric_info:
                    numerical_data[metric_key] = {
                        "value": metric_info.get('value'),
                        "unit": metric_info.get('unit', 'unknown'),
                        "original_name": metric_info.get('original_name', metric_key)
                    }
            
            return numerical_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing metrics extraction response: {e}")
            return numerical_data
        except Exception as e:
            print(f"Error extracting metrics with LLM: {e}")
            return numerical_data
    
    def _format_available_data(self, available_data: Dict[str, Any]) -> str:
        """Format extracted metrics for display in LLM prompt"""
        if not available_data:
            return "No financial metrics extracted from the content."
        
        formatted_lines = []
        for metric_key, metric_info in available_data.items():
            if isinstance(metric_info, dict):
                value = metric_info.get('value', 'N/A')
                unit = metric_info.get('unit', 'unknown')
                original_name = metric_info.get('original_name', metric_key)
                formatted_lines.append(f"- {original_name}: {value} {unit}")
            else:
                formatted_lines.append(f"- {metric_key}: {metric_info}")
        
        return "\n".join(formatted_lines)
    
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