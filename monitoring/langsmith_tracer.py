# monitoring/langsmith_tracer.py
import langsmith
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_core.callbacks import CallbackManager
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime
import json

class LangSmithTracer:
    """Custom tracer for LangSmith integration"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.client = Client()
        self.tracer = LangChainTracer()
        self.callback_manager = CallbackManager([self.tracer])
    
    def trace_operation(self, 
                       agent_name: str, 
                       operation: str, 
                       inputs: Dict[str, Any], 
                       outputs: Dict[str, Any],
                       metadata: Dict[str, Any] = None) -> str:
        """Trace a single operation to LangSmith"""
        
        trace_id = str(uuid.uuid4())
        operation_metadata = {
            "agent": agent_name,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": trace_id,
            **(metadata or {})
        }
        
        try:
            # Create a custom run in LangSmith
            self.client.create_run(
                name=f"{agent_name}.{operation}",
                inputs=inputs,
                outputs=outputs,
                run_type="chain",  # or "llm", "tool", etc.
                extra={"metadata": operation_metadata}
            )
            
            # Log to console for local debugging
            print(f"🔍 [{agent_name}] {operation} - Trace ID: {trace_id}")
            
        except Exception as e:
            print(f"LangSmith tracing error: {e}")
        
        return trace_id
    
    def trace_llm_call(self, 
                      agent_name: str, 
                      prompt: str, 
                      response: str,
                      model: str = None,
                      tokens_used: int = None) -> str:
        """Trace LLM calls specifically"""
        
        inputs = {
            "prompt": prompt,
            "model": model,
            "agent": agent_name
        }
        
        outputs = {
            "response": response,
            "tokens_used": tokens_used
        }
        
        metadata = {
            "operation_type": "llm_call",
            "model": model
        }
        
        return self.trace_operation(agent_name, "llm_call", inputs, outputs, metadata)
    
    def trace_retrieval(self,
                       agent_name: str,
                       query: str,
                       results: List[Dict],
                       filters: Dict = None) -> str:
        """Trace retrieval operations"""
        
        inputs = {
            "query": query,
            "filters": filters,
            "agent": agent_name
        }
        
        outputs = {
            "results_count": len(results),
            "sources": [result.get('metadata', {}).get('source_document', 'Unknown') for result in results],
            "top_result_preview": results[0]['content'][:200] + "..." if results else "No results"
        }
        
        metadata = {
            "operation_type": "retrieval",
            "results_count": len(results)
        }
        
        return self.trace_operation(agent_name, "retrieval", inputs, outputs, metadata)
    
    def trace_calculation(self,
                        agent_name: str,
                        request: str,
                        result: Dict,
                        formula: str = None) -> str:
        """Trace calculation operations"""
        
        inputs = {
            "calculation_request": request,
            "formula": formula,
            "agent": agent_name
        }
        
        outputs = {
            "result": result.get('result'),
            "calculation_type": result.get('calculation_type'),
            "units": result.get('units'),
            "success": result.get('execution_success', False)
        }
        
        metadata = {
            "operation_type": "calculation",
            "formula_used": formula
        }
        
        return self.trace_operation(agent_name, "calculation", inputs, outputs, metadata)

    def get_agent_callback_manager(self):
        """Get callback manager for LangChain components"""
        return self.callback_manager