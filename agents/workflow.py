from langgraph.graph import StateGraph, END
from .agent_definitions import AgentState
from .orchestrator_agent import OrchestratorAgent
from .retrieval_agent import RetrievalAgent
from .calculation_agent import CalculationAgent
from .synthesis_agent import SynthesisAgent
from monitoring.langsmith_tracer import LangSmithTracer
import uuid

from typing import Dict, Any

class FABWorkflow:
    """Main multi-agent workflow for financial analysis"""
    
    def __init__(self, orchestrator_llm, specialist_llm, chunker):
        self.orchestrator = OrchestratorAgent(orchestrator_llm)
        self.retrieval_agent = RetrievalAgent(specialist_llm, chunker)
        self.calculation_agent = CalculationAgent(specialist_llm)
        self.synthesis_agent = SynthesisAgent(specialist_llm)
        self.workflow_tracer = LangSmithTracer(agent_name="FABWorkflow")

        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("calculation", self._calculation_node)
        workflow.add_node("synthesis", self._synthesis_node)
        
        # Define entry point
        workflow.set_entry_point("orchestrator")
        
        # Define conditional edges
        workflow.add_conditional_edges(
            "orchestrator",
            self.orchestrator.should_continue,
            {
                "retrieval": "retrieval",
                "calculation": "calculation", 
                "synthesis": "synthesis",
                "finish": END
            }
        )
        
        workflow.add_conditional_edges(
            "retrieval",
            self._should_continue_after_retrieval,
            {
                "continue": "orchestrator",
                "finish": END
            }
        )
        
        workflow.add_conditional_edges(
            "calculation",
            self._should_continue_after_calculation,
            {
                "continue": "orchestrator",
                "finish": END
            }
        )
        
        workflow.add_edge("synthesis", END)
        
        return workflow.compile()
    
    def _orchestrator_node(self, state: AgentState) -> Dict[str, Any]:
        """Orchestrator node execution with tracing"""
        trace_inputs = {"query": state['query'], "current_step": state.get('current_step', 0)}
        
        if not state.get('plan'):
            # First time - create plan
            plan = self.orchestrator.create_plan(state['query'])
            result = {
                "plan": plan,
                "current_step": 0,
                "context": [],
                "retrieval_history": [],
                "calculation_results": {}
            }
            # Trace orchestrator node execution
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="orchestrator_node_initial",
                inputs=trace_inputs,
                outputs={"plan_created": len(plan), "plan_steps": plan},
                metadata={"node": "orchestrator", "action": "plan_creation"}
            )
        else:
            # Update step and decide next action
            current_step = state.get('current_step', 0)
            result = {"current_step": current_step + 1}
            # Trace orchestrator step increment
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="orchestrator_node_step_update",
                inputs=trace_inputs,
                outputs={"new_step": current_step + 1},
                metadata={"node": "orchestrator", "action": "step_increment"}
            )
        
        return result

    
    def _retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieval node execution with tracing"""
        plan = state['plan']
        current_step = state['current_step']
        current_step_text = plan[current_step] if current_step < len(plan) else ""
        
        trace_inputs = {
            "query": state['query'],
            "current_step": current_step,
            "step_text": current_step_text
        }
        
        try:
            # Execute retrieval
            results = self.retrieval_agent.execute_retrieval(
                query=state['query'],
                current_step=current_step_text,
                context=state['context']
            )
            
            # Update context
            new_context = state['context'] + [
                {
                    "type": "retrieval_result",
                    "step": current_step_text,
                    "results": results,
                    "timestamp": len(state['context'])
                }
            ]
            
            retrieval_history = state['retrieval_history'] + [{
                "step": current_step,
                "query": current_step_text,
                "results_count": len(results),
                "timestamp": len(state['retrieval_history'])
            }]
            
            # Trace retrieval node execution
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="retrieval_node",
                inputs=trace_inputs,
                outputs={
                    "results_count": len(results),
                    "step": current_step,
                    "sources": [r.get('metadata', {}).get('source_document', 'Unknown') for r in results]
                },
                metadata={"node": "retrieval", "success": True}
            )
            
            return {
                "context": new_context,
                "retrieval_history": retrieval_history
            }
        except Exception as e:
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="retrieval_node_error",
                inputs=trace_inputs,
                outputs={"error": str(e)},
                metadata={"node": "retrieval", "success": False}
            )
            raise

    
    def _calculation_node(self, state: AgentState) -> Dict[str, Any]:
        """Calculation node execution with tracing"""
        plan = state['plan']
        current_step = state['current_step']
        current_step_text = plan[current_step] if current_step < len(plan) else ""
        
        trace_inputs = {
            "query": state['query'],
            "current_step": current_step,
            "step_text": current_step_text
        }
        
        try:
            # Execute calculation
            calculation_result = self.calculation_agent.perform_calculation(
                calculation_request=current_step_text,
                data_context=state['context']
            )
            
            # Update context and results
            new_context = state['context'] + [
                {
                    "type": "calculation_result",
                    "step": current_step_text,
                    "result": calculation_result,
                    "timestamp": len(state['context'])
                }
            ]
            
            calculation_results = state['calculation_results'].copy()
            calculation_key = f"step_{current_step}"
            calculation_results[calculation_key] = calculation_result
            
            # Trace calculation node execution
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="calculation_node",
                inputs=trace_inputs,
                outputs={
                    "calculation_type": calculation_result.get('calculation_type'),
                    "result": calculation_result.get('validated_result'),
                    "units": calculation_result.get('units'),
                    "success": calculation_result.get('execution_success', False)
                },
                metadata={"node": "calculation", "success": True}
            )
            
            return {
                "context": new_context,
                "calculation_results": calculation_results
            }
        except Exception as e:
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="calculation_node_error",
                inputs=trace_inputs,
                outputs={"error": str(e)},
                metadata={"node": "calculation", "success": False}
            )
            raise

    
    def _synthesis_node(self, state: AgentState) -> Dict[str, Any]:
        """Synthesis node execution with tracing"""
        trace_inputs = {
            "query": state['query'],
            "context_size": len(state['context']),
            "calculation_results": len(state['calculation_results'])
        }
        
        try:
            final_answer = self.synthesis_agent.synthesize_answer(
                query=state['query'],
                context=state['context'],
                calculation_results=state['calculation_results']
            )
            
            # Validate the answer
            validation = self.synthesis_agent.validate_answer(final_answer, state['context'])
            
            # Trace synthesis node execution
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="synthesis_node",
                inputs=trace_inputs,
                outputs={
                    "answer_length": len(final_answer) if final_answer else 0,
                    "is_valid": validation.get('is_valid', False),
                    "unsupported_claims": len(validation.get('unsupported_claims', []))
                },
                metadata={"node": "synthesis", "success": True}
            )
            
            return {
                "final_answer": final_answer,
                "validation": validation,
                "context": state['context'] + [
                    {
                        "type": "final_synthesis",
                        "answer": final_answer,
                        "validation": validation,
                        "timestamp": len(state['context'])
                    }
                ]
            }
        except Exception as e:
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="synthesis_node_error",
                inputs=trace_inputs,
                outputs={"error": str(e)},
                metadata={"node": "synthesis", "success": False}
            )
            raise

    
    def _should_continue_after_retrieval(self, state: AgentState) -> str:
        """Determine if we should continue after retrieval"""
        current_step = state.get('current_step', 0)
        plan = state.get('plan', [])
        
        if current_step >= len(plan):
            return "finish"
        return "continue"
    
    def _should_continue_after_calculation(self, state: AgentState) -> str:
        """Determine if we should continue after calculation"""
        current_step = state.get('current_step', 0)
        plan = state.get('plan', [])
        
        if current_step >= len(plan):
            return "finish"
        return "continue"
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a query through the full workflow with tracing"""
        trace_inputs = {"query": query}
        
        try:
            initial_state = AgentState(
                query=query,
                plan=[],
                context=[],
                current_step=0,
                final_answer=None,
                retrieval_history=[],
                calculation_results={},
                metadata={}
            )
            
            # Trace workflow start
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="execute_query_start",
                inputs=trace_inputs,
                outputs={},
                metadata={"workflow": "start"}
            )
            
            result = self.graph.invoke(initial_state)
            
            # Trace workflow completion
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="execute_query_complete",
                inputs=trace_inputs,
                outputs={
                    "final_answer_length": len(result.get('final_answer', '')) if result.get('final_answer') else 0,
                    "context_items": len(result.get('context', [])),
                    "calculation_steps": len(result.get('calculation_results', {})),
                    "validation": result.get('validation', {}).get('is_valid', False)
                },
                metadata={"workflow": "complete", "success": True}
            )
            
            return result
        except Exception as e:
            # Trace workflow error
            self.workflow_tracer.trace_operation(
                agent_name="FABWorkflow",
                operation="execute_query_error",
                inputs=trace_inputs,
                outputs={"error": str(e)},
                metadata={"workflow": "error", "success": False}
            )
            raise
