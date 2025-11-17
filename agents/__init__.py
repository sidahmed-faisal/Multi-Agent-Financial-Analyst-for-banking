# agents/__init__.py
from .workflow import FABWorkflow
from .orchestrator_agent import OrchestratorAgent
from .retrieval_agent import RetrievalAgent
from .calculation_agent import CalculationAgent
from .synthesis_agent import SynthesisAgent
from typing import List, Dict, Any


class FABAgentManager:
    """Main manager for the FAB multi-agent system"""
    
    def __init__(self, orchestrator_llm, specialist_llm, chunker):
        self.workflow = FABWorkflow(orchestrator_llm, specialist_llm, chunker)
        self.chunker = chunker
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Main method to analyze financial queries"""
        print(f"🤖 Processing query: {query}")
        
        try:
            result = self.workflow.execute_query(query)
            
            # Extract key information for response
            response = {
                "query": query,
                "final_answer": result.get('final_answer', 'No answer generated'),
                "sources_used": self._extract_sources(result.get('context', [])),
                "calculations_performed": result.get('calculation_results', {}),
                "retrieval_steps": result.get('retrieval_history', []),
                "validation": result.get('validation', {}),
                "processing_steps": len(result.get('context', [])),
                "success": True
            }
            
            return response
            
        except Exception as e:
            print(f"Error in agent workflow: {e}")
            return {
                "query": query,
                "final_answer": f"Error processing query: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _extract_sources(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context"""
        sources = []
        
        for item in context:
            if isinstance(item, dict) and item.get('type') == 'retrieval_result':
                for result in item.get('results', []):
                    if 'metadata' in result:
                        source_info = {
                            'document': result['metadata'].get('source_document'),
                            'section': result['metadata'].get('section_name'),
                            'page': result['metadata'].get('page_number'),
                            'quarter': result['metadata'].get('quarter'),
                            'year': result['metadata'].get('year'),
                            'content_preview': result['content'][:100] + '...' if len(result['content']) > 100 else result['content']
                        }
                        sources.append(source_info)
        
        return sources
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        embedding_stats = self.chunker.get_embedding_stats()
        section_index = self.chunker.create_section_index()
        
        return {
            "embedding_stats": embedding_stats,
            "sections_indexed": len(section_index),
            "total_chunks": embedding_stats.get('total_chunks', 0),
            "embedding_model": embedding_stats.get('embedding_model'),
            "embedding_dimension": embedding_stats.get('embedding_dimension')
        }