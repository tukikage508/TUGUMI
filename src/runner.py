from typing import Dict, Any
from src.graph import TUGUMIGraph
from src.agent_state import AgentState, TaskStatus
from src.logger import TUGUMILogger
from src.memory import TaskMemory
from src.llm_client import LLMClient
import json
from datetime import datetime
import uuid

class AgentRunner:
    """Execute agent tasks end-to-end"""
    
    def __init__(
        self,
        llm_server_url: str = "http://0.0.0.0:8080",
        llm_timeout: int = 300,
        memory_file: str = "data/memory.json"
    ):
        self.logger = TUGUMILogger("TUGUMI")
        self.memory = TaskMemory(memory_file=memory_file)
        self.llm = LLMClient(server_url=llm_server_url, timeout=llm_timeout)
        
        # Start health monitoring
        self.llm.start_health_monitor()
        
        self.graph = TUGUMIGraph(self.llm, self.memory, self.logger)
        self.compiled_graph = self.graph.compile()
    
    def run_task(self, goal: str) -> Dict[str, Any]:
        """Run a complete task from goal to completion"""
        task_id = str(uuid.uuid4())[:8]
        
        self.logger.info("\n" + "="*70)
        self.logger.info(f"TUGUMI AUTONOMOUS AGENT - Task {task_id}")
        self.logger.info(f"Goal: {goal}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("="*70)
        
        # Check LLM health
        health = self.llm.get_health_status()
        self.logger.info(f"LLM Server Status: {health['status']}")
        
        if not self.llm.is_healthy:
            self.logger.error("⚠️  LLM server is not responding!")
            self.logger.info("Waiting for server to be ready...")
            
            # Try to reconnect
            import time
            for attempt in range(5):
                time.sleep(5)
                health = self.llm.get_health_status()
                self.logger.info(f"Reconnect attempt {attempt+1}: {health['status']}")
                if self.llm.is_healthy:
                    break
        
        # Create agent state
        agent_state = AgentState(task_id, goal)
        agent_state.mark_started()
        
        # Initialize state
        state = {
            "agent_state": agent_state,
            "evaluation": {},
            "summary": None
        }
        
        try:
            # Execute graph
            self.logger.info("\n[EXECUTING AGENT LOOP]")
            final_state = self.compiled_graph.invoke(state)
            
            self.logger.info("\n[TASK RESULTS]")
            self.logger.info(json.dumps(final_state.get("summary", {}), ensure_ascii=False, indent=2))
            
            return {
                "success": True,
                "task_id": task_id,
                "summary": final_state.get("summary"),
                "state": final_state["agent_state"].to_dict()
            }
        
        except Exception as e:
            self.logger.error(f"Task execution failed: {str(e)}")
            agent_state.mark_failed(str(e))
            
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "state": agent_state.to_dict()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "llm_health": self.llm.get_health_status(),
            "memory_summary": self.memory.get_summary(),
            "timestamp": datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.llm.stop_health_monitor()
