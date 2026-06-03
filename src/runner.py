from typing import Dict, Any, Optional
from src.graph import TUGUMIGraph
from src.agent_state import AgentState
from src.logger import TUGUMILogger
from src.memory import TaskMemory
from src.llm_client import LLMClient
import json
from datetime import datetime
import uuid
import time


class AgentRunner:
    """Execute agent tasks end-to-end"""
    
    def __init__(
        self,
        llm_server_url: str = "http://127.0.0.1:8080",
        llm_timeout: int = 300,
        memory_file: str = "data/memory.json",
        max_reconnect_attempts: int = 5
    ):
        self.logger = TUGUMILogger("TUGUMI")
        self.memory = TaskMemory(memory_file=memory_file)
        self.llm = LLMClient(server_url=llm_server_url, timeout=llm_timeout)
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Start health monitoring
        self.llm.start_health_monitor()
        
        self.graph = TUGUMIGraph(self.llm, self.memory, self.logger)
        self.compiled_graph = self.graph.compile()
    
    def run_task(self, goal: str) -> Dict[str, Any]:
        """Run a complete task from goal to completion"""
        if not goal or not goal.strip():
            return {
                "success": False,
                "task_id": "",
                "error": "Goal cannot be empty"
            }
        
        task_id = str(uuid.uuid4())[:8]
        
        self.logger.info("\n" + "="*70)
        self.logger.info(f"TUGUMI AUTONOMOUS AGENT - Task {task_id}")
        self.logger.info(f"Goal: {goal}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("="*70)
        
        # Check LLM health
        health = self.llm.get_health_status()
        self.logger.info(f"LLM Server Status: {health.get('status', 'unknown')}")
        
        if not self.llm.is_healthy:
            self.logger.warning("⚠️  LLM server is not responding!")
            self.logger.info("Waiting for server to be ready...")
            
            # Try to reconnect
            if not self._wait_for_llm():
                self.logger.error("✗ LLM server failed to respond after retries")
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": "LLM server is not available"
                }
        
        self.logger.info("✓ LLM server is ready")
        
        # Create agent state
        agent_state = AgentState(task_id, goal)
        agent_state.mark_started()
        
        # Initialize state
        state = {
            "agent_state": agent_state,
            "evaluation_status": "progress",
            "summary": None,
            "reached_end": False
        }
        
        try:
            # Execute graph
            self.logger.info("\n[EXECUTING AGENT LOOP]")
            final_state = self.compiled_graph.invoke(state)
            
            agent_state = final_state.get("agent_state")
            if agent_state:
                summary = final_state.get("summary", {})
                self.logger.info("\n[TASK RESULTS]")
                self.logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "summary": summary,
                    "state": agent_state.to_dict()
                }
            else:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": "No agent state in final result"
                }
        
        except Exception as e:
            self.logger.error(f"✗ Task execution failed: {str(e)}")
            if agent_state:
                agent_state.mark_failed(str(e))
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": str(e),
                    "state": agent_state.to_dict()
                }
            else:
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": str(e)
                }
    
    def _wait_for_llm(self, interval: int = 3) -> bool:
        """Wait for LLM server to be ready with retries"""
        for attempt in range(self.max_reconnect_attempts):
            self.logger.info(f"Reconnect attempt {attempt + 1}/{self.max_reconnect_attempts}...")
            time.sleep(interval)
            
            health = self.llm.get_health_status()
            self.logger.info(f"  Status: {health.get('status', 'unknown')}")
            
            if self.llm.is_healthy:
                return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "llm_health": self.llm.get_health_status(),
            "memory_summary": self.memory.get_summary(),
            "timestamp": datetime.now().isoformat()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up resources...")
        self.llm.stop_health_monitor()
        self.logger.info("✓ Cleanup complete")
