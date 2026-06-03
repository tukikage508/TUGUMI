from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    LEARNING = "learning"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class AgentState:
    """Complete agent state for persistent task execution"""
    
    def __init__(self, task_id: str, goal: str):
        self.task_id = task_id
        self.goal = goal
        self.status = TaskStatus.PENDING
        
        # Planning
        self.plan: List[str] = []
        self.current_step = 0
        
        # Execution
        self.tool_calls: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        
        # Context
        self.context: Dict[str, Any] = {}
        self.thoughts: List[str] = []
        
        # Timing
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        # Retry
        self.retry_count = 0
        self.max_retries = 3
        self.last_error = None
        
        # Learning
        self.learned_solutions: Dict[str, str] = {}
        self.search_history: List[str] = []
    
    def set_plan(self, plan: List[str]):
        """Set execution plan"""
        self.plan = plan
        self.current_step = 0
        self.status = TaskStatus.PLANNING
    
    def add_thought(self, thought: str):
        """Record agent thought process"""
        self.thoughts.append(f"[{datetime.now().isoformat()}] {thought}")
    
    def record_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Record tool execution"""
        self.tool_calls.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "params": params,
            "result": result,
            "success": result.get("success", False)
        })
        
        if result.get("success"):
            self.results.append(result)
        else:
            error_entry = {
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "error": result.get("error", "Unknown error")
            }
            self.errors.append(error_entry)
            self.last_error = error_entry
    
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self):
        """Increment retry counter"""
        self.retry_count += 1
        self.status = TaskStatus.RETRYING
    
    def mark_started(self):
        """Mark task as started"""
        self.started_at = datetime.now()
        self.status = TaskStatus.IN_PROGRESS
    
    def mark_completed(self):
        """Mark task as completed"""
        self.completed_at = datetime.now()
        self.status = TaskStatus.COMPLETED
    
    def mark_failed(self, reason: str):
        """Mark task as failed"""
        self.completed_at = datetime.now()
        self.status = TaskStatus.FAILED
        self.last_error = {"reason": reason}
    
    def get_duration(self) -> float:
        """Get execution duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for logging"""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status.value,
            "plan": self.plan,
            "current_step": self.current_step,
            "tool_calls_count": len(self.tool_calls),
            "results_count": len(self.results),
            "errors_count": len(self.errors),
            "retry_count": self.retry_count,
            "duration": self.get_duration(),
            "thoughts_count": len(self.thoughts),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
