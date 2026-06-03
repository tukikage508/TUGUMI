from typing import Any, Dict, List
from datetime import datetime
from collections import deque
import json
from pathlib import Path

class TaskMemory:
    """Persistent memory system for agent learning and context"""
    
    def __init__(self, max_items: int = 100, memory_file: str = "data/memory.json"):
        self.max_items = max_items
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.task_history = deque(maxlen=max_items)
        self.tool_usage = {}  # Track tool effectiveness
        self.error_patterns = []  # Learn from errors
        self.solution_cache = {}  # Cache successful solutions
        
        self.load_memory()
    
    def add_task(self, task_id: str, goal: str, status: str, result: str = None):
        """Record task execution"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "goal": goal,
            "status": status,
            "result": result
        }
        self.task_history.append(entry)
        self.save_memory()
    
    def record_tool_usage(self, tool_name: str, success: bool, duration: float):
        """Track tool effectiveness for optimization"""
        if tool_name not in self.tool_usage:
            self.tool_usage[tool_name] = {
                "count": 0,
                "success": 0,
                "total_time": 0,
                "avg_time": 0
            }
        
        stats = self.tool_usage[tool_name]
        stats["count"] += 1
        if success:
            stats["success"] += 1
        stats["total_time"] += duration
        stats["avg_time"] = stats["total_time"] / stats["count"]
        self.save_memory()
    
    def add_error_pattern(self, error_type: str, context: str, solution: str):
        """Learn from errors and store solutions"""
        pattern = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "context": context,
            "solution": solution
        }
        self.error_patterns.append(pattern)
        self.save_memory()
    
    def cache_solution(self, problem_key: str, solution: str, metadata: Dict = None):
        """Cache successful solutions for similar problems"""
        self.solution_cache[problem_key] = {
            "solution": solution,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.save_memory()
    
    def get_cached_solution(self, problem_key: str) -> str | None:
        """Retrieve cached solution if available"""
        return self.solution_cache.get(problem_key, {}).get("solution")
    
    def get_similar_errors(self, error_type: str) -> List[Dict]:
        """Find similar error patterns for learning"""
        return [e for e in self.error_patterns if error_type in e["error_type"].lower()]
    
    def get_tool_stats(self, tool_name: str) -> Dict | None:
        """Get tool usage statistics"""
        return self.tool_usage.get(tool_name)
    
    def save_memory(self):
        """Persist memory to disk"""
        data = {
            "task_history": list(self.task_history),
            "tool_usage": self.tool_usage,
            "error_patterns": self.error_patterns,
            "solution_cache": self.solution_cache
        }
        with open(self.memory_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_memory(self):
        """Load memory from disk"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.task_history = deque(data.get("task_history", []), maxlen=self.max_items)
                    self.tool_usage = data.get("tool_usage", {})
                    self.error_patterns = data.get("error_patterns", [])
                    self.solution_cache = data.get("solution_cache", {})
            except Exception as e:
                print(f"Error loading memory: {e}")
    
    def get_summary(self) -> Dict:
        """Get memory summary for agent decision-making"""
        return {
            "total_tasks": len(self.task_history),
            "tool_count": len(self.tool_usage),
            "error_patterns_count": len(self.error_patterns),
            "cached_solutions": len(self.solution_cache),
            "most_used_tools": sorted(
                self.tool_usage.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:5]
        }
