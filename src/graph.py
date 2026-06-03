from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, Literal
from src.logger import TUGUMILogger
from src.memory import TaskMemory
from src.tools import ToolExecutor
from src.agent_state import AgentState, TaskStatus
from src.llm_client import LLMClient
import json
from datetime import datetime


class TUGUMIGraph:
    """LangGraph-based agent execution graph"""
    
    def __init__(self, llm_client: LLMClient, memory: TaskMemory, logger: TUGUMILogger):
        self.llm = llm_client
        self.memory = memory
        self.logger = logger
        self.tools = ToolExecutor(memory=memory)
        self.max_steps = 10  # Prevent infinite loops
        
        self.graph = StateGraph(dict)
        self._build_graph()
    
    def _build_graph(self):
        """Build the agentic loop graph"""
        
        # Add nodes
        self.graph.add_node("understand_goal", self._understand_goal)
        self.graph.add_node("plan_steps", self._plan_steps)
        self.graph.add_node("execute_step", self._execute_step)
        self.graph.add_node("evaluate_result", self._evaluate_result)
        self.graph.add_node("learn_from_failure", self._learn_from_failure)
        self.graph.add_node("adapt_plan", self._adapt_plan)
        self.graph.add_node("complete_task", self._complete_task)
        
        # Add edges
        self.graph.set_entry_point("understand_goal")
        self.graph.add_edge("understand_goal", "plan_steps")
        self.graph.add_edge("plan_steps", "execute_step")
        self.graph.add_edge("execute_step", "evaluate_result")
        
        self.graph.add_conditional_edges(
            "evaluate_result",
            self._should_continue,
            {
                "next_step": "execute_step",
                "adapt": "adapt_plan",
                "fail_retry": "learn_from_failure",
                "success": "complete_task",
                "end": END
            }
        )
        
        self.graph.add_edge("learn_from_failure", "adapt_plan")
        self.graph.add_edge("adapt_plan", "plan_steps")
        self.graph.add_edge("complete_task", END)
    
    def _understand_goal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 1: Understand the goal from user instruction"""
        self.logger.info("=" * 50)
        self.logger.info("[UNDERSTAND GOAL] Analyzing user instruction...")
        
        agent_state = state.get("agent_state")
        if not agent_state:
            self.logger.error("No agent state found")
            return state
        
        agent_state.add_thought("Analyzing goal requirements...")
        agent_state.status = TaskStatus.PLANNING
        
        prompt = f"""You are an autonomous AI agent. Analyze the following goal and break it down into key understanding points:

Goal: {agent_state.goal}

Provide:
1. What needs to be done?
2. Required information or resources
3. Potential challenges
4. Success criteria

Be concise and actionable."""
        
        result = self.llm.completion(prompt, max_tokens=1000)
        
        if result.get("success"):
            analysis_text = result.get("text", "")[:200]
            agent_state.add_thought(f"Goal analysis: {analysis_text}...")
            agent_state.context["goal_analysis"] = result.get("text", "")
            self.logger.info(f"✓ Goal understood: {analysis_text}...")
        else:
            error_msg = result.get("error", "Unknown error")
            self.logger.warning(f"✗ Failed to analyze goal: {error_msg}")
            agent_state.add_thought(f"Goal analysis error: {error_msg}")
        
        state["agent_state"] = agent_state
        return state
    
    def _plan_steps(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 2: Create execution plan"""
        self.logger.info("[PLANNING] Creating execution plan...")
        
        agent_state = state.get("agent_state")
        if not agent_state:
            return state
        
        agent_state.status = TaskStatus.PLANNING
        agent_state.status_text = "Creating plan"
        
        prompt = f"""Based on this goal, create a step-by-step action plan:

Goal: {agent_state.goal}
Context: {json.dumps(agent_state.context, ensure_ascii=False, indent=2)[:500]}

Provide numbered steps (max 7 steps) that are:
- Specific and actionable
- Executable by an AI agent
- Include tool usage where necessary

Format each step as: [STEP N] Description"""
        
        result = self.llm.completion(prompt, max_tokens=1500)
        
        if result.get("success"):
            # Parse steps from response
            text = result.get("text", "")
            steps = [line.strip() for line in text.split('\n') if '[STEP' in line and line.strip()]
            
            if not steps:
                # Fallback: use generic steps
                steps = [
                    "[STEP 1] Research and gather information",
                    "[STEP 2] Analyze the requirements",
                    "[STEP 3] Execute the solution"
                ]
            
            agent_state.set_plan(steps[:7])  # Limit to 7 steps
            
            self.logger.info(f"✓ Plan created with {len(steps)} steps")
            for i, step in enumerate(steps, 1):
                step_preview = step[:60] if len(step) > 60 else step
                self.logger.info(f"  Step {i}: {step_preview}...")
        else:
            error_msg = result.get("error", "Unknown error")
            self.logger.error(f"Failed to create plan: {error_msg}")
            agent_state.mark_failed(f"Cannot create execution plan: {error_msg}")
        
        state["agent_state"] = agent_state
        return state
    
    def _execute_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 3: Execute current step"""
        agent_state = state.get("agent_state")
        if not agent_state:
            return state
        
        # Check for infinite loop - improved boundary check
        if agent_state.current_step >= len(agent_state.plan) or agent_state.current_step >= self.max_steps:
            self.logger.info(f"[EXECUTING] Reached end of plan or max steps ({self.max_steps})")
            state["reached_end"] = True
            return state
        
        # Boundary check before accessing plan
        if agent_state.current_step < 0 or agent_state.current_step >= len(agent_state.plan):
            self.logger.warning(f"[EXECUTING] Invalid step index: {agent_state.current_step}")
            state["reached_end"] = True
            return state
        
        step = agent_state.plan[agent_state.current_step]
        self.logger.info(f"[EXECUTING] Step {agent_state.current_step + 1}/{len(agent_state.plan)}: {step[:70]}")
        agent_state.status = TaskStatus.EXECUTING
        agent_state.add_thought(f"Executing: {step}")
        
        # Analyze step and choose tools
        prompt = f"""Based on this step, decide what tools to use:

Step: {step}
Goal: {agent_state.goal}

Available tools:
- web_search(query): Search the internet
- fetch_and_summarize(url): Get webpage content  
- execute_command(cmd): Run shell commands
- install_package(name): Install Python packages
- save_output(content, filename): Save results

Respond with:
TOOL: [tool_name or NONE]
PARAMS: [json params or empty]
REASON: [why this tool or step complete]"""
        
        result = self.llm.completion(prompt, max_tokens=500)
        
        if result.get("success"):
            # Parse and execute tool
            self._execute_suggested_tools(result.get("text", ""), agent_state)
        else:
            self.logger.warning(f"Failed to suggest tools: {result.get('error', '')}")
        
        agent_state.current_step += 1
        state["agent_state"] = agent_state
        return state
    
    def _execute_suggested_tools(self, llm_response: str, agent_state: AgentState):
        """Execute tools suggested by LLM"""
        if not llm_response:
            return
        
        lines = llm_response.split('\n')
        current_tool = None
        params = {}
        
        for line in lines:
            if "TOOL:" in line:
                current_tool = line.split("TOOL:")[1].strip().lower()
            elif "PARAMS:" in line:
                try:
                    params_str = line.split("PARAMS:")[1].strip()
                    if params_str and params_str != "empty":
                        params = json.loads(params_str)
                except (json.JSONDecodeError, IndexError):
                    params = {}
        
        if current_tool and current_tool != "none":
            self.logger.info(f"  → Calling tool: {current_tool}")
            tool_result = self._call_tool(current_tool, params)
            agent_state.record_tool_call(current_tool, params, tool_result)
            self.logger.info(f"  → Result: {'✓ Success' if tool_result.get('success') else '✗ Failed'}")
        else:
            self.logger.info(f"  → No tool needed for this step")
    
    def _call_tool(self, tool_name: str, params: Dict) -> Dict[str, Any]:
        """Call specific tool"""
        try:
            if tool_name == "web_search":
                return self.tools.web_search(
                    params.get("query", ""),
                    params.get("max_results", 5)
                )
            elif tool_name == "fetch_and_summarize":
                return self.tools.fetch_and_summarize(params.get("url", ""))
            elif tool_name == "execute_command":
                return self.tools.execute_command(
                    params.get("cmd", ""),
                    params.get("timeout", 60)
                )
            elif tool_name == "install_package":
                return self.tools.install_package(params.get("name", ""))
            elif tool_name == "save_output":
                return self.tools.save_output(
                    params.get("content", ""),
                    params.get("filename", "output.txt")
                )
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _evaluate_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 4: Evaluate execution results"""
        self.logger.info("[EVALUATING] Checking progress...")
        
        agent_state = state.get("agent_state")
        if not agent_state:
            return state
        
        agent_state.status = TaskStatus.EVALUATING
        
        recent_calls = agent_state.tool_calls[-3:] if agent_state.tool_calls else []
        recent_errors = len(agent_state.errors)
        
        prompt = f"""Evaluate if we're making progress toward the goal:

Goal: {agent_state.goal}
Step {agent_state.current_step}/{len(agent_state.plan)}
Recent errors: {recent_errors}
Recent actions: {len(recent_calls)} tool calls

Respond with ONE of:
STATUS: success
STATUS: progress
STATUS: stuck

ANALYSIS: [brief 1-line assessment]"""
        
        result = self.llm.completion(prompt, max_tokens=300)
        
        if result.get("success"):
            text = result.get("text", "").upper()
            if "SUCCESS" in text:
                state["evaluation_status"] = "success"
            elif "STUCK" in text:
                state["evaluation_status"] = "stuck"
            else:
                state["evaluation_status"] = "progress"
            agent_state.context["last_evaluation"] = result.get("text", "")
            self.logger.info(f"  Evaluation: {state['evaluation_status']}")
        else:
            state["evaluation_status"] = "progress"
            self.logger.warning(f"Evaluation error: {result.get('error', '')}")
        
        state["agent_state"] = agent_state
        return state
    
    def _should_continue(self, state: Dict[str, Any]) -> Literal["next_step", "adapt", "fail_retry", "success", "end"]:
        """Decision point: should we continue, adapt, or complete?"""
        agent_state = state.get("agent_state")
        if not agent_state:
            return "end"
        
        evaluation_status = state.get("evaluation_status", "progress")
        
        # Check if task is already completed
        if agent_state.status == TaskStatus.COMPLETED or agent_state.status_text == "COMPLETED":
            return "success"
        
        # Check if we've reached the end of the plan
        if state.get("reached_end") or agent_state.current_step >= len(agent_state.plan):
            return "success"
        
        # Check for max retries exceeded
        if len(agent_state.errors) > 3:
            if agent_state.can_retry():
                return "fail_retry"
            else:
                return "success"  # Try to complete with what we have
        
        # Check evaluation result
        if evaluation_status == "stuck":
            if agent_state.can_retry():
                return "adapt"
            else:
                return "success"
        
        # Continue to next step
        return "next_step"
    
    def _learn_from_failure(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 5: Learn from failures and search for solutions"""
        self.logger.info("[LEARNING] Analyzing failures and searching for solutions...")
        
        agent_state = state.get("agent_state")
        if not agent_state:
            return state
        
        agent_state.status = TaskStatus.LEARNING
        
        if not agent_state.can_retry():
            agent_state.mark_failed("Max retries exceeded")
            self.logger.warning("Max retries exceeded, marking as failed")
            return state
        
        # Get error pattern
        if agent_state.errors:
            last_error = agent_state.errors[-1].get("error", "Unknown error")
            self.logger.info(f"Last error: {last_error}")
            
            # Search for solution
            search_result = self.tools.web_search(f"how to fix {last_error[:50]}")
            
            if search_result.get("success"):
                agent_state.search_history.append(last_error)
                count = search_result.get("count", 0)
                self.logger.info(f"Found {count} potential solutions")
                
                # Cache the search results
                if self.memory and search_result.get("results"):
                    solution_text = json.dumps(search_result["results"][:2], ensure_ascii=False)
                    self.memory.add_error_pattern(
                        "execution_error",
                        last_error,
                        solution_text
                    )
        
        agent_state.increment_retry()
        state["agent_state"] = agent_state
        return state
    
    def _adapt_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 6: Adapt the plan based on feedback"""
        self.logger.info("[ADAPTING] Modifying plan based on feedback...")
        
        agent_state = state.get("agent_state")
        if not agent_state:
            return state
        
        agent_state.status = TaskStatus.PLANNING
        agent_state.add_thought("Plan needs adjustment")
        
        # Re-plan with learned information
        error_summary = ""
        if agent_state.errors:
            errors = [e.get("error", "")[:50] for e in agent_state.errors[-3:]]
            error_summary = "; ".join(errors)
        
        prompt = f"""The original plan isn't working. Adapt it based on failures:

Goal: {agent_state.goal}
Current step: {agent_state.current_step}/{len(agent_state.plan)}
Errors: {error_summary[:200]}

Provide 3-5 revised steps that:
- Try different approaches
- Are more specific/actionable
- Address the errors encountered

Format: [STEP N] Description"""
        
        result = self.llm.completion(prompt, max_tokens=1000)
        
        if result.get("success"):
            text = result.get("text", "")
            steps = [line.strip() for line in text.split('\n') if '[STEP' in line and line.strip()]
            
            if steps:
                agent_state.set_plan(steps[:7])
                agent_state.current_step = 0  # Reset to start with new plan
                self.logger.info(f"✓ Plan adapted with {len(steps)} new steps")
            else:
                self.logger.warning("No valid steps in adapted plan")
        else:
            self.logger.error(f"Failed to adapt plan: {result.get('error', '')}")
        
        state["agent_state"] = agent_state
        return state
    
    def _complete_task(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 7: Mark task as completed"""
        self.logger.info("[COMPLETE] Task execution completed")
        
        agent_state = state.get("agent_state")
        if not agent_state:
            return state
        
        agent_state.mark_completed()
        
        # Summary
        summary = {
            "task_id": agent_state.task_id,
            "goal": agent_state.goal,
            "status": "completed",
            "steps_executed": agent_state.current_step,
            "steps_planned": len(agent_state.plan),
            "tools_used": len(agent_state.tool_calls),
            "successful_tools": len(agent_state.results),
            "failed_tools": len(agent_state.errors),
            "duration_seconds": agent_state.get_duration(),
            "retries": agent_state.retry_count
        }
        
        self.logger.info(f"✓ Task completed: {json.dumps(summary, ensure_ascii=False)}")
        
        # Save to memory
        if self.memory:
            self.memory.add_task(
                agent_state.task_id,
                agent_state.goal,
                "completed",
                json.dumps(summary, ensure_ascii=False)
            )
        
        state["agent_state"] = agent_state
        state["summary"] = summary
        return state
    
    def compile(self):
        """Compile the graph"""
        return self.graph.compile()
