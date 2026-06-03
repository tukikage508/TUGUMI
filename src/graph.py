from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
from src.logger import TUGUMILogger
from src.memory import TaskMemory
from src.tools import ToolExecutor
from src.agent_state import AgentState
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
        self.graph.add_edge("adapt_plan", "execute_step")
        self.graph.add_edge("complete_task", END)
    
    def _understand_goal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 1: Understand the goal from user instruction"""
        self.logger.info("=" * 50)
        self.logger.info("[UNDERSTAND GOAL] Analyzing user instruction...")
        
        agent_state = state["agent_state"]
        agent_state.add_thought("Analyzing goal requirements...")
        
        prompt = f"""You are an autonomous AI agent. Analyze the following goal and break it down into key understanding points:

Goal: {agent_state.goal}

Provide:
1. What needs to be done?
2. Required information or resources
3. Potential challenges
4. Success criteria

Be concise and actionable."""
        
        result = self.llm.completion(prompt, max_tokens=1000)
        
        if result["success"]:
            agent_state.add_thought(f"Goal analysis: {result['text'][:200]}...")
            agent_state.context["goal_analysis"] = result["text"]
            self.logger.info(f"✓ Goal understood: {result['text'][:100]}...")
        else:
            self.logger.warning(f"✗ Failed to analyze goal: {result.get('error')}")
        
        state["agent_state"] = agent_state
        return state
    
    def _plan_steps(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 2: Create execution plan"""
        self.logger.info("[PLANNING] Creating execution plan...")
        
        agent_state = state["agent_state"]
        agent_state.status_text = "Creating plan"
        
        prompt = f"""Based on this goal, create a step-by-step action plan:

Goal: {agent_state.goal}
Context: {json.dumps(agent_state.context, ensure_ascii=False, indent=2)}

Provide numbered steps that are:
- Specific and actionable
- Executable by an AI agent
- Include tool usage where necessary

Format each step as: [STEP N] Description"""
        
        result = self.llm.completion(prompt, max_tokens=1500)
        
        if result["success"]:
            # Parse steps from response
            text = result["text"]
            steps = [line.strip() for line in text.split('\n') if line.strip().startswith('[STEP')]
            agent_state.set_plan(steps)
            
            self.logger.info(f"✓ Plan created with {len(steps)} steps")
            for i, step in enumerate(steps, 1):
                self.logger.info(f"  Step {i}: {step[:60]}...")
        else:
            self.logger.error(f"Failed to create plan: {result.get('error')}")
            agent_state.mark_failed("Cannot create execution plan")
        
        state["agent_state"] = agent_state
        return state
    
    def _execute_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 3: Execute current step"""
        agent_state = state["agent_state"]
        
        if agent_state.current_step >= len(agent_state.plan):
            return state
        
        step = agent_state.plan[agent_state.current_step]
        self.logger.info(f"[EXECUTING] Step {agent_state.current_step + 1}: {step}")
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
TOOL: [tool_name]
PARAMS: [json params]
REASON: [why this tool]"""
        
        result = self.llm.completion(prompt, max_tokens=500)
        
        if result["success"]:
            # Parse and execute tool
            self._execute_suggested_tools(result["text"], agent_state)
        
        agent_state.current_step += 1
        state["agent_state"] = agent_state
        return state
    
    def _execute_suggested_tools(self, llm_response: str, agent_state: AgentState):
        """Execute tools suggested by LLM"""
        lines = llm_response.split('\n')
        current_tool = None
        params = {}
        
        for line in lines:
            if line.startswith("TOOL:"):
                current_tool = line.replace("TOOL:", "").strip()
            elif line.startswith("PARAMS:"):
                try:
                    params_str = line.replace("PARAMS:", "").strip()
                    params = json.loads(params_str)
                except:
                    params = {}
        
        if current_tool:
            self.logger.info(f"  Executing tool: {current_tool}")
            tool_result = self._call_tool(current_tool, params)
            agent_state.record_tool_call(current_tool, params, tool_result)
            self.logger.info(f"  Result: {'Success' if tool_result.get('success') else 'Failed'}")
    
    def _call_tool(self, tool_name: str, params: Dict) -> Dict[str, Any]:
        """Call specific tool"""
        try:
            if tool_name == "web_search":
                return self.tools.web_search(params.get("query", ""), params.get("max_results", 5))
            elif tool_name == "fetch_and_summarize":
                return self.tools.fetch_and_summarize(params.get("url", ""))
            elif tool_name == "execute_command":
                return self.tools.execute_command(params.get("cmd", ""), params.get("timeout", 60))
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
        self.logger.info("[EVALUATING] Checking results...")
        
        agent_state = state["agent_state"]
        recent_calls = agent_state.tool_calls[-3:] if agent_state.tool_calls else []
        
        prompt = f"""Evaluate if we're making progress toward the goal:

Goal: {agent_state.goal}
Recent actions: {json.dumps(recent_calls, ensure_ascii=False, indent=2)}

Respond with:
STATUS: [success|progress|stuck|error]
ANALYSIS: [brief assessment]
NEXT: [what to do next]"""
        
        result = self.llm.completion(prompt, max_tokens=500)
        
        if result["success"]:
            agent_state.context["last_evaluation"] = result["text"]
            self.logger.info(f"Evaluation: {result['text'][:100]}...")
        
        state["evaluation"] = result
        state["agent_state"] = agent_state
        return state
    
    def _should_continue(self, state: Dict[str, Any]) -> str:
        """Decision point: should we continue, adapt, or complete?"""
        agent_state = state["agent_state"]
        evaluation = state.get("evaluation", {})
        
        if agent_state.status_text == "COMPLETED":
            return "success"
        
        if len(agent_state.errors) > 3 and agent_state.can_retry():
            return "fail_retry"
        
        if agent_state.current_step >= len(agent_state.plan):
            return "success"
        
        if "stuck" in evaluation.get("text", "").lower():
            return "adapt"
        
        return "next_step"
    
    def _learn_from_failure(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 5: Learn from failures and search for solutions"""
        self.logger.info("[LEARNING] Analyzing failures and searching for solutions...")
        
        agent_state = state["agent_state"]
        
        if not agent_state.can_retry():
            agent_state.mark_failed("Max retries exceeded")
            return state
        
        # Get error pattern
        if agent_state.errors:
            last_error = agent_state.errors[-1]["error"]
            
            # Search for solution
            search_result = self.tools.web_search(f"how to fix {last_error}")
            
            if search_result["success"]:
                agent_state.search_history.append(last_error)
                self.logger.info(f"Found {search_result['count']} potential solutions")
                
                # Cache the search results
                if self.memory:
                    self.memory.add_error_pattern(
                        "execution_error",
                        last_error,
                        json.dumps(search_result["results"], ensure_ascii=False)
                    )
        
        agent_state.increment_retry()
        state["agent_state"] = agent_state
        return state
    
    def _adapt_plan(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 6: Adapt the plan based on feedback"""
        self.logger.info("[ADAPTING] Modifying plan based on feedback...")
        
        agent_state = state["agent_state"]
        agent_state.add_thought("Plan needs adjustment")
        
        # Re-plan with learned information
        prompt = f"""The original plan isn't working. Adapt it:

Original goal: {agent_state.goal}
Previous plan: {json.dumps(agent_state.plan, ensure_ascii=False)}
Errors encountered: {json.dumps([e.get('error', '') for e in agent_state.errors], ensure_ascii=False)}
Learned: {json.dumps(agent_state.context.get('learned_solutions', {}), ensure_ascii=False)}

Provide a revised plan with different approaches."""
        
        result = self.llm.completion(prompt, max_tokens=1500)
        
        if result["success"]:
            steps = [line.strip() for line in result["text"].split('\n') if line.strip().startswith('[STEP')]
            agent_state.set_plan(steps)
            self.logger.info(f"✓ Plan adapted with {len(steps)} new steps")
        
        state["agent_state"] = agent_state
        return state
    
    def _complete_task(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 7: Mark task as completed"""
        self.logger.info("[COMPLETE] Task execution completed")
        
        agent_state = state["agent_state"]
        agent_state.mark_completed()
        
        # Summary
        summary = {
            "task_id": agent_state.task_id,
            "goal": agent_state.goal,
            "status": "completed",
            "steps_executed": agent_state.current_step,
            "tools_used": len(agent_state.tool_calls),
            "duration_seconds": agent_state.get_duration(),
            "results": len(agent_state.results),
            "errors": len(agent_state.errors)
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
