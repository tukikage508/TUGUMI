#!/usr/bin/env python3
"""
TUGUMI - Self-Autonomous AI Agent Framework
Main entry point for running autonomous tasks
"""

from src.runner import AgentRunner
from src.logger import TUGUMILogger
import argparse
import sys
from datetime import datetime


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TUGUMI - Self-Autonomous AI Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Create a Python script that prints Hello World"
  python main.py -u http://localhost:8080 "Search for Python tips and save results"
  python main.py --timeout 600 "Complex research task"
        """
    )
    
    parser.add_argument(
        "goal",
        nargs="?",
        help="The goal for the autonomous agent to accomplish"
    )
    parser.add_argument(
        "-u", "--url",
        default="http://127.0.0.1:8080",
        help="LLM server URL (default: http://127.0.0.1:8080)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=300,
        help="LLM request timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "-m", "--memory",
        default="data/memory.json",
        help="Memory file path (default: data/memory.json)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show agent status and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = TUGUMILogger("TUGUMI_MAIN")
    
    logger.info("="*70)
    logger.info("TUGUMI - Self-Autonomous AI Agent Framework")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("="*70)
    
    # Initialize runner
    try:
        runner = AgentRunner(
            llm_server_url=args.url,
            llm_timeout=args.timeout,
            memory_file=args.memory
        )
        
        # Show status if requested
        if args.status:
            status = runner.get_status()
            print("\nAgent Status:")
            print(f"  LLM: {status['llm_health']['status']}")
            print(f"  Tasks: {status['memory_summary']['total_tasks']}")
            print(f"  Tools Used: {status['memory_summary']['tool_count']}")
            runner.cleanup()
            return 0
        
        # Require goal for task execution
        if not args.goal:
            parser.print_help()
            logger.warning("No goal provided. Use -h for help.")
            runner.cleanup()
            return 1
        
        # Run task
        logger.info(f"Goal: {args.goal}")
        result = runner.run_task(args.goal)
        
        # Print results
        print("\n" + "="*70)
        if result.get("success"):
            print("✓ TASK COMPLETED")
            summary = result.get("summary", {})
            if summary:
                print(f"\nSummary:")
                print(f"  Task ID: {summary.get('task_id')}")
                print(f"  Status: {summary.get('status')}")
                print(f"  Duration: {summary.get('duration_seconds', 0):.1f}s")
                print(f"  Steps: {summary.get('steps_executed', 0)}/{summary.get('steps_planned', 0)}")
                print(f"  Tools: {summary.get('successful_tools', 0)} successful, {summary.get('failed_tools', 0)} failed")
        else:
            print("✗ TASK FAILED")
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("="*70)
        
        runner.cleanup()
        return 0 if result.get("success") else 1
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"✗ Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
