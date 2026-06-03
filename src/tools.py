import subprocess
import json
import time
from typing import Dict, List, Any
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from datetime import datetime

class ToolExecutor:
    """Execute tools and manage external processes"""
    
    def __init__(self, memory=None):
        self.memory = memory
        self.last_tool_execution = None
    
    def web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web using DuckDuckGo"""
        try:
            start_time = time.time()
            results = []
            
            with DDGS() as ddgs:
                for result in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": result.get("title"),
                        "body": result.get("body"),
                        "link": result.get("href")
                    })
            
            duration = time.time() - start_time
            if self.memory:
                self.memory.record_tool_usage("web_search", True, duration)
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "duration": duration
            }
        
        except Exception as e:
            if self.memory:
                self.memory.record_tool_usage("web_search", False, 0)
            return {
                "success": False,
                "error": str(e)
            }
    
    def fetch_and_summarize(self, url: str) -> Dict[str, Any]:
        """Fetch webpage and extract main content"""
        try:
            start_time = time.time()
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            content = '\n'.join(lines[:100])  # First 100 lines
            
            duration = time.time() - start_time
            if self.memory:
                self.memory.record_tool_usage("fetch_and_summarize", True, duration)
            
            return {
                "success": True,
                "title": soup.title.string if soup.title else "No title",
                "content": content[:2000],  # Max 2000 chars
                "url": url,
                "duration": duration
            }
        
        except Exception as e:
            if self.memory:
                self.memory.record_tool_usage("fetch_and_summarize", False, 0)
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_output(
        self,
        content: str,
        filename: str,
        output_dir: str = "/storage/emulated/0/TUGUMIDesk"
    ) -> Dict[str, Any]:
        """Save output to file on device storage"""
        try:
            start_time = time.time()
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            file_path = output_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            duration = time.time() - start_time
            if self.memory:
                self.memory.record_tool_usage("save_output", True, duration)
            
            return {
                "success": True,
                "path": str(file_path),
                "size": len(content),
                "duration": duration
            }
        
        except Exception as e:
            if self.memory:
                self.memory.record_tool_usage("save_output", False, 0)
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_command(
        self,
        command: str,
        timeout: int = 60,
        shell: bool = True
    ) -> Dict[str, Any]:
        """Execute shell command with timeout and error handling"""
        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            if self.memory:
                self.memory.record_tool_usage("execute_command", success, duration)
            
            return {
                "success": success,
                "stdout": result.stdout[:2000],  # Limit output
                "stderr": result.stderr[:2000],
                "return_code": result.returncode,
                "duration": duration,
                "command": command
            }
        
        except subprocess.TimeoutExpired:
            if self.memory:
                self.memory.record_tool_usage("execute_command", False, timeout)
            return {
                "success": False,
                "error": f"Command timeout after {timeout}s",
                "command": command
            }
        except Exception as e:
            if self.memory:
                self.memory.record_tool_usage("execute_command", False, 0)
            return {
                "success": False,
                "error": str(e),
                "command": command
            }
    
    def install_package(self, package_name: str) -> Dict[str, Any]:
        """Autonomously install Python packages"""
        try:
            result = self.execute_command(
                f"pip install --upgrade {package_name}",
                timeout=300
            )
            
            if result["success"]:
                if self.memory:
                    self.memory.cache_solution(
                        f"install_{package_name}",
                        result["stdout"]
                    )
            
            return result
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
