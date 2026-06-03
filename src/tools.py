import subprocess
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from datetime import datetime


class ToolExecutor:
    """Execute tools and manage external processes"""
    
    def __init__(self, memory=None):
        self.memory = memory
        self.last_tool_execution: Optional[datetime] = None
        self.max_retries = 3
    
    def web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web using DuckDuckGo"""
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Query cannot be empty",
                "results": [],
                "count": 0
            }
        
        try:
            start_time = time.time()
            results = []
            
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=max_results))
                for result in search_results:
                    results.append({
                        "title": result.get("title", "No title"),
                        "body": result.get("body", "No description"),
                        "link": result.get("href", "No link")
                    })
            
            duration = time.time() - start_time
            self.last_tool_execution = datetime.now()
            
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
                "error": str(e),
                "results": [],
                "count": 0
            }
    
    def fetch_and_summarize(self, url: str) -> Dict[str, Any]:
        """Fetch webpage and extract main content"""
        if not url or not url.strip():
            return {
                "success": False,
                "error": "URL cannot be empty",
                "title": "",
                "content": ""
            }
        
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
            self.last_tool_execution = datetime.now()
            
            if self.memory:
                self.memory.record_tool_usage("fetch_and_summarize", True, duration)
            
            return {
                "success": True,
                "title": soup.title.string if soup.title else "No title",
                "content": content[:2000],  # Max 2000 chars
                "url": url,
                "duration": duration
            }
        
        except requests.exceptions.Timeout:
            if self.memory:
                self.memory.record_tool_usage("fetch_and_summarize", False, 0)
            return {
                "success": False,
                "error": "Request timeout",
                "title": "",
                "content": ""
            }
        except requests.exceptions.ConnectionError:
            if self.memory:
                self.memory.record_tool_usage("fetch_and_summarize", False, 0)
            return {
                "success": False,
                "error": "Connection error",
                "title": "",
                "content": ""
            }
        except Exception as e:
            if self.memory:
                self.memory.record_tool_usage("fetch_and_summarize", False, 0)
            return {
                "success": False,
                "error": str(e),
                "title": "",
                "content": ""
            }
    
    def save_output(
        self,
        content: str,
        filename: str,
        output_dir: str = "output"
    ) -> Dict[str, Any]:
        """Save output to file"""
        if not content or not filename:
            return {
                "success": False,
                "error": "Content and filename are required",
                "path": "",
                "size": 0
            }
        
        try:
            start_time = time.time()
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Sanitize filename
            safe_filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_', '.'))
            if not safe_filename:
                safe_filename = "output.txt"
            
            file_path = output_path / safe_filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            duration = time.time() - start_time
            self.last_tool_execution = datetime.now()
            
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
                "error": str(e),
                "path": "",
                "size": 0
            }
    
    def execute_command(
        self,
        command: str,
        timeout: int = 60,
        shell: bool = True
    ) -> Dict[str, Any]:
        """Execute shell command with timeout and error handling"""
        if not command or not command.strip():
            return {
                "success": False,
                "error": "Command cannot be empty",
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "duration": 0
            }
        
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
            self.last_tool_execution = datetime.now()
            
            if self.memory:
                self.memory.record_tool_usage("execute_command", success, duration)
            
            return {
                "success": success,
                "stdout": result.stdout[:2000] if result.stdout else "",
                "stderr": result.stderr[:2000] if result.stderr else "",
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
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "duration": timeout,
                "command": command
            }
        except Exception as e:
            if self.memory:
                self.memory.record_tool_usage("execute_command", False, 0)
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "duration": 0,
                "command": command
            }
    
    def install_package(self, package_name: str) -> Dict[str, Any]:
        """Autonomously install Python packages"""
        if not package_name or not package_name.strip():
            return {
                "success": False,
                "error": "Package name cannot be empty"
            }
        
        try:
            # Validate package name
            safe_name = "".join(c for c in package_name if c.isalnum() or c in ('-', '_', '.'))
            if not safe_name:
                return {
                    "success": False,
                    "error": "Invalid package name"
                }
            
            result = self.execute_command(
                f"pip install --upgrade {safe_name}",
                timeout=300
            )
            
            if result["success"] and self.memory:
                self.memory.cache_solution(
                    f"install_{safe_name}",
                    result.get("stdout", "")
                )
            
            return result
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
