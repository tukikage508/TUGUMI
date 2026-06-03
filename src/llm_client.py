import requests
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import threading

class LLMClient:
    """Client for local llama.cpp server with health monitoring"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:8080", timeout: int = 300):
        self.server_url = server_url
        self.timeout = timeout
        self.health_check_interval = 30  # seconds
        self.last_health_check = None
        self.is_healthy = False
        self.start_time = None
        self.last_response_time = None
        self._health_monitor_thread = None
        self._stop_monitor = False
    
    def start_health_monitor(self):
        """Start background health monitoring"""
        if not self._health_monitor_thread or not self._health_monitor_thread.is_alive():
            self._stop_monitor = False
            self._health_monitor_thread = threading.Thread(target=self._monitor_health, daemon=True)
            self._health_monitor_thread.start()
    
    def stop_health_monitor(self):
        """Stop background health monitoring"""
        self._stop_monitor = True
        if self._health_monitor_thread:
            self._health_monitor_thread.join(timeout=5)
    
    def _monitor_health(self):
        """Continuous health monitoring in background"""
        while not self._stop_monitor:
            try:
                response = requests.get(
                    f"{self.server_url}/health",
                    timeout=5
                )
                self.is_healthy = response.status_code == 200
                self.last_health_check = datetime.now()
            except Exception:
                self.is_healthy = False
                self.last_health_check = datetime.now()
            
            time.sleep(self.health_check_interval)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status"""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=5
            )
            self.is_healthy = response.status_code == 200
            self.last_health_check = datetime.now()
            
            return {
                "status": "healthy" if self.is_healthy else "unhealthy",
                "server_url": self.server_url,
                "last_check": self.last_health_check.isoformat() if self.last_health_check else None,
                "is_running": self.is_healthy,
                "last_response_time": self.last_response_time
            }
        except requests.exceptions.ConnectionError:
            self.is_healthy = False
            return {
                "status": "unreachable",
                "server_url": self.server_url,
                "error": "Cannot connect to llama.cpp server",
                "last_response_time": self.last_response_time
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def completion(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: list = None
    ) -> Dict[str, Any]:
        """Request completion from llama.cpp server with progress tracking"""
        
        if not self.is_healthy:
            # Try to reconnect
            health = self.get_health_status()
            if not self.is_healthy:
                raise ConnectionError(f"LLM server not responding: {health}")
        
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                timeout=self.timeout
            )
            self.last_response_time = time.time() - start_time
            self.is_healthy = True
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "text": response.json().get("content", ""),
                    "response_time": self.last_response_time,
                    "model_metadata": response.json().get("model", "")
                }
            else:
                return {
                    "success": False,
                    "error": f"Server returned {response.status_code}",
                    "response_time": self.last_response_time
                }
        
        except requests.exceptions.Timeout:
            self.is_healthy = False
            return {
                "success": False,
                "error": f"Request timeout after {self.timeout}s. LLM may be thinking...",
                "response_time": time.time() - start_time
            }
        except requests.exceptions.ConnectionError:
            self.is_healthy = False
            return {
                "success": False,
                "error": "Cannot connect to LLM server. Is it running?"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def streaming_completion(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        callback = None
    ) -> str:
        """Stream completion for progress tracking"""
        
        if not self.is_healthy:
            health = self.get_health_status()
            if not self.is_healthy:
                raise ConnectionError(f"LLM server not responding: {health}")
        
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        full_response = ""
        try:
            response = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    content = data.get("content", "")
                    full_response += content
                    
                    if callback:
                        callback(content, data.get("stop", False))
            
            self.is_healthy = True
            return full_response
        
        except Exception as e:
            self.is_healthy = False
            raise Exception(f"Streaming error: {str(e)}")
