import math
import re
from datetime import datetime
from typing import Dict, Optional

class ToolRegistry:
    """Registry for tools (calculator, date, etc.), from your script."""
    def __init__(self):
        self.tools = {
            'calculator': self._calculator,
            'date': self._current_date,
            'math': self._math_solver
        }
    
    def _calculator(self, expression: str) -> str:
        """Simple calculator with safety"""
        try:
            if re.search(r"[a-zA-Z]", expression):
                return "Error: Unsafe expression. Only numbers and operators are allowed."
            return f"Result: {eval(expression, {'__builtins__': {}}, {})}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _current_date(self, _: str) -> str:
        """Get current date"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _math_solver(self, problem: str) -> str:
        """Basic math problem solver"""
        try:
            if "sqrt" in problem.lower():
                match = re.search(r'sqrt\s*\(?\s*(\d+\.?\d*)\s*\)?', problem)
                if match:
                    num = float(match.group(1))
                    return f"√{num} = {math.sqrt(num)}"
            return "Math solver: Problem type not recognized."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def route_query(self, query: str) -> Optional[Dict]:
        """Detect if a query requires a tool and return the tool and input."""
        query_lower = query.lower()
        if any(op in query for op in ['+', '-', '*', '/']) and re.search(r'\d', query):
            return {'name': 'calculator', 'input': query}
        elif any(word in query_lower for word in ['date', 'time', 'today']):
            return {'name': 'date', 'input': ''}
        elif 'sqrt' in query_lower:
            return {'name': 'math', 'input': query}
        return None
    
    def execute_tool(self, tool_name: str, input_data: str) -> str:
        """Execute a tool"""
        if tool_name in self.tools:
            return self.tools[tool_name](input_data)
        return f"Tool '{tool_name}' not found."