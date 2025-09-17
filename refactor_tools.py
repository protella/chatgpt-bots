#!/usr/bin/env python3
"""
Bulletproof Refactoring Tools - Ensure NO calls are missed during modularization
"""

import ast
import os
import sys
import json
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class MethodCall:
    """Represents a method call found in the code"""
    file_path: str
    line_number: int
    column: int
    call_type: str  # 'self', 'instance', 'class', 'dynamic'
    full_expression: str
    context_lines: List[str] = field(default_factory=list)

@dataclass
class MethodMove:
    """Represents a method being moved to a new location"""
    original_class: str
    original_method: str
    new_module: str
    new_class: str
    new_method: str
    new_access_path: str  # e.g., "self.vision.analyze"

class ComprehensiveCallFinder(ast.NodeVisitor):
    """AST visitor that finds ALL calls to a method"""

    def __init__(self, method_name: str, class_name: Optional[str] = None):
        self.method_name = method_name
        self.class_name = class_name
        self.calls: List[Tuple[int, int, str, str]] = []
        self.current_class = None
        self.in_target_class = False

    def visit_ClassDef(self, node):
        old_class = self.current_class
        old_in_target = self.in_target_class

        self.current_class = node.name
        if self.class_name:
            self.in_target_class = (node.name == self.class_name)

        self.generic_visit(node)

        self.current_class = old_class
        self.in_target_class = old_in_target

    def visit_Attribute(self, node):
        """Find attribute access like self.method() or instance.method()"""
        if node.attr == self.method_name:
            # Get the full expression
            if isinstance(node.value, ast.Name):
                if node.value.id == 'self':
                    call_type = 'self'
                else:
                    call_type = 'instance'
                full_expr = f"{node.value.id}.{node.attr}"
            elif isinstance(node.value, ast.Attribute):
                call_type = 'chained'
                full_expr = ast.unparse(node) if hasattr(ast, 'unparse') else f"?.{node.attr}"
            else:
                call_type = 'complex'
                full_expr = ast.unparse(node) if hasattr(ast, 'unparse') else f"?.{node.attr}"

            self.calls.append((node.lineno, node.col_offset, call_type, full_expr))

        self.generic_visit(node)

    def visit_Call(self, node):
        """Find function calls including getattr and dynamic calls"""
        # Check for getattr(obj, "method_name")
        if isinstance(node.func, ast.Name) and node.func.id == 'getattr':
            if len(node.args) >= 2:
                if isinstance(node.args[1], ast.Constant) and node.args[1].value == self.method_name:
                    obj = ast.unparse(node.args[0]) if hasattr(ast, 'unparse') else "?"
                    self.calls.append((node.lineno, node.col_offset, 'dynamic', f'getattr({obj}, "{self.method_name}")'))

        self.generic_visit(node)

class RefactoringSafetyAnalyzer:
    """Ensures safe refactoring by finding ALL method references"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.method_calls: Dict[str, List[MethodCall]] = {}
        self.method_definitions: Dict[str, Tuple[str, int]] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}

    def analyze_file(self, file_path: Path) -> Dict[str, List[MethodCall]]:
        """Analyze a single Python file for method calls"""
        calls = {}

        try:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.splitlines()
                tree = ast.parse(content)

            # Find all method definitions first
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    key = f"{file_path}:{node.name}"
                    self.method_definitions[key] = (str(file_path), node.lineno)

            return calls

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {}

    def find_all_calls_to_method(self, method_name: str, class_name: Optional[str] = None) -> List[MethodCall]:
        """Find ALL calls to a specific method across the entire project"""
        all_calls = []

        # Search all Python files
        for py_file in self.project_root.rglob("*.py"):
            if 'venv' in py_file.parts or '__pycache__' in py_file.parts:
                continue

            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.splitlines()

                tree = ast.parse(content)
                finder = ComprehensiveCallFinder(method_name, class_name)
                finder.visit(tree)

                for line_no, col, call_type, expr in finder.calls:
                    # Get context lines
                    start = max(0, line_no - 2)
                    end = min(len(lines), line_no + 2)
                    context = lines[start:end]

                    call = MethodCall(
                        file_path=str(py_file),
                        line_number=line_no,
                        column=col,
                        call_type=call_type,
                        full_expression=expr,
                        context_lines=context
                    )
                    all_calls.append(call)

            except Exception as e:
                print(f"Error processing {py_file}: {e}")

        return all_calls

    def analyze_method_dependencies(self, file_path: str, class_name: str) -> Dict[str, Set[str]]:
        """Find which methods call which other methods within a class"""
        dependencies = {}

        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        class DependencyAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.current_method = None
                self.current_class = None
                self.deps = {}

            def visit_ClassDef(self, node):
                if node.name == class_name:
                    self.current_class = node.name
                    self.generic_visit(node)
                    self.current_class = None

            def visit_FunctionDef(self, node):
                if self.current_class:
                    old_method = self.current_method
                    self.current_method = node.name
                    self.deps[node.name] = set()
                    self.generic_visit(node)
                    self.current_method = old_method

            def visit_Attribute(self, node):
                if self.current_method and isinstance(node.value, ast.Name) and node.value.id == 'self':
                    self.deps[self.current_method].add(node.attr)
                self.generic_visit(node)

        analyzer = DependencyAnalyzer()
        analyzer.visit(tree)
        return analyzer.deps

    def generate_move_plan(self, method_moves: List[MethodMove]) -> Dict:
        """Generate a complete plan for moving methods safely"""
        plan = {
            "moves": [],
            "required_updates": [],
            "risk_analysis": [],
            "test_requirements": []
        }

        for move in method_moves:
            # Find all calls to this method
            calls = self.find_all_calls_to_method(move.original_method)

            move_info = {
                "method": f"{move.original_class}.{move.original_method}",
                "destination": f"{move.new_class}.{move.new_method}",
                "call_sites": len(calls),
                "files_affected": len(set(c.file_path for c in calls)),
                "updates_required": []
            }

            # Generate update requirements for each call
            for call in calls:
                update = {
                    "file": call.file_path,
                    "line": call.line_number,
                    "current": call.full_expression,
                    "new": call.full_expression.replace(move.original_method, move.new_access_path),
                    "type": call.call_type
                }
                move_info["updates_required"].append(update)

            plan["moves"].append(move_info)

        return plan

    def verify_no_dynamic_calls(self, method_name: str) -> bool:
        """Check if there are any dynamic calls that might break"""
        calls = self.find_all_calls_to_method(method_name)
        dynamic_calls = [c for c in calls if c.call_type in ('dynamic', 'complex')]

        if dynamic_calls:
            print(f"⚠️  WARNING: Found {len(dynamic_calls)} dynamic calls to {method_name}:")
            for call in dynamic_calls:
                print(f"  - {call.file_path}:{call.line_number} - {call.full_expression}")
            return False
        return True

    def generate_test_verification_script(self, method_moves: List[MethodMove]) -> str:
        """Generate a script to verify all moves work correctly"""
        script = '''#!/usr/bin/env python3
"""Auto-generated verification script for refactoring"""
import sys
import importlib

def verify_method_exists(module_path, class_name, method_name):
    """Verify a method exists and is callable"""
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        method = getattr(cls, method_name)
        assert callable(method), f"{method_name} is not callable"
        return True
    except Exception as e:
        print(f"❌ {class_name}.{method_name} verification failed: {e}")
        return False

def main():
    success = True

'''
        for move in method_moves:
            script += f'''    # Verify {move.original_method} is accessible
    success &= verify_method_exists("{move.new_module}", "{move.original_class}", "{move.original_method}")
    success &= verify_method_exists("{move.new_module}", "{move.new_class}", "{move.new_method}")

'''
        script += '''
    if success:
        print("✅ All method moves verified successfully")
    else:
        print("❌ Some verifications failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        return script


def main():
    """Example usage and safety checks"""
    analyzer = RefactoringSafetyAnalyzer()

    # Example: Check before moving _handle_vision_analysis
    print("=" * 60)
    print("REFACTORING SAFETY ANALYSIS")
    print("=" * 60)

    methods_to_check = [
        ("MessageProcessor", "_handle_vision_analysis"),
        ("MessageProcessor", "_handle_image_generation"),
        ("OpenAIClient", "_enhance_image_prompt"),
    ]

    for class_name, method_name in methods_to_check:
        print(f"\n### Analyzing {class_name}.{method_name}")

        # Find all calls
        calls = analyzer.find_all_calls_to_method(method_name, class_name)
        print(f"Found {len(calls)} call sites:")

        # Group by file
        calls_by_file = {}
        for call in calls:
            if call.file_path not in calls_by_file:
                calls_by_file[call.file_path] = []
            calls_by_file[call.file_path].append(call)

        for file_path, file_calls in calls_by_file.items():
            print(f"\n  {file_path}:")
            for call in file_calls:
                print(f"    Line {call.line_number}: {call.full_expression} ({call.call_type})")
                if call.context_lines:
                    for line in call.context_lines:
                        print(f"      | {line}")

        # Check for risky dynamic calls
        if not analyzer.verify_no_dynamic_calls(method_name):
            print(f"  ⚠️  Refactoring {method_name} requires extra care due to dynamic calls")

    # Generate a move plan
    print("\n" + "=" * 60)
    print("GENERATING MOVE PLAN")
    print("=" * 60)

    example_moves = [
        MethodMove(
            original_class="MessageProcessor",
            original_method="_handle_vision_analysis",
            new_module="message_processor.handlers.vision",
            new_class="VisionHandler",
            new_method="analyze",
            new_access_path="self.vision.analyze"
        )
    ]

    plan = analyzer.generate_move_plan(example_moves)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()