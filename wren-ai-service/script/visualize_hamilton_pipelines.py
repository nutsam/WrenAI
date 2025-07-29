#!/usr/bin/env python3
"""
Hamilton Pipeline Visualization Script for WrenAI

This script generates DAG visualizations for all Hamilton pipelines in the WrenAI project.
It can visualize individual pipelines or create a comprehensive overview of all pipelines.

Usage:
    python visualize_hamilton_pipelines.py --pipeline all
    python visualize_hamilton_pipelines.py --pipeline sql_generation
    python visualize_hamilton_pipelines.py --pipeline db_schema_retrieval --format png
"""

import argparse
import sys
import os
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root and src to path to import WrenAI modules
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

try:
    from hamilton import base
    from hamilton.async_driver import AsyncDriver
    from hamilton.graph import create_graphviz_graph
    import graphviz
except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("Please install: pip install hamilton-sdk graphviz")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HamiltonPipelineVisualizer:
    """Visualizes Hamilton pipelines in the WrenAI project."""
    
    def __init__(self):
        self.pipeline_modules = self._discover_pipeline_modules()
        self.output_dir = Path("pipeline_visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def _discover_pipeline_modules(self) -> Dict[str, Path]:
        """Discover all Hamilton pipeline modules in the project."""
        pipeline_modules = {}
        
        # Define pipeline directories (relative to project root)
        project_root = Path(__file__).parent.parent
        pipeline_dirs = [
            project_root / "src/pipelines/indexing",
            project_root / "src/pipelines/retrieval", 
            project_root / "src/pipelines/generation"
        ]
        
        for pipeline_path in pipeline_dirs:
            if pipeline_path.exists():
                for py_file in pipeline_path.glob("*.py"):
                    if py_file.name != "__init__.py":
                        module_name = py_file.stem
                        category = pipeline_path.name
                        full_name = f"{category}.{module_name}"
                        pipeline_modules[full_name] = py_file
        
        return pipeline_modules
    
    def _load_module(self, module_path: Path) -> Optional[Any]:
        """Dynamically load a Python module from file path."""
        try:
            # Ensure the project root is in sys.path for src imports
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # Use a unique module name to avoid conflicts
            module_name = f"hamilton_viz_{module_path.stem}_{id(module_path)}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Set the module name properly before execution
                module.__name__ = module_name
                spec.loader.exec_module(module)
                return module
        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {e}")
            logger.debug(f"Current sys.path: {sys.path[:3]}...")  # Show first 3 paths for debugging
        return None
    
    def _extract_hamilton_functions(self, module: Any) -> dict:
        """Extract Hamilton pipeline functions from a module."""
        functions = {}
        
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(module, attr_name)
            if not callable(attr):
                continue
                
            # Check if it's likely a Hamilton function by looking for annotations
            # and excluding classes, imported functions, etc.
            if (hasattr(attr, '__annotations__') and 
                attr.__annotations__ and 
                hasattr(attr, '__module__') and
                attr.__module__ == module.__name__ and
                not isinstance(attr, type)):  # Exclude classes like BaseModel
                functions[attr_name] = attr
        
        return functions
    
    def _analyze_function_dependencies(self, hamilton_functions: dict) -> dict:
        """Analyze dependencies between Hamilton functions."""
        dependencies = {}
        
        for func_name, func in hamilton_functions.items():
            deps = []
            if hasattr(func, '__annotations__'):
                # Extract parameter names as dependencies
                for param_name, param_type in func.__annotations__.items():
                    if param_name != 'return' and param_name in hamilton_functions:
                        deps.append(param_name)
            dependencies[func_name] = deps
                        
        return dependencies
    
    def _create_simple_dag_visualization(self, pipeline_name: str, hamilton_functions: dict, output_format: str) -> bool:
        """Create a simple DAG visualization without Hamilton driver."""
        try:
            import graphviz
            
            # Analyze dependencies
            dependencies = self._analyze_function_dependencies(hamilton_functions)
            
            # Create graphviz graph
            graph = graphviz.Digraph(f'Hamilton_Pipeline_{pipeline_name}')
            graph.attr(rankdir='TB')
            graph.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
            graph.attr('edge', color='darkblue')
            graph.attr(label=f'Hamilton Pipeline: {pipeline_name}', fontsize='16', fontname='Arial Bold')
            
            # Add nodes
            for func_name in hamilton_functions:
                graph.node(func_name, func_name)
            
            # Add edges based on dependencies
            for func_name, deps in dependencies.items():
                for dep in deps:
                    if dep in hamilton_functions:
                        graph.edge(dep, func_name)
            
            # Save the graph
            output_path = self.output_dir / f"{pipeline_name.replace('.', '_')}"
            graph.render(str(output_path), format=output_format, cleanup=True)
            
            logger.info(f"Pipeline visualization saved: {output_path}.{output_format}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create simple DAG visualization: {e}")
            return False
    
    def visualize_pipeline(self, pipeline_name: str, output_format: str = "png") -> bool:
        """Visualize a specific pipeline."""
        if pipeline_name not in self.pipeline_modules:
            logger.error(f"Pipeline '{pipeline_name}' not found. Available: {list(self.pipeline_modules.keys())}")
            return False
        
        module_path = self.pipeline_modules[pipeline_name]
        logger.info(f"Visualizing pipeline: {pipeline_name} from {module_path}")
        
        # Load the module
        module = self._load_module(module_path)
        if not module:
            return False
        
        # Extract functions
        functions = self._extract_hamilton_functions(module)
        logger.info(f"Found functions: {list(functions.keys())}")
        
        # Create simple DAG visualization directly
        return self._create_simple_dag_visualization(pipeline_name, functions, output_format)
    
    def visualize_all_pipelines(self, output_format: str = "png") -> Dict[str, bool]:
        """Visualize all discovered pipelines."""
        results = {}
        
        logger.info(f"Visualizing {len(self.pipeline_modules)} pipelines...")
        
        for pipeline_name in self.pipeline_modules:
            logger.info(f"\n{'='*50}")
            results[pipeline_name] = self.visualize_pipeline(pipeline_name, output_format)
        
        # Generate summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Visualization Summary: {successful}/{total} successful")
        
        if successful > 0:
            logger.info(f"Visualizations saved in: {self.output_dir}")
        
        return results
    
    def create_pipeline_overview(self, output_format: str = "png") -> bool:
        """Create a high-level overview of all pipelines."""
        try:
            # Create a high-level graph
            overview = graphviz.Digraph('WrenAI_Pipeline_Overview')
            overview.attr(rankdir='TB')
            overview.attr('node', shape='box', style='rounded,filled')
            overview.attr('edge', color='darkgreen', style='bold')
            overview.attr(label='WrenAI Hamilton Pipelines Overview', fontsize='20', fontname='Arial Bold')
            
            # Add categories
            categories = {
                'indexing': {'color': 'lightgreen', 'pipelines': []},
                'retrieval': {'color': 'lightblue', 'pipelines': []},
                'generation': {'color': 'lightyellow', 'pipelines': []}
            }
            
            # Categorize pipelines
            for pipeline_name in self.pipeline_modules:
                category = pipeline_name.split('.')[0]
                if category in categories:
                    categories[category]['pipelines'].append(pipeline_name)
            
            # Add category nodes
            for category, info in categories.items():
                with overview.subgraph(name=f'cluster_{category}') as cluster:
                    cluster.attr(label=f'{category.title()} Pipelines', fontsize='14', style='filled', fillcolor=info['color'])
                    
                    for pipeline in info['pipelines']:
                        node_name = pipeline.replace('.', '_')
                        display_name = pipeline.split('.')[1] if '.' in pipeline else pipeline
                        cluster.node(node_name, display_name, fillcolor='white')
            
            # Add flow connections (conceptual)
            overview.edge('indexing_db_schema', 'retrieval_db_schema_retrieval', label='schema')
            overview.edge('retrieval_db_schema_retrieval', 'generation_sql_generation', label='context')
            overview.edge('generation_sql_generation', 'generation_sql_correction', label='validate')
            
            # Save overview
            overview_path = self.output_dir / "pipeline_overview"
            overview.render(str(overview_path), format=output_format, cleanup=True)
            
            logger.info(f"Pipeline overview saved: {overview_path}.{output_format}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create pipeline overview: {e}")
            return False
    
    def list_pipelines(self) -> None:
        """List all discovered pipelines."""
        print("\nDiscovered Hamilton Pipelines:")
        print("=" * 50)
        
        categories = {}
        for pipeline_name in sorted(self.pipeline_modules.keys()):
            category = pipeline_name.split('.')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(pipeline_name.split('.')[1])
        
        for category, pipelines in categories.items():
            print(f"\n{category.upper()} ({len(pipelines)} pipelines):")
            for pipeline in sorted(pipelines):
                print(f"  - {pipeline}")
        
        print(f"\nTotal: {len(self.pipeline_modules)} pipelines")


def main():
    parser = argparse.ArgumentParser(description="Visualize Hamilton pipelines in WrenAI")
    parser.add_argument(
        "--pipeline", 
        default="all",
        help="Pipeline to visualize (use 'all' for all pipelines, 'overview' for high-level view, or specific pipeline name)"
    )
    parser.add_argument(
        "--format", 
        default="png",
        choices=["png", "svg", "pdf", "dot"],
        help="Output format for the visualization"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available pipelines"
    )
    
    args = parser.parse_args()
    
    visualizer = HamiltonPipelineVisualizer()
    
    if args.list:
        visualizer.list_pipelines()
        return
    
    if args.pipeline == "all":
        visualizer.visualize_all_pipelines(args.format)
        visualizer.create_pipeline_overview(args.format)
    elif args.pipeline == "overview":
        visualizer.create_pipeline_overview(args.format)
    else:
        success = visualizer.visualize_pipeline(args.pipeline, args.format)
        if not success:
            print(f"\nUse --list to see available pipelines")


if __name__ == "__main__":
    main()