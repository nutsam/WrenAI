#!/usr/bin/env python3
"""
Dataset Schema Analyzer

This script analyzes TOML dataset files to extract table and column information.

Usage:
    # Analyze a single file
    python dataset_analyzer.py eval/dataset/ecommerce_eval_dataset.toml
    
    # Analyze all files in a directory
    python dataset_analyzer.py eval/dataset/
    
    # Show only summary (no detailed table info)
    python dataset_analyzer.py eval/dataset/ --summary-only

Features:
    - Extracts dataset metadata (ID, date, catalog, schema, data source)
    - Lists all tables with their primary keys
    - Shows all columns with their types and constraints
    - Provides summary statistics
    - Supports both English and Chinese table/column names
    - Handles multiple file analysis with summary reports
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import toml
from tabulate import tabulate


class DatasetAnalyzer:
    """Analyzes TOML dataset files to extract schema information."""
    
    def __init__(self):
        self.dataset_info = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single TOML file and extract schema information."""
        try:
            # Load TOML file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = toml.load(f)
            
            # Extract basic info
            dataset_id = data.get('dataset_id', 'Unknown')
            date = data.get('date', 'Unknown')
            
            # Extract MDL information
            mdl = data.get('mdl', {})
            catalog = mdl.get('catalog', 'Unknown')
            schema = mdl.get('schema', 'Unknown')
            data_source = mdl.get('dataSource', 'Unknown')
            
            # Extract tables and columns
            tables = []
            models = mdl.get('models', [])
            
            for model in models:
                table_name = model.get('name', 'Unknown')
                primary_key = model.get('primaryKey', None)
                
                # Extract columns
                columns = []
                model_columns = model.get('columns', [])
                
                for col in model_columns:
                    col_name = col.get('name', 'Unknown')
                    col_type = col.get('type', 'Unknown')
                    not_null = col.get('notNull', False)
                    
                    columns.append({
                        'name': col_name,
                        'type': col_type,
                        'notNull': not_null,
                        'isPrimaryKey': col_name == primary_key
                    })
                
                tables.append({
                    'name': table_name,
                    'primaryKey': primary_key,
                    'columns': columns
                })
            
            # Store analysis results
            result = {
                'file_path': file_path,
                'dataset_id': dataset_id,
                'date': date,
                'catalog': catalog,
                'schema': schema,
                'data_source': data_source,
                'tables': tables,
                'total_tables': len(tables),
                'total_columns': sum(len(table['columns']) for table in tables)
            }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return None
    
    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print analysis results in a formatted way."""
        if not analysis:
            return
            
        print("=" * 80)
        print(f"Dataset Analysis: {Path(analysis['file_path']).name}")
        print("=" * 80)
        
        # Basic information
        print(f"Dataset ID: {analysis['dataset_id']}")
        print(f"Date: {analysis['date']}")
        print(f"Catalog: {analysis['catalog']}")
        print(f"Schema: {analysis['schema']}")
        print(f"Data Source: {analysis['data_source']}")
        print(f"Total Tables: {analysis['total_tables']}")
        print(f"Total Columns: {analysis['total_columns']}")
        print()
        
        # Tables and columns
        for i, table in enumerate(analysis['tables'], 1):
            print(f"Table {i}: {table['name']}")
            if table['primaryKey']:
                print(f"  Primary Key: {table['primaryKey']}")
            print(f"  Columns ({len(table['columns'])}):")
            
            # Prepare table data for tabulate
            table_data = []
            for col in table['columns']:
                pk_marker = "🔑" if col['isPrimaryKey'] else ""
                not_null_marker = "✓" if col['notNull'] else ""
                table_data.append([
                    col['name'],
                    col['type'],
                    not_null_marker,
                    pk_marker
                ])
            
            # Print columns table
            headers = ['Column Name', 'Type', 'Not Null', 'Primary Key']
            print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='left'))
            print()
    
    def analyze_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Analyze all TOML files in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            print(f"Directory {directory_path} does not exist.")
            return []
        
        results = []
        toml_files = list(directory.glob("*.toml"))
        
        if not toml_files:
            print(f"No TOML files found in {directory_path}")
            return []
        
        print(f"Found {len(toml_files)} TOML files:")
        for file_path in toml_files:
            print(f"  - {file_path.name}")
        print()
        
        for file_path in toml_files:
            analysis = self.analyze_file(str(file_path))
            if analysis:
                results.append(analysis)
        
        return results
    
    def print_summary(self, analyses: List[Dict[str, Any]]) -> None:
        """Print a summary of all analyses."""
        if not analyses:
            return
            
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        summary_data = []
        for analysis in analyses:
            summary_data.append([
                Path(analysis['file_path']).name,
                analysis['dataset_id'],
                analysis['catalog'],
                analysis['total_tables'],
                analysis['total_columns']
            ])
        
        headers = ['File', 'Dataset ID', 'Catalog', 'Tables', 'Columns']
        print(tabulate(summary_data, headers=headers, tablefmt='grid'))
        print()
        
        # Overall statistics
        total_files = len(analyses)
        total_tables = sum(a['total_tables'] for a in analyses)
        total_columns = sum(a['total_columns'] for a in analyses)
        
        print(f"Total Files Analyzed: {total_files}")
        print(f"Total Tables: {total_tables}")
        print(f"Total Columns: {total_columns}")
        print(f"Average Tables per Dataset: {total_tables / total_files:.1f}")
        print(f"Average Columns per Dataset: {total_columns / total_files:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze TOML dataset files to extract schema information")
    parser.add_argument(
        'path',
        help='Path to TOML file or directory containing TOML files'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Show only summary, skip detailed analysis'
    )
    
    args = parser.parse_args()
    
    analyzer = DatasetAnalyzer()
    path = Path(args.path)
    
    if path.is_file():
        # Analyze single file
        if path.suffix.lower() != '.toml':
            print(f"Error: {path} is not a TOML file")
            sys.exit(1)
        
        analysis = analyzer.analyze_file(str(path))
        if analysis:
            if not args.summary_only:
                analyzer.print_analysis(analysis)
            analyzer.print_summary([analysis])
    
    elif path.is_dir():
        # Analyze directory
        analyses = analyzer.analyze_directory(str(path))
        if analyses:
            if not args.summary_only:
                for analysis in analyses:
                    analyzer.print_analysis(analysis)
            analyzer.print_summary(analyses)
    
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)


if __name__ == "__main__":
    main()