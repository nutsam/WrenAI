import asyncio
import itertools
import os
import random
import re
from collections import defaultdict
from itertools import chain, product
from typing import Any, Iterator, List, Set, Tuple, Dict
import psycopg2
import sqlparse
import tqdm

from eval.metrics.spider.process_sql import get_sql

# Flag to disable value evaluation
DISABLE_VALUE = True
# Flag to disable distinct in select evaluation
DISABLE_DISTINCT = True

TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}

WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)

# Import evaluation functions from the original module
from eval.metrics.spider import (
    get_scores, eval_sel, eval_where, eval_group, eval_having, eval_order,
    eval_and_or, get_nestedSQL, eval_nested, eval_IUEN, get_keywords,
    eval_keywords, Evaluator, rebuild_col_unit_col, rebuild_val_unit_col,
    rebuild_table_unit_col, rebuild_cond_unit_col, rebuild_condition_col,
    rebuild_select_col, rebuild_from_col, rebuild_group_by_col, rebuild_order_by_col,
    rebuild_sql_col, rebuild_cond_unit_val, rebuild_condition_val, rebuild_sql_val,
    build_valid_col_units, rewrite_sql, tokenize, build_foreign_key_map,
    build_foreign_key_map_from_json, VALUE_NUM_SYMBOL, plugin, plugin_all_permutations,
    strip_query, reformat_query, replace_values, extract_query_values,
    get_all_preds_for_execution, remove_distinct, postprocess, replace_cur_year,
    permute_tuple, unorder_row, quick_rej, get_constraint_permutation,
    multiset_eq, result_eq
)

TIMEOUT = 60

# PostgreSQL connection pool (unused with psycopg2 approach)
_connection_pools = {}

def exec_on_db_sync(db_config: Dict[str, Any], query: str) -> Tuple[str, Any]:
    """Execute a query on PostgreSQL database synchronously."""
    query = replace_cur_year(query)
    
    connection = None
    cursor = None
    try:
        # Use psycopg2 for more stable connection
        connection = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        cursor = connection.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        return "result", result
    except Exception as e:
        return "exception", e
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

async def exec_on_db_(db_config: Dict[str, Any], query: str) -> Tuple[str, Any]:
    """Execute a query on PostgreSQL database."""
    # Run synchronous function in thread pool to maintain async interface
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, exec_on_db_sync, db_config, query)

async def exec_on_db(
    db_config: Dict[str, Any], query: str, process_id: str = "", timeout: int = TIMEOUT
) -> Tuple[str, Any]:
    """Execute a query on PostgreSQL database with timeout."""
    try:
        return await asyncio.wait_for(exec_on_db_(db_config, query), timeout)
    except asyncio.TimeoutError:
        return ("exception", TimeoutError)
    except Exception as e:
        return ("exception", e)

async def get_database_schemas(db_config: Dict[str, Any]) -> List[str]:
    """Get list of available schemas in the PostgreSQL database."""
    query = """
    SELECT schema_name 
    FROM information_schema.schemata 
    WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
    """
    
    try:
        flag, result = await exec_on_db_(db_config, query)
        if flag == "result":
            return [row[0] for row in result]
        else:
            raise result
    except Exception as e:
        print(f"Error getting schemas: {e}")
        return ['public']  # Default to public schema

async def get_databases_from_server(base_db_config: Dict[str, Any]) -> List[str]:
    """Get list of databases from PostgreSQL server."""
    query = """
    SELECT datname FROM pg_database 
    WHERE datistemplate = false AND datallowconn = true
    """
    
    try:
        # Connect to postgres database to get list of databases
        server_config = base_db_config.copy()
        server_config['database'] = 'postgres'
        
        flag, result = await exec_on_db_(server_config, query)
        if flag == "result":
            return [row[0] for row in result]
        else:
            raise result
    except Exception as e:
        print(f"Error getting databases: {e}")
        return [base_db_config['database']]  # Return the original database

async def eval_exec_match(
    db_config: Dict[str, Any],
    p_str: str,
    g_str: str,
    plug_value: bool = False,
    keep_distinct: bool = False,
    progress_bar_for_each_datapoint: bool = False,
    use_multiple_databases: bool = False,
) -> int:
    """
    Evaluate execution match between predicted and gold SQL queries on PostgreSQL.
    
    Args:
        db_config: Dictionary containing PostgreSQL connection parameters
        p_str: Predicted SQL query
        g_str: Gold/expected SQL query
        plug_value: Whether to plug in values from gold query
        keep_distinct: Whether to keep DISTINCT in queries
        progress_bar_for_each_datapoint: Whether to show progress bar
        use_multiple_databases: Whether to test on multiple databases
    
    Returns:
        1 if queries are equivalent, 0 otherwise
    """
    # Post-process the prediction
    p_str, g_str = postprocess(p_str), postprocess(g_str)
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)

    # Check if order matters
    order_matters = "order by" in g_str.lower()

    # Determine which databases to test on
    if use_multiple_databases:
        try:
            db_names = await get_databases_from_server(db_config)
        except Exception as e:
            print(f"Could not get multiple databases, using single database: {e}")
            db_names = [db_config['database']]
    else:
        db_names = [db_config['database']]

    preds = [p_str]
    # If plug in value, enumerate all ways to plug in values
    if plug_value:
        _, preds = get_all_preds_for_execution(g_str, p_str)
        preds = chain([p_str], preds)

    for pred in preds:
        pred_passes = 1
        
        # Test on each database
        if progress_bar_for_each_datapoint:
            ranger = tqdm.tqdm(db_names)
        else:
            ranger = db_names

        for db_name in ranger:
            # Create config for this specific database
            current_db_config = db_config.copy()
            current_db_config['database'] = db_name
            
            try:
                g_flag, g_denotation = await exec_on_db(current_db_config, g_str)
                p_flag, p_denotation = await exec_on_db(current_db_config, pred)
                
                # Gold query should execute successfully
                if g_flag == "exception":
                    print(f"Warning: Gold query failed on database {db_name}: {g_denotation}")
                    continue

                # If prediction fails, mark as wrong
                if p_flag == "exception":
                    pred_passes = 0
                # If denotations are not equivalent, mark as wrong
                elif not result_eq(g_denotation, p_denotation, order_matters=order_matters):
                    pred_passes = 0
                
                if pred_passes == 0:
                    break
                    
            except Exception as e:
                print(f"Error testing on database {db_name}: {e}")
                continue

        # If this prediction passed all databases, return success
        if pred_passes == 1:
            return 1

    # None of the predictions passed
    return 0

async def close_all_pools():
    """Close all connection pools (not used with psycopg2)."""
    _connection_pools.clear()

# Utility function for testing
async def test_connection(db_config: Dict[str, Any]) -> bool:
    """Test PostgreSQL connection."""
    try:
        flag, result = await exec_on_db_(db_config, "SELECT 1")
        return flag == "result"
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False