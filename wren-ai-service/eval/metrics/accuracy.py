import asyncio
import re
import traceback

import orjson
import pandas as pd
from deepeval.evaluate import TestResult
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deprecated import deprecated

from eval.utils import get_data_from_wren_engine, get_openai_client


class AccuracyMetric(BaseMetric):
    def __init__(self, engine_info: dict, enable_semantics_comparison: bool = False):
        self.threshold = 0
        self.score = 0
        self.engine_info = engine_info
        self.enable_semantics_comparison = enable_semantics_comparison
        if self.enable_semantics_comparison:
            self._openai_client = get_openai_client()

    def measure(self, test_case: LLMTestCase):
        return asyncio.run(self.a_measure(test_case))

    def _normalize_dataframes_for_comparison(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Normalize two dataframes for comparison by sorting values and resetting column names.
        This handles cases where column names are different but data is the same.
        """
        if len(df1.columns) != len(df2.columns):
            return df1, df2, False
            
        # Create copies to avoid modifying original dataframes
        df1_normalized = df1.copy()
        df2_normalized = df2.copy()
        
        # Convert all values to string for consistent sorting
        for col in df1_normalized.columns:
            df1_normalized[col] = df1_normalized[col].astype(str)
        for col in df2_normalized.columns:
            df2_normalized[col] = df2_normalized[col].astype(str)
        
        # Sort both dataframes by all columns (converted to string for comparison)
        df1_sorted = df1_normalized.sort_values(by=list(df1_normalized.columns)).reset_index(drop=True)
        df2_sorted = df2_normalized.sort_values(by=list(df2_normalized.columns)).reset_index(drop=True)
        
        # Assign generic column names
        generic_cols = [f"col_{i}" for i in range(len(df1.columns))]
        df1_sorted.columns = generic_cols
        df2_sorted.columns = generic_cols
        
        return df1_sorted, df2_sorted, True

    def _is_subset(self, expected: pd.DataFrame, actual: pd.DataFrame) -> bool:
        """
        Check if expected is a subset of actual (expected ⊆ actual).
        Returns True if all rows in expected can be found in actual.
        Handles cases where:
        1. Column names are different but data is the same
        2. Expected has fewer columns than actual (projection subset)
        """
        # If expected has more columns than actual, it can't be a subset
        if len(expected.columns) > len(actual.columns):
            return False
        
        # If same number of columns, use existing normalization logic
        if len(expected.columns) == len(actual.columns):
            expected_norm, actual_norm, can_compare = self._normalize_dataframes_for_comparison(expected, actual)
            
            if not can_compare:
                return False

            # Check if expected ⊆ actual: merge expected with actual
            merged = pd.merge(
                expected_norm,
                actual_norm,
                on=list(expected_norm.columns),
                how="left",
                indicator=True,
            )
            
            # All expected rows should be found in actual
            return all(merged["_merge"] == "both")
        
        # Case: expected has fewer columns than actual
        # Try to find if expected data exists as a projection of actual
        expected_values = expected.values.tolist()
        
        # For each combination of columns in actual that matches expected's column count
        from itertools import combinations
        actual_col_count = len(actual.columns)
        expected_col_count = len(expected.columns)
        
        for col_combo in combinations(range(actual_col_count), expected_col_count):
            # Extract the subset of columns
            actual_subset = actual.iloc[:, list(col_combo)]
            
            # Convert to comparable format
            actual_subset_values = actual_subset.values.tolist()
            
            # Check if all expected rows exist in this subset
            all_found = True
            for expected_row in expected_values:
                if expected_row not in actual_subset_values:
                    all_found = False
                    break
            
            if all_found:
                print(f"Found subset match with columns {[actual.columns[i] for i in col_combo]}")
                return True
        
        return False
    
    def _count_partial_matches(self, expected: pd.DataFrame, actual: pd.DataFrame) -> float:
        """
        Calculate partial match score based on column structure and data content.
        Handles semantic column name differences (e.g., payment_type vs 支付方式).
        Returns score based on data content match regardless of column names.
        """
        expected_col_count = len(expected.columns)
        actual_col_count = len(actual.columns)
        
        if expected_col_count == 0:
            return 0.0
            
        # Case 1: Same number of columns - try data-based matching
        if expected_col_count == actual_col_count:
            try:
                # Normalize and compare
                expected_norm, actual_norm, can_compare = self._normalize_dataframes_for_comparison(expected, actual)
                
                if can_compare:
                    # Check if data content matches (ignoring column names)
                    if expected_norm.equals(actual_norm):
                        print(f"Partial match (columns): {expected_col_count}/{expected_col_count} = 1.000 (data match)")
                        print(f"Expected columns: {list(expected.columns)}")
                        print(f"Actual columns: {list(actual.columns)}")
                        return 1.0
                    
                    # Check if expected is subset of actual (for cases like LIMIT queries)
                    merged = pd.merge(
                        expected_norm,
                        actual_norm,
                        on=list(expected_norm.columns),
                        how="left",
                        indicator=True,
                    )
                    if all(merged["_merge"] == "both"):
                        print(f"Partial match (columns): {expected_col_count}/{expected_col_count} = 1.000 (subset match)")
                        print(f"Expected columns: {list(expected.columns)}")
                        print(f"Actual columns: {list(actual.columns)}")
                        return 1.0
                    
            except Exception as e:
                print(f"Error in data-based matching: {e}")
                traceback.print_exc()
        
        # Case 2: Different number of columns or data doesn't match - use exact column name matching
        expected_cols = set(expected.columns)
        actual_cols = set(actual.columns)
        common_cols = expected_cols.intersection(actual_cols)
        
        if len(common_cols) == 0:
            print(f"Partial match (columns): 0/{expected_col_count} = 0.000")
            print(f"Expected columns: {expected_cols}")
            print(f"Actual columns: {actual_cols}")
            return 0.0
            
        # Calculate score based on common columns
        column_score = len(common_cols) / expected_col_count
        
        # Verify data in common columns
        try:
            expected_subset = expected[list(common_cols)].sort_values(by=list(common_cols)).reset_index(drop=True)
            actual_subset = actual[list(common_cols)].sort_values(by=list(common_cols)).reset_index(drop=True)
            
            if not expected_subset.equals(actual_subset):
                # Check if expected is subset of actual
                merged = pd.merge(
                    expected_subset,
                    actual_subset,
                    on=list(common_cols),
                    how="left",
                    indicator=True,
                )
                if not all(merged["_merge"] == "both"):
                    column_score *= 0.5  # Penalize for data mismatch
                    
        except Exception as e:
            print(f"Error comparing common columns data: {e}")
            column_score *= 0.5
        
        print(f"Partial match (columns): {len(common_cols)}/{expected_col_count} = {column_score:.3f}")
        print(f"Common columns: {common_cols}")
        print(f"Expected columns: {expected_cols}")
        print(f"Actual columns: {actual_cols}")
        
        return column_score

    def _rewrite_sql(self, sql: str) -> str:
        # Pattern to match double quotes after WHERE clause, including multiple occurrences
        pattern = r'(WHERE\s+.*?)(")(.+?)(")(.*)$'
        replacement = r"\1'\3'\5"

        # Apply the replacement repeatedly until no more changes
        new_sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE | re.DOTALL)
        while new_sql != sql:
            sql = new_sql
            new_sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE | re.DOTALL)

        return sql

    async def _retrieve_data(self, sql: str) -> pd.DataFrame:
        response = await get_data_from_wren_engine(sql=sql, **self.engine_info)

        df = pd.DataFrame(**response)
        # Don't sort columns here - preserve original order for better comparison
        return df

    async def _check_sql_semantics(self, expected_sql: str, actual_sql: str):
        _system_prompt = (
            "### TASK ### \n"
            + "You are a great data anlyst, please carefully check the semantics of two given SQLs if they are the same. \n"
            + "The output should be a JSON format with the following schema: \n"
            + "{ \n"
            + '   "reasoning": <REASONING_STRING> \n'
            + '   "same": <BOOL> \n'
            + "}"
        )

        _user_prompt = (
            "### QUESTION ### \n"
            + f"Expected SQL: {expected_sql} \n"
            + f"Actual SQL: {actual_sql} \n"
            + "\n"
            + "Please think step by step"
        )

        response = await self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _system_prompt},
                {"role": "user", "content": _user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        print(
            f"response of _check_sql_semantics: {response.choices[0].message.content}"
        )

        return 1 if orjson.loads(response.choices[0].message.content)["same"] else 0

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        try:
            enable_rewrite = test_case.additional_metadata.get("enable_rewrite", False)
            rewritten_expected_output = test_case.expected_output

            if enable_rewrite:
                rewritten_expected_output = self._rewrite_sql(test_case.expected_output)

            expected_dataset = await self._retrieve_data(rewritten_expected_output)
            actual_dataset = await self._retrieve_data(test_case.actual_output)
            
            if len(expected_dataset) == 0:
                self.success = False
                self.score = 0.0
                return self.score

            print(f"expected shape: {expected_dataset.shape}")
            print(f"actual shape: {actual_dataset.shape}")
            print(f"expected columns: {list(expected_dataset.columns)}")
            print(f"actual columns: {list(actual_dataset.columns)}")

            # 1. Check for exact match (with column name normalization)
            expected_norm, actual_norm, can_compare = self._normalize_dataframes_for_comparison(
                expected_dataset, actual_dataset
            )
            
            if can_compare and expected_norm.equals(actual_norm):
                print("Match type: Exact match (after normalization)")
                self.success = True
                self.score = 1.0
                return self.score

            # 2. Check for subset match (expected ⊆ actual)
            if self._is_subset(expected_dataset, actual_dataset):
                print("Match type: Subset match (expected ⊆ actual)")
                self.success = True
                self.score = 1.0
                return self.score

            # 3. Calculate partial match score
            print("Match type: Checking partial match")
            self.score = self._count_partial_matches(expected_dataset, actual_dataset)
            
            # 4. Use LLM to check SQL semantics if score is 0
            if self.score == 0 and self.enable_semantics_comparison:
                print(f"before _check_sql_semantics: {self.score}")
                print(f"expected sql: {rewritten_expected_output}")
                print(f"actual sql: {test_case.actual_output}")
                self.score = await self._check_sql_semantics(
                    rewritten_expected_output, test_case.actual_output
                )
                print(f"after _check_sql_semantics: {self.score}")
                
        except Exception as e:
            self.error = f"Error occurred while evaluating the metric: {e}"
            traceback.print_exc()
            self.score = 0.0

        # if didn't pass any of the above checks
        self.success = self.score > 0
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Accuracy(column-based)"


@deprecated(
    reason="We don't generate multiple candidates for Text to SQL task, so don't need this metric"
)
class AccuracyMultiCandidateMetric(BaseMetric):
    def __init__(self):
        self.threshold = 0
        self.score = 0
        self._questions = {}

    def collect(self, test_case: LLMTestCase, result: TestResult):
        for metric in result.metrics_data:
            if metric.name != "Accuracy(column-based)":
                continue

            # or 0 to avoid when metric.error is exist
            self._questions[test_case.input] = (
                self._questions.get(test_case.input, 0) or metric.score or 0
            )

    def measure(self):
        if not self._questions:
            return 0
        self.score = sum(self._questions.values()) / len(self._questions)
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Accuracy(question-based)"