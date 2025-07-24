import asyncio
import os
from typing import Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from eval.metrics.spider.postgres_eval import eval_exec_match


class PostgreSQLExecutionAccuracy(BaseMetric):
    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_user: str = "postgres",
        db_password: str = "postgres",
        db_name: Optional[str] = None,
    ):
        self.threshold = 0
        self.score = 0
        
        self.db_host = db_host
        self.db_port = db_port
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name

    def measure(self, test_case: LLMTestCase):
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        # Use database name from metadata if available, otherwise use configured db_name
        db_name = test_case.additional_metadata.get("catalog") or self.db_name
        if not db_name:
            raise ValueError("Database name must be provided either in constructor or test case metadata")

        db_config = {
            "host": self.db_host,
            "port": self.db_port,
            "user": self.db_user,
            "password": self.db_password,
            "database": db_name,
        }

        self.score = await eval_exec_match(
            db_config=db_config,
            p_str=test_case.actual_output,
            g_str=test_case.expected_output,
        )

        self.success = self.score >= self.threshold

        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "PostgreSQLExecutionAccuracy"


if __name__ == "__main__":
    # Example usage
    metric = PostgreSQLExecutionAccuracy(
        db_host="localhost",
        db_port=5432,
        db_user="postgres",
        db_password="postgres",
        db_name="ecommerce"
    )
    
    test_case = LLMTestCase(
        input="",
        expected_output="SELECT COUNT(*) FROM olist_customers_dataset;",
        actual_output="SELECT COUNT(*) FROM olist_customers_dataset;",
        additional_metadata={"catalog": "ecommerce"},
    )
    
    print(metric.measure(test_case))