import asyncio

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from haystack.components.builders.prompt_builder import PromptBuilder
from pydantic import BaseModel

from src.providers import LLMProvider


class EvalResult(BaseModel):
    score: int  # Changed from float to int
    reason: str


_MODEL_KWARGS = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "eval_result",
            "schema": EvalResult.model_json_schema(),
        },
    }
}


def format(response: dict) -> EvalResult:
    reply = response.get("replies", [])[0]
    return EvalResult.model_validate_json(reply)


class QuestionToReasoningJudge(BaseMetric):
    _system_prompt = """
    You are an expert evaluator. Your task is to analyze the reasoning provided for a given question and determine if it makes sense. 
    Provide a score of either 0 or 1 and a detailed explanation for your evaluation.
    
    Scoring:
    - Score 1: The reasoning is logical, coherent, and appropriately addresses the question
    - Score 0: The reasoning is illogical, incoherent, or does not appropriately address the question
    """
    _test_case_prompt = """
    Question: 
    {{ question }}
    
    Reasoning:
    {{ reasoning }}
    """

    def __init__(self, llm_provider: LLMProvider, **_):
        self.threshold = 1  # Changed threshold to 1 for binary scoring
        self.score = 0
        self.llm_provider = llm_provider
        self.llm = llm_provider.get_generator(
            system_prompt=self._system_prompt,
            generation_kwargs=_MODEL_KWARGS,
        )
        self.prompt_builder = PromptBuilder(template=self._test_case_prompt)

    def measure(self, test_case: LLMTestCase):
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        prompt = self.prompt_builder.run(
            question=test_case.input,
            reasoning=test_case.reasoning,
        )
        response = await self.llm(prompt.get("prompt"))
        result = format(response)

        self.score = result.score
        self.reason = result.reason

        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "QuestionToReasoningJudge"


class ReasoningToSqlJudge(BaseMetric):
    _system_prompt = """
    You are an expert evaluator. Your task is to analyze the reasoning provided for a given SQL query and determine if it makes sense. 
    Provide a score of either 0 or 1 and a detailed explanation for your evaluation.
    
    Scoring:
    - Score 1: The reasoning correctly explains the SQL query logic and is coherent
    - Score 0: The reasoning does not correctly explain the SQL query logic or is incoherent
    """
    _test_case_prompt = """
    Actual Output: 
    {{ actual_output }}

    Reasoning:
    {{ reasoning }}
    """

    def __init__(self, llm_provider: LLMProvider, **_):
        self.threshold = 1  # Changed threshold to 1 for binary scoring
        self.score = 0
        self.llm_provider = llm_provider
        self.llm = llm_provider.get_generator(
            system_prompt=self._system_prompt,
            generation_kwargs=_MODEL_KWARGS,
        )
        self.prompt_builder = PromptBuilder(template=self._test_case_prompt)

    def measure(self, test_case: LLMTestCase):
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        prompt = self.prompt_builder.run(
            actual_output=test_case.actual_output,
            reasoning=test_case.reasoning,
        )
        response = await self.llm(prompt.get("prompt"))
        result = format(response)

        self.score = result.score
        self.reason = result.reason

        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "ReasoningToSqlJudge"


class SqlSemanticsJudge(BaseMetric):
    _system_prompt = """
    You are an expert evaluator. Your task is to analyze the actual SQL query and the expected SQL query and determine if they are semantically equivalent. 
    Provide a score of either 0 or 1 and a detailed explanation for your evaluation.
    
    Scoring:
    - Score 1: The queries are semantically equivalent (produce the same result set)
    - Score 0: The queries are NOT semantically equivalent (produce different result sets)
    
    Evaluation Criteria for Score 1:
    - Identical output: Same columns, same rows, same data types
    - Logical equivalence: Different syntax but same logical meaning
    - Order independence: Different ORDER BY clauses or missing ORDER BY (unless order is specifically required)
    - Alias differences: Different column or table aliases that don't affect functionality
    - Whitespace/formatting: Different formatting, spacing, or capitalization
    - Equivalent expressions: Different but mathematically/logically equivalent conditions
    - Join variations: Different join syntax producing same results
    
    Evaluation Criteria for Score 0:
    - Different result sets: Produces different rows or columns
    - Missing/extra columns: Different SELECT clause columns
    - Different filtering: Different WHERE conditions affecting results
    - Different aggregations: Different GROUP BY, HAVING, or aggregate functions
    - Different data sources: Different tables or joins
    - Syntax errors: Query contains syntax errors preventing execution
    - Different business logic: Fundamentally different approach to solving the problem
    """
    _test_case_prompt = """
    Actual SQL: 
    {{ actual_sql }}

    Expected SQL: 
    {{ expected_sql }}
    """

    def __init__(self, llm_provider: LLMProvider, **_):
        self.threshold = 1  # Changed threshold to 1 for binary scoring
        self.score = 0
        self.llm_provider = llm_provider
        self.llm = llm_provider.get_generator(
            system_prompt=self._system_prompt,
            generation_kwargs=_MODEL_KWARGS,
        )
        self.prompt_builder = PromptBuilder(template=self._test_case_prompt)

    def measure(self, test_case: LLMTestCase):
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        prompt = self.prompt_builder.run(
            actual_sql=test_case.actual_output,
            expected_sql=test_case.expected_output,
        )
        response = await self.llm(prompt.get("prompt"))
        result = format(response)

        self.score = result.score
        self.reason = result.reason

        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "SqlSemanticsJudge"