import logging
import sys
from typing import Any

from hamilton import base
from hamilton.async_driver import AsyncDriver
from haystack.components.builders.prompt_builder import PromptBuilder
from langfuse.decorators import observe

from src.core.engine import Engine
from src.core.pipeline import BasicPipeline
from src.core.provider import DocumentStoreProvider, LLMProvider
from src.pipelines.common import retrieve_metadata
from src.pipelines.generation.utils.sql import (
    SQL_GENERATION_MODEL_KWARGS,
    SQLGenPostProcessor,
    calculated_field_instructions,
    construct_instructions,
    metric_instructions,
    sql_generation_system_prompt,
)
from src.pipelines.retrieval.sql_functions import SqlFunction
from src.utils import trace_cost
from src.web.v1.services import Configuration

logger = logging.getLogger("wren-ai-service")


sql_generation_user_prompt_template = """
### DATABASE SCHEMA ###
{% for document in documents %}
    {{ document }}
{% endfor %}

{% if calculated_field_instructions %}
{{ calculated_field_instructions }}
{% endif %}

{% if metric_instructions %}
{{ metric_instructions }}
{% endif %}

{% if sql_functions %}
### SQL FUNCTIONS ###
{% for function in sql_functions %}
{{ function }}
{% endfor %}
{% endif %}

{% if sql_samples %}
### SQL SAMPLES ###
{% for sample in sql_samples %}
Question:
{{sample.question}}
SQL:
{{sample.sql}}
{% endfor %}
{% endif %}

{% if instructions %}
### USER INSTRUCTIONS ###
{% for instruction in instructions %}
{{ loop.index }}. {{ instruction }}
{% endfor %}
{% endif %}

### QUESTION ###
User's Question: {{ query }}

### CORE PRINCIPLES ###
1. **SIMPLICITY FIRST**: Always choose the simplest query structure that answers the question correctly
2. **EXACT REQUIREMENTS**: Return only what the question asks for - no extra columns or information
3. **VERIFY LOGIC**: Ensure the query logic directly matches the question's intent

### CONSTRAINTS ###
- Always use **table aliases** (e.g., `FROM TABLE1 AS T1`).
- All columns in the `SELECT` clause **must either be**:
  1. included in the `GROUP BY` clause,
  2. or wrapped with an aggregate function like `COUNT`, `MAX`, `MIN`, or `ANY_VALUE`.
- Do not use columns in `ORDER BY` or `SELECT` that are not grouped or aggregated.
- Use `ANY_VALUE(column)` when the exact value is not important in `GROUP BY` queries.
- Do not use double quotes around alias names. Use `AS 別名` for aliasing.
- The SQL must be **syntactically correct** and compatible with SQLite or DuckDB.
- Avoid ambiguous column references. Always use `T1.column_name` or `T2.column_name`.

### ADVANCED RULES ###
- In any SQL query with `GROUP BY`, all columns in the `SELECT` clause must appear in the `GROUP BY` clause or be wrapped in aggregate functions.
- This rule applies independently to each SELECT in compound queries (e.g., `UNION`, `INTERSECT`, `EXCEPT`).
- For compound conditions involving different grouping levels, **split the logic into multiple subqueries**, and join on the correct key level (e.g., country-level vs manufacturer-level).
- If filtering based on substrings like brand or model (e.g., `fiat`), always cast to lowercase and use `LIKE` with wildcards: `LOWER(column) LIKE '%fiat%'`.
- When joining across multiple tables, use explicit join chains and table aliases to avoid incorrect mappings.

### QUERY OPTIMIZATION GUIDELINES ###
1. **Avoid Over-Engineering**:
   - Don't use CTEs unless absolutely necessary for complex logic
   - Prefer simple WHERE conditions over complex subqueries when possible
   - Use direct aggregation instead of window functions when applicable

2. **JOIN Strategy**:
   - Verify each JOIN relationship matches the actual foreign key constraints
   - Use the minimum number of JOINs required to answer the question
   - When in doubt about table relationships, use the most direct path

3. **Question Interpretation**:
   - If asking for "minimum value", return the value (not the record)
   - If asking for "which car", return identifying information (model/name)
   - If asking for "count", return only the count number
   - If asking for "countries without X", use NOT EXISTS or LEFT JOIN ... WHERE IS NULL

4. **Common Patterns**:
   - For "最小/最大值" questions: Use MIN()/MAX() directly
   - For "哪個/哪些" questions: Use ORDER BY ... LIMIT or appropriate filtering
   - For "多少個" questions: Use COUNT() with appropriate grouping
   - For "沒有X的Y" questions: Use exclusion patterns (NOT IN, NOT EXISTS, LEFT JOIN with NULL check)

### VALIDATION CHECKLIST ###
Before finalizing the query, verify:
- [ ] Does the query answer the exact question asked?
- [ ] Are all JOINs using correct foreign key relationships?
- [ ] Is this the simplest possible query structure?
- [ ] Are all GROUP BY rules followed correctly?
- [ ] Will this query actually execute without syntax errors?

{% if sql_generation_reasoning %}
### REASONING PLAN ###
{{ sql_generation_reasoning }}
{% endif %}

Let's think step by step.
"""


## Start of Pipeline
@observe(capture_input=False)
def prompt(
    query: str,
    documents: list[str],
    prompt_builder: PromptBuilder,
    sql_generation_reasoning: str | None = None,
    configuration: Configuration | None = None,
    sql_samples: list[dict] | None = None,
    instructions: list[dict] | None = None,
    has_calculated_field: bool = False,
    has_metric: bool = False,
    sql_functions: list[SqlFunction] | None = None,
) -> dict:
    return prompt_builder.run(
        query=query,
        documents=documents,
        sql_generation_reasoning=sql_generation_reasoning,
        instructions=construct_instructions(
            instructions=instructions,
        ),
        calculated_field_instructions=calculated_field_instructions
        if has_calculated_field
        else "",
        metric_instructions=metric_instructions if has_metric else "",
        sql_samples=sql_samples,
        sql_functions=sql_functions,
    )


@observe(as_type="generation", capture_input=False)
@trace_cost
async def generate_sql(
    prompt: dict,
    generator: Any,
    generator_name: str,
) -> dict:
    return await generator(prompt=prompt.get("prompt")), generator_name


@observe(capture_input=False)
async def post_process(
    generate_sql: dict,
    post_processor: SQLGenPostProcessor,
    engine_timeout: float,
    data_source: str,
    project_id: str | None = None,
    use_dry_plan: bool = False,
    allow_dry_plan_fallback: bool = True,
) -> dict:
    return await post_processor.run(
        generate_sql.get("replies"),
        timeout=engine_timeout,
        project_id=project_id,
        use_dry_plan=use_dry_plan,
        data_source=data_source,
        allow_dry_plan_fallback=allow_dry_plan_fallback,
    )


## End of Pipeline


class SQLGeneration(BasicPipeline):
    def __init__(
        self,
        llm_provider: LLMProvider,
        document_store_provider: DocumentStoreProvider,
        engine: Engine,
        engine_timeout: float = 30.0,
        **kwargs,
    ):
        self._retriever = document_store_provider.get_retriever(
            document_store_provider.get_store("project_meta")
        )

        self._components = {
            "generator": llm_provider.get_generator(
                system_prompt=sql_generation_system_prompt,
                generation_kwargs=SQL_GENERATION_MODEL_KWARGS,
            ),
            "generator_name": llm_provider.get_model(),
            "prompt_builder": PromptBuilder(
                template=sql_generation_user_prompt_template
            ),
            "post_processor": SQLGenPostProcessor(engine=engine),
        }

        self._configs = {
            "engine_timeout": engine_timeout,
        }

        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    @observe(name="SQL Generation")
    async def run(
        self,
        query: str,
        contexts: list[str],
        sql_generation_reasoning: str | None = None,
        configuration: Configuration = Configuration(),
        sql_samples: list[dict] | None = None,
        instructions: list[dict] | None = None,
        project_id: str | None = None,
        has_calculated_field: bool = False,
        has_metric: bool = False,
        sql_functions: list[SqlFunction] | None = None,
        use_dry_plan: bool = False,
        allow_dry_plan_fallback: bool = True,
    ):
        logger.info("SQL Generation pipeline is running...")

        if use_dry_plan:
            metadata = await retrieve_metadata(project_id or "", self._retriever)
        else:
            metadata = {}

        return await self._pipe.execute(
            ["post_process"],
            inputs={
                "query": query,
                "documents": contexts,
                "sql_generation_reasoning": sql_generation_reasoning,
                "sql_samples": sql_samples,
                "instructions": instructions,
                "project_id": project_id,
                "configuration": configuration,
                "has_calculated_field": has_calculated_field,
                "has_metric": has_metric,
                "sql_functions": sql_functions,
                "use_dry_plan": use_dry_plan,
                "allow_dry_plan_fallback": allow_dry_plan_fallback,
                "data_source": metadata.get("data_source", "local_file"),
                **self._components,
                **self._configs,
            },
        )


if __name__ == "__main__":
    from src.pipelines.common import dry_run_pipeline

    dry_run_pipeline(
        SQLGeneration,
        "sql_generation",
        query="this is a test query",
        contexts=[],
    )
