from graph_numeric.core.attribute_graph import graph_from_csv_text
from graph_numeric.pipeline import run_operator_pipeline


def main() -> None:
    graph = graph_from_csv_text(
        """company_name,industry,year,revenue,net_profit
Alpha,TECH,2024,1200 million USD,120 million USD
Beta,TECH,2024,950 million USD,95 million USD
Gamma,FINANCE,2024,3200 million USD,320 million USD
"""
    )
    result = run_operator_pipeline("2024 年 TECH 行业 revenue 总和是多少", graph)
    assert result.status == "ok", result
    assert result.plan is not None and result.plan.operator == "SUM", result
    assert abs(float(result.answer) - 2150.0) < 1e-9, result
    print(f"ok answer={result.answer} operator={result.plan.operator}")


if __name__ == "__main__":
    main()
