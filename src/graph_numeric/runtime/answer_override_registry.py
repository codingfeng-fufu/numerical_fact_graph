from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MAX_OVERRIDE_ERROR_RATE = 0.30


@dataclass(frozen=True)
class AnswerOverrideCondition:
    condition_id: str
    justification: str
    hit_count: int
    error_count: int
    absorb_debt: bool
    absorb_path: str

    @property
    def error_rate(self) -> float:
        if self.hit_count <= 0:
            return 0.0
        return self.error_count / self.hit_count

    @property
    def enabled(self) -> bool:
        return self.error_rate <= MAX_OVERRIDE_ERROR_RATE

    def to_gate_payload(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "enabled": self.enabled,
            "hit_count": self.hit_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 6),
            "max_error_rate": MAX_OVERRIDE_ERROR_RATE,
            "justification": self.justification,
            "absorb_debt": self.absorb_debt,
            "absorb_path": self.absorb_path,
        }


ANSWER_OVERRIDE_CONDITION_REGISTRY: dict[str, AnswerOverrideCondition] = {
    "average_fallback_overrides_non_average_pipeline": AnswerOverrideCondition(
        condition_id="average_fallback_overrides_non_average_pipeline",
        justification="题面明确要求平均值，但 pipeline 输出非 AVG；当前由平均值模板临时接管。",
        hit_count=1,
        error_count=0,
        absorb_debt=True,
        absorb_path="把 AVG 多值槽位编译进主流水线，支持同题多年份/多列均值聚合。",
    ),
    "sentence_percent_change_overrides_pipeline_binding": AnswerOverrideCondition(
        condition_id="sentence_percent_change_overrides_pipeline_binding",
        justification="pipeline 和句级百分比变化模板同算子但数值显著冲突，已知失败集中在操作数绑定。",
        hit_count=1,
        error_count=0,
        absorb_debt=True,
        absorb_path="修复 PERCENT_CHANGE 的年份/分母绑定，优先使用同句显式数值对。",
    ),
    "tax_position_activity_overrides_zero_difference": AnswerOverrideCondition(
        condition_id="tax_position_activity_overrides_zero_difference",
        justification="税项活动题要求多行活动汇总，pipeline 误用零差值时由活动净变化模板接管。",
        hit_count=1,
        error_count=0,
        absorb_debt=True,
        absorb_path="实现多行活动汇总子图，支持 tax position activity 的 SUM/NET_CHANGE 主流水线求解。",
    ),
}


def override_condition_gate(condition_id: str) -> dict[str, Any]:
    condition = ANSWER_OVERRIDE_CONDITION_REGISTRY.get(condition_id)
    if condition is None:
        return {
            "condition_id": condition_id,
            "enabled": False,
            "hit_count": 0,
            "error_count": 0,
            "error_rate": 1.0,
            "max_error_rate": MAX_OVERRIDE_ERROR_RATE,
            "justification": "unregistered override condition",
            "absorb_debt": True,
            "absorb_path": "register this override before enabling it",
        }
    return condition.to_gate_payload()
