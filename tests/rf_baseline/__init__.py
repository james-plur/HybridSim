"""Frozen rf_op costing, used only to compare against the live primitive DAG."""

from rf_baseline.rf_block_cost import rf_block_cost, sum_dag_blocks

__all__ = ["rf_block_cost", "sum_dag_blocks"]
