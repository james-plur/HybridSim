"""Frontier architecture case definitions and CLI builders."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from frontier_bridge.config import frontier_root

FRONTIER_ROOT = frontier_root()
CASES = [
    ("co-location", "offline/dense_model_basic.sh"),
    ("co-location", "offline/moe_model_basic.sh"),
    ("co-location", "offline/thinking_mode_basic.sh"),
    ("co-location", "offline/moe_spec_dec.sh"),
    ("co-location", "offline/moe_prefix_caching.sh"),
    ("co-location", "online/dense_model_basic_online.sh"),
    ("co-location", "online/moe_model_basic_online.sh"),
    ("co-location", "online/thinking_mode_basic_online.sh"),
    ("co-location", "online/moe_spec_dec_online.sh"),
    ("co-location", "online/moe_prefix_caching_online.sh"),
    ("pdd", "offline/dense_model_basic.sh"),
    ("pdd", "offline/moe_model_basic.sh"),
    ("pdd", "offline/thinking_mode_basic.sh"),
    ("pdd", "offline/moe_spec_dec.sh"),
    ("pdd", "offline/moe_prefix_caching.sh"),
    ("pdd", "online/dense_model_basic_online.sh"),
    ("pdd", "online/moe_model_basic_online.sh"),
    ("pdd", "online/thinking_mode_basic_online.sh"),
    ("pdd", "online/moe_spec_dec_online.sh"),
    ("pdd", "online/moe_prefix_caching_online.sh"),
]


@dataclass(frozen=True)
class ArchitectureCase:
    arch: str
    script_rel: str

    @property
    def script_path(self) -> Path:
        return FRONTIER_ROOT / "examples" / "architecture" / self.arch / self.script_rel

    @property
    def case_id(self) -> str:
        return f"{self.arch}/{self.script_rel}"

    @property
    def is_online(self) -> bool:
        return self.script_rel.startswith("online/")

    @property
    def is_pdd(self) -> bool:
        return self.arch == "pdd"


def list_cases() -> list[ArchitectureCase]:
    return [ArchitectureCase(arch, script_rel) for arch, script_rel in CASES]


def parse_bash_defaults(script_path: Path) -> dict[str, str]:
    text = script_path.read_text(encoding="utf-8")
    defaults: dict[str, str] = {}
    for match in re.finditer(
        r'^([A-Z0-9_]+)="\$\{\1:-([^}]*)\}"', text, flags=re.MULTILINE
    ):
        defaults[match.group(1)] = match.group(2)

    repo_root = str(FRONTIER_ROOT)
    for key, value in list(defaults.items()):
        defaults[key] = value.replace("$REPO_ROOT", repo_root)
    return defaults


def _bool_flag(value: str, enable_arg: str, disable_arg: str) -> list[str]:
    return [enable_arg if value.lower() == "true" else disable_arg]


def build_cli_args(
    case: ArchitectureCase,
    *,
    metrics_output_dir: Path,
    enable_trace: bool = True,
) -> list[str]:
    env = parse_bash_defaults(case.script_path)
    run_id = env.get("RUN_ID", case.script_rel.replace(".sh", ""))
    args: list[str] = []

    if case.is_pdd:
        args.extend(
            [
                "--simulation_mode",
                "online" if case.is_online else "offline",
                "--sys_arch",
                env.get("SYS_ARCH", "pd-disaggregation"),
                "--no-enable_parallel_clusters",
                "--cluster_config_prefill_cluster_num_replicas",
                env.get("PREFILL_REPLICAS", "1"),
                "--cluster_config_decode_cluster_num_replicas",
                env.get("DECODE_REPLICAS", "1"),
                "--cluster_config_prefill_replica_config_num_pipeline_stages",
                env.get("PREFILL_PP", "1"),
                "--cluster_config_prefill_replica_config_attn_tensor_parallel_size",
                env.get("PREFILL_ATTN_TP", "1"),
                "--cluster_config_prefill_replica_config_attn_data_parallel_size",
                env.get("PREFILL_ATTN_DP", "1"),
                "--cluster_config_prefill_replica_config_moe_tensor_parallel_size",
                env.get("PREFILL_MOE_TP", "1"),
                "--cluster_config_prefill_replica_config_moe_expert_parallel_size",
                env.get("PREFILL_MOE_EP", "1"),
                "--cluster_config_prefill_replica_config_device",
                env.get("PREFILL_DEVICE", "a800"),
                "--cluster_config_decode_replica_config_num_pipeline_stages",
                env.get("DECODE_PP", "1"),
                "--cluster_config_decode_replica_config_attn_tensor_parallel_size",
                env.get("DECODE_ATTN_TP", "1"),
                "--cluster_config_decode_replica_config_attn_data_parallel_size",
                env.get("DECODE_ATTN_DP", "1"),
                "--cluster_config_decode_replica_config_moe_tensor_parallel_size",
                env.get("DECODE_MOE_TP", "1"),
                "--cluster_config_decode_replica_config_moe_expert_parallel_size",
                env.get("DECODE_MOE_EP", "1"),
                "--cluster_config_decode_replica_config_device",
                env.get("DECODE_DEVICE", "a800"),
                "--cc_backend_config_type",
                "analytical",
                "--replica_config_model_name",
                env.get("MODEL_NAME", "meta-llama/Llama-2-7b-hf"),
                "--replica_config_total_expert_num",
                env.get("TOTAL_EXPERTS", "1"),
                "--replica_config_router_topk",
                env.get("ROUTER_TOPK", "1"),
                "--replica_config_moe_routing_mode",
                env.get("MOE_ROUTING_MODE", "simulation"),
                "--replica_config_moe_routing_seed",
                env.get("MOE_ROUTING_SEED", "42"),
                "--analytical_kv_cache_transfer_config_network_bandwidth_gbps",
                env.get("KV_TRANSFER_BANDWIDTH_GBPS", "200.0"),
                "--analytical_kv_cache_transfer_config_network_latency_ms",
                env.get("KV_TRANSFER_LATENCY_MS", "0.5"),
            ]
        )
    else:
        args.extend(
            [
                "--simulation_mode",
                "online" if case.is_online else "offline",
                "--sys_arch",
                env.get("SYS_ARCH", "co-location"),
                "--cc_backend_config_type",
                env.get("CC_BACKEND", "analytical"),
                "--cluster_config_num_replicas",
                env.get("NUM_REPLICAS", "1"),
                "--replica_config_device",
                env.get("DEVICE", "a100"),
                "--replica_config_model_name",
                env.get("MODEL_NAME", "meta-llama/Llama-2-7b-hf"),
                "--replica_config_attn_tensor_parallel_size",
                env.get("ATTN_TP", "1"),
                "--replica_config_num_pipeline_stages",
                env.get("PP", "1"),
                "--replica_config_attn_data_parallel_size",
                env.get("DP", "1"),
            ]
        )
        if "MOE_TP" in env:
            args.extend(
                [
                    "--replica_config_moe_tensor_parallel_size",
                    env.get("MOE_TP", "1"),
                    "--replica_config_moe_expert_parallel_size",
                    env.get("MOE_EP", "1"),
                    "--replica_config_total_expert_num",
                    env.get("TOTAL_EXPERTS", "1"),
                    "--replica_config_router_topk",
                    env.get("ROUTER_TOPK", "1"),
                    "--replica_config_moe_routing_mode",
                    env.get("MOE_ROUTING_MODE", "simulation"),
                    "--replica_config_moe_routing_seed",
                    env.get("MOE_ROUTING_SEED", "42"),
                ]
            )

    args.extend(
        [
            "--replica_scheduler_config_type",
            env.get("REPLICA_SCHEDULER", "vllm_v1"),
            "--decode_cuda_graph_mode",
            env.get("DECODE_CUDA_GRAPH_MODE", "none"),
            "--vllm_v1_scheduler_config_max_tokens_in_batch",
            env.get("MAX_TOKENS_IN_BATCH", "1024"),
            "--vllm_v1_scheduler_config_long_prefill_token_threshold",
            env.get("LONG_PREFILL_TOKEN_THRESHOLD", "64"),
        ]
    )

    if "BLOCK_SIZE" in env:
        args.extend(
            [
                "--vllm_v1_scheduler_config_block_size",
                env["BLOCK_SIZE"],
                "--vllm_v1_scheduler_config_num_blocks",
                env.get("NUM_BLOCKS", "128"),
            ]
        )

    if "thinking_mode" in run_id:
        args.extend(
            [
                "--enable_thinking_mode",
                "--thinking_depth",
                env.get("THINKING_DEPTH", "2"),
                "--tool_call_latency",
                env.get("TOOL_CALL_LATENCY", "0.001"),
                "--thinking_round_prefill_tokens",
                env.get("THINKING_ROUND_PREFILL_TOKENS", "3"),
                "--thinking_round_decode_tokens",
                env.get("THINKING_ROUND_DECODE_TOKENS", "1"),
            ]
        )

    if "prefix_caching" in run_id:
        trace_file = env.get(
            "TRACE_FILE",
            str(FRONTIER_ROOT / "examples/fixtures/prefix_cache_shared_session_trace.csv"),
        )
        args.extend(
            [
                "--cluster_scheduler_config_type",
                "sticky_round_robin",
                "--request_generator_config_type",
                "trace_replay",
                "--trace_request_generator_config_trace_file",
                trace_file,
                "--trace_request_generator_config_max_tokens",
                env.get("MAX_TOKENS", "128"),
            ]
        )
    else:
        args.extend(
            [
                "--request_generator_config_type",
                "synthetic",
                "--synthetic_request_generator_config_num_requests",
                env.get("NUM_REQUESTS", "8"),
                "--length_generator_config_type",
                "fixed",
                "--fixed_request_length_generator_config_prefill_tokens",
                env.get("PREFILL_TOKENS", "512"),
                "--fixed_request_length_generator_config_decode_tokens",
                env.get("DECODE_TOKENS", "64"),
                "--interval_generator_config_type",
                "poisson",
                "--poisson_request_interval_generator_config_qps",
                env.get("QPS", "1.0"),
            ]
        )

    if "spec_dec" in run_id:
        args.extend(
            [
                "--speculative_decoding_config_enabled",
                "--speculative_decoding_config_method",
                env.get("SPEC_METHOD", "ngram"),
                "--speculative_decoding_config_spec_model_name",
                env.get("SPEC_MODEL_NAME", ""),
                "--speculative_decoding_config_num_speculative_tokens",
                env.get("NUM_SPECULATIVE_TOKENS", "2"),
                "--speculative_decoding_config_committed_tokens_per_iteration",
                env.get("COMMITTED_TOKENS_PER_ITERATION", "2"),
                "--speculative_decoding_config_proposer_overhead_ms_by_method",
                env.get(
                    "PROPOSER_OVERHEAD_MS_BY_METHOD",
                    '{"ngram":0.0,"qwen3_next_mtp":0.0,"deepseek_mtp":0.0}',
                ),
            ]
        )
        if env.get("MTP_N_PREDICT", "0") != "0":
            args.extend(
                [
                    "--speculative_decoding_config_mtp_n_predict",
                    env["MTP_N_PREDICT"],
                    "--speculative_decoding_config_mtp_num_layers",
                    env.get("MTP_NUM_LAYERS", "1"),
                ]
            )

    args.extend(
        _bool_flag(
            env.get("ENABLE_CHUNKED_PREFILL", "true"),
            "--vllm_v1_scheduler_config_enable_chunked_prefill",
            "--no-vllm_v1_scheduler_config_enable_chunked_prefill",
        )
    )

    if env.get("ENABLE_DUMMY_MODE", "true").lower() == "true":
        args.extend(
            [
                "--random_forrest_execution_time_predictor_config_enable_dummy_mode",
                "--random_forrest_execution_time_predictor_config_dummy_execution_time_ms",
                env.get("DUMMY_EXEC_TIME_MS", "1.0"),
            ]
        )

    if "prefix_caching" in run_id:
        args.extend(
            [
                "--vllm_v1_scheduler_config_enable_prefix_caching",
            ]
        )

    args.extend(
        [
            "--metrics_config_output_dir",
            str(metrics_output_dir),
            "--metrics_config_run_id",
            run_id,
            "--metrics_config_write_metrics",
            "--metrics_config_store_request_metrics",
            "--metrics_config_store_batch_metrics",
            "--metrics_config_store_token_completion_metrics",
            "--metrics_config_store_utilization_metrics",
            "--no-metrics_config_store_plots",
        ]
    )

    if enable_trace:
        args.extend(
            [
                "--metrics_config_enable_chrome_trace",
                "--metrics_config_write_json_trace",
            ]
        )
    else:
        args.extend(
            [
                "--no-metrics_config_enable_chrome_trace",
                "--no-metrics_config_write_json_trace",
            ]
        )

    return args
