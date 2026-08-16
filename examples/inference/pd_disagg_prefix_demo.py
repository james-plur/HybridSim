#!/usr/bin/env python3
"""PD disaggregation demo (cluster_type=pd) with local prefix caching.

Flow (per request):
  Cluster → Prefill pool (stamp do_remote_decode)
    → local schedule / optional prefix hit
    → handoff (RequestHandoffMsg)
  → Decode pool (stamp do_remote_prefill + remote_replica_id=source P)
    → control-plane lookup (RTT, no Store hash match) + RDMA pull sim
    → decode → finish

Replicas are homogeneous: role is only implied by request stamps from the
ClusterManager. Store may be enabled alongside PD (orthogonal).
"""

from __future__ import annotations

from hybridsim_infer import (
    InferenceConfig,
    InferenceRequest,
    build_inference_simulation,
)


def main() -> None:
    block_size = 8
    shared_prefix = list(range(100, 100 + 16))  # 2 blocks
    suffix_a = list(range(200, 208))
    suffix_b = list(range(300, 308))

    prompt_full = shared_prefix + suffix_a  # 24 tokens
    prompt_reuse = list(prompt_full)  # identical → Prefill local prefix hit
    prompt_partial = shared_prefix + suffix_b  # shares first 16 tokens only

    cfg = InferenceConfig(
        cluster_type="pd",
        num_prefill_replicas=1,
        num_decode_replicas=1,
        enable_kv_client=True,
        enable_prefix_caching=True,
        model_preset="llama-3.1-8b",
        block_size=block_size,
        num_gpu_blocks=256,
        tokens_per_step=8,
        decode_tokens_per_step=1,
        max_num_scheduled_tokens=64,
        step_interval=1e-3,
        dummy_exec_s=0.01,
        kv_transfer_s=1e-4,
        kv_bandwidth_gbps=100.0,
        kv_lookup_rtt_s=1e-3,
        duration_mode="token_proportional",
        prefill_s_per_token=5e-5,
        decode_s_per_token=2e-4,
        enable_request_profile=True,
    )
    from hybridsim.request_profile import default_profile_dir

    cfg.request_profile_path = default_profile_dir() / "pd_disagg_prefix_demo.json"
    infra = build_inference_simulation(cfg)
    assert infra.kv_store is not None  # enable_kv_client wires shared Store
    assert len(infra.replicas) == 2

    requests = [
        InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=len(prompt_full),
            num_decode_tokens=4,
            prompt_token_ids=list(prompt_full),
        ),
        # Same prompt after req1 finished Prefill → Prefill local prefix reuse.
        InferenceRequest(
            request_id=2,
            arrived_at=0.15,
            num_prefill_tokens=len(prompt_reuse),
            num_decode_tokens=3,
            prompt_token_ids=list(prompt_reuse),
        ),
        # Partial shared prefix with req1 (first 16 tokens).
        InferenceRequest(
            request_id=3,
            arrived_at=0.30,
            num_prefill_tokens=len(prompt_partial),
            num_decode_tokens=2,
            prompt_token_ids=list(prompt_partial),
        ),
    ]

    print("=== PD disagg + prefix cache demo ===")
    print(
        f"cluster_type=pd P={cfg.num_prefill_replicas} D={cfg.num_decode_replicas} "
        f"block_size={block_size} prefix_caching=on store=on"
    )
    print(
        f"prompts: req1/2 full={len(prompt_full)} tok; "
        f"req3 partial share={len(shared_prefix)} tok"
    )

    infra.schedule_arrivals(requests)
    infra.run()
    infra.check_errors()

    finished = sorted(infra.finished_requests, key=lambda r: r.request_id)
    print(
        f"arrived={infra.cluster.arrived_count} finished={len(finished)} "
        f"now={infra.now:.4f}s"
    )

    for req in finished:
        params = req.kv_transfer_params or {}
        print(
            f"  req={req.request_id} computed={req.num_computed_tokens}/"
            f"{req.num_tokens_with_output} completed={req.completed} "
            f"xfer={params.get('transfer_id', '')} "
            f"do_remote_prefill={bool(params.get('do_remote_prefill'))} "
            f"src_P={params.get('remote_replica_id')} "
            f"handed_off={bool(params.get('_handed_off'))}"
        )
        if not req.completed:
            raise SystemExit(f"req {req.request_id} did not complete")
        expect = req.num_prefill_tokens + req.num_decode_tokens
        if req.num_computed_tokens != expect:
            raise SystemExit(
                f"req {req.request_id}: computed={req.num_computed_tokens} "
                f"expected={expect}"
            )

    # Prefill-side local prefix entries should retain shared prompts.
    prefill_kv = infra.replicas[0]._kv
    n_prefixes = len(getattr(prefill_kv, "_prefix_hash_chains", None) or [])
    if n_prefixes < 1:
        n_prefixes = len(getattr(prefill_kv, "_prefix_entries", []) or [])
    print(f"prefill replica local prefix entries={n_prefixes}")
    if n_prefixes < 1:
        raise SystemExit("expected Prefill replica to cache at least one prefix")

    if len(finished) != len(requests):
        raise SystemExit(f"expected {len(requests)} finished, got {len(finished)}")
    if infra.profile_path is not None:
        print(f"request profile: {infra.profile_path}")
    print("ok")


if __name__ == "__main__":
    main()
