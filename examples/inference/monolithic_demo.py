#!/usr/bin/env python3
"""NO_NETWORK monolithic inference demo: request → step → workload → finish."""

from __future__ import annotations

from hybridsim_infer import (
    InferenceConfig,
    InferenceRequest,
    build_inference_simulation,
)


def main() -> None:
    cfg = InferenceConfig(
        num_replicas=1,
        step_interval=1e-3,
        dummy_exec_s=0.02,
        tokens_per_step=8,
    )
    infra = build_inference_simulation(cfg)

    requests = [
        InferenceRequest(
            request_id=1,
            arrived_at=0.0,
            num_prefill_tokens=16,
            num_decode_tokens=8,
        ),
        InferenceRequest(
            request_id=2,
            arrived_at=0.01,
            num_prefill_tokens=8,
            num_decode_tokens=4,
        ),
        InferenceRequest(
            request_id=3,
            arrived_at=0.02,
            num_prefill_tokens=4,
            num_decode_tokens=4,
        ),
    ]
    infra.schedule_arrivals(requests)
    infra.run()
    infra.check_errors()

    finished = infra.finished_requests
    print(f"arrived={infra.cluster.arrived_count} finished={len(finished)} now={infra.now:.4f}")
    for req in finished:
        print(
            f"  req={req.request_id} computed={req.num_computed_tokens}/"
            f"{req.num_tokens_with_output} completed={req.completed}"
        )
    if len(finished) != len(requests):
        raise SystemExit(
            f"expected {len(requests)} finished, got {len(finished)}"
        )


if __name__ == "__main__":
    main()
