"""Unit tests for RingCommAnalyzer and per-rank analyzer expansion."""

from __future__ import annotations

import unittest

from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analyzer import (
    analyze_split,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.analyzer import (
    AnalyticAnalyzer,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.analytic.types import (
    CommCollective,
    OperatorDAG,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.comm import (
    KERNEL_PUT,
    KERNEL_WAIT,
    RingCommAnalyzer,
    encode_conn,
)
from hybridsim_infer.workload_generators.infer_workload_generator.op_level.mock.ops import (
    CommOp,
    GemmOp,
)


class TestRingCommAnalyzer(unittest.TestCase):
    def test_allreduce_n4_conn_pairs(self) -> None:
        analyzer = RingCommAnalyzer(replica_id=0, num_ranks=4)
        op = CommOp(
            name="allreduce",
            payload_shape=(8,),
            dtype_bytes=2,
            collective=CommCollective.ALLREDUCE,
            num_ranks=4,
        )
        by_rank = {
            r: analyzer.expand(op, rank=r, op_index=3) for r in range(4)
        }
        for r, kernels in by_rank.items():
            self.assertEqual(len(kernels), 12, msg=f"rank {r}")
            puts = [k for k in kernels if k["type"] == KERNEL_PUT]
            waits = [k for k in kernels if k["type"] == KERNEL_WAIT]
            self.assertEqual(len(puts), 6)
            self.assertEqual(len(waits), 6)

        puts_by_conn = {}
        waits_by_conn = {}
        for r, kernels in by_rank.items():
            for k in kernels:
                conn = int(k["params"]["conn_id"])
                if k["type"] == KERNEL_PUT:
                    puts_by_conn.setdefault(conn, []).append(r)
                    self.assertEqual(
                        k["params"]["dst_addr"], f"0:{(r + 1) % 4}"
                    )
                else:
                    waits_by_conn.setdefault(conn, []).append(r)

        self.assertEqual(set(puts_by_conn), set(waits_by_conn))
        for conn, senders in puts_by_conn.items():
            self.assertEqual(len(senders), 1)
            self.assertEqual(len(waits_by_conn[conn]), 1)
            sender = senders[0]
            waiter = waits_by_conn[conn][0]
            self.assertEqual(waiter, (sender + 1) % 4)

    def test_p2p_sender_receiver(self) -> None:
        analyzer = RingCommAnalyzer(replica_id=2, num_ranks=2)
        op = CommOp(
            name="p2p",
            payload_shape=(4,),
            dtype_bytes=2,
            collective=CommCollective.P2P,
            num_ranks=2,
        )
        r0 = analyzer.expand(op, rank=0, op_index=1)
        r1 = analyzer.expand(op, rank=1, op_index=1)
        self.assertEqual(r0[0]["type"], KERNEL_PUT)
        self.assertEqual(r0[0]["params"]["dst_addr"], "2:1")
        self.assertEqual(r1[0]["type"], KERNEL_WAIT)
        self.assertEqual(
            r0[0]["params"]["conn_id"], r1[0]["params"]["conn_id"]
        )

    def test_all_to_all_pairs(self) -> None:
        analyzer = RingCommAnalyzer(replica_id=0, num_ranks=3)
        op = CommOp(
            name="dispatch",
            payload_shape=(3,),
            dtype_bytes=2,
            collective=CommCollective.DISPATCH,
            num_ranks=3,
        )
        by_rank = {r: analyzer.expand(op, rank=r, op_index=0) for r in range(3)}
        for r, kernels in by_rank.items():
            self.assertEqual(len(kernels), 4)
        conn = encode_conn(0, 0, 0, 1)
        puts = [
            k
            for k in by_rank[0]
            if k["type"] == KERNEL_PUT and k["params"]["conn_id"] == conn
        ]
        waits = [
            k
            for k in by_rank[1]
            if k["type"] == KERNEL_WAIT and k["params"]["conn_id"] == conn
        ]
        self.assertEqual(len(puts), 1)
        self.assertEqual(len(waits), 1)


class TestSplitAnalyzers(unittest.TestCase):
    def test_compute_analytic_comm_ring(self) -> None:
        gemm = GemmOp(
            name="gemm",
            a_shape=(4, 4),
            b_shape=(4, 4),
            dtype_bytes=2,
        )
        comm = CommOp(
            name="allreduce",
            payload_shape=(8,),
            dtype_bytes=2,
            collective=CommCollective.ALLREDUCE,
            num_ranks=2,
            deps=[0],
        )
        dag = OperatorDAG(operators=[gemm, comm])
        compute = AnalyticAnalyzer()
        ring = RingCommAnalyzer(replica_id=0, num_ranks=2)
        wl = analyze_split(
            dag,
            compute=compute,
            comm=ring,
            workload_id=7,
            rank=0,
            replica_id=0,
            num_ranks=2,
        )
        kernels = wl["kernels"]
        self.assertEqual(kernels[0]["name"], "gemm")
        self.assertGreater(float(kernels[0]["duration"]), 0.0)
        self.assertNotIn("type", kernels[0])
        puts = [k for k in kernels[1:] if k.get("type") == KERNEL_PUT]
        waits = [k for k in kernels[1:] if k.get("type") == KERNEL_WAIT]
        self.assertEqual(len(puts), 2)
        self.assertEqual(len(waits), 2)
        self.assertEqual(puts[0]["dependencies"], [0])

    def test_analytic_only_keeps_timeout_comm(self) -> None:
        comm = CommOp(
            name="allreduce",
            payload_shape=(8,),
            dtype_bytes=2,
            collective=CommCollective.ALLREDUCE,
            num_ranks=2,
        )
        dag = OperatorDAG(operators=[comm])
        wl = AnalyticAnalyzer().analyze(dag, workload_id=1)
        self.assertEqual(len(wl["kernels"]), 1)
        self.assertGreater(float(wl["kernels"][0]["duration"]), 0.0)
        self.assertNotIn("type", wl["kernels"][0])


class TestWorkerEngineBarrier(unittest.TestCase):
    def test_per_rank_waits_for_all(self) -> None:
        from hybridsim_infer.actors.worker_engine import WorkerEngine
        from hybridsim_infer.config import InferenceConfig
        from hybridsim_infer.schedule_types import ScheduleBatch

        class FakeEngine:
            def __init__(self) -> None:
                self.handler = None
                self.sent: list[dict] = []

            def set_on_workload_complete(self, handler) -> None:
                self.handler = handler

            def send_workload(self, workload) -> None:
                self.sent.append(workload)

            def start(self) -> None:
                return None

            def check_error(self) -> None:
                return None

        engines = [FakeEngine(), FakeEngine()]
        done: list[int] = []
        worker = WorkerEngine(
            engines,
            on_batch_complete=lambda wid, _b: done.append(wid),
            config=InferenceConfig(),
        )
        batch = ScheduleBatch(batch_id=1, requests=[], tokens_per_request={})
        worker.submit(
            {
                "workload_id": 9,
                "per_rank": {
                    0: {"kernels": []},
                    1: {"kernels": []},
                },
            },
            batch,
        )
        self.assertEqual(done, [])
        engines[0].handler(9)
        self.assertEqual(done, [])
        engines[1].handler(9)
        self.assertEqual(done, [9])


if __name__ == "__main__":
    unittest.main()
