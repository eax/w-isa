"""Versioned flow admission, priority ordering, expiration and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time

from .schema import ModelIdentity, NeuralRule


@dataclass
class _Entry:
    rule: NeuralRule
    status: str
    order: int
    created_at: float
    activated_at: float | None = None
    last_used_at: float | None = None
    hits: int = 0
    evidence: dict | None = None


class FlowTable:
    def __init__(self, identity: ModelIdentity, clock=time.time):
        self.identity = identity
        self.clock = clock
        self.lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self.version = 0

    def submit(self, rule: NeuralRule) -> None:
        with self.lock:
            if rule.identity != self.identity:
                raise ValueError("rule model identity differs from the flow table")
            if rule.rule_id in self._entries:
                raise ValueError(f"rule ID already exists: {rule.rule_id}")
            self._entries[rule.rule_id] = _Entry(rule.clone(), "candidate", len(self._entries), self.clock())

    def contains(self, rule_id: str) -> bool:
        with self.lock:
            return rule_id in self._entries

    def _expire(self):
        now = self.clock()
        for entry in self._entries.values():
            if entry.status != "active":
                continue
            if entry.activated_at is None or entry.last_used_at is None:
                raise ValueError("active rule timestamps are missing")
            hard, idle = entry.rule.hard_timeout_seconds, entry.rule.idle_timeout_seconds
            hard_expired = hard is not None and now - entry.activated_at >= hard
            idle_expired = idle is not None and now - entry.last_used_at >= idle
            if hard_expired or idle_expired:
                entry.status = "expired"
                self.version += 1

    def snapshot(self) -> tuple[int, tuple[NeuralRule, ...]]:
        with self.lock:
            self._expire()
            entries = sorted((e for e in self._entries.values() if e.status == "active"),
                             key=lambda e: (-e.rule.priority, e.order))
            return self.version, tuple(e.rule.clone() for e in entries)

    def promote(self, rule_id: str, evidence: dict) -> None:
        with self.lock:
            self._expire()
            entry = self._entries[rule_id]
            if entry.status != "candidate":
                raise ValueError("only candidate rules can be promoted")
            if (evidence.get("accepted") is not True
                    or evidence.get("rule_fingerprint") != entry.rule.fingerprint
                    or evidence.get("table_version") != self.version):
                raise ValueError("promotion requires accepted evidence for this rule and table version")
            saved_evidence = json.loads(json.dumps(evidence, allow_nan=False))
            entry.status = "active"
            entry.activated_at = entry.last_used_at = self.clock()
            entry.evidence = saved_evidence
            self.version += 1

    def reject(self, rule_id: str, evidence: dict) -> None:
        with self.lock:
            entry = self._entries[rule_id]
            if entry.status != "candidate":
                raise ValueError("only candidate rules can be rejected")
            saved_evidence = json.loads(json.dumps(evidence, allow_nan=False))
            entry.status = "rejected"
            entry.evidence = saved_evidence

    def trial_snapshot(self, rule_id: str):
        with self.lock:
            self._expire()
            candidate = self._entries[rule_id]
            if candidate.status != "candidate":
                raise ValueError("trial requires a candidate rule")
            active = [entry for entry in self._entries.values() if entry.status == "active"]

            def ordered(entries):
                return tuple(entry.rule.clone() for entry in sorted(entries,
                             key=lambda entry: (-entry.rule.priority, entry.order)))

            return self.version, ordered(active), ordered([*active, candidate])

    def retire(self, rule_id: str) -> None:
        with self.lock:
            entry = self._entries[rule_id]
            if entry.status == "active":
                self.version += 1
            entry.status = "retired"

    def record_hit(self, rule_id: str) -> None:
        with self.lock:
            entry = self._entries[rule_id]
            if entry.status == "active":
                entry.hits += 1
                entry.last_used_at = self.clock()

    def export(self) -> dict:
        with self.lock:
            self._expire()
            return {"schema_version": 1, "identity": asdict(self.identity), "version": self.version,
                    "entries": [{"rule": e.rule.payload(), "fingerprint": e.rule.fingerprint,
                                 "status": e.status, "order": e.order, "created_at": e.created_at,
                                 "activated_at": e.activated_at, "last_used_at": e.last_used_at,
                                 "hits": e.hits, "evidence": e.evidence} for e in self._entries.values()]}

    def save(self, path: str | Path) -> None:
        payload = self.export()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                             delete=False) as handle:
                temporary = handle.name
                json.dump(payload, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path, identity: ModelIdentity, clock=time.time) -> FlowTable:
        data = json.loads(Path(path).read_text())
        if data["schema_version"] != 1 or ModelIdentity(**data["identity"]) != identity:
            raise ValueError("flow-table schema or model identity mismatch")
        if type(data["version"]) is not int or data["version"] < 0:
            raise ValueError("flow-table version must be a nonnegative integer")
        table = cls(identity, clock)
        orders = set()
        for record in data["entries"]:
            rule = NeuralRule.from_payload(record["rule"])
            if rule.fingerprint != record["fingerprint"]:
                raise ValueError("rule checksum mismatch")
            if rule.rule_id in table._entries or rule.identity != identity:
                raise ValueError("duplicate rule ID or inconsistent model identity")
            state = {k: v for k, v in record.items() if k not in {"rule", "fingerprint"}}
            if type(state["order"]) is not int or state["order"] < 0 or state["order"] in orders:
                raise ValueError("flow registration order must be unique and nonnegative")
            orders.add(state["order"])
            if type(state["hits"]) is not int or state["hits"] < 0:
                raise ValueError("flow hit count must be a nonnegative integer")
            if state["created_at"] is None:
                raise ValueError("flow creation time is required")
            for name in ("created_at", "activated_at", "last_used_at"):
                value = state[name]
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError("flow timestamps must be finite numbers")
            if state["status"] not in {"candidate", "active", "rejected", "expired", "retired"}:
                raise ValueError("unknown rule status")
            if state["status"] == "active":
                evidence = state["evidence"] or {}
                if (state["activated_at"] is None or state["last_used_at"] is None
                        or evidence.get("accepted") is not True
                        or evidence.get("rule_fingerprint") != rule.fingerprint):
                    raise ValueError("active rule has incomplete admission state")
            table._entries[rule.rule_id] = _Entry(rule=rule, **state)
        table.version = data["version"]
        table._expire()
        return table