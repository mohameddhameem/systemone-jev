"""Use a published System One router: P(escalate to the large model) for one question, built exactly as in training.

    from systemone_jev.route import Router
    router = Router.from_pretrained("models/qp-large-helps-nanojev-A/seed0/model")   # or a Hub id[@revision]
    p = router.p_escalate(question, passages)                  # qp: before the small model runs
    p = router.p_escalate(question, passages, answer="1987")   # qpa: also reads the small model's answer
    use_large = router.route(question, passages) == "large"   # p >= the calib-picked threshold

`passages` are the small model's "title: text" paragraphs in the order it read them (in training: HotpotQA's
distractor paragraphs, nano-jev's top 2). The router's `systemone.json`, next to its weights, names its input set
and its operating threshold.
"""

import json
from dataclasses import dataclass

from systemone_jev.evidence import resolve_checkpoint
from systemone_jev.judge import INPUTS, Judge, route_items

CARD = "systemone.json"  # {"inputs": "qp" | "qpa", "threshold": float, ...}


@dataclass
class Router:
    judge: Judge
    inputs: str
    threshold: float = 0.5

    @classmethod
    def from_pretrained(cls, path: str, device: str | None = None, inputs: str | None = None,
                        threshold: float | None = None) -> "Router":
        card_path = resolve_checkpoint(path) / CARD
        card = json.loads(card_path.read_text()) if card_path.exists() else {}
        inputs = inputs or card.get("inputs")
        if inputs not in INPUTS:
            raise ValueError(f"{path}: set inputs to one of {list(INPUTS)} (no {CARD} names it)")
        return cls(Judge(path, device), inputs, threshold if threshold is not None else card.get("threshold", 0.5))

    def p_escalate_batch(self, rows: list[dict]) -> list[float]:
        """rows: {"question", "passages", and "small_pred" for an input set that reads the answer}."""
        if INPUTS[self.inputs] and any(r.get("small_pred") is None for r in rows):
            raise ValueError(f"the {self.inputs} router reads the small model's answer: pass answer=")
        import pandas as pd

        probs, _ = self.judge.probabilities(route_items(pd.DataFrame(rows), self.inputs))
        return probs[:, 1].tolist()  # option 1 is "large model"

    def p_escalate(self, question: str, passages: list[str], answer: str | None = None) -> float:
        return self.p_escalate_batch([{"question": question, "passages": passages, "small_pred": answer}])[0]

    def route(self, question: str, passages: list[str], answer: str | None = None) -> str:
        return "large" if self.p_escalate(question, passages, answer) >= self.threshold else "small"
