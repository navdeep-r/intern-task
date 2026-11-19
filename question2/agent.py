from __future__ import annotations

import json
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution


@dataclass
class TicketResolution:
    action: str
    kb_article: str


@dataclass(frozen=True)
class TriageResult:
    """Small, serializable handoff between the pipeline's first two nodes."""

    facts: dict[str, object]
    candidate_articles: tuple[str, ...]


_ALLOWED_ACTIONS = {"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"}


class SupportPipeline:
    """Same product as Question 1 — triage, retrieve, compose, audit."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        triage = self._triage(order)
        retrieved = self._retrieve(triage)
        draft = self._compose(ticket_id, thread, snapshot, triage, retrieved)
        return self._audit(draft, triage, retrieved)

    def _triage(self, order: dict) -> TriageResult:
        """Select policy candidates locally from structured order facts.

        This is retrieval routing, not a customer-facing decision: the model still
        produces the final action and citation after receiving the source policy.
        """
        facts = {
            key: order.get(key)
            for key in (
                "product_type",
                "delivered_days_ago",
                "subscription_age_days",
                "opened",
                "license_revealed",
                "customized",
                "damage_reported",
            )
            if key in order
        }

        if order.get("product_type") == "subscription":
            candidates = ("subscription-cooling-off",)
        elif order.get("license_revealed"):
            candidates = ("digital-license-no-refund",)
        elif order.get("customized") and order.get("damage_reported"):
            candidates = ("custom-items-no-replace",)
        elif order.get("damage_reported"):
            candidates = ("damage-replacement-7-days", "returns-within-30-days")
        else:
            candidates = ("returns-within-30-days",)
        return TriageResult(facts=facts, candidate_articles=candidates)

    def _retrieve(self, triage: TriageResult) -> dict[str, str]:
        """Load only the policy documents selected by triage."""
        return {article_id: load_article(article_id) for article_id in triage.candidate_articles}

    def _compose(
        self,
        ticket_id: str,
        thread: str,
        snapshot: str,
        triage: TriageResult,
        retrieved: dict[str, str],
    ) -> str:
        messages = [
            ChatMessage(
                "system",
                (
                    "Compose node: decide using only the supplied policy and order facts. "
                    "Return only JSON with action, kb_article, and customer_reply."
                ),
            ),
            ChatMessage(
                "user",
                "\n\n".join(
                    [
                        f"TICKET {ticket_id}",
                        f"THREAD:\n{thread}",
                        f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
                        f"TRIAGE_FACTS:\n{json.dumps(triage.facts, separators=(',', ':'))}",
                        f"POLICY_ARTICLES:\n{json.dumps(retrieved, separators=(',', ':'))}",
                    ]
                ),
            ),
        ]
        return self.llm.chat("compose", messages)

    def _audit(
        self,
        draft: str,
        triage: TriageResult,
        retrieved: dict[str, str],
    ) -> TicketResolution:
        """Validate the generated answer without another costly model completion."""
        parsed = parse_resolution(draft)
        if parsed.action not in _ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported action from compose node: {parsed.action}")
        if parsed.kb_article not in retrieved:
            raise ValueError(
                "Compose node cited an article that was not retrieved: "
                f"{parsed.kb_article}; candidates={triage.candidate_articles}"
            )
        return TicketResolution(parsed.action, parsed.kb_article)
