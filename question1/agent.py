from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


# Keep the policy knowledge small and stable.  The model only needs the routing
# rules and article ids to make a decision; sending the full article prose and
# worked examples on every ticket adds cost without adding decision signal.
_POLICY_GUIDE = """POLICIES:
returns-within-30-days: physical, unopened, delivered <=30 days -> REFUND.
digital-license-no-refund: revealed digital license -> SEND_ARTICLE.
custom-items-no-replace: damaged customized item -> ESCALATE_TIER2.
damage-replacement-7-days: damaged non-custom item, delivered <=7 days -> REPLACE.
subscription-cooling-off: subscription <=30 days -> REFUND; otherwise SEND_ARTICLE.
"""


class SupportCopilot:
    """
    ShopAssist Tier-1 copilot (baseline).

    Given a support ticket + order snapshot, choose an action, cite the help-center
    article, and draft the customer reply.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        system = "\n".join(
            [
                "You are ShopAssist, a Tier-1 customer support copilot.",
                _POLICY_GUIDE.rstrip(),
                (
                    "Use the order facts and applicable policy. Return only a JSON object with "
                    "action, kb_article, and customer_reply. Do not include an explanation."
                ),
            ]
        )
        user = "\n\n".join(
            [
                f"TICKET_ID: {ticket_id}",
                f"THREAD:\n{thread}",
                f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT",
            ]
        )
        return [ChatMessage("system", system), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat("handle_ticket", messages)
        return parse_resolution(completion)
