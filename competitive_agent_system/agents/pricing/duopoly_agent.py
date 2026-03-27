from __future__ import annotations

from typing import Any

from transformers import PreTrainedTokenizer

from competitive_agent_system.agents.base import CompetitiveBaseAgent
from competitive_agent_system.agents.registry import CompetitiveAgentRegistry


@CompetitiveAgentRegistry.register("Firm 1 Agent")
class Firm1PricingAgent(CompetitiveBaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Firm 1 Agent", wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)


@CompetitiveAgentRegistry.register("Firm 2 Agent")
class Firm2PricingAgent(CompetitiveBaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Firm 2 Agent", wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
