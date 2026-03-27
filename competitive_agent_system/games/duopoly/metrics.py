from __future__ import annotations

from competitive_agent_system.games.base import CompetitiveEpisodeSummary


class DuopolyMetricComputer:
    def finalize(self, game) -> CompetitiveEpisodeSummary:
        return game.summarize_episode()
