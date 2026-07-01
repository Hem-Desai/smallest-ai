"""
In-memory event store for pushing call transcript events to SSE subscribers.

Events:
  - call_connected     — Twilio media stream connected
  - user_transcript    — STT returned transcript
  - agent_thinking     — LLM is generating a response
  - agent_response     — LLM response text ready
  - agent_speaking     — TTS started speaking
  - agent_done         — TTS completed this utterance
  - call_ended         — call terminated
  - call_status        — Twilio status webhook update
  - error              — any error
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger("smallest.events")


@dataclass
class CallEvent:
    type: str        # event type string
    payload: dict    # event data
    timestamp: float = field(default_factory=time.time)


class EventStore:
    """Thread-safe in-memory pub/sub for call events."""

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._history: Dict[str, list] = defaultdict(list)  # full event log per call

    def get_queue(self, execution_id: str) -> asyncio.Queue:
        return self._queues[execution_id]

    async def push(self, execution_id: str, event_type: str, **payload) -> None:
        event = CallEvent(type=event_type, payload=payload)
        self._history[execution_id].append(event)
        q = self._queues[execution_id]
        await q.put(event)
        logger.debug(
            "EVENT PUSH | execution_id=%s | type=%s | payload_keys=%s",
            execution_id, event_type, list(payload.keys()),
        )

    async def subscribe(self, execution_id: str):
        """Async generator that yields events as they arrive."""
        q = self._queues[execution_id]
        try:
            while True:
                event = await q.get()
                yield event
                if event.type == "call_ended":
                    break
        except asyncio.CancelledError:
            pass

    def get_history(self, execution_id: str) -> list:
        """Return all events for a call so far."""
        return self._history.get(execution_id, [])

    def cleanup(self, execution_id: str) -> None:
        """Remove queues/history for a completed call."""
        self._queues.pop(execution_id, None)
        self._history.pop(execution_id, None)


# Global singleton
event_store = EventStore()
