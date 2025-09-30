"""
In-memory repositories for tickets and priority configuration.

Provides simple thread-safe storage abstractions to keep request handlers
decoupled from storage implementation. This can be replaced by a database layer
without changing route handlers significantly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import threading
import uuid
import time


@dataclass
class Ticket:
    """Represents a support ticket stored in the repository."""
    id: str
    title: str
    description: str
    created_at: float = field(default_factory=lambda: time.time())
    label: Optional[str] = None
    probability: Optional[Dict[str, float]] = None


class TicketRepository:
    """Thread-safe in-memory ticket storage."""
    def __init__(self) -> None:
        self._store: Dict[str, Ticket] = {}
        self._lock = threading.RLock()

    # PUBLIC_INTERFACE
    def create(self, title: str, description: str) -> Ticket:
        """Create and store a new ticket."""
        with self._lock:
            ticket_id = str(uuid.uuid4())
            ticket = Ticket(id=ticket_id, title=title, description=description)
            self._store[ticket_id] = ticket
            return ticket

    # PUBLIC_INTERFACE
    def update_classification(self, ticket_id: str, label: str, probability: Dict[str, float]) -> Optional[Ticket]:
        """Update a ticket classification data."""
        with self._lock:
            ticket = self._store.get(ticket_id)
            if ticket is None:
                return None
            ticket.label = label
            ticket.probability = probability
            return ticket

    # PUBLIC_INTERFACE
    def get(self, ticket_id: str) -> Optional[Ticket]:
        """Retrieve a ticket."""
        with self._lock:
            return self._store.get(ticket_id)

    # PUBLIC_INTERFACE
    def list(self) -> List[Ticket]:
        """List all tickets."""
        with self._lock:
            return list(self._store.values())


class PriorityConfig:
    """
    Thread-safe store of priority configuration mapping labels to numeric priority.
    Lower number indicates higher urgency, e.g., critical: 1, major: 2, minor: 3
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._map: Dict[str, int] = {
            "critical": 1,
            "major": 2,
            "minor": 3,
        }

    # PUBLIC_INTERFACE
    def get_all(self) -> Dict[str, int]:
        """Return the current priority mapping."""
        with self._lock:
            return dict(self._map)

    # PUBLIC_INTERFACE
    def update(self, label: str, priority: int) -> Dict[str, int]:
        """Update the priority for a label; returns the full map."""
        with self._lock:
            self._map[label] = int(priority)
            return dict(self._map)
