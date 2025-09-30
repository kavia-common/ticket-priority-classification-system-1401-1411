"""
Service layer providing operations on tickets and priorities by composing
repositories with ML model inference and training logic.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..models.model import TicketPriorityModel, TrainingSample
from .repository import TicketRepository, PriorityConfig, Ticket


class TicketService:
    """Coordinates ticket CRUD operations with model predictions."""
    def __init__(self, repo: TicketRepository, model: TicketPriorityModel, priority_cfg: PriorityConfig) -> None:
        self.repo = repo
        self.model = model
        self.priority_cfg = priority_cfg

    # PUBLIC_INTERFACE
    def submit_ticket(self, title: str, description: str) -> Dict:
        """
        Create a ticket and classify it immediately, returning the enriched ticket.
        """
        ticket = self.repo.create(title=title, description=description)
        label, prob = self.model.predict(f"{title}. {description}")
        self.repo.update_classification(ticket.id, label, prob)
        return self._serialize_ticket(self.repo.get(ticket.id))

    # PUBLIC_INTERFACE
    def classify_ticket(self, ticket_id: str) -> Optional[Dict]:
        """
        Run classification on an existing ticket by id and update it.
        """
        ticket = self.repo.get(ticket_id)
        if not ticket:
            return None
        label, prob = self.model.predict(f"{ticket.title}. {ticket.description}")
        self.repo.update_classification(ticket.id, label, prob)
        return self._serialize_ticket(self.repo.get(ticket.id))

    # PUBLIC_INTERFACE
    def get_ticket(self, ticket_id: str) -> Optional[Dict]:
        """Get a ticket by id."""
        return self._serialize_ticket(self.repo.get(ticket_id))

    # PUBLIC_INTERFACE
    def list_tickets(self) -> List[Dict]:
        """List all tickets."""
        return [self._serialize_ticket(t) for t in self.repo.list()]

    # PUBLIC_INTERFACE
    def get_priorities(self) -> Dict[str, int]:
        """Get priority mapping."""
        return self.priority_cfg.get_all()

    # PUBLIC_INTERFACE
    def update_priority(self, label: str, priority: int) -> Dict[str, int]:
        """Update a label->priority mapping."""
        return self.priority_cfg.update(label, priority)

    # PUBLIC_INTERFACE
    def train_model(self, samples: List[TrainingSample]) -> Dict[str, str]:
        """
        Retrain the model combining base data with additional samples.
        """
        self.model.retrain_with_augmented(samples)
        # After retraining, we can optionally re-classify existing tickets if needed
        for t in self.repo.list():
            label, prob = self.model.predict(f"{t.title}. {t.description}")
            self.repo.update_classification(t.id, label, prob)
        return {"status": "ok", "message": "Model retrained successfully."}

    def _serialize_ticket(self, ticket: Optional[Ticket]) -> Optional[Dict]:
        if not ticket:
            return None
        return {
            "id": ticket.id,
            "title": ticket.title,
            "description": ticket.description,
            "created_at": ticket.created_at,
            "classification": {
                "label": ticket.label,
                "probabilities": ticket.probability or {},
                "priority_value": self.priority_cfg.get_all().get(ticket.label or "", None),
            },
        }
