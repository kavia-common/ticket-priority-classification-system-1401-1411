"""
Application startup and dependency wiring.

Creates singleton instances of repositories, model, and service.
Exposes a `services` module-level instance for route handlers to import.
"""
from __future__ import annotations

from dataclasses import dataclass
from flask_smorest import Api

from .services.repository import TicketRepository, PriorityConfig
from .models.model import TicketPriorityModel
from .services.ticket_service import TicketService


@dataclass
class Services:
    ticket_repo: TicketRepository
    priority_config: PriorityConfig
    model: TicketPriorityModel
    ticket_service: TicketService


# Initialize singletons for app lifetime
ticket_repo = TicketRepository()
priority_config = PriorityConfig()
model = TicketPriorityModel()
ticket_service = TicketService(ticket_repo, model, priority_config)

# Exported container of services for easy access
services = Services(
    ticket_repo=ticket_repo,
    priority_config=priority_config,
    model=model,
    ticket_service=ticket_service,
)


def register_blueprints(api: Api) -> None:
    """Register all blueprints on the given Api instance."""
    from .routes.health import blp as health_blp
    from .routes.tickets import blp as tickets_blp
    from .routes.priority import blp as priority_blp

    api.register_blueprint(health_blp)
    api.register_blueprint(tickets_blp)
    api.register_blueprint(priority_blp)
