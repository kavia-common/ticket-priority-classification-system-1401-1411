"""
Tickets and classification routes.

Provides endpoints to:
- Submit a ticket
- Get a ticket by id
- List tickets
- Re-classify a ticket
- Train model with additional labeled samples
"""
from flask_smorest import Blueprint
from flask.views import MethodView

from ..schemas import (
    SubmitTicketSchema,
    TicketResponseSchema,
    TicketsListResponseSchema,
    TrainRequestSchema,
)
from ..services.ticket_service import TicketService
from ..models.model import TrainingSample


blp = Blueprint(
    "Tickets",
    "tickets",
    url_prefix="/tickets",
    description="Endpoints for submitting tickets and retrieving classifications",
)


def get_service() -> TicketService:
    # Lazy import to avoid circular imports
    from ..startup import services
    return services.ticket_service


@blp.route("/")
class TicketCollection(MethodView):
    @blp.arguments(SubmitTicketSchema, as_kwargs=True)
    @blp.response(201, TicketResponseSchema)
    def post(self, title: str, description: str):
        """
        Submit a new ticket and receive its classification.

        Returns the created ticket with classification label, probabilities,
        and mapped priority value.
        """
        svc = get_service()
        return svc.submit_ticket(title=title, description=description)

    @blp.response(200, TicketsListResponseSchema)
    def get(self):
        """
        List all tickets with their classification details.
        """
        svc = get_service()
        items = svc.list_tickets()
        return {"items": items}


@blp.route("/<string:ticket_id>")
class TicketResource(MethodView):
    @blp.response(200, TicketResponseSchema)
    def get(self, ticket_id: str):
        """
        Retrieve a ticket by id.
        """
        svc = get_service()
        ticket = svc.get_ticket(ticket_id)
        if not ticket:
            blp.abort(404, message="Ticket not found")
        return ticket


@blp.route("/<string:ticket_id>/classify")
class TicketClassify(MethodView):
    @blp.response(200, TicketResponseSchema)
    def post(self, ticket_id: str):
        """
        Re-classify an existing ticket by id.
        """
        svc = get_service()
        ticket = svc.classify_ticket(ticket_id)
        if not ticket:
            blp.abort(404, message="Ticket not found")
        return ticket


@blp.route("/train")
class ModelTraining(MethodView):
    @blp.arguments(TrainRequestSchema, as_kwargs=True)
    def post(self, samples):
        """
        Retrain the model with additional labeled samples.

        This endpoint accepts a list of training samples (text, label) and
        retrains the model combining built-in bootstrap data with provided
        samples. Returns a status message upon completion.
        """
        svc = get_service()
        extra = [TrainingSample(text=s["text"], label=s["label"]) for s in samples]
        return svc.train_model(extra)
