"""
Priority configuration routes.

Provides endpoints to:
- Get current label->priority mapping
- Update a label's priority value
"""
from flask_smorest import Blueprint
from flask.views import MethodView

from ..schemas import PriorityMapSchema, LabelPrioritySchema
from ..services.ticket_service import TicketService


blp = Blueprint(
    "Priority",
    "priority",
    url_prefix="/priority",
    description="Endpoints for priority configuration and management",
)


def get_service() -> TicketService:
    # Lazy import to avoid circular imports
    from ..startup import services
    return services.ticket_service


@blp.route("/")
class PriorityCollection(MethodView):
    @blp.response(200, PriorityMapSchema)
    def get(self):
        """
        Get the current priority mapping of labels to numeric priority values.
        """
        svc = get_service()
        return {"priorities": svc.get_priorities()}

    @blp.arguments(LabelPrioritySchema, as_kwargs=True)
    @blp.response(200, PriorityMapSchema)
    def put(self, label: str, priority: int):
        """
        Update or create a priority mapping for a given label.
        """
        svc = get_service()
        mapping = svc.update_priority(label, priority)
        return {"priorities": mapping}
