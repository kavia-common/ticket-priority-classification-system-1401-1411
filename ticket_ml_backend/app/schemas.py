"""
Request and response schemas using marshmallow for flask-smorest integration.
"""
from __future__ import annotations

from marshmallow import Schema, fields, validate


class SubmitTicketSchema(Schema):
    """Schema for ticket submission requests."""
    title = fields.Str(
        required=True,
        description="Short title for the ticket",
        validate=validate.Length(min=1, max=200),
    )
    description = fields.Str(
        required=True,
        description="Detailed description of the issue",
        validate=validate.Length(min=1, max=5000),
    )


class TicketIdSchema(Schema):
    """Schema with a ticket id path parameter."""
    ticket_id = fields.Str(required=True, description="Ticket identifier (UUID)")


class LabelPrioritySchema(Schema):
    """Schema for updating label priority."""
    label = fields.Str(
        required=True,
        description="Label to update (e.g., critical, major, minor)",
    )
    priority = fields.Int(
        required=True,
        description="Priority value (lower is higher priority)",
        validate=validate.Range(min=1, max=100),
    )


class TrainingSampleSchema(Schema):
    """Schema for a single training sample."""
    text = fields.Str(required=True, description="Text content describing the ticket")
    label = fields.Str(
        required=True,
        description="Label for the sample",
        validate=validate.OneOf(["critical", "major", "minor"]),
    )


class TrainRequestSchema(Schema):
    """Schema for training request with additional samples."""
    samples = fields.List(
        fields.Nested(TrainingSampleSchema),
        required=True,
        description="List of labeled samples to augment training data",
    )


class ClassificationSchema(Schema):
    """Schema for classification info in a ticket response."""
    label = fields.Str(allow_none=True, description="Predicted label")
    probabilities = fields.Dict(
        keys=fields.Str(),
        values=fields.Float(),
        description="Per-label probabilities",
    )
    priority_value = fields.Int(
        allow_none=True,
        description="Mapped priority numeric value",
    )


class TicketResponseSchema(Schema):
    """Schema for a single ticket response."""
    id = fields.Str(required=True, description="Ticket ID")
    title = fields.Str(required=True, description="Ticket title")
    description = fields.Str(required=True, description="Ticket description")
    created_at = fields.Float(required=True, description="Creation timestamp (epoch seconds)")
    classification = fields.Nested(
        ClassificationSchema,
        required=True,
        description="Classification result for this ticket",
    )


class TicketsListResponseSchema(Schema):
    """Schema for listing tickets."""
    items = fields.List(fields.Nested(TicketResponseSchema), required=True)


class PriorityMapSchema(Schema):
    """Schema for returning the full priority map."""
    priorities = fields.Dict(keys=fields.Str(), values=fields.Int(), required=True)
