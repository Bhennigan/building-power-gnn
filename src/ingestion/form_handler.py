"""Form handling utilities for web UI integration.

Provides schema definitions and validation for manual data entry forms.
"""

from typing import Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import json

from pydantic import BaseModel, Field, field_validator


class FormFieldType(str, Enum):
    """Types of form fields."""
    TEXT = "text"
    NUMBER = "number"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    CHECKBOX = "checkbox"
    DATE = "date"
    DATETIME = "datetime"
    TEXTAREA = "textarea"


@dataclass
class FormFieldOption:
    """Option for select/multi-select fields."""
    value: str
    label: str
    disabled: bool = False


@dataclass
class FormField:
    """Definition of a form field."""
    name: str
    label: str
    field_type: FormFieldType
    required: bool = False
    default_value: Any = None
    placeholder: str = ""
    help_text: str = ""
    options: list[FormFieldOption] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    depends_on: Optional[str] = None  # Field visibility depends on another field

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "name": self.name,
            "label": self.label,
            "type": self.field_type.value,
            "required": self.required,
            "placeholder": self.placeholder,
            "helpText": self.help_text,
        }

        if self.default_value is not None:
            result["defaultValue"] = self.default_value

        if self.options:
            result["options"] = [
                {"value": o.value, "label": o.label, "disabled": o.disabled}
                for o in self.options
            ]

        if self.min_value is not None:
            result["min"] = self.min_value
        if self.max_value is not None:
            result["max"] = self.max_value
        if self.pattern:
            result["pattern"] = self.pattern
        if self.depends_on:
            result["dependsOn"] = self.depends_on

        return result


@dataclass
class FormSchema:
    """Complete form schema definition."""
    title: str
    description: str
    fields: list[FormField]
    submit_label: str = "Submit"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "fields": [f.to_dict() for f in self.fields],
            "submitLabel": self.submit_label,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


# Pre-defined form schemas for building graph entities

NODE_TYPE_OPTIONS = [
    FormFieldOption("HVAC", "HVAC System"),
    FormFieldOption("Lighting", "Lighting"),
    FormFieldOption("Sensor", "Sensor"),
    FormFieldOption("Room", "Room/Space"),
    FormFieldOption("Meter", "Power Meter"),
    FormFieldOption("WeatherStation", "Weather Station"),
]

SENSOR_SUBTYPE_OPTIONS = [
    FormFieldOption("temperature", "Temperature"),
    FormFieldOption("humidity", "Humidity"),
    FormFieldOption("occupancy", "Occupancy"),
    FormFieldOption("co2", "CO2 Level"),
    FormFieldOption("light", "Light Level"),
    FormFieldOption("power", "Power"),
]

EDGE_TYPE_OPTIONS = [
    FormFieldOption("serves", "Serves (HVAC/Lighting -> Room)"),
    FormFieldOption("monitors", "Monitors (Sensor -> Room/HVAC)"),
    FormFieldOption("feeds", "Feeds (Meter -> Equipment)"),
    FormFieldOption("adjacent", "Adjacent (Room -> Room)"),
    FormFieldOption("controls", "Controls (Sensor -> HVAC/Lighting)"),
]


def get_node_form_schema() -> FormSchema:
    """Get the form schema for creating nodes."""
    return FormSchema(
        title="Add Building Node",
        description="Add a new component to the building graph",
        fields=[
            FormField(
                name="node_id",
                label="Node ID",
                field_type=FormFieldType.TEXT,
                required=True,
                placeholder="e.g., hvac_floor1_01",
                help_text="Unique identifier for this node",
                pattern=r"^[a-zA-Z0-9_-]+$",
            ),
            FormField(
                name="node_type",
                label="Node Type",
                field_type=FormFieldType.SELECT,
                required=True,
                options=NODE_TYPE_OPTIONS,
                help_text="Type of building component",
            ),
            FormField(
                name="subtype",
                label="Sensor Subtype",
                field_type=FormFieldType.SELECT,
                options=SENSOR_SUBTYPE_OPTIONS,
                depends_on="node_type:Sensor",
                help_text="Type of sensor measurement",
            ),
            FormField(
                name="zone",
                label="Zone",
                field_type=FormFieldType.TEXT,
                placeholder="e.g., floor_1, zone_a",
                help_text="Building zone or area",
            ),
            FormField(
                name="floor",
                label="Floor",
                field_type=FormFieldType.TEXT,
                placeholder="e.g., 1, basement, roof",
            ),
            FormField(
                name="capacity_kw",
                label="Capacity (kW)",
                field_type=FormFieldType.NUMBER,
                depends_on="node_type:HVAC",
                min_value=0,
                help_text="HVAC system capacity in kilowatts",
            ),
            FormField(
                name="wattage",
                label="Wattage (W)",
                field_type=FormFieldType.NUMBER,
                depends_on="node_type:Lighting",
                min_value=0,
                help_text="Lighting power consumption",
            ),
            FormField(
                name="area_sqft",
                label="Area (sq ft)",
                field_type=FormFieldType.NUMBER,
                depends_on="node_type:Room",
                min_value=0,
                help_text="Room floor area",
            ),
            FormField(
                name="occupancy_max",
                label="Max Occupancy",
                field_type=FormFieldType.NUMBER,
                depends_on="node_type:Room",
                min_value=0,
                help_text="Maximum room occupancy",
            ),
        ],
        submit_label="Add Node",
    )


def get_edge_form_schema(available_nodes: Optional[list[dict]] = None) -> FormSchema:
    """Get the form schema for creating edges.

    Args:
        available_nodes: List of existing nodes for dropdowns.
    """
    node_options = []
    if available_nodes:
        node_options = [
            FormFieldOption(n["node_id"], f"{n['node_id']} ({n.get('node_type', 'unknown')})")
            for n in available_nodes
        ]

    return FormSchema(
        title="Add Connection",
        description="Define a relationship between building components",
        fields=[
            FormField(
                name="source",
                label="Source Node",
                field_type=FormFieldType.SELECT,
                required=True,
                options=node_options,
                help_text="Starting node of the connection",
            ),
            FormField(
                name="target",
                label="Target Node",
                field_type=FormFieldType.SELECT,
                required=True,
                options=node_options,
                help_text="Ending node of the connection",
            ),
            FormField(
                name="edge_type",
                label="Connection Type",
                field_type=FormFieldType.SELECT,
                required=True,
                options=EDGE_TYPE_OPTIONS,
                help_text="Type of relationship",
            ),
            FormField(
                name="weight",
                label="Weight",
                field_type=FormFieldType.NUMBER,
                default_value=1.0,
                min_value=0,
                help_text="Connection strength/importance (optional)",
            ),
            FormField(
                name="bidirectional",
                label="Bidirectional",
                field_type=FormFieldType.CHECKBOX,
                default_value=False,
                help_text="Create connection in both directions",
            ),
        ],
        submit_label="Add Connection",
    )


def get_timeseries_form_schema(available_nodes: Optional[list[dict]] = None) -> FormSchema:
    """Get the form schema for adding time series readings."""
    node_options = []
    if available_nodes:
        node_options = [
            FormFieldOption(n["node_id"], f"{n['node_id']} ({n.get('node_type', 'unknown')})")
            for n in available_nodes
        ]

    return FormSchema(
        title="Add Sensor Reading",
        description="Record a time series measurement",
        fields=[
            FormField(
                name="node_id",
                label="Node",
                field_type=FormFieldType.SELECT,
                required=True,
                options=node_options,
                help_text="Node that recorded this reading",
            ),
            FormField(
                name="timestamp",
                label="Timestamp",
                field_type=FormFieldType.DATETIME,
                required=True,
                help_text="When the reading was taken",
            ),
            FormField(
                name="value",
                label="Value",
                field_type=FormFieldType.NUMBER,
                required=True,
                help_text="Measured value",
            ),
            FormField(
                name="metric",
                label="Metric Type",
                field_type=FormFieldType.TEXT,
                default_value="default",
                placeholder="e.g., temperature, power, humidity",
            ),
            FormField(
                name="unit",
                label="Unit",
                field_type=FormFieldType.TEXT,
                placeholder="e.g., kWh, °F, %",
            ),
        ],
        submit_label="Add Reading",
    )


class DynamicFormHandler:
    """Handles dynamic form generation based on node types."""

    def __init__(self):
        """Initialize form handler."""
        self._type_specific_fields: dict[str, list[FormField]] = {
            "HVAC": [
                FormField(
                    name="attr_efficiency_rating",
                    label="Efficiency Rating",
                    field_type=FormFieldType.NUMBER,
                    min_value=0,
                    max_value=1,
                    default_value=0.85,
                ),
                FormField(
                    name="attr_age_years",
                    label="Age (years)",
                    field_type=FormFieldType.NUMBER,
                    min_value=0,
                ),
                FormField(
                    name="attr_cop",
                    label="Coefficient of Performance",
                    field_type=FormFieldType.NUMBER,
                    min_value=0,
                    default_value=3.0,
                ),
                FormField(
                    name="attr_variable_speed",
                    label="Variable Speed",
                    field_type=FormFieldType.CHECKBOX,
                    default_value=False,
                ),
            ],
            "Lighting": [
                FormField(
                    name="attr_type",
                    label="Light Type",
                    field_type=FormFieldType.SELECT,
                    options=[
                        FormFieldOption("LED", "LED"),
                        FormFieldOption("fluorescent", "Fluorescent"),
                        FormFieldOption("incandescent", "Incandescent"),
                        FormFieldOption("halogen", "Halogen"),
                    ],
                    default_value="LED",
                ),
                FormField(
                    name="attr_dimmable",
                    label="Dimmable",
                    field_type=FormFieldType.CHECKBOX,
                    default_value=False,
                ),
                FormField(
                    name="attr_occupancy_sensor",
                    label="Has Occupancy Sensor",
                    field_type=FormFieldType.CHECKBOX,
                    default_value=False,
                ),
            ],
            "Sensor": [
                FormField(
                    name="attr_accuracy",
                    label="Accuracy",
                    field_type=FormFieldType.NUMBER,
                    min_value=0,
                    default_value=0.1,
                ),
                FormField(
                    name="attr_sample_rate_hz",
                    label="Sample Rate (Hz)",
                    field_type=FormFieldType.NUMBER,
                    min_value=0,
                    default_value=1.0,
                ),
            ],
            "Room": [
                FormField(
                    name="attr_ceiling_height",
                    label="Ceiling Height (ft)",
                    field_type=FormFieldType.NUMBER,
                    min_value=0,
                    default_value=10.0,
                ),
                FormField(
                    name="attr_window_ratio",
                    label="Window Ratio",
                    field_type=FormFieldType.NUMBER,
                    min_value=0,
                    max_value=1,
                    default_value=0.2,
                ),
            ],
        }

    def get_fields_for_type(self, node_type: str) -> list[FormField]:
        """Get type-specific form fields."""
        return self._type_specific_fields.get(node_type, [])

    def get_extended_node_schema(self, node_type: str) -> FormSchema:
        """Get full form schema including type-specific fields."""
        base_schema = get_node_form_schema()

        # Add type-specific fields
        extra_fields = self.get_fields_for_type(node_type)
        base_schema.fields.extend(extra_fields)

        return base_schema
