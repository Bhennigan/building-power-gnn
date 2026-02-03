"""Database models for multi-tenant building power management."""

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime,
    ForeignKey, Text, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Tenant(Base):
    """Organization/company using the platform."""
    __tablename__ = "tenants"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    plan = Column(String(50), default="free")  # free, starter, pro, enterprise
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Settings
    settings = Column(JSON, default=dict)

    # Relationships
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    buildings = relationship("Building", back_populates="tenant", cascade="all, delete-orphan")
    models = relationship("TrainedModel", back_populates="tenant", cascade="all, delete-orphan")


class User(Base):
    """User account."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255))
    role = Column(String(50), default="member")  # admin, member, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")

    __table_args__ = (
        Index("ix_users_tenant_email", "tenant_id", "email"),
    )


class Building(Base):
    """A building or facility being monitored."""
    __tablename__ = "buildings"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)
    name = Column(String(255), nullable=False)
    address = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    timezone = Column(String(50), default="UTC")
    total_area_sqft = Column(Float)
    floors = Column(Integer)
    building_type = Column(String(100))  # office, warehouse, hospital, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="buildings")
    nodes = relationship("Node", back_populates="building", cascade="all, delete-orphan")
    edges = relationship("Edge", back_populates="building", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_buildings_tenant", "tenant_id"),
    )


class Node(Base):
    """A node in the building graph (HVAC, sensor, room, etc.)."""
    __tablename__ = "nodes"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    building_id = Column(String(36), ForeignKey("buildings.id"), nullable=False)
    node_id = Column(String(100), nullable=False)  # User-defined ID
    node_type = Column(String(50), nullable=False)
    subtype = Column(String(50))
    zone = Column(String(100))
    floor = Column(String(50))

    # Common attributes
    capacity_kw = Column(Float)
    wattage = Column(Float)
    area_sqft = Column(Float)
    occupancy_max = Column(Integer)

    # Flexible attributes
    attributes = Column(JSON, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    building = relationship("Building", back_populates="nodes")
    readings = relationship("Reading", back_populates="node", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("building_id", "node_id", name="uq_building_node"),
        Index("ix_nodes_building_type", "building_id", "node_type"),
    )


class Edge(Base):
    """An edge/connection between nodes."""
    __tablename__ = "edges"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    building_id = Column(String(36), ForeignKey("buildings.id"), nullable=False)
    source_node_id = Column(String(100), nullable=False)
    target_node_id = Column(String(100), nullable=False)
    edge_type = Column(String(50), nullable=False)
    weight = Column(Float, default=1.0)
    bidirectional = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    building = relationship("Building", back_populates="edges")

    __table_args__ = (
        Index("ix_edges_building", "building_id"),
    )


class Reading(Base):
    """Time-series sensor reading."""
    __tablename__ = "readings"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    node_id = Column(String(36), ForeignKey("nodes.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    metric = Column(String(50), default="default")
    unit = Column(String(20))

    # Relationships
    node = relationship("Node", back_populates="readings")

    __table_args__ = (
        Index("ix_readings_node_time", "node_id", "timestamp"),
    )


class TrainedModel(Base):
    """Trained ML model for a tenant."""
    __tablename__ = "trained_models"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)
    building_id = Column(String(36), ForeignKey("buildings.id"))

    name = Column(String(255))
    version = Column(Integer, default=1)
    status = Column(String(50), default="pending")  # pending, training, ready, failed
    model_path = Column(String(500))  # S3 or local path

    # Training metrics
    metrics = Column(JSON, default=dict)  # mape, rmse, etc.
    config = Column(JSON, default=dict)  # hyperparameters

    trained_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="models")

    __table_args__ = (
        Index("ix_models_tenant_status", "tenant_id", "status"),
    )


class AuditLog(Base):
    """Audit trail for important actions."""
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    tenant_id = Column(String(36), ForeignKey("tenants.id"), nullable=False)
    user_id = Column(String(36))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(36))
    details = Column(JSON)
    ip_address = Column(String(45))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_audit_tenant_time", "tenant_id", "created_at"),
    )
