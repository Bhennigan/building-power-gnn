"""CRUD operations for database models."""

from datetime import datetime
from typing import Optional
import re

from sqlalchemy.orm import Session
from sqlalchemy import and_
import pandas as pd

from .models import Tenant, User, Building, Node, Edge, Reading, TrainedModel


# --- Tenant Operations ---

def create_tenant(db: Session, name: str, slug: Optional[str] = None) -> Tenant:
    """Create a new tenant."""
    if not slug:
        slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

    tenant = Tenant(name=name, slug=slug)
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return tenant


def get_tenant(db: Session, tenant_id: str) -> Optional[Tenant]:
    """Get tenant by ID."""
    return db.query(Tenant).filter(Tenant.id == tenant_id).first()


def get_tenant_by_slug(db: Session, slug: str) -> Optional[Tenant]:
    """Get tenant by slug."""
    return db.query(Tenant).filter(Tenant.slug == slug).first()


# --- User Operations ---

def create_user(
    db: Session,
    tenant_id: str,
    email: str,
    password_hash: str,
    name: Optional[str] = None,
    role: str = "member"
) -> User:
    """Create a new user."""
    user = User(
        tenant_id=tenant_id,
        email=email,
        password_hash=password_hash,
        name=name,
        role=role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email."""
    return db.query(User).filter(User.email == email).first()


def get_user(db: Session, user_id: str) -> Optional[User]:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def update_user_login(db: Session, user_id: str) -> None:
    """Update user's last login time."""
    db.query(User).filter(User.id == user_id).update({"last_login": datetime.utcnow()})
    db.commit()


# --- Building Operations ---

def create_building(
    db: Session,
    tenant_id: str,
    name: str,
    **kwargs
) -> Building:
    """Create a new building."""
    building = Building(tenant_id=tenant_id, name=name, **kwargs)
    db.add(building)
    db.commit()
    db.refresh(building)
    return building


def get_buildings(db: Session, tenant_id: str) -> list[Building]:
    """Get all buildings for a tenant."""
    return db.query(Building).filter(Building.tenant_id == tenant_id).all()


def get_building(db: Session, building_id: str) -> Optional[Building]:
    """Get building by ID."""
    return db.query(Building).filter(Building.id == building_id).first()


def delete_building(db: Session, building_id: str) -> bool:
    """Delete a building and all its data."""
    building = get_building(db, building_id)
    if building:
        db.delete(building)
        db.commit()
        return True
    return False


# --- Node Operations ---

def create_node(
    db: Session,
    building_id: str,
    node_id: str,
    node_type: str,
    **kwargs
) -> Node:
    """Create a new node."""
    node = Node(
        building_id=building_id,
        node_id=node_id,
        node_type=node_type,
        **kwargs
    )
    db.add(node)
    db.commit()
    db.refresh(node)
    return node


def get_nodes(db: Session, building_id: str, node_type: Optional[str] = None) -> list[Node]:
    """Get all nodes for a building."""
    query = db.query(Node).filter(Node.building_id == building_id)
    if node_type:
        query = query.filter(Node.node_type == node_type)
    return query.all()


def get_node_by_node_id(db: Session, building_id: str, node_id: str) -> Optional[Node]:
    """Get node by user-defined node_id."""
    return db.query(Node).filter(
        and_(Node.building_id == building_id, Node.node_id == node_id)
    ).first()


def bulk_create_nodes(db: Session, building_id: str, nodes_df: pd.DataFrame) -> int:
    """Bulk create nodes from DataFrame."""
    count = 0
    for _, row in nodes_df.iterrows():
        node_data = row.to_dict()
        node_id = node_data.pop("node_id")
        node_type = node_data.pop("node_type")

        # Extract known columns
        subtype = node_data.pop("subtype", None)
        zone = node_data.pop("zone", None)
        floor = node_data.pop("floor", None)
        capacity_kw = node_data.pop("capacity_kw", None)
        wattage = node_data.pop("wattage", None)
        area_sqft = node_data.pop("area_sqft", None)
        occupancy_max = node_data.pop("occupancy_max", None)

        # Remaining columns go to attributes
        attributes = {k: v for k, v in node_data.items() if pd.notna(v)}

        node = Node(
            building_id=building_id,
            node_id=node_id,
            node_type=node_type,
            subtype=subtype,
            zone=zone,
            floor=floor,
            capacity_kw=capacity_kw,
            wattage=wattage,
            area_sqft=area_sqft,
            occupancy_max=int(occupancy_max) if pd.notna(occupancy_max) else None,
            attributes=attributes
        )
        db.add(node)
        count += 1

    db.commit()
    return count


# --- Edge Operations ---

def create_edge(
    db: Session,
    building_id: str,
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    weight: float = 1.0,
    bidirectional: bool = False
) -> Edge:
    """Create a new edge."""
    edge = Edge(
        building_id=building_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        weight=weight,
        bidirectional=bidirectional
    )
    db.add(edge)
    db.commit()
    db.refresh(edge)
    return edge


def get_edges(db: Session, building_id: str) -> list[Edge]:
    """Get all edges for a building."""
    return db.query(Edge).filter(Edge.building_id == building_id).all()


def bulk_create_edges(db: Session, building_id: str, edges_df: pd.DataFrame) -> int:
    """Bulk create edges from DataFrame."""
    count = 0
    for _, row in edges_df.iterrows():
        edge = Edge(
            building_id=building_id,
            source_node_id=row["source"],
            target_node_id=row["target"],
            edge_type=row["edge_type"],
            weight=float(row.get("weight", 1.0)),
            bidirectional=bool(row.get("bidirectional", False))
        )
        db.add(edge)
        count += 1

    db.commit()
    return count


# --- Reading Operations ---

def create_reading(
    db: Session,
    node_db_id: str,  # Database ID, not user node_id
    timestamp: datetime,
    value: float,
    metric: str = "default",
    unit: Optional[str] = None
) -> Reading:
    """Create a new reading."""
    reading = Reading(
        node_id=node_db_id,
        timestamp=timestamp,
        value=value,
        metric=metric,
        unit=unit
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


def get_readings(
    db: Session,
    node_db_id: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
) -> list[Reading]:
    """Get readings for a node."""
    query = db.query(Reading).filter(Reading.node_id == node_db_id)

    if start_time:
        query = query.filter(Reading.timestamp >= start_time)
    if end_time:
        query = query.filter(Reading.timestamp <= end_time)

    return query.order_by(Reading.timestamp.desc()).limit(limit).all()


def bulk_create_readings(
    db: Session,
    building_id: str,
    readings_df: pd.DataFrame
) -> int:
    """Bulk create readings from DataFrame."""
    # Build node_id -> db_id mapping
    nodes = get_nodes(db, building_id)
    node_map = {n.node_id: n.id for n in nodes}

    count = 0
    for _, row in readings_df.iterrows():
        user_node_id = row["node_id"]
        if user_node_id not in node_map:
            continue

        reading = Reading(
            node_id=node_map[user_node_id],
            timestamp=pd.to_datetime(row["timestamp"]),
            value=float(row["value"]),
            metric=row.get("metric", "default"),
            unit=row.get("unit")
        )
        db.add(reading)
        count += 1

    db.commit()
    return count


# --- Model Operations ---

def create_trained_model(
    db: Session,
    tenant_id: str,
    building_id: Optional[str] = None,
    name: Optional[str] = None
) -> TrainedModel:
    """Create a new model record."""
    model = TrainedModel(
        tenant_id=tenant_id,
        building_id=building_id,
        name=name or f"Model {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def update_model_status(
    db: Session,
    model_id: str,
    status: str,
    model_path: Optional[str] = None,
    metrics: Optional[dict] = None
) -> None:
    """Update model training status."""
    updates = {"status": status}
    if model_path:
        updates["model_path"] = model_path
    if metrics:
        updates["metrics"] = metrics
    if status == "ready":
        updates["trained_at"] = datetime.utcnow()

    db.query(TrainedModel).filter(TrainedModel.id == model_id).update(updates)
    db.commit()


def get_latest_model(db: Session, tenant_id: str, building_id: Optional[str] = None) -> Optional[TrainedModel]:
    """Get the latest ready model for a tenant."""
    query = db.query(TrainedModel).filter(
        and_(TrainedModel.tenant_id == tenant_id, TrainedModel.status == "ready")
    )
    if building_id:
        query = query.filter(TrainedModel.building_id == building_id)

    return query.order_by(TrainedModel.trained_at.desc()).first()


# --- Stats Operations ---

def get_building_stats(db: Session, building_id: str) -> dict:
    """Get statistics for a building."""
    nodes = get_nodes(db, building_id)
    edges = get_edges(db, building_id)

    # Count by type
    nodes_by_type = {}
    for node in nodes:
        nodes_by_type[node.node_type] = nodes_by_type.get(node.node_type, 0) + 1

    edges_by_type = {}
    for edge in edges:
        edges_by_type[edge.edge_type] = edges_by_type.get(edge.edge_type, 0) + 1

    # Count readings
    total_readings = sum(
        db.query(Reading).filter(Reading.node_id == node.id).count()
        for node in nodes
    )

    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "total_readings": total_readings,
        "nodes_by_type": nodes_by_type,
        "edges_by_type": edges_by_type
    }
