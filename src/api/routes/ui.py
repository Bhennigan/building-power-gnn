"""UI routes for serving HTML templates."""

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import random

from ...db import get_db, crud
from ...auth.dependencies import get_current_user_optional, get_current_user
from ...db.models import User

router = APIRouter()

# Get templates directory relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, user: User = Depends(get_current_user_optional)):
    """Home page - redirect to dashboard if logged in, else login."""
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return RedirectResponse(url="/login", status_code=302)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user: User = Depends(get_current_user_optional)):
    """Login page."""
    if user:
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/logout")
async def logout(request: Request):
    """Logout and redirect to login."""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("access_token")
    return response


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Dashboard page."""
    buildings = crud.get_buildings(db, user.tenant_id)

    # Calculate stats
    total_nodes = 0
    for building in buildings:
        nodes = crud.get_nodes(db, building.id)
        total_nodes += len(nodes)
        building.node_count = len(nodes)

    # Calculate mock efficiency score based on data
    efficiency_score = 0.0
    if buildings:
        # Simple mock: random-ish score based on node count
        efficiency_score = min(95, 50 + (total_nodes * 2) + random.uniform(-5, 10))

    # Calculate mock savings
    estimated_savings = efficiency_score * 15 if efficiency_score > 0 else 0

    # Generate recommendations
    recommendations = []
    if not buildings:
        recommendations.append("Upload your first building data to get started")
    else:
        if total_nodes < 10:
            recommendations.append("Add more equipment nodes for better analysis")
        if efficiency_score < 70:
            recommendations.append("Consider upgrading older HVAC equipment")
        recommendations.append("Schedule regular maintenance for optimal efficiency")
        recommendations.append("Review sensor placement for accurate readings")

    # Generate mock forecast data
    forecast = None
    forecast_labels = []
    forecast_values = []
    if buildings and total_nodes > 0:
        forecast = True
        for i in range(24):
            hour = i
            forecast_labels.append(f"{hour:02d}:00")
            # Simple sinusoidal pattern for demo
            base = 50 + 30 * abs((i - 12) / 12)
            forecast_values.append(round(base + random.uniform(-5, 5), 1))

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "active_page": "dashboard",
        "buildings": buildings,
        "total_nodes": total_nodes,
        "efficiency_score": efficiency_score,
        "estimated_savings": estimated_savings,
        "recommendations": recommendations[:4],
        "forecast": forecast,
        "forecast_labels": forecast_labels,
        "forecast_values": forecast_values,
    })


@router.get("/buildings", response_class=HTMLResponse)
async def buildings_list(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Buildings list page."""
    buildings = crud.get_buildings(db, user.tenant_id)

    for building in buildings:
        stats = crud.get_building_stats(db, building.id)
        building.node_count = stats["total_nodes"]
        building.edge_count = stats["total_edges"]
        building.reading_count = stats["total_readings"]

    return templates.TemplateResponse("buildings.html", {
        "request": request,
        "user": user,
        "active_page": "buildings",
        "buildings": buildings
    })


@router.get("/buildings/{building_id}", response_class=HTMLResponse)
async def building_detail(
    request: Request,
    building_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Building detail page."""
    building = crud.get_building(db, building_id)
    if not building or building.tenant_id != user.tenant_id:
        return RedirectResponse(url="/buildings", status_code=302)

    stats = crud.get_building_stats(db, building_id)
    nodes = crud.get_nodes(db, building_id)
    edges = crud.get_edges(db, building_id)

    return templates.TemplateResponse("building_detail.html", {
        "request": request,
        "user": user,
        "active_page": "buildings",
        "building": building,
        "stats": stats,
        "nodes": nodes,
        "edges": edges
    })


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload data page."""
    buildings = crud.get_buildings(db, user.tenant_id)

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "user": user,
        "active_page": "upload",
        "buildings": buildings
    })


@router.get("/insights", response_class=HTMLResponse)
async def insights_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Insights page."""
    buildings = crud.get_buildings(db, user.tenant_id)

    # Get latest model status
    model = crud.get_latest_model(db, user.tenant_id)

    return templates.TemplateResponse("insights.html", {
        "request": request,
        "user": user,
        "active_page": "insights",
        "buildings": buildings,
        "model": model
    })


@router.get("/integrations", response_class=HTMLResponse)
async def integrations_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Integrations management page."""
    return templates.TemplateResponse("integrations.html", {
        "request": request,
        "user": user,
        "active_page": "integrations",
    })
