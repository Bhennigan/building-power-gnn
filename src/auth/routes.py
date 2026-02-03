"""Authentication routes."""

from fastapi import APIRouter, Depends, HTTPException, status, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from ..db import get_db, crud
from .security import verify_password, get_password_hash, create_access_token, Token

router = APIRouter()


class RegisterRequest(BaseModel):
    """Registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=1)
    company_name: str = Field(..., min_length=1)


class LoginRequest(BaseModel):
    """Login request."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User response."""
    id: str
    email: str
    name: str | None
    role: str
    tenant_id: str
    tenant_name: str


@router.post("/register", response_model=Token)
async def register(
    request: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """Register a new user and tenant."""
    # Check if email exists
    existing = crud.get_user_by_email(db, request.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create tenant
    tenant = crud.create_tenant(db, request.company_name)

    # Create user
    password_hash = get_password_hash(request.password)
    user = crud.create_user(
        db,
        tenant_id=tenant.id,
        email=request.email,
        password_hash=password_hash,
        name=request.name,
        role="admin"  # First user is admin
    )

    # Create token
    token = create_access_token(
        user_id=user.id,
        tenant_id=tenant.id,
        email=user.email,
        role=user.role
    )

    # Set cookie
    response.set_cookie(
        key="access_token",
        value=token.access_token,
        httponly=True,
        max_age=token.expires_in,
        samesite="lax"
    )

    return token


@router.post("/login", response_model=Token)
async def login(
    request: LoginRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """Login and get access token."""
    user = crud.get_user_by_email(db, request.email)

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )

    # Update last login
    crud.update_user_login(db, user.id)

    # Create token
    token = create_access_token(
        user_id=user.id,
        tenant_id=user.tenant_id,
        email=user.email,
        role=user.role
    )

    # Set cookie
    response.set_cookie(
        key="access_token",
        value=token.access_token,
        httponly=True,
        max_age=token.expires_in,
        samesite="lax"
    )

    return token


@router.post("/logout")
async def logout(response: Response):
    """Logout and clear cookie."""
    response.delete_cookie("access_token")
    return {"message": "Logged out"}


@router.get("/me", response_model=UserResponse)
async def get_me(
    db: Session = Depends(get_db),
    current_user = Depends(lambda: None)  # Will be replaced
):
    """Get current user info."""
    from .dependencies import get_current_user

    # Re-import to avoid circular dependency
    user = await get_current_user(
        token=None,  # Will be populated by dependency
        db=db
    )

    tenant = crud.get_tenant(db, user.tenant_id)

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        tenant_id=user.tenant_id,
        tenant_name=tenant.name if tenant else ""
    )
