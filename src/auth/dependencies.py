"""Authentication dependencies for FastAPI."""

from typing import Optional

from fastapi import Depends, HTTPException, status, Request, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ..db import get_db, crud
from ..db.models import User
from .security import decode_token, is_token_expired, TokenData

# HTTP Bearer scheme
security = HTTPBearer(auto_error=False)


async def get_token_from_header_or_cookie(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token: Optional[str] = Cookie(None)
) -> Optional[str]:
    """Extract token from Authorization header or cookie."""
    # Try Authorization header first
    if credentials:
        return credentials.credentials

    # Fall back to cookie
    if access_token:
        return access_token

    return None


async def get_current_user(
    token: Optional[str] = Depends(get_token_from_header_or_cookie),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    token_data = decode_token(token)
    if token_data is None or is_token_expired(token_data):
        raise credentials_exception

    user = crud.get_user(db, token_data.user_id)
    if user is None or not user.is_active:
        raise credentials_exception

    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(get_token_from_header_or_cookie),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""
    if not token:
        return None

    token_data = decode_token(token)
    if token_data is None or is_token_expired(token_data):
        return None

    user = crud.get_user(db, token_data.user_id)
    if user is None or not user.is_active:
        return None

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Ensure user is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def require_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """Require admin role."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


class TenantContext:
    """Context holder for current tenant."""

    def __init__(self, user: User):
        self.tenant_id = user.tenant_id
        self.user_id = user.id
        self.user_role = user.role


async def get_tenant_context(
    current_user: User = Depends(get_current_user)
) -> TenantContext:
    """Get tenant context from current user."""
    return TenantContext(current_user)
