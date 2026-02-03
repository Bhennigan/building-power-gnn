"""Authentication module."""

from .security import (
    verify_password,
    get_password_hash,
    create_access_token,
    decode_token,
    Token,
    TokenData
)
from .dependencies import (
    get_current_user,
    get_current_user_optional,
    get_current_active_user,
    require_admin,
    get_tenant_context,
    TenantContext
)
from .routes import router as auth_router

__all__ = [
    "verify_password",
    "get_password_hash",
    "create_access_token",
    "decode_token",
    "Token",
    "TokenData",
    "get_current_user",
    "get_current_user_optional",
    "get_current_active_user",
    "require_admin",
    "get_tenant_context",
    "TenantContext",
    "auth_router",
]
