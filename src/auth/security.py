"""Security utilities for authentication."""

from datetime import datetime, timedelta
from typing import Optional
import os

from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenData(BaseModel):
    """JWT token payload."""
    user_id: str
    tenant_id: str
    email: str
    role: str
    exp: Optional[datetime] = None


class Token(BaseModel):
    """Token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(
    user_id: str,
    tenant_id: str,
    email: str,
    role: str,
    expires_delta: Optional[timedelta] = None
) -> Token:
    """Create a JWT access token."""
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    to_encode = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "email": email,
        "role": role,
        "exp": expire
    }

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return Token(
        access_token=encoded_jwt,
        expires_in=int((expire - datetime.utcnow()).total_seconds())
    )


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenData(
            user_id=payload.get("sub"),
            tenant_id=payload.get("tenant_id"),
            email=payload.get("email"),
            role=payload.get("role"),
            exp=datetime.fromtimestamp(payload.get("exp"))
        )
    except JWTError:
        return None


def is_token_expired(token_data: TokenData) -> bool:
    """Check if token is expired."""
    if token_data.exp is None:
        return True
    return datetime.utcnow() > token_data.exp
