"""
dependencies.py — FastAPI dependency injection.
Provides current_user, supabase client, and admin guards.
"""
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.database import get_supabase
from app.utils.jwt import verify_access_token
from app.models.user import UserProfile
import structlog

log = structlog.get_logger()

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
    supabase=Depends(get_supabase),
) -> UserProfile:
    """
    Dependency that extracts and validates the JWT bearer token.
    Raises 401 if token is missing or invalid.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    user_id = verify_access_token(token)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Fetch profile from DB
    try:
        result = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found.",
            )
        profile = result.data

        # Also get email from auth.users via Supabase admin
        from app.database import get_supabase_admin
        admin = get_supabase_admin()
        auth_user = admin.auth.admin.get_user_by_id(user_id)
        email = auth_user.user.email if auth_user.user else ""

        return UserProfile(
            id=profile["id"],
            email=email,
            full_name=profile.get("full_name"),
            age=profile.get("age"),
            gender=profile.get("gender"),
            blood_type=profile.get("blood_type"),
            created_at=profile.get("created_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error("auth.profile_fetch_failed", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load user profile.",
        )


# Type alias for clean route signatures
CurrentUser = Annotated[UserProfile, Depends(get_current_user)]
SupabaseClient = Annotated[object, Depends(get_supabase)]


def get_pagination(skip: int = 0, limit: int = 20):
    """Pagination dependency with sane defaults."""
    limit = min(limit, 100)
    return {"skip": skip, "limit": limit}


Pagination = Annotated[dict, Depends(get_pagination)]
