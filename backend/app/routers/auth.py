"""
routers/auth.py — Authentication endpoints.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from app.database import get_supabase
from app.dependencies import CurrentUser, SupabaseClient
from app.models.user import (
    UserRegisterRequest, UserLoginRequest, UserProfileUpdate,
    TokenResponse, UserProfile, RefreshTokenRequest,
)
from app.utils.jwt import create_access_token, create_refresh_token, verify_refresh_token
from app.config import settings
import structlog
import uuid
from datetime import datetime, timezone

log = structlog.get_logger()
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: UserRegisterRequest, supabase=Depends(get_supabase)):
    """
    Register a new user.
    1. Create Supabase Auth user
    2. Create profile row
    3. Return JWT tokens
    """
    from app.database import get_supabase_admin
    admin = get_supabase_admin()

    # Check if email already exists
    try:
        existing = admin.auth.admin.list_users()
        if any(u.email == request.email for u in existing):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # list_users may fail on some plans — proceed

    # Create auth user
    try:
        auth_response = admin.auth.admin.create_user({
            "email": request.email,
            "password": request.password,
            "email_confirm": True,  # auto-confirm for dev
        })
        user_id = auth_response.user.id
    except Exception as e:
        log.error("auth.register_failed", email=request.email, error=str(e))
        error_msg = str(e).lower()
        if "already" in error_msg or "duplicate" in error_msg:
            raise HTTPException(status_code=409, detail="Email already registered.")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

    # Create profile (use admin client to bypass RLS on insert)
    try:
        admin.table("profiles").insert({
            "id": user_id,
            "full_name": request.full_name,
            "age": request.age,
            "gender": request.gender,
            "blood_type": request.blood_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        log.error("auth.profile_create_failed", user_id=user_id, error=str(e))
        # Clean up auth user if profile creation failed
        try:
            admin.auth.admin.delete_user(user_id)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail="Failed to create user profile.")

    access_token = create_access_token(user_id, request.email)
    refresh_token = create_refresh_token(user_id)

    log.info("auth.register_success", user_id=user_id, email=request.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_expiry_minutes * 60,
        user=UserProfile(
            id=user_id,
            email=request.email,
            full_name=request.full_name,
            age=request.age,
            gender=request.gender,
            blood_type=request.blood_type,
        ),
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: UserLoginRequest, supabase=Depends(get_supabase)):
    """Authenticate with email + password, return JWT tokens."""
    try:
        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password,
        })
        if not response.user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        user = response.user
        user_id = user.id

        # Load profile
        profile_result = supabase.table("profiles").select("*").eq("id", user_id).execute()
        profile = profile_result.data[0] if profile_result.data else {}

        access_token = create_access_token(user_id, user.email)
        refresh_token = create_refresh_token(user_id)

        log.info("auth.login_success", user_id=user_id)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt_expiry_minutes * 60,
            user=UserProfile(
                id=user_id,
                email=user.email,
                full_name=profile.get("full_name"),
                age=profile.get("age"),
                gender=profile.get("gender"),
                blood_type=profile.get("blood_type"),
                created_at=profile.get("created_at"),
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        log.warning("auth.login_failed", email=request.email, error=str(e))
        raise HTTPException(status_code=401, detail="Invalid email or password.")


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest, supabase=Depends(get_supabase)):
    """Exchange a refresh token for a new access token."""
    user_id = verify_refresh_token(request.refresh_token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token.")

    # Get user info
    from app.database import get_supabase_admin
    admin = get_supabase_admin()
    try:
        auth_user = admin.auth.admin.get_user_by_id(user_id)
        email = auth_user.user.email
    except Exception as e:
        raise HTTPException(status_code=401, detail="User not found.")

    profile_result = supabase.table("profiles").select("*").eq("id", user_id).execute()
    profile = profile_result.data[0] if profile_result.data else {}

    new_access = create_access_token(user_id, email)
    new_refresh = create_refresh_token(user_id)

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=settings.jwt_expiry_minutes * 60,
        user=UserProfile(
            id=user_id,
            email=email,
            full_name=profile.get("full_name"),
            age=profile.get("age"),
            gender=profile.get("gender"),
            blood_type=profile.get("blood_type"),
            created_at=profile.get("created_at"),
        ),
    )


@router.get("/me", response_model=UserProfile)
async def get_me(current_user: CurrentUser):
    """Get the currently authenticated user's profile."""
    return current_user


@router.put("/profile", response_model=UserProfile)
async def update_profile(
    request: UserProfileUpdate,
    current_user: CurrentUser,
    supabase: SupabaseClient,
):
    """Update the current user's profile."""
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    if not updates:
        return current_user

    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        result = supabase.table("profiles").update(updates).eq("id", current_user.id).execute()
        updated = result.data[0]
        return UserProfile(
            id=updated["id"],
            email=current_user.email,
            full_name=updated.get("full_name"),
            age=updated.get("age"),
            gender=updated.get("gender"),
            blood_type=updated.get("blood_type"),
            created_at=updated.get("created_at"),
        )
    except Exception as e:
        log.error("auth.profile_update_failed", user_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update profile.")


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(current_user: CurrentUser, supabase: SupabaseClient):
    """
    Sign out the current user.
    JWT tokens are stateless so the client must discard them.
    We also call Supabase sign_out to invalidate server-side sessions.
    """
    try:
        supabase.auth.sign_out()
    except Exception:
        pass  # Best-effort
    log.info("auth.logout", user_id=current_user.id)
