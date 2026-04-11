"""
database.py — Supabase client singleton.
Uses service-role key for server-side operations (bypasses RLS when needed).
"""
from supabase import create_client, Client
from app.config import settings
import structlog

log = structlog.get_logger()

_supabase_client: Client | None = None
_supabase_admin: Client | None = None


def get_supabase() -> Client:
    """
    Anon client — respects Row Level Security.
    Use for user-scoped queries where JWT is forwarded.
    """
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_anon_key,
        )
        log.info("supabase.anon_client_initialized")
    return _supabase_client


def get_supabase_admin() -> Client:
    """
    Service-role client — bypasses RLS.
    Use ONLY for admin operations (ingestion, background jobs).
    Never expose this to user-controlled input paths.
    """
    global _supabase_admin
    if _supabase_admin is None:
        _supabase_admin = create_client(
            settings.supabase_url,
            settings.supabase_service_key,
        )
        log.info("supabase.admin_client_initialized")
    return _supabase_admin
