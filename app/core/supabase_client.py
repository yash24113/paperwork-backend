from functools import lru_cache

from supabase import Client, create_client

from app.core.config import get_settings


@lru_cache
def get_supabase_client() -> Client:
    """Returns a singleton Supabase client authenticated with the service-role key.

    The service-role key is used because all writes happen from trusted backend
    code, bypassing row level security intentionally.
    """
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_key)
