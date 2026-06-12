# demo/supabase_client.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")


def get_client() -> Client:
    """Return a Supabase client instance."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_top_candidates(limit: int = 100):
    """Fetch top ranked candidates from the ranked_submission table."""
    client = get_client()
    res = client.table("ranked_submission").select("*").order("rank").limit(limit).execute()
    return res.data


def search_candidates(min_score: float = 0.0, min_yoe: int = 0, max_yoe: int = 30,
                       open_to_work_only: bool = False, limit: int = 50):
    """Search scored candidates with filters."""
    client = get_client()
    query = (client.table("candidate_scores")
             .select("*")
             .gte("final_score", min_score)
             .gte("yoe", min_yoe)
             .lte("yoe", max_yoe)
             .order("final_score", desc=True)
             .limit(limit))
    if open_to_work_only:
        query = query.eq("open_to_work", True)
    res = query.execute()
    return res.data
