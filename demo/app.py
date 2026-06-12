# demo/app.py
import streamlit as st
import pandas as pd
import json
import os
from supabase import create_client
from pathlib import Path

st.set_page_config(
    page_title="Redrob AI Ranker",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Redrob AI — Intelligent Candidate Ranker")
st.caption("Demo sandbox for the India Runs Data & AI Challenge")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Sidebar: JD Context ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Job Description")
    st.markdown("""
    **Role:** Senior AI Engineer — Founding Team  
    **Company:** Redrob AI (Series A)  
    **Location:** Pune/Noida, India (Hybrid)  
    **Experience:** 5–9 years  
    
    **Must have:**
    - Production embeddings/retrieval systems
    - Vector DB experience (Pinecone, Qdrant, etc.)
    - Ranking eval frameworks (NDCG, MRR, MAP)
    - Strong Python
    """)
    
    st.divider()
    st.header("⚙️ Scoring Weights")
    st.markdown("""
    | Signal | Weight |
    |--------|--------|
    | Skills match | 35% |
    | Career quality | 25% |
    | Experience fit | 15% |
    | Behavioral signals | 15% |
    | Education | 5% |
    | Assessments | Bonus |
    """)

# ── Main Content ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Run on Sample", "🏆 Full Results (Top 100)", "🔍 Candidate Explorer"])

with tab1:
    st.header("Upload & Rank Sample Candidates")
    uploaded = st.file_uploader("Upload a JSON file with candidates (≤100)", type=["json", "jsonl"])
    
    if uploaded:
        content = uploaded.read().decode("utf-8")
        candidates = []
        
        # Handle both JSON array and JSONL
        try:
            data = json.loads(content)
            candidates = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            for line in content.strip().split("\n"):
                if line.strip():
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        if candidates:
            st.info(f"Loaded {len(candidates)} candidates. Running ranker...")
            
            # Import and run scoring
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from rank import score_candidate
            from scoring.reasoning_generator import generate_reasoning
            
            results = []
            for c in candidates[:100]:  # cap at 100 for sandbox
                score, breakdown = score_candidate(c)
                results.append({
                    "candidate_id": c["candidate_id"],
                    "score": score,
                    "title": c["profile"]["current_title"],
                    "yoe": c["profile"]["years_of_experience"],
                    "location": c["profile"]["location"],
                    "breakdown": breakdown,
                    "candidate": c,
                })
            
            results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
            
            df = pd.DataFrame([{
                "Rank": i + 1,
                "Candidate ID": r["candidate_id"],
                "Score": f"{r['score']:.3f}",
                "Title": r["title"],
                "YoE": r["yoe"],
                "Location": r["location"],
                "Skills": f"{r['breakdown'].get('skills', 0):.2f}",
                "Career": f"{r['breakdown'].get('career', 0):.2f}",
                "Behavioral": f"{r['breakdown'].get('behavioral', 0):.2f}",
                "Honeypot": "⚠️" if r["breakdown"].get("honeypot") else "✅",
            } for i, r in enumerate(results)])
            
            st.dataframe(df, use_container_width=True)
            
            # Download
            import csv, io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for i, r in enumerate(results[:100], start=1):
                reasoning = generate_reasoning(r["candidate"], r["breakdown"], i)
                writer.writerow([r["candidate_id"], i, f"{r['score']:.4f}", reasoning])
            
            st.download_button(
                "⬇️ Download Ranked CSV",
                output.getvalue().encode("utf-8"),
                "ranked_output.csv",
                "text/csv"
            )

with tab2:
    st.header("Full Submission — Top 100 Candidates")
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = get_supabase()
        try:
            res = supabase.table("ranked_submission").select("*").order("rank").execute()
            if res.data:
                df = pd.DataFrame(res.data)
                st.dataframe(df[["rank", "candidate_id", "score", "reasoning"]], use_container_width=True)
            else:
                st.warning("No ranked submission found. Run rank.py first and upload results.")
        except Exception as e:
            st.error(f"Supabase error: {e}")
    else:
        st.warning("Set SUPABASE_URL and SUPABASE_KEY in environment to load stored results.")

with tab3:
    st.header("Explore Scored Candidates")
    st.markdown("Filter and explore from the full 100K pool (Supabase-backed).")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Min score", 0.0, 1.0, 0.6, 0.01)
    with col2:
        min_yoe = st.number_input("Min YoE", 0, 20, 4)
    with col3:
        max_yoe = st.number_input("Max YoE", 0, 30, 12)
    
    open_only = st.checkbox("Open to work only", value=False)
    
    if st.button("🔍 Search"):
        if SUPABASE_URL and SUPABASE_KEY:
            supabase = get_supabase()
            query = (supabase.table("candidate_scores")
                     .select("*")
                     .gte("final_score", min_score)
                     .gte("yoe", min_yoe)
                     .lte("yoe", max_yoe)
                     .order("final_score", desc=True)
                     .limit(50))
            if open_only:
                query = query.eq("open_to_work", True)
            
            res = query.execute()
            if res.data:
                df = pd.DataFrame(res.data)
                st.dataframe(
                    df[["candidate_id", "final_score", "title", "yoe", "location", 
                        "skills_score", "career_score", "behavioral_score", "is_honeypot"]],
                    use_container_width=True
                )
                st.caption(f"Showing top 50 of matching candidates")
            else:
                st.info("No candidates match these filters.")
        else:
            st.warning("Connect Supabase to use this feature.")
