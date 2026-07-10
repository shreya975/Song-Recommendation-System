import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SONIX AI — Next Generation Music Discovery",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# LOAD ML ARTIFACTS  (same pipeline as the training notebook —
# NearestNeighbors + StandardScaler + cosine distance, unchanged)
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("spotify_recommender.pkl")
    df = joblib.load("spotify_dataset.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_columns.pkl")
    X_scaled = scaler.transform(df[features])
    return model, df, scaler, features, X_scaled

try:
    model, df, scaler, FEATURES, X_SCALED = load_artifacts()
    ARTIFACTS_OK = True
    LOAD_ERROR = None
except Exception as e:
    ARTIFACTS_OK = False
    LOAD_ERROR = str(e)
    df = pd.DataFrame()

# ============================================================
# RECOMMENDATION LOGIC — identical to the notebook's recommend_songs()
# ============================================================
def recommend_songs(song_name: str, n_recommendations: int = 6):
    matches = df[df["track_name"].str.lower() == song_name.lower()]
    if matches.empty:
        return None, None

    idx = matches.index[0]
    row_pos = df.index.get_loc(idx)

    distances, indices = model.kneighbors(
        [X_SCALED[row_pos]],
        n_neighbors=n_recommendations + 1,
    )

    recs = []
    for dist, i in zip(distances[0][1:], indices[0][1:]):
        r = df.iloc[i]
        recs.append({
            "track_name": r.get("track_name", "Unknown"),
            "artists": r.get("artists", "Unknown Artist"),
            "album_name": r.get("album_name", "Unknown Album"),
            "track_genre": r.get("track_genre", "—"),
            "popularity": r.get("popularity", 0),
            "energy": r.get("energy", 0),
            "danceability": r.get("danceability", 0),
            "valence": r.get("valence", 0),
            "tempo": r.get("tempo", 0),
            "similarity": round(1 - dist, 3),
        })
    return matches.iloc[0], recs


def mood_tags(rec):
    tags = []
    if rec["energy"] > 0.7:
        tags.append("⚡ High Energy")
    if rec["danceability"] > 0.7:
        tags.append("🕺 Party Ready")
    if rec["valence"] > 0.7:
        tags.append("😊 Happy Mood")
    if rec["valence"] < 0.35 and rec["energy"] < 0.45:
        tags.append("🌙 Late Night")
    if rec["acousticness"] > 0.6 if "acousticness" in rec else False:
        tags.append("🎻 Acoustic Chill")
    if rec["energy"] > 0.6 and rec["tempo"] > 120:
        tags.append("🏋️ Workout Friendly")
    if not tags:
        tags.append("✨ Balanced Vibe")
    return tags[:3]


# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "page": "Home",
    "search_history": [],
    "favorites": set(),
    "last_query": "",
    "results": None,
    "searched_song": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# GLOBAL CSS — DESIGN SYSTEM
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
  --bg:#07070b;
  --card:rgba(255,255,255,0.045);
  --card-border:rgba(255,255,255,0.09);
  --green:#1ed760;
  --cyan:#3df5ff;
  --purple:#a06bff;
  --pink:#ff5fb1;
  --text:#f5f6fa;
  --muted:#9a9cae;
}

/* ---- Hide default Streamlit chrome ---- */
#MainMenu, header, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  display:none !important;
}
.block-container{
  padding-top:0rem !important;
  padding-bottom:2rem !important;
  max-width:1200px !important;
}
[data-testid="stAppViewContainer"]{background:transparent !important;}
[data-testid="stHeader"]{background:transparent !important;}

html, body, [class*="css"]{
  font-family:'Manrope', sans-serif;
  color:var(--text);
}
h1,h2,h3,h4{font-family:'Space Grotesk', sans-serif !important;}

/* ---- Animated aurora background ---- */
.stApp{
  background: var(--bg);
  background-image:
    radial-gradient(circle at 15% 20%, rgba(30,215,96,0.20), transparent 40%),
    radial-gradient(circle at 85% 15%, rgba(160,107,255,0.22), transparent 45%),
    radial-gradient(circle at 50% 85%, rgba(61,245,255,0.16), transparent 45%),
    radial-gradient(circle at 90% 80%, rgba(255,95,177,0.16), transparent 40%);
  background-attachment:fixed;
  animation: auroraShift 18s ease-in-out infinite alternate;
}
@keyframes auroraShift{
  0%{background-position:0% 0%, 100% 0%, 50% 100%, 100% 100%;}
  100%{background-position:10% 10%, 90% 20%, 40% 90%, 85% 70%;}
}

/* ---- floating orbs ---- */
.orb{position:fixed; border-radius:50%; filter:blur(60px); opacity:0.35; z-index:0; pointer-events:none;}
.orb1{width:280px;height:280px;background:var(--green);top:8%;left:6%;animation:float1 12s ease-in-out infinite;}
.orb2{width:220px;height:220px;background:var(--purple);top:60%;right:8%;animation:float2 14s ease-in-out infinite;}
.orb3{width:180px;height:180px;background:var(--cyan);bottom:5%;left:35%;animation:float1 16s ease-in-out infinite;}
@keyframes float1{0%,100%{transform:translateY(0px);}50%{transform:translateY(-30px);}}
@keyframes float2{0%,100%{transform:translateY(0px);}50%{transform:translateY(30px);}}

/* ---- nav bar ---- */
.sonix-nav{
  display:flex; align-items:center; justify-content:space-between;
  padding:16px 26px; margin:10px 0 28px 0;
  background:rgba(255,255,255,0.045);
  border:1px solid var(--card-border);
  border-radius:20px;
  backdrop-filter: blur(18px);
  position:sticky; top:10px; z-index:999;
}
.sonix-logo{font-family:'Space Grotesk',sans-serif; font-weight:800; font-size:22px;
  background:linear-gradient(90deg, var(--green), var(--cyan));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;}

/* ---- nav buttons (streamlit buttons restyled) ---- */
div[data-testid="column"] .stButton>button{
  background:transparent !important;
  border:1px solid transparent !important;
  color:var(--muted) !important;
  border-radius:12px !important;
  font-weight:600 !important;
  font-size:14px !important;
  padding:8px 16px !important;
  transition:all .25s ease !important;
}
div[data-testid="column"] .stButton>button:hover{
  color:var(--text) !important;
  background:rgba(255,255,255,0.06) !important;
  border:1px solid var(--card-border) !important;
  transform:translateY(-1px);
}

/* ---- hero ---- */
.hero{text-align:center; padding:70px 10px 40px 10px;}
.hero h1{
  font-size:64px; font-weight:800; line-height:1.05; margin-bottom:14px;
  background:linear-gradient(90deg, var(--green), var(--cyan) 45%, var(--purple) 80%, var(--pink));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-size:200% auto; animation:shine 6s linear infinite;
}
@keyframes shine{to{background-position:200% center;}}
.hero p.tagline{color:var(--muted); font-size:19px; max-width:620px; margin:0 auto 30px auto; line-height:1.55;}

/* ---- glass card ---- */
.glass{
  background:var(--card);
  border:1px solid var(--card-border);
  border-radius:20px;
  padding:22px;
  backdrop-filter: blur(16px);
  transition: all .3s ease;
}
.glass:hover{
  transform:translateY(-4px);
  border-color:rgba(30,215,96,0.45);
  box-shadow:0 12px 40px rgba(30,215,96,0.12);
}

/* ---- stat cards ---- */
.stat-card{
  text-align:center; padding:26px 10px;
  background:var(--card); border:1px solid var(--card-border);
  border-radius:18px; backdrop-filter:blur(14px);
}
.stat-num{font-family:'Space Grotesk',sans-serif; font-size:32px; font-weight:700; color:var(--text);}
.stat-label{color:var(--muted); font-size:13px; letter-spacing:.06em; text-transform:uppercase; margin-top:4px;}

/* ---- song card ---- */
.song-card{
  background:linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border:1px solid var(--card-border);
  border-radius:18px;
  padding:18px;
  position:relative;
  overflow:hidden;
  transition: all .3s ease;
  height:100%;
}
.song-card:hover{
  transform:translateY(-6px) scale(1.01);
  border-color:rgba(61,245,255,0.5);
  box-shadow:0 16px 40px rgba(61,245,255,0.15);
}
.cover{
  width:100%; aspect-ratio:1/1; border-radius:14px; margin-bottom:12px;
  background:linear-gradient(135deg, var(--green), var(--cyan), var(--purple));
  display:flex; align-items:center; justify-content:center; font-size:34px;
  animation: coverPulse 5s ease-in-out infinite;
}
@keyframes coverPulse{0%,100%{filter:brightness(1);}50%{filter:brightness(1.18);}}
.song-title{font-weight:700; font-size:16px; margin-bottom:2px; color:var(--text);}
.song-artist{color:var(--muted); font-size:13px; margin-bottom:10px;}
.song-meta{display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px;}
.pill{
  font-size:11px; padding:4px 9px; border-radius:999px;
  background:rgba(255,255,255,0.07); border:1px solid var(--card-border); color:var(--muted);
}
.pill-green{color:var(--green); border-color:rgba(30,215,96,0.4);}
.match-badge{
  position:absolute; top:14px; right:14px; font-size:11px; font-weight:700;
  padding:4px 10px; border-radius:999px; color:#00120a;
  background:linear-gradient(90deg, var(--green), var(--cyan));
}

/* ---- insight card ---- */
.insight{
  padding:14px 16px; border-radius:14px; font-size:13px; font-weight:600;
  background:rgba(255,255,255,0.05); border:1px solid var(--card-border);
  text-align:center; color:var(--text);
}

/* ---- section headers ---- */
.section-title{font-size:26px; font-weight:800; margin:10px 0 18px 0; font-family:'Space Grotesk',sans-serif;}
.section-sub{color:var(--muted); margin-bottom:20px; font-size:14px;}

/* ---- inputs ---- */
.stTextInput>div>div>input{
  background:rgba(255,255,255,0.055) !important;
  border:1px solid var(--card-border) !important;
  border-radius:16px !important;
  color:var(--text) !important;
  padding:14px 18px !important;
  font-size:15px !important;
}
.stTextInput>div>div>input:focus{
  border-color:var(--green) !important;
  box-shadow:0 0 0 3px rgba(30,215,96,0.18) !important;
}

/* ---- primary CTA buttons ---- */
.stButton>button[kind="primary"], .cta .stButton>button{
  background:linear-gradient(90deg, var(--green), var(--cyan)) !important;
  color:#001a0d !important; font-weight:800 !important;
  border:none !important; border-radius:14px !important;
  padding:12px 26px !important; font-size:15px !important;
  box-shadow:0 8px 24px rgba(30,215,96,0.25) !important;
  transition: all .25s ease !important;
}
.cta .stButton>button:hover{transform:translateY(-2px) scale(1.02); box-shadow:0 12px 32px rgba(30,215,96,0.4) !important;}

/* ---- fake player ---- */
.player{
  position:sticky; bottom:14px; margin-top:36px;
  background:rgba(15,15,20,0.75); border:1px solid var(--card-border);
  border-radius:20px; padding:16px 24px; backdrop-filter:blur(20px);
  display:flex; align-items:center; gap:18px;
}
.eq{display:flex; align-items:flex-end; gap:3px; height:22px;}
.eq span{width:4px; background:var(--green); border-radius:2px; animation:eqbar 1s ease-in-out infinite;}
.eq span:nth-child(2){animation-delay:.15s; background:var(--cyan);}
.eq span:nth-child(3){animation-delay:.3s; background:var(--purple);}
.eq span:nth-child(4){animation-delay:.45s; background:var(--pink);}
@keyframes eqbar{0%,100%{height:6px;}50%{height:22px;}}
.progress-track{height:5px; background:rgba(255,255,255,0.1); border-radius:6px; flex:1; overflow:hidden;}
.progress-fill{height:100%; width:38%; background:linear-gradient(90deg, var(--green), var(--cyan)); border-radius:6px;}

/* misc text */
.muted{color:var(--muted);}
hr{border-color:var(--card-border) !important;}
</style>

<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
""", unsafe_allow_html=True)


# ============================================================
# NAV BAR
# ============================================================
def nav_bar():
    st.markdown('<div class="sonix-nav">', unsafe_allow_html=True)
    cols = st.columns([2, 1, 1, 1, 1, 1])
    with cols[0]:
        st.markdown('<div class="sonix-logo">🎵 SONIX AI™</div>', unsafe_allow_html=True)
    pages = ["Home", "Discover", "Analytics", "About"]
    for i, p in enumerate(pages):
        with cols[i + 1]:
            if st.button(p, key=f"nav_{p}", use_container_width=True):
                st.session_state.page = p
    st.markdown('</div>', unsafe_allow_html=True)

nav_bar()

# ============================================================
# HOME PAGE
# ============================================================
def render_home():
    st.markdown("""
    <div class="hero">
        <h1>SONIX AI</h1>
        <p class="tagline">Discover your next favorite song using artificial intelligence,
        audio-feature analysis and intelligent similarity search — trained on real Spotify data.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 0.6, 1])
    with c2:
        st.markdown('<div class="cta">', unsafe_allow_html=True)
        if st.button("🚀 Start Discovering", use_container_width=True):
            st.session_state.page = "Discover"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    n_songs = f"{len(df):,}" if ARTIFACTS_OK else "114,000+"
    n_genres = df["track_genre"].nunique() if ARTIFACTS_OK else 114
    n_artists = f"{df['artists'].nunique():,}" if ARTIFACTS_OK else "10,000+"

    stats = [
        (n_songs, "Songs Indexed"),
        (str(n_genres), "Genres"),
        (n_artists, "Artists"),
        ("Cosine kNN", "AI Engine"),
    ]
    cols = st.columns(4)
    for col, (num, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num">{num}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-title">Why SONIX AI</div>', unsafe_allow_html=True)
    feat_cols = st.columns(3)
    feats = [
        ("🧠", "Audio Intelligence", "14 audio features — energy, valence, tempo, acousticness — standardized and compared."),
        ("🎯", "Nearest-Neighbor Search", "Cosine-similarity kNN finds the songs closest to your taste in feature space."),
        ("⚡", "Instant Results", "Cached model + vectorized scaling deliver recommendations in milliseconds."),
    ]
    for col, (icon, title, desc) in zip(feat_cols, feats):
        with col:
            st.markdown(f"""
            <div class="glass">
                <div style="font-size:28px;margin-bottom:8px;">{icon}</div>
                <div style="font-weight:700;font-size:16px;margin-bottom:6px;">{title}</div>
                <div class="muted" style="font-size:13px;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# DISCOVER PAGE
# ============================================================
def render_discover():
    st.markdown('<div class="section-title">Find Your Next Favorite Song</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Type an exact track name from the catalog to get AI-powered recommendations.</div>', unsafe_allow_html=True)

    if not ARTIFACTS_OK:
        st.error(f"Model artifacts not found next to app.py: {LOAD_ERROR}\n\n"
                  f"Make sure spotify_recommender.pkl, spotify_dataset.pkl, scaler.pkl and "
                  f"feature_columns.pkl are in the same folder as app.py.")
        return

    c1, c2 = st.columns([4, 1])
    with c1:
        query = st.text_input("search", value=st.session_state.last_query,
                               placeholder="🔍  Try: Shape of You, Believer, Blinding Lights...",
                               label_visibility="collapsed")
    with c2:
        surprise = st.button("🎲 Surprise Me", use_container_width=True)

    if surprise:
        query = df["track_name"].dropna().sample(1).values[0]
        st.session_state.last_query = query

    go = st.button("✨ Get Recommendations", use_container_width=False)

    if go or surprise:
        if not query.strip():
            st.warning("Type a song name first.")
        else:
            steps = [
                "🎵 Searching song...",
                "🧠 Understanding audio features...",
                "🎧 Finding similar music...",
                "✨ Generating personalized recommendations...",
            ]
            ph = st.empty()
            for s in steps:
                ph.markdown(f"<div class='glass' style='text-align:center;font-weight:600;'>{s}</div>",
                            unsafe_allow_html=True)
                time.sleep(0.28)
            ph.empty()

            original, recs = recommend_songs(query, n_recommendations=6)
            st.session_state.last_query = query
            if original is None:
                st.error(f"'{query}' wasn't found in the catalog. Try an exact track title.")
                st.session_state.results = None
            else:
                st.session_state.results = recs
                st.session_state.searched_song = original
                if query not in st.session_state.search_history:
                    st.session_state.search_history.insert(0, query)
                    st.session_state.search_history = st.session_state.search_history[:8]

    if st.session_state.results:
        original = st.session_state.searched_song
        st.markdown(f"""
        <div class="glass" style="margin:20px 0;">
            <div class="muted" style="font-size:12px;letter-spacing:.05em;text-transform:uppercase;">Seed Track</div>
            <div style="font-size:20px;font-weight:800;">{original['track_name']}</div>
            <div class="muted">{original['artists']} · {original.get('album_name','')} · {original.get('track_genre','')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="font-size:20px;">Recommended For You</div>', unsafe_allow_html=True)
        recs = st.session_state.results
        cols = st.columns(3)
        for i, rec in enumerate(recs):
            with cols[i % 3]:
                tags = mood_tags(rec)
                pills = "".join(f'<span class="pill">{t}</span>' for t in tags)
                st.markdown(f"""
                <div class="song-card">
                    <div class="match-badge">{int(rec['similarity']*100)}% match</div>
                    <div class="cover">🎵</div>
                    <div class="song-title">{rec['track_name']}</div>
                    <div class="song-artist">{rec['artists']}</div>
                    <div class="song-meta">
                        <span class="pill pill-green">Pop {int(rec['popularity'])}</span>
                        <span class="pill">Energy {rec['energy']:.2f}</span>
                        <span class="pill">Dance {rec['danceability']:.2f}</span>
                        <span class="pill">Valence {rec['valence']:.2f}</span>
                    </div>
                    <div class="song-meta">{pills}</div>
                </div>
                """, unsafe_allow_html=True)
                bc1, bc2, bc3 = st.columns(3)
                key_base = f"{rec['track_name']}_{i}"
                with bc1:
                    st.button("▶ Play", key=f"play_{key_base}", use_container_width=True)
                with bc2:
                    liked = key_base in st.session_state.favorites
                    if st.button("♥ Liked" if liked else "♡ Like", key=f"like_{key_base}", use_container_width=True):
                        if liked:
                            st.session_state.favorites.discard(key_base)
                        else:
                            st.session_state.favorites.add(key_base)
                        st.rerun()
                with bc3:
                    st.button("＋ Save", key=f"save_{key_base}", use_container_width=True)

        csv = pd.DataFrame(recs).to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Recommendations (CSV)", csv, "sonix_recommendations.csv", "text/csv")

    if st.session_state.search_history:
        st.write("")
        st.markdown('<div class="section-title" style="font-size:18px;">Recently Searched</div>', unsafe_allow_html=True)
        hist_cols = st.columns(len(st.session_state.search_history))
        for col, h in zip(hist_cols, st.session_state.search_history):
            with col:
                if st.button(h, key=f"hist_{h}", use_container_width=True):
                    st.session_state.last_query = h
                    st.rerun()

    # fake player
    st.markdown("""
    <div class="player">
        <div class="cover" style="width:52px;height:52px;margin:0;font-size:20px;">🎵</div>
        <div style="min-width:160px;">
            <div style="font-weight:700;font-size:14px;">Now Playing Preview</div>
            <div class="muted" style="font-size:12px;">SONIX AI Radio</div>
        </div>
        <div class="progress-track"><div class="progress-fill"></div></div>
        <div class="eq"><span></span><span></span><span></span><span></span></div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ANALYTICS PAGE
# ============================================================
def render_analytics():
    st.markdown('<div class="section-title">Catalog Analytics</div>', unsafe_allow_html=True)
    if not ARTIFACTS_OK:
        st.error("Dataset not loaded — analytics unavailable.")
        return

    plotly_theme = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f5f6fa", family="Manrope"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    palette = ["#1ed760", "#3df5ff", "#a06bff", "#ff5fb1"]

    c1, c2 = st.columns(2)
    with c1:
        top_genres = df["track_genre"].value_counts().head(10).reset_index()
        top_genres.columns = ["genre", "count"]
        fig = px.bar(top_genres, x="genre", y="count", color="count",
                     color_continuous_scale=palette, title="Top 10 Genres")
        fig.update_layout(**plotly_theme)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(df, x="popularity", nbins=30, title="Popularity Distribution",
                            color_discrete_sequence=[palette[1]])
        fig.update_layout(**plotly_theme)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        sample = df.sample(min(3000, len(df)), random_state=1)
        fig = px.scatter(sample, x="energy", y="valence", color="track_genre",
                          opacity=0.55, title="Energy vs Valence")
        fig.update_layout(**plotly_theme, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(df, x="tempo", nbins=30, title="Tempo Distribution",
                            color_discrete_sequence=[palette[2]])
        fig.update_layout(**plotly_theme)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title" style="font-size:20px;">Audio Feature Radar (Catalog Average)</div>', unsafe_allow_html=True)
    radar_feats = ["danceability", "energy", "valence", "acousticness", "liveness", "speechiness"]
    avg = df[radar_feats].mean().tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=avg + [avg[0]], theta=radar_feats + [radar_feats[0]],
                                   fill="toself", line_color="#1ed760"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                  radialaxis=dict(visible=True, color="#9a9cae")),
                       **plotly_theme, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    top_artists = df["artists"].value_counts().head(8).reset_index()
    top_artists.columns = ["artist", "count"]
    fig = px.pie(top_artists, names="artist", values="count", title="Top Artists Share",
                 color_discrete_sequence=palette)
    fig.update_layout(**plotly_theme)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ABOUT PAGE
# ============================================================
def render_about():
    st.markdown('<div class="section-title">About SONIX AI</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
    <p>SONIX AI is a music discovery engine built on a <b>Nearest Neighbors</b> model with
    <b>cosine similarity</b> over 14 standardized audio features
    (popularity, duration, danceability, energy, key, loudness, mode, speechiness,
    acousticness, instrumentalness, liveness, valence, tempo, time signature).</p>
    <p class="muted">Pipeline: pandas cleaning → StandardScaler → sklearn NearestNeighbors
    (metric="cosine", algorithm="brute") → top-N similarity lookup, exactly as trained
    in the original notebook.</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ROUTER
# ============================================================
page = st.session_state.page
if page == "Home":
    render_home()
elif page == "Discover":
    render_discover()
elif page == "Analytics":
    render_analytics()
else:
    render_about()