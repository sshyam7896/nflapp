import os
import io
import pandas as pd
import numpy as np
import streamlit as st

# Data loader: nflreadpy (nflverse)
try:
    import nflreadpy as nfl
except Exception:
    st.error("nflreadpy is required. Please install dependencies from requirements.txt")
    st.stop()

# ---------------------
# Page setup
# ---------------------
st.set_page_config(page_title="NFL Weekly Benchmarks (2025)", layout="wide", page_icon="🏈")

# ---------------------
# Dark theme CSS + UI helpers
# ---------------------
PRIMARY_ACCENT = "#7aa2ff"     # soft blue
POS_RECEIVE = "#34d399"        # green
POS_RUSH    = "#fbbf24"        # amber
POS_PASS    = "#a78bfa"        # purple
BG_DARK     = "#0b0f1a"        # app background
PANEL_DARK  = "#121826"        # cards/sidebar
BORDER_DARK = "rgba(255,255,255,0.06)"
TEXT_MAIN   = "#e5e7eb"
TEXT_MUTED  = "#9ca3af"

def inject_css():
    st.markdown(
        f"""
<style>
/* Base font smoothing */
html, body, [class*="css"] {{
  -webkit-font-smoothing: antialiased !important;
  -moz-osx-font-smoothing: grayscale !important;
  color: {TEXT_MAIN};
}}
/* App background (dark) */
.stApp {{
  background:
     radial-gradient(1200px 600px at 10% -10%, rgba(122,162,255,0.10), transparent 60%),
     radial-gradient(1200px 600px at 110% 0%, rgba(52,211,153,0.08), transparent 60%),
     linear-gradient(180deg, {BG_DARK}, {BG_DARK});
}}

/* Containers */
.block-container {{ padding-top: 1.2rem; }}
#MainMenu, footer {{ visibility: hidden; }}

/* Headings */
h1, h2, h3, h4 {{
  letter-spacing: .2px; color: {TEXT_MAIN};
}}
h1 {{ font-weight: 800; }}
h2, h3 {{ font-weight: 700; }}

/* Glass / card */
.glass {{
  background: linear-gradient(180deg, rgba(18,24,38,0.88), rgba(18,24,38,0.78));
  border: 1px solid {BORDER_DARK};
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 8px 28px rgba(0,0,0,0.35);
}}

/* KPI stat cards */
.stat-card {{
  border-radius: 16px;
  padding: 14px 16px;
  background: linear-gradient(180deg, rgba(23,31,49,0.92), rgba(23,31,49,0.86));
  border: 1px solid {BORDER_DARK};
  box-shadow: 0 10px 24px rgba(0,0,0,0.35);
}}
.stat-card .label {{
  font-size: 0.85rem; color: {TEXT_MUTED}; margin-bottom: 6px;
}}
.stat-card .value {{
  font-size: 1.6rem; font-weight: 800; line-height: 1.2; color: {TEXT_MAIN};
}}
.stat-card .sub {{
  font-size: 0.8rem; color: {TEXT_MUTED};
}}

/* Buttons */
.stDownloadButton button, .stButton button {{
  border-radius: 12px !important;
  padding: 0.6rem 0.9rem !important;
  font-weight: 700 !important;
  border: 1px solid {BORDER_DARK} !important;
  box-shadow: 0 8px 22px rgba(122,162,255,0.22) !important;
  background: {PRIMARY_ACCENT} !important;
  color: #0b1020 !important;
}}
.stDownloadButton button:hover, .stButton button:hover {{
  filter: brightness(1.04);
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; }}
.stTabs [data-baseweb="tab"] {{
  background: rgba(18,24,38, 0.82);
  border: 1px solid {BORDER_DARK};
  border-bottom: 2px solid transparent;
  border-radius: 12px 12px 0 0;
  padding: .5rem .9rem;
  color: {TEXT_MUTED};
}}
.stTabs [aria-selected="true"] {{
  border-bottom: 2px solid {PRIMARY_ACCENT} !important;
  background: #182036;
  color: {TEXT_MAIN};
}}

/* Sidebar */
section[data-testid="stSidebar"] > div {{
  background: linear-gradient(180deg, rgba(18,24,38,0.96), rgba(18,24,38,0.92));
  border-right: 1px solid {BORDER_DARK};
  color: {TEXT_MAIN};
}}
.sidebar-title {{
  font-weight: 800; font-size: .95rem; letter-spacing: .3px; margin-bottom: .5rem; color: {TEXT_MAIN};
}}
.sidebar-card {{
  background: rgba(18,24,38,0.78);
  border: 1px solid {BORDER_DARK};
  border-radius: 14px;
  padding: .6rem .7rem;
}}

/* Dataframe dark tweaks */
[data-testid="stDataFrame"] div[role="table"] {{ border: none !important; color: {TEXT_MAIN}; }}
[data-testid="stDataFrame"] thead th {{
  background: #141b2e !important;
  color: {TEXT_MAIN} !important;
  border-bottom: 1px solid {BORDER_DARK} !important;
}}
[data-testid="stDataFrame"] tbody tr:nth-child(odd) td {{ background: rgba(20,27,46,0.60); }}
[data-testid="stDataFrame"] tbody tr:nth-child(even) td {{ background: rgba(20,27,46,0.48); }}

/* Hit background (green tint) */
.cell-hit {{ background-color: rgba(34,197,94,0.28) !important; }}

/* Streak badges */
.badge {{
  display: inline-block; padding: 2px 8px; font-size: 12px;
  border-radius: 999px; background: #1f2539; color: {TEXT_MAIN};
  border: 1px solid {BORDER_DARK};
}}
.badge.green {{ background: rgba(16,185,129,.20); color: #34d399; border-color: rgba(16,185,129,.35); }}
.badge.orange {{ background: rgba(251,191,36,.18); color: #fbbf24; border-color: rgba(251,191,36,.35); }}

.player-card {{
  display: flex; align-items: center; gap: 14px;
  background: linear-gradient(180deg, rgba(23,31,49,0.92), rgba(23,31,49,0.86));
  border: 1px solid {BORDER_DARK};
  border-radius: 16px; padding: 12px 14px; margin: 6px 0 14px 0;
}}
.player-card img {{
  border-radius: 12px; background: #0e1322; border: 1px solid {BORDER_DARK};
}}
.player-card .meta {{
  display: flex; flex-direction: column;
}}
.player-card .meta .name {{
  font-weight: 800; font-size: 1.2rem; color: {TEXT_MAIN};
}}
.player-card .meta .team {{
  font-weight: 600; font-size: .95rem; color: {TEXT_MUTED};
}}
</style>
        """,
        unsafe_allow_html=True,
    )

def stat_card(label, value, sub=None, icon=None, color=None):
    ico_html = f"<span class='badge' style='margin-right:8px'>{icon}</span>" if icon else ""
    style = "" if not color else f"border-left: 6px solid {color}; padding-left: 10px;"
    st.markdown(
        f"""
<div class="stat-card" style="{style}">
  <div class="label">{ico_html}{label}</div>
  <div class="value">{value}</div>
  <div class="sub">{sub or ""}</div>
</div>
""",
        unsafe_allow_html=True,
    )

inject_css()

# ---------------------
# Config
# ---------------------
SEASON = 2025
QB_THRESHOLDS   = [175, 200, 225, 250, 275, 300]
RUSH_THRESHOLDS = [15, 25, 40, 50, 60, 70, 80, 90, 100]
REC_THRESHOLDS  = [15, 25, 40, 50, 60, 70, 80, 90, 100]
RECEPT_THRESHOLDS = [2, 3, 4, 5, 6]
ALLOWED_POS = {"QB", "RB", "WR", "TE", "K"}

# ---- Display renames (UI/export only; internal column names unchanged) ----
DISPLAY_NAME_MAP = {
    "player_name": "Name",
    "recent_team": "Team",
    "position": "Position",
    "week": "Week",
    "receiving_yards": "Receiving Yards",
    "receptions": "Receptions",
    "rushing_yards": "Rushing Yards",
    "passing_yards": "Passing Yards",
}

def _bench_display_name(col: str) -> str:
    # Map internal hit columns (e.g., QB_175+) to requested labels
    import re
    m = re.match(r"^(QB|Rush|RecYds|Receptions)_(\d+)\+$", col)
    if not m:
        return col
    typ, n = m.group(1), m.group(2)
    if typ == "QB":
        return f"{n} Passing Yards"
    if typ == "Rush":
        return f"{n} Rushing Yards"
    if typ == "RecYds":
        return f"{n} Receiving Yards"
    if typ == "Receptions":
        return f"{n}+ Receptions"
    return col

def rename_columns_for_display(df_in: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (renamed_df, renamed_hit_cols) for UI styling that still highlights hits."""
    hit_cols = [c for c in df_in.columns if c.endswith("+")]
    rename_map = {**DISPLAY_NAME_MAP, **{c: _bench_display_name(c) for c in hit_cols}}
    df_out = df_in.rename(columns=rename_map)
    renamed_hits = [rename_map.get(c, c) for c in hit_cols]
    return df_out, renamed_hits

# ESPN team slug map for logos (slug used in ESPN logo path)
ESPN_TEAM_SLUG = {
    "ARI": "ari", "ATL": "atl", "BAL": "bal", "BUF": "buf",
    "CAR": "car", "CHI": "chi", "CIN": "cin", "CLE": "cle",
    "DAL": "dal", "DEN": "den", "DET": "det", "GB": "gb",
    "HOU": "hou", "IND": "ind", "JAX": "jac", "KC": "kc",
    "LV": "lv", "LAC": "lac", "LAR": "lar", "MIA": "mia",
    "MIN": "min", "NE": "ne", "NO": "no", "NYG": "nyg",
    "NYJ": "nyj", "PHI": "phi", "PIT": "pit", "SEA": "sea",
    "SF": "sf", "TB": "tb", "TEN": "ten", "WAS": "wsh"
}
DEFAULT_HEADSHOT = "https://a.espncdn.com/i/headshots/nophoto/league/500.png"
DEFAULT_LOGO = "https://a.espncdn.com/i/teamlogos/leagues/500/nfl.png"

# ---------------------
# Cached data loading / prep
# ---------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def load_player_game_stats(season: int) -> pd.DataFrame:
    df = nfl.load_player_stats([season]).to_pandas()

    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]

    # Map columns and ensure we capture a stable player id for joining
    rename_map = {
        "player": "player_name",
        "player_display_name": "player_name",
        "team": "recent_team",
        "recent_team": "recent_team",
        "position": "position",
        "week": "week",
        "game_id": "game_id",
        "game_date": "game_date",
        "opponent": "opponent",
        "passing_yards": "passing_yards",
        "rushing_yards": "rushing_yards",
        "receiving_yards": "receiving_yards",
        "receptions": "receptions",
        # stable id variants
        "player_id": "player_id",
        "gsis_id": "player_id",
        "nfl_id": "player_id",
        "nfl_api_id": "player_id",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    need = ["player_id","player_name","recent_team","position","week","game_id","game_date","opponent",
            "passing_yards","rushing_yards","receiving_yards","receptions"]
    for c in need:
        if c not in df.columns:
            df[c] = 0 if c in ["passing_yards","rushing_yards","receiving_yards","receptions"] else ""

    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
    for c in ["passing_yards","rushing_yards","receiving_yards","receptions"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["player_id"] = df["player_id"].astype(str).fillna("")
    df = df[df["week"].between(1, 18, inclusive="both")]
    df = df[df["player_name"].astype(str).str.strip().ne("")]
    df["position"] = df["position"].astype(str)
    df = df[df["position"].str.upper().isin(ALLOWED_POS)]
    df = df[need].copy()

    for c in ["player_name","recent_team","position","opponent","game_id"]:
        df[c] = df[c].astype("category")

    return df

@st.cache_data(ttl=24*60*60, show_spinner=False)
def load_players_meta() -> pd.DataFrame:
    """
    Returns a players table with:
      - full_name (str)
      - team (abbr if available)
      - espn_id (str, digits)
      - player_id (str; gsis/nfl id)
      - name_norm (str; normalized for fallback name matching)
    """
    try:
        meta = nfl.load_players().to_pandas()
    except Exception:
        return pd.DataFrame({"full_name": [], "team": [], "espn_id": [], "player_id": [], "name_norm": []})

    if meta.empty:
        return pd.DataFrame({"full_name": [], "team": [], "espn_id": [], "player_id": [], "name_norm": []})

    cols_lower = {c.lower(): c for c in meta.columns}

    def pick(*cands):
        for c in cands:
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    # full name
    full_name_col = pick("full_name", "display_name", "player_display_name")
    if full_name_col:
        full_name = meta[full_name_col].astype(str)
    else:
        first = pick("first_name", "first")
        last  = pick("last_name", "last")
        full_name = (
            meta[first].astype(str).str.strip() + " " + meta[last].astype(str).str.strip()
            if (first and last) else meta.index.astype(str)
        )

    # team abbr
    team_col = pick("team", "team_abbr", "recent_team", "team_short_name", "team_abbreviation", "current_team")
    team = (meta[team_col].astype(str).str.upper() if team_col else pd.Series([""] * len(meta)))
    team = team.replace({"WFT": "WAS", "WSH": "WAS", "LA": "LAR"}).fillna("")

    # espn id
    espn_id_col = pick("espn_id", "espn_player_id", "espn")
    espn_id = (meta[espn_id_col].astype(str).str.extract(r"(\d+)", expand=False)
               if espn_id_col else pd.Series([pd.NA] * len(meta)))

    # stable player id
    pid_col = pick("player_id", "gsis_id", "nfl_id", "nfl_api_id")
    player_id = (meta[pid_col].astype(str) if pid_col else pd.Series([""] * len(meta))).fillna("")

    out = pd.DataFrame({
        "full_name": full_name.astype(str),
        "team": team.astype(str),
        "espn_id": espn_id,
        "player_id": player_id.astype(str),
    })

    # normalized name fallback
    def norm_name(s: pd.Series) -> pd.Series:
        return (s.astype(str)
                 .str.replace(r"[^A-Za-z0-9 ]", "", regex=True)
                 .str.replace(r"\b(JR|SR|II|III|IV|V)\b\.?", "", regex=True)
                 .str.replace(r"\s+", " ", regex=True)
                 .str.strip()
                 .str.lower())

    out["name_norm"] = norm_name(out["full_name"])
    out = out.drop_duplicates(ignore_index=True)

    for c in ["full_name","team","espn_id","player_id","name_norm"]:
        if c not in out.columns:
            out[c] = ""

    return out

players_meta = load_players_meta()

# --- NEW: force full names in the game logs by joining on player_id
@st.cache_data(show_spinner=False)
def apply_full_names(df_in: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty or meta is None or meta.empty:
        return df_in
    m = meta[["player_id", "full_name"]].dropna().copy()
    out = df_in.merge(m, on="player_id", how="left")
    # replace player_name with full_name when available
    full = out["full_name"].astype(str).str.strip()
    out["player_name"] = np.where(full.ne(""), full, out["player_name"].astype(str))
    out = out.drop(columns=["full_name"])
    # recategorize to refresh the categories with full names
    for c in ["player_name","recent_team","position","opponent","game_id"]:
        out[c] = out[c].astype("category")
    return out

@st.cache_data(show_spinner=False)
def add_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    is_qb = out["position"].astype(str).str.upper().eq("QB")
    for t in QB_THRESHOLDS:
        out[f"QB_{t}+"] = (is_qb & (out["passing_yards"] >= t)).astype(np.int8)
    for t in RUSH_THRESHOLDS:
        out[f"Rush_{t}+"] = (out["rushing_yards"] >= t).astype(np.int8)
    for t in REC_THRESHOLDS:
        out[f"RecYds_{t}+"] = (out["receiving_yards"] >= t).astype(np.int8)
    for t in RECEPT_THRESHOLDS:
        out[f"Receptions_{t}+"] = (out["receptions"] >= t).astype(np.int8)
    return out

# --------- Vectorized streak helpers ----------
def tail_streak_ones(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    rev = arr[::-1]
    zeros = np.where(rev == 0)[0]
    return int(len(rev) if zeros.size == 0 else zeros[0])

@st.cache_data(show_spinner=False)
def compute_streaks(df: pd.DataFrame, hit_cols: list[str]) -> pd.DataFrame:
    if df.empty or not hit_cols:
        return pd.DataFrame()
    sdf = df.sort_values(["player_name","recent_team","position","week"])
    g = sdf.groupby(["player_name","recent_team","position"], observed=True)
    frames = []
    for col in hit_cols:
        streak = g[col].apply(lambda s: tail_streak_ones(s.to_numpy()))
        frames.append(streak.reset_index().rename(columns={col: f"{col} (streak)"}))
    out = frames[0]
    for add in frames[1:]:
        out = out.merge(add, on=["player_name","recent_team","position"], how="left")
    return out

@st.cache_data(show_spinner=False)
def summary_fraction_streaks(df: pd.DataFrame, hit_cols: list[str]) -> pd.DataFrame:
    if df.empty or not hit_cols:
        return pd.DataFrame(columns=["player_name","recent_team","position","Games"])
    base = df.groupby(["player_name","recent_team","position"], observed=True, as_index=False).agg(
        Games=("week","nunique")
    )
    sums = df.groupby(["player_name","recent_team","position"], observed=True)[hit_cols].sum().reset_index()
    out = base.merge(sums, on=["player_name","recent_team","position"], how="left")
    streaks = compute_streaks(df, hit_cols)
    if not streaks.empty:
        out = out.merge(streaks, on=["player_name","recent_team","position"], how="left")
    for c in hit_cols:
        out[f"{c} (Hit Rate)"] = (out[c].fillna(0).astype(int).astype(str) + "/" + out["Games"].astype(int).astype(str))
        sc = f"{c} (streak)"
        if sc in out.columns:
            out[sc] = out[sc].fillna(0).astype(int)
            out[sc] = out[sc].apply(lambda x: f"{x} 🔥" if x >= 2 else (f"{x}" if x > 0 else "0"))
    return out.sort_values(["position","player_name"]).reset_index(drop=True)

# ----- UPDATED: vertical table with requested display labels -----
@st.cache_data(show_spinner=False)
def build_player_vertical_table(pdf: pd.DataFrame, thresholds: list[int], col_prefix: str, label_prefix: str) -> pd.DataFrame:
    """
    Builds a 4-column vertical table: Metric | Hits | Metric Streak | Streak.
    Display labels are humanized per request; internal hit math is unchanged.
    """
    if pdf.empty:
        return pd.DataFrame(columns=["Metric","Hits","Metric Streak","Streak"])

    def label_from_prefix(pfx: str, t: int) -> str:
        if pfx == "QB_":
            return f"{t} Passing Yards"
        if pfx == "Rush_":
            return f"{t} Rushing Yards"
        if pfx == "RecYds_":
            return f"{t} Receiving Yards"
        if pfx == "Receptions_":
            return f"{t}+ Receptions"
        # Fallback
        return f"{pfx}{t}+"

    games = int(pdf["week"].nunique())
    rows = []
    for t in thresholds:
        col = f"{col_prefix}{t}+"
        label = label_from_prefix(col_prefix, t)
        h = pdf[col].to_numpy() if col in pdf.columns else np.zeros(len(pdf), dtype=np.int8)
        hits = int(h.sum())
        streak_n = tail_streak_ones(h)
        rows.append({
            "Metric": label,
            "Hits": f"{hits}/{games}",
            "Metric Streak": f"{label} Streak",
            "Streak": f"{streak_n} 🔥" if streak_n >= 2 else (f"{streak_n}" if streak_n > 0 else "0")
        })
    return pd.DataFrame(rows)

def to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    import xlsxwriter
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()

# ---------------------
# Load & Prepare Data
# ---------------------
df_raw = load_player_game_stats(SEASON)
# enforce full names using players_meta (player_id join)
players_meta = load_players_meta()
df_raw = apply_full_names(df_raw, players_meta)

df = add_benchmarks(df_raw)

# ---------------------
# Sidebar: Filters
# ---------------------
st.sidebar.markdown("<div class='sidebar-title'>Filters</div>", unsafe_allow_html=True)
pos = st.sidebar.multiselect("Position", sorted(df["position"].cat.categories.tolist()), default=None)
team = st.sidebar.multiselect("Team", sorted(df["recent_team"].cat.categories.tolist()), default=None)

all_players = sorted(df["player_name"].cat.categories.tolist())
search_text = st.sidebar.text_input(
    "Search player (any part or last name)",
    value="",
    help="Type part of the first/last name; the list filters live."
).strip().lower()

if search_text:
    filtered_player_options = [p for p in all_players if (search_text in p.lower()) or (search_text in p.lower().split()[-1])]
else:
    filtered_player_options = all_players

player = st.sidebar.multiselect(
    "Player (filtered by search)",
    options=filtered_player_options,
    default=None
)

@st.cache_data(show_spinner=False)
def apply_filters(df: pd.DataFrame, pos, team, player) -> pd.DataFrame:
    f = df
    if pos:
        f = f[f["position"].isin(pos)]
    if team:
        f = f[f["recent_team"].isin(team)]
    if player:
        f = f[f["player_name"].isin(player)]
    return f

fdf = apply_filters(df, pos, team, player)

# ---------------------
# Sidebar: 📊 Box Scores (ESPN only; direct Box Score links)
# ---------------------
st.sidebar.divider()
st.sidebar.markdown("<div class='sidebar-title'>Box Scores</div>", unsafe_allow_html=True)

def pick_first_col(d: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in d.columns:
            return c
    return None

@st.cache_data(ttl=60*60, show_spinner=False)
def load_schedule(season: int) -> pd.DataFrame:
    """Schedule for week/game listing in Box Scores section, with ESPN id if present."""
    try:
        sch = nfl.load_schedules([season]).to_pandas()
    except Exception:
        sch = nfl.load_schedule([season]).to_pandas()

    rename_pairs = {
        "gameday": "game_date",
        "home": "home_team",
        "away": "away_team",
        "home_team_name": "home_team",
        "away_team_name": "away_team",
    }
    for src, dst in rename_pairs.items():
        if src in sch.columns and dst not in sch.columns:
            sch[dst] = sch[src]

    espn_id_col = pick_first_col(sch, ["espn_game_id", "game_id_espn", "game_id2", "espn", "eid"])

    keep = [c for c in ["game_id","week","home_team","away_team","game_date"] if c in sch.columns]
    if espn_id_col and espn_id_col not in keep:
        keep.append(espn_id_col)

    sch = sch[keep].copy()
    sch["week"] = pd.to_numeric(sch["week"], errors="coerce").astype("Int64")
    sch = sch[sch["week"].between(1, 18, inclusive="both")]

    sch.attrs["espn_id_col"] = espn_id_col
    return sch

schedule_df = load_schedule(SEASON)
if schedule_df.empty:
    st.sidebar.info("No schedule available yet.")
else:
    weeks = sorted(schedule_df["week"].dropna().unique().tolist())
    sel_week = st.sidebar.selectbox("Week", weeks, index=(len(weeks)-1 if weeks else 0))

    # ESPN weekly scoreboard (regular season = seasontype/2)
    espn_week_url = f"https://www.espn.com/nfl/scoreboard/_/week/{int(sel_week)}/year/{SEASON}/seasontype/2"
    st.sidebar.markdown(
        f"<div class='sidebar-card'>📅 <a href='{espn_week_url}' target='_blank'><b>Open ESPN Week {int(sel_week)} Scoreboard</b></a></div>",
        unsafe_allow_html=True
    )

    # Per-game direct ESPN Box Score links
    st.sidebar.markdown("**This Week’s Games (ESPN Box Scores)**")
    wk = schedule_df[schedule_df["week"] == sel_week].copy()
    if wk.empty:
        st.sidebar.write("_No games found for this week._")
    else:
        wk = wk.sort_values(["game_date","away_team","home_team"])
        espn_col = schedule_df.attrs.get("espn_id_col")

        for _, r in wk.iterrows():
            away = str(r.get("away_team", "")).strip()
            home = str(r.get("home_team", "")).strip()
            gdate = str(r.get("game_date", ""))[:10] if pd.notna(r.get("game_date", "")) else ""
            espn_box_link = None
            if espn_col and pd.notna(r.get(espn_col, None)):
                espn_id = str(r[espn_col]).strip()
                if espn_id:
                    espn_box_link = f"https://www.espn.com/nfl/boxscore/_/gameId/{espn_id}"

            if espn_box_link:
                st.sidebar.markdown(
                    f"- **{away} @ {home}** {('— ' + gdate) if gdate else ''}  \n"
                    f"  <a href='{espn_box_link}' target='_blank'>ESPN Box Score</a>",
                    unsafe_allow_html=True
                )
            else:
                st.sidebar.markdown(
                    f"- **{away} @ {home}** {('— ' + gdate) if gdate else ''}  \n"
                    f"  <em>ESPN game id not in schedule — open the</em> <a href='{espn_week_url}' target='_blank'>Week {int(sel_week)} Scoreboard</a>",
                    unsafe_allow_html=True
                )

# ---------------------
# Priority logic (controls column order & charts emphasis)
# ---------------------
def infer_priority_mode(_df: pd.DataFrame, selected_players, pos_filter) -> str:
    def norm(s): return {x.upper() for x in s} if s else set()
    if selected_players:
        sel_pos = _df[_df["player_name"].isin(selected_players)]["position"].astype(str).str.upper().unique().tolist()
        sp = norm(sel_pos)
        if sp and sp.issubset({"WR","TE"}): return "receiving"
        if sp == {"RB"}: return "rushing"
        if sp == {"QB"}: return "passing"
    if pos_filter:
        sp = norm(pos_filter)
        if sp.issubset({"WR","TE"}): return "receiving"
        if sp == {"RB"}: return "rushing"
        if sp == {"QB"}: return "passing"
    return "receiving"

priority_mode = infer_priority_mode(fdf, player if player else None, pos if pos else None)

# ---------------------
# Header
# ---------------------
st.markdown(f"""
<div class="glass" style="margin-bottom: .6rem;">
  <h1>🏈 NFL Weekly Benchmarks — 2025</h1>
  <div class="badge">Auto-updating from nflverse</div>
</div>
""", unsafe_allow_html=True)

# ---------------------
# KPIs (stat cards)
# ---------------------
c1, c2, c3, c4 = st.columns(4)
with c1: stat_card("Players", f"{fdf['player_name'].nunique():,}", "unique players", icon="👤")
with c2: stat_card("Teams", f"{fdf['recent_team'].nunique():,}", "active this view", icon="🏟️")
with c3: stat_card("Games (rows)", f"{len(fdf):,}", "log entries", icon="🧾")
with c4: stat_card("Latest Week in Data", "-" if fdf.empty else int(fdf["week"].max()), "season 2025", icon="🗓️")

# ---------------------
# Helper: resolve ESPN media for a player (ID-first; robust fallback)
# ---------------------
@st.cache_data(show_spinner=False)
def get_player_media(player_name: str, team_abbr: str, player_id: str | None = None):
    """
    Return (headshot_url, team_logo_url) for a given player.
    Prefer exact ID match to get espn_id. Fallback to normalized name lookup.
    """
    headshot = DEFAULT_HEADSHOT
    logo = DEFAULT_LOGO

    slug = ESPN_TEAM_SLUG.get(str(team_abbr).upper())
    if slug:
        logo = f"https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"

    pm = players_meta
    if pm is None or pm.empty:
        return headshot, logo

    # 1) Try direct ID match
    if player_id:
        pid = str(player_id)
        cand = pm[pm["player_id"] == pid]
        if not cand.empty:
            espn_ids = cand["espn_id"].dropna().astype(str)
            if not espn_ids.empty and espn_ids.iloc[0]:
                return (f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_ids.iloc[0]}.png", logo)

    # 2) Exact name (case-insensitive), prefer same team
    cand = pm[pm["full_name"].str.casefold() == str(player_name).casefold()]
    if not cand.empty:
        if team_abbr:
            c_team = cand[cand["team"].astype(str).str.upper() == str(team_abbr).upper()]
            if not c_team.empty:
                cand = c_team
        espn_ids = cand["espn_id"].dropna().astype(str)
        if not espn_ids.empty and espn_ids.iloc[0]:
            return (f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_ids.iloc[0]}.png", logo)

    # 3) Fallback: normalized name match
    def norm_one(s: str) -> str:
        import re
        s2 = re.sub(r"[^A-Za-z0-9 ]", "", s)
        s2 = re.sub(r"\b(JR|SR|II|III|IV|V)\b\.?", "", s2, flags=re.IGNORECASE)
        s2 = re.sub(r"\s+", " ", s2).strip().lower()
        return s2

    name_norm = norm_one(player_name)
    cand = pm[pm["name_norm"] == name_norm]
    if not cand.empty:
        if team_abbr:
            c_team = cand[cand["team"].astype(str).str.upper() == str(team_abbr).upper()]
            if not c_team.empty:
                cand = c_team
        espn_ids = cand["espn_id"].dropna().astype(str)
        if not espn_ids.empty and espn_ids.iloc[0]:
            return (f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_ids.iloc[0]}.png", logo)

    return headshot, logo

def render_player_card(name: str, team_abbr: str, pos: str, player_id: str | None = None):
    headshot, logo = get_player_media(name, team_abbr, player_id)
    st.markdown(
        f"""
<div class="player-card">
  <img src="{headshot}" alt="headshot" width="76" height="76"/>
  <div class="meta">
    <div class="name">{name}</div>
    <div class="team">{team_abbr} • {pos}</div>
  </div>
  <div style="margin-left:auto">
    <img src="{logo}" alt="team" width="64" height="64"/>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

# If a single player is selected in sidebar, show the card below KPIs
if player and len(player) == 1:
    p = player[0]
    row = fdf[fdf["player_name"] == p].head(1)
    if not row.empty:
        render_player_card(
            p,
            str(row["recent_team"].iloc[0]),
            str(row["position"].iloc[0]),
            player_id=str(row["player_id"].iloc[0]) if "player_id" in row.columns else None
        )

# ---------------------
# Weekly Game Log (with visual highlighting)  — player_id hidden
# ---------------------
st.markdown("### Weekly Game Log")

# keep player_id in the dataframe for lookups, but don't display it
base_cols_display = ["player_name", "recent_team", "position", "week"]

if priority_mode == "passing":
    ordered_metrics = ["passing_yards", "rushing_yards", "receiving_yards", "receptions"]
elif priority_mode == "rushing":
    ordered_metrics = ["rushing_yards", "receiving_yards", "receptions", "passing_yards"]
else:
    ordered_metrics = ["receiving_yards", "receptions", "rushing_yards", "passing_yards"]

bench_cols = [c for c in fdf.columns if c.endswith("+")]
cols_show = [c for c in base_cols_display if c in fdf.columns] + ordered_metrics + bench_cols

def style_weekly(df_show_display: pd.DataFrame, hit_cols_display: list[str]):
    def hit_color(v):
        try:
            return "background-color: rgba(34,197,94,0.28)" if int(v) == 1 else ""
        except Exception:
            return ""
    base_bold_cols = [c for c in ["Name","Team","Position"] if c in df_show_display.columns]
    styled = (df_show_display
              .style
              .applymap(lambda _: "font-weight:600; color:#fff", subset=base_bold_cols)
              .applymap(hit_color, subset=[c for c in hit_cols_display if c in df_show_display.columns])
              .format(precision=0, na_rep="—"))
    return styled

# --- Build the weekly table for display, with renamed headers but same underlying data ---
_df_weekly = fdf[cols_show].sort_values(["player_name","week"]).reset_index(drop=True)
_df_weekly_display, _hit_cols_display = rename_columns_for_display(_df_weekly)

st.dataframe(
    style_weekly(_df_weekly_display, _hit_cols_display),
    use_container_width=True,
    height=430
)

# ---------------------
# Precompute summaries (cached column lists)
# ---------------------
bench_cols_all = [c for c in fdf.columns if c.endswith("+")]
qb_cols   = [c for c in bench_cols_all if c.startswith("QB_")]
rush_cols = [c for c in bench_cols_all if c.startswith("Rush_")]
recy_cols = [c for c in bench_cols_all if c.startswith("RecYds_")]
recv_cols = [c for c in bench_cols_all if c.startswith("Receptions_")]

# ---------------------
# Tabs: Passing / Rushing / Receiving
# ---------------------
tab_pass, tab_rush, tab_recv = st.tabs([
    f"🏈 Passing", f"🏃 Rushing", f"🎯 Receiving"
])

players_to_render = sorted(fdf["player_name"].unique()) if not player else sorted(player)
if not player and len(players_to_render) > 15:
    with st.sidebar:
        show_all = st.checkbox(f"Show all {len(players_to_render)} players in tables (slower)", value=False)
    if not show_all:
        players_to_render = players_to_render[:15]

def style_vertical_table(dfv: pd.DataFrame, accent="#34d399"):
    def streak_tint(s):
        out = []
        for v in s:
            try:
                n = int(str(v).split()[0])
                if n >= 3:
                    out.append("background-color: rgba(16,185,129,.30); font-weight:700; color:#eafff6;")
                elif n == 2:
                    out.append("background-color: rgba(251,191,36,.26); font-weight:600; color:#fff7da;")
                else:
                    out.append("")
            except Exception:
                out.append("")
        return out
    sty = (dfv.style
           .set_table_styles([{"selector":"th","props":[("font-weight","700"), ("color", "#e5e7eb"), ("background","#141b2e")]}])
           .apply(streak_tint, subset=["Streak"]))
    return sty

with tab_pass:
    st.markdown("#### Passing Yard Benchmarks — Per Player")
    if fdf.empty or not qb_cols:
        st.info("No passing data after filters.")
    else:
        for pname in players_to_render:
            pdf_player = fdf[fdf["player_name"] == pname].sort_values("week")
            pass_table = build_player_vertical_table(pdf_player, QB_THRESHOLDS, "QB_", "qb_")
            st.markdown(f"**{pname} — Passing Yards**")
            st.dataframe(style_vertical_table(pass_table, POS_PASS), use_container_width=True, height=330)

with tab_rush:
    st.markdown("#### Rushing Yard Benchmarks — Per Player")
    if fdf.empty or not rush_cols:
        st.info("No rushing data after filters.")
    else:
        for pname in players_to_render:
            pdf_player = fdf[fdf["player_name"] == pname].sort_values("week")
            rush_table = build_player_vertical_table(pdf_player, RUSH_THRESHOLDS, "Rush_", "rush_")
            st.markdown(f"**{pname} — Rushing Yards**")
            st.dataframe(style_vertical_table(rush_table, POS_RUSH), use_container_width=True, height=330)

with tab_recv:
    st.markdown("#### Receiving Metrics — Per Player")
    if fdf.empty or (not recy_cols and not recv_cols):
        st.info("No receiving data after filters.")
    else:
        for pname in players_to_render:
            pdf_player = fdf[fdf["player_name"] == pname].sort_values("week")
            if recy_cols:
                recy_table = build_player_vertical_table(pdf_player, REC_THRESHOLDS, "RecYds_", "recyds_")
                st.markdown(f"**{pname} — Receiving Yards**")
                st.dataframe(style_vertical_table(recy_table, POS_RECEIVE), use_container_width=True, height=330)
            if recv_cols:
                recv_table = build_player_vertical_table(pdf_player, RECEPT_THRESHOLDS, "Receptions_", "receptions_")
                st.markdown(f"**{pname} — Receptions**")
                st.dataframe(style_vertical_table(recv_table, POS_RECEIVE), use_container_width=True, height=300)

# ---------------------
# Trend Explorer (averages + 3-game moving averages; includes receptions)
# ---------------------
st.markdown("### Player Trend Explorer")

pdf = pd.DataFrame()  # ensure defined for export
sel_player = None
if fdf.empty:
    st.info("No data after filters.")
else:
    sel_player = st.selectbox("Choose a player to visualize weekly trends", sorted(fdf["player_name"].unique()))

    # Show player card here too (using this player's most recent team/pos in filtered data)
    row_for_card = fdf[fdf["player_name"] == sel_player].sort_values("week", ascending=False).head(1)
    if not row_for_card.empty:
        render_player_card(
            sel_player,
            str(row_for_card["recent_team"].iloc[0]),
            str(row_for_card["position"].iloc[0]),
            player_id=str(row_for_card["player_id"].iloc[0]) if "player_id" in row_for_card.columns else None
        )

    pdf = fdf[fdf["player_name"] == sel_player].sort_values("week").copy()
    sel_pos = str(pdf["position"].iloc[0]).upper() if not pdf.empty else ""

    # Averages
    avg_rec  = float(pdf["receiving_yards"].mean()) if not pdf.empty else 0.0
    avg_rush = float(pdf["rushing_yards"].mean())   if not pdf.empty else 0.0
    avg_pass = float(pdf["passing_yards"].mean())   if not pdf.empty else 0.0
    avg_recp = float(pdf["receptions"].mean())      if not pdf.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1: stat_card("Avg Receiving Yds", f"{avg_rec:.1f}", icon="🎯", color=POS_RECEIVE)
    with k2: stat_card("Avg Rushing Yds",   f"{avg_rush:.1f}", icon="🏃", color=POS_RUSH)
    with k3: stat_card("Avg Passing Yds",   f"{avg_pass:.1f}", icon="🏈", color=POS_PASS)
    with k4: stat_card("Avg Receptions",    f"{avg_recp:.2f}", icon="🫳", color=POS_RECEIVE)

    # MA3
    pdf["recv_ma3"]  = pdf["receiving_yards"].rolling(3).mean()
    pdf["rush_ma3"]  = pdf["rushing_yards"].rolling(3).mean()
    pdf["pass_ma3"]  = pdf["passing_yards"].rolling(3).mean()
    pdf["recp_ma3"]  = pdf["receptions"].rolling(3).mean()

    def last_ma(series: pd.Series) -> str:
        s = series.dropna()
        return f"{s.iloc[-1]:.1f}" if not s.empty else "—"

    m1, m2, m3, m4 = st.columns(4)
    with m1: stat_card("MA3 Receiving Yds", last_ma(pdf["recv_ma3"]), icon="📈", color=POS_RECEIVE)
    with m2: stat_card("MA3 Rushing Yds",   last_ma(pdf["rush_ma3"]), icon="📈", color=POS_RUSH)
    with m3: stat_card("MA3 Passing Yds",   last_ma(pdf["pass_ma3"]), icon="📈", color=POS_PASS)
    with m4: stat_card("MA3 Receptions",    (f"{pdf['recp_ma3'].dropna().iloc[-1]:.2f}" if not pdf['recp_ma3'].dropna().empty else "—"), icon="📈", color=POS_RECEIVE)

    # Priority chart layout
    mode = "receiving" if sel_pos in {"WR","TE"} else ("rushing" if sel_pos=="RB" else ("passing" if sel_pos=="QB" else "receiving"))

    glass_left, glass_right = st.columns([1,1])
    if mode == "passing":
        with glass_left:
            st.markdown("<div class='glass'><b>Passing Yards (raw + MA3)</b></div>", unsafe_allow_html=True)
            st.line_chart(pdf.set_index("week")[["passing_yards","pass_ma3"]], height=260)
        with glass_right:
            st.markdown("<div class='glass'><b>Rushing Yds (raw + MA3)</b></div>", unsafe_allow_html=True)
            st.line_chart(pdf.set_index("week")[["rushing_yards","rush_ma3"]], height=240)
            st.markdown("<div class='glass'><b>Receiving Yds (raw + MA3)</b></div>", unsafe_allow_html=True)
            st.line_chart(pdf.set_index("week")[["receiving_yards","recv_ma3"]], height=240)
    elif mode == "rushing":
        with glass_left:
            st.markdown("<div class='glass'><b>Rushing Yds (raw + MA3)</b></div>", unsafe_allow_html=True)
            st.line_chart(pdf.set_index("week")[["rushing_yards","rush_ma3"]], height=260)
        with glass_right:
            st.markdown("<div class='glass'><b>Receiving Yds (raw + MA3)</b></div>", unsafe_allow_html=True)
            st.line_chart(pdf.set_index("week")[["receiving_yards","recv_ma3"]], height=240)
            st.markdown("<div class='glass'><b>Passing Yds (raw + MA3)</b></div>", unsafe_allow_html=True)
            st.line_chart(pdf.set_index("week")[["passing_yards","pass_ma3"]], height=240)
    else:
        with glass_left:
            st.markdown("<div class='glass'><b>Receiving Yds (raw + MA3)</b></div>", unsafe_allow_html=True)
            st.line_chart(pdf.set_index("week")[["receiving_yards","recv_ma3"]], height=260)
        with glass_right:
            st.markdown("<div class='glass'><b>Rushing Yds (raw + MA3)</b></div>", unsafe_allow_html=True)
            st.line_chart(pdf.set_index("week")[["rushing_yards","rush_ma3"]], height=240)
            st.markdown("<div class='glass'><b>Passing Yds (raw + MA3)</b></div>", unsafe_allow_html=True)
            st.line_chart(pdf.set_index("week")[["passing_yards","pass_ma3"]], height=240)

# ---------------------
# PDF Export (snapshot)
# ---------------------
def safe_str(x):
    try:
        return "" if pd.isna(x) else str(x)
    except Exception:
        return str(x)

def _col_width_pts(col: str) -> int:
    c = col.lower()
    # Support both internal and display names
    if c in ("player_name", "name"):
        return 140
    if c in ("recent_team", "team", "position"):
        return 70
    if c in ("week",):
        return 40
    # Wider for descriptive hit labels & streaks
    if "yards" in c or "receptions" in c or "streak" in c or "hit rate" in c:
        return 90
    return 60

def _chunk_columns_by_width(columns: list[str], max_width_pts: int) -> list[list[str]]:
    chunks, cur, cur_w = [], [], 0
    for col in columns:
        w = _col_width_pts(col)
        if cur and cur_w + w > max_width_pts:
            chunks.append(cur)
            cur, cur_w = [col], w
        else:
            cur.append(col); cur_w += w
    if cur: chunks.append(cur)
    return chunks

def df_to_rl_table_data(df_in: pd.DataFrame, columns: list[str], max_rows: int = 800):
    if df_in is None or df_in.empty:
        return [["(no rows)"]]
    use = df_in[columns].head(max_rows).copy()
    data = [list(map(safe_str, use.columns.tolist()))]
    for _, r in use.iterrows():
        data.append([safe_str(v) for v in r.tolist()])
    return data

def build_pdf_bytes(
    page_title: str,
    filters: dict,
    weekly_df: pd.DataFrame,
    columns_shown: list[str],
    players_for_tables: list[str],
    fdf_all: pd.DataFrame,
    rec_thresholds, rush_thresholds, qb_thresholds,
    sel_player_name: str | None,
    sel_avgs: dict | None,
    sel_ma3: dict | None
) -> bytes:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, Table, TableStyle, PageBreak

    page_size = landscape(letter)
    margin = 18
    usable_w = page_size[0] - margin * 2  # points
    styles = getSampleStyleSheet()
    H = styles["Heading2"]; H.spaceAfter = 8
    P = styles["BodyText"]; P.fontSize = 8; P.leading = 10
    HEADER_BG = colors.Color(0.08, 0.10, 0.18)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=page_size, leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin
    )

    story = []
    story.append(Paragraph(page_title, H))

    # Filters summary
    filt_txt = " | ".join([f"<b>{k}</b>: {', '.join(map(safe_str, v)) if isinstance(v, (list, tuple, set)) else safe_str(v)}"
                           for k, v in filters.items()])
    story.append(Paragraph(filt_txt if filt_txt else "(No filters)", P))
    story.append(Spacer(1, 6))

    # Weekly Game Log (already display-renamed)
    if weekly_df is not None:
        story.append(Paragraph("<b>Weekly Game Log (filtered)</b>", P))
        wk = weekly_df.reset_index(drop=True)
        col_chunks = _chunk_columns_by_width(columns_shown, max_width_pts=int(usable_w))
        total_parts = len(col_chunks)
        for i, cols in enumerate(col_chunks, start=1):
            story.append(Paragraph(f"Columns {i}/{total_parts}", P))
            data = df_to_rl_table_data(wk, cols, max_rows=800)
            col_widths = [_col_width_pts(c) for c in cols]
            tbl = LongTable(data, repeatRows=1, colWidths=col_widths)
            tbl.setStyle(TableStyle([
                ("FONT", (0,0), (-1,-1), "Helvetica", 7),
                ("LEFTPADDING", (0,0), (-1,-1), 2),
                ("RIGHTPADDING", (0,0), (-1,-1), 2),
                ("TOPPADDING", (0,0), (-1,-1), 1),
                ("BOTTOMPADDING", (0,0), (-1,-1), 1),
                ("BACKGROUND", (0,0), (-1,0), HEADER_BG),
                ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
                ("LINEABOVE", (0,0), (-1,0), 0.5, colors.white),
                ("LINEBELOW", (0,0), (-1,0), 0.5, colors.white),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 6))

    # Per-player vertical tables (already with display headers)
    def add_player_table(title, df_table):
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>{title}</b>", P))
        if df_table is None or df_table.empty:
            story.append(Paragraph("(no rows)", P))
            return
        data = df_to_rl_table_data(df_table, df_table.columns.tolist(), max_rows=200)
        col_widths = [120, 70, 140, 70]
        t = LongTable(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONT", (0,0), (-1,-1), "Helvetica", 7),
            ("LEFTPADDING", (0,0), (-1,-1), 2),
            ("RIGHTPADDING", (0,0), (-1,-1), 2),
            ("TOPPADDING", (0,0), (-1,-1), 1),
            ("BOTTOMPADDING", (0,0), (-1,-1), 1),
            ("BACKGROUND", (0,0), (-1,0), HEADER_BG),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(t)

    for pname in players_to_render:
        pdf_player = fdf_all[fdf_all["player_name"] == pname].sort_values("week")
        recy_table = build_player_vertical_table(pdf_player, REC_THRESHOLDS, "RecYds_", "recyds_")
        add_player_table(f"{pname} — Receiving Yards", recy_table)
        recv_table = build_player_vertical_table(pdf_player, RECEPT_THRESHOLDS, "Receptions_", "receptions_")
        add_player_table(f"{pname} — Receptions", recv_table)
        rush_table = build_player_vertical_table(pdf_player, RUSH_THRESHOLDS, "Rush_", "rush_")
        add_player_table(f"{pname} — Rushing Yards", rush_table)
        pass_table = build_player_vertical_table(pdf_player, QB_THRESHOLDS, "QB_", "qb_")
        add_player_table(f"{pname} — Passing Yards", pass_table)

    # Trend KPIs page
    if sel_player_name and sel_avgs and sel_ma3:
        from reportlab.platypus import PageBreak
        story.append(PageBreak())
        story.append(Paragraph(f"<b>Player Trend KPIs — {sel_player_name}</b>", H))
        kpi_html = (
            f"Avg Receiving: <b>{sel_avgs.get('rec', '—')}</b> &nbsp;|&nbsp; "
            f"Avg Rushing: <b>{sel_avgs.get('rush', '—')}</b> &nbsp;|&nbsp; "
            f"Avg Passing: <b>{sel_avgs.get('pass', '—')}</b> &nbsp;|&nbsp; "
            f"Avg Receptions: <b>{sel_avgs.get('recp', '—')}</b><br/>"
            f"MA3 Receiving: <b>{sel_ma3.get('rec', '—')}</b> &nbsp;|&nbsp; "
            f"MA3 Rushing: <b>{sel_ma3.get('rush', '—')}</b> &nbsp;|&nbsp; "
            f"MA3 Passing: <b>{sel_ma3.get('pass', '—')}</b> &nbsp;|&nbsp; "
            f"MA3 Receptions: <b>{sel_ma3.get('recp', '—')}</b>"
        )
        story.append(Paragraph(kpi_html, P))

    doc.build(story)
    return buf.getvalue()

# ---------------------
# Export Snapshot (PDF)
# ---------------------
st.markdown("### Export Snapshot")
try:
    players_snapshot = players_to_render

    sel_avgs_vals = None
    sel_ma3_vals = None
    if 'pdf' in locals() and not pdf.empty and sel_player:
        avg_rec  = float(pdf["receiving_yards"].mean())
        avg_rush = float(pdf["rushing_yards"].mean())
        avg_pass = float(pdf["passing_yards"].mean())
        avg_recp = float(pdf["receptions"].mean())
        recv_ma3_last = pdf["receiving_yards"].rolling(3).mean().dropna()
        rush_ma3_last = pdf["rushing_yards"].rolling(3).mean().dropna()
        pass_ma3_last = pdf["passing_yards"].rolling(3).mean().dropna()
        recp_ma3_last = pdf["receptions"].rolling(3).mean().dropna()
        sel_avgs_vals = {
            "rec":  f"{avg_rec:.1f}",
            "rush": f"{avg_rush:.1f}",
            "pass": f"{avg_pass:.1f}",
            "recp": f"{avg_recp:.2f}",
        }
        sel_ma3_vals  = {
            "rec":  f"{recv_ma3_last.iloc[-1]:.1f}" if not recv_ma3_last.empty else "—",
            "rush": f"{rush_ma3_last.iloc[-1]:.1f}" if not rush_ma3_last.empty else "—",
            "pass": f"{pass_ma3_last.iloc[-1]:.1f}" if not pass_ma3_last.empty else "—",
            "recp": f"{recp_ma3_last.iloc[-1]:.2f}" if not recp_ma3_last.empty else "—",
        }

    # Prepare a display-version of the weekly df for the PDF (headers only)
    _wk_pdf = fdf.sort_values(["player_name","week"]).reset_index(drop=True)
    _wk_pdf = _wk_pdf[cols_show]
    _wk_pdf_display, _ = rename_columns_for_display(_wk_pdf)

    pdf_bytes = build_pdf_bytes(
        page_title="NFL Weekly Benchmarks — Snapshot",
        filters={
            "Positions": pos if pos else ["All"],
            "Teams": team if team else ["All"],
            "Players": player if player else ["All"],
            "Priority": priority_mode.title()
        },
        weekly_df=_wk_pdf_display,                        # display headers
        columns_shown=_wk_pdf_display.columns.tolist(),   # display headers
        players_for_tables=players_snapshot,
        fdf_all=fdf,
        rec_thresholds=REC_THRESHOLDS,
        rush_thresholds=RUSH_THRESHOLDS,
        qb_thresholds=QB_THRESHOLDS,
        sel_player_name=sel_player,
        sel_avgs=sel_avgs_vals,
        sel_ma3=sel_ma3_vals
    )

    st.download_button(
        "📥 Download current view as PDF",
        data=pdf_bytes,
        file_name=f"nfl_benchmarks_snapshot_{SEASON}.pdf",
        mime="application/pdf"
    )
except Exception as e:
    st.error(f"PDF export failed. Make sure reportlab is installed. Details: {e}")

st.caption("Dark mode with ESPN headshots & team logos. ESPN box scores in sidebar. Tables highlight benchmark hits and streaks.")
