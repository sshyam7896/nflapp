# NFL Weekly Benchmarks — Auto-updating Streamlit Dashboard (2025)

This app pulls 2025 **player game-level stats** from the **nflverse** (via `nflreadpy`), builds your
**QB passing / rushing yards / receiving yards / receptions** benchmarks, and exposes hit counts,
hit rates, and trend charts. When nflverse publishes new weekly stats (usually overnight during the season),
just run the app again and it refreshes automatically.

## How to run
1) Create a fresh Python 3.10+ environment
2) `pip install -r requirements.txt`
3) `streamlit run app.py`

The app will download the latest 2025 data at launch. Use the sidebar to filter by **Position**, **Team**, or **Player**.
Export the filtered view + summaries to Excel from the **Export** section.

## Notes
- Data source: nflverse via `nflreadpy` (see docs).
- Season is set to 2025; change `SEASON` in `app.py` if needed.
- If you want scheduled refresh/exports, run this in a cron or Windows Task Scheduler daily.