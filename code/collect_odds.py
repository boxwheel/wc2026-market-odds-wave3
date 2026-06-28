"""
Collect pre-match 1X2 odds for all 64 WC-2026 group-stage matches.
Strategy:
1. Use Polymarket data from amirdaraee/world-cup-predictions (56 matches)
2. Supplement with odds-api.io or web-scraped historical odds
3. Manually curated fallback for remaining matches
"""
import json, requests, os, re
import pandas as pd
import numpy as np

DATA_DIR = "/root/research/wave3-market-odds/fifa_data"
OUT_DIR = "/root/research/wave3-market-odds/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# Load canonical match list
matches = pd.read_csv(f"{DATA_DIR}/matches_detailed.csv")
completed = matches[matches['status'] == 'Completed'].copy()
completed = completed.reset_index(drop=True)
print(f"Total completed matches: {len(completed)}")
print(completed[['match_id','home_team_name','away_team_name','date']].to_string())
