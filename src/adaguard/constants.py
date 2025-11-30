from pathlib import Path

# __file__ is repo/src/adaguard/constants.py
# parent is ada, p2 is src, p3 is repo
REPO_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = REPO_ROOT / ".env"
DATA_DIR = REPO_ROOT / "data"
WILDJAIL_SUBSET_PATH = DATA_DIR / "wildjail_subset"

GPT_OSS_20B = "openai/gpt-oss-20b"
