import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

def format_timestamp(dt):
    """Format datetime for display"""
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "N/A"

def time_ago(dt):
    """Get human-readable time ago string"""
    now = datetime.utcnow()
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600} hours ago"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60} minutes ago"
    else:
        return "just now"

def chunk_list(lst, n):
    """Split a list into chunks of size n"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def safe_json_loads(json_str, default=None):
    """Safely parse JSON string"""
    if default is None:
        default = {}
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default