from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    # Input from previous node
    messages: Annotated[List[str], add_messages]
    
    # Processed data
    raw_posts: List[dict]
    processed_posts: List[dict]
    social_insights: List[dict]
    competitor_mentions: List[dict]
    market_insights: List[dict]
    
    # Final output
    final_report: Optional[dict]
    errors: List[str]