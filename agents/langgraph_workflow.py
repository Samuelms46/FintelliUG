from langgraph.graph import StateGraph, END
from .state import AgentState
from database.db_manager import DatabaseManager
from .competitor_agent import CompetitorAnalysisAgent
from .coordinator import CoordinatorAgent
import json

class MultiAgentWorkflow:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.competitor_agent = CompetitorAnalysisAgent()
        self.coordinator_agent = CoordinatorAgent()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # the nodes
        workflow.add_node("fetch_data", self.fetch_data)
        workflow.add_node("process_posts", self.process_posts)
        workflow.add_node("analyze_competitors", self.analyze_competitors)
        workflow.add_node("generate_insights", self.generate_insights)
        workflow.add_node("compile_report", self.compile_report)
        
        # Define edges between nodes to define the flow
        workflow.set_entry_point("fetch_data")
        workflow.add_edge("fetch_data", "process_posts")
        workflow.add_edge("process_posts", "analyze_competitors")
        workflow.add_edge("analyze_competitors", "generate_insights")
        workflow.add_edge("generate_insights", "compile_report")
        workflow.add_edge("compile_report", END)
        
        return workflow.compile()
    
    def fetch_data(self, state: AgentState) -> AgentState:
        """Fetch new social media data"""
        # In production, this would fetch from APIs
        # For now, get unprocessed posts from database
        raw_posts = self.db_manager.get_unprocessed_posts(limit=50)
        
        # Convert SQLAlchemy objects to dictionaries
        posts_data = []
        for post in raw_posts:
            posts_data.append({
                "id": post.id,
                "source": post.source,
                "content": post.content,
                "author": post.author,
                "url": post.url,
                "timestamp": post.timestamp
            })
        
        return {"raw_posts": posts_data}
    
    def process_posts(self, state: AgentState) -> AgentState:
        """Process raw posts through the data pipeline"""
        from data_processing.processor import DataProcessor
        
        processor = DataProcessor()
        processed_posts = []
        
        for raw_post in state["raw_posts"]:
            try:
                result = processor.process_post(raw_post)
                if result:
                    processed_posts.append(result)
                    # Mark post as processed in database
                    self.db_manager.mark_post_processed(raw_post["id"])
            except Exception as e:
                print(f"Error processing post {raw_post['id']}: {e}")
        
        return {"processed_posts": processed_posts}
    
    def analyze_competitors(self, state: AgentState) -> AgentState:
        """Analyze competitor mentions in processed posts"""
        competitor_mentions = []
        
        for post in state["processed_posts"]:
            try:
                # Get full post from database for analysis
                session = self.db_manager.get_session()
                db_post = session.query(SocialMediaPost).get(post["post_id"])
                
                if db_post and db_post.cleaned_content:
                    insights = self.competitor_agent.extract_competitor_insights(
                        post["post_id"], db_post.cleaned_content
                    )
                    competitor_mentions.extend(insights)
            except Exception as e:
                print(f"Error analyzing competitors for post {post['post_id']}: {e}")
        
        return {"competitor_mentions": competitor_mentions}
    
    def generate_insights(self, state: AgentState) -> AgentState:
        """Generate market insights from processed data"""
        market_insights = []
        
        # Generate insights from competitor data
        if state["competitor_mentions"]:
            competitor_report = self.competitor_agent.generate_competitive_intelligence(hours=24)
            if competitor_report.get("summary"):
                for insight in competitor_report["summary"]:
                    market_insights.append({
                        "type": "competitor_intelligence",
                        "content": insight.get("text", ""),
                        "confidence": insight.get("confidence", 0.5)
                    })
        
        # Generate basic sentiment insights
        positive_count = 0
        negative_count = 0
        total = len(state["processed_posts"])
        
        for post in state["processed_posts"]:
            # Get sentiment from database
            session = self.db_manager.get_session()
            db_post = session.query(SocialMediaPost).get(post["post_id"])
            if db_post and db_post.sentiment:
                if db_post.sentiment == "positive":
                    positive_count += 1
                elif db_post.sentiment == "negative":
                    negative_count += 1
        
        if total > 0:
            positive_pct = (positive_count / total) * 100
            negative_pct = (negative_count / total) * 100
            
            market_insights.append({
                "type": "market_sentiment",
                "content": f"Market sentiment: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative",
                "confidence": min(0.9, max(0.5, (abs(positive_pct - negative_pct) / 100) + 0.5))
            })
        
        return {"market_insights": market_insights}
    
    def compile_report(self, state: AgentState) -> AgentState:
        """Compile final intelligence report"""
        from datetime import datetime
        
        # Use coordinator to synthesize all insights
        synthesized = self.coordinator_agent.synthesize_insights(
            [],  # social_insights would come from another agent
            state["competitor_mentions"],
            state["market_insights"]
        )
        
        # Generate daily briefing
        briefing = self.coordinator_agent.generate_daily_briefing(synthesized)
        
        final_report = {
            "timestamp": datetime.now().isoformat(),
            "synthesized_insights": synthesized,
            "daily_briefing": briefing,
            "metrics": {
                "posts_processed": len(state["processed_posts"]),
                "competitor_mentions": len(state["competitor_mentions"]),
                "market_insights": len(state["market_insights"])
            }
        }
        
        # Save to database
        self.db_manager.add_insight({
            "type": "workflow_report",
            "content": json.dumps(final_report),
            "confidence": synthesized.get("confidence", 0.7),
            "source_data": [post["post_id"] for post in state["processed_posts"]]
        })
        
        return {"final_report": final_report}
    
    def run(self):
        """Run the complete workflow"""
        initial_state = AgentState(
            messages=[],
            raw_posts=[],
            processed_posts=[],
            competitor_mentions=[],
            market_insights=[],
            final_report=None,
            errors=[]
        )
        
        return self.workflow.invoke(initial_state)