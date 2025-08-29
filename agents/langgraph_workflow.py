from langgraph.graph import StateGraph, END
from .state import AgentState
from database.db_manager import DatabaseManager
from database.models import SocialMediaPost
from .competitor_agent import CompetitorAnalysisAgent
from .market_sentiment_agent import MarketSentimentAgent
from .coordinator import CoordinatorAgent
from .social_intel_agent import SocialIntelAgent
import json
from datetime import datetime

class MultiAgentWorkflow:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.competitor_agent = CompetitorAnalysisAgent()
        self.coordinator_agent = CoordinatorAgent()
        self.social_agent = SocialIntelAgent({})  # Initialize with empty config
        self.market_sentiment_agent = MarketSentimentAgent()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add the nodes
        workflow.add_node("fetch_data", self.fetch_data)
        workflow.add_node("process_posts", self.process_posts)
        workflow.add_node("social_intelligence", self.social_intelligence)
        workflow.add_node("analyze_market_sentiment", self.analyze_market_sentiment)
        workflow.add_node("analyze_competitors", self.analyze_competitors)
        workflow.add_node("generate_insights", self.generate_insights)
        workflow.add_node("compile_report", self.compile_report)
        
        # Define edges between nodes to define the flow
        workflow.set_entry_point("fetch_data")
        workflow.add_edge("fetch_data", "process_posts")
        workflow.add_edge("process_posts", "social_intelligence")
        workflow.add_edge("social_intelligence", "analyze_market_sentiment")
        workflow.add_edge("analyze_market_sentiment", "analyze_competitors")
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
        
        # If no unprocessed posts, create some mock data for demonstration
        if not posts_data:
            posts_data = [
                {
                    "id": 1,
                    "source": "reddit",
                    "content": "MTN Mobile Money fees are getting too high. Thinking of switching to Airtel.",
                    "author": "uganda_user123",
                    "url": "https://reddit.com/r/uganda/123",
                    "timestamp": datetime.now()
                },
                {
                    "id": 2,
                    "source": "twitter",
                    "content": "Airtel Money has better network coverage in my village than MTN. Finally can send money home easily!",
                    "author": "rural_user",
                    "url": "https://twitter.com/user/123",
                    "timestamp": datetime.now()
                }
            ]
        
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
    
    def social_intelligence(self, state: AgentState) -> AgentState:
        """Run social intelligence analysis"""
        try:
            # Convert processed posts to format expected by social agent
            posts_for_analysis = []
            for post in state["processed_posts"]:
                # Get full post from database
                session = self.db_manager.get_session()
                try:
                    db_post = session.query(SocialMediaPost).get(post["post_id"])
                    if db_post:
                        posts_for_analysis.append({
                            "text": db_post.cleaned_content or db_post.content,
                            "source": db_post.source,
                            "timestamp": db_post.timestamp.isoformat() if db_post.timestamp else datetime.now().isoformat()
                        })
                finally:
                    session.close()
            
            # Run social intelligence analysis
            social_result = self.social_agent.process({
                "posts": posts_for_analysis
            })
            
            return {"social_insights": social_result.get("insights", [])}
            
        except Exception as e:
            print(f"Error in social intelligence: {e}")
            return {"social_insights": []}
    
    def analyze_market_sentiment(self, state: AgentState) -> AgentState:
        """Analyze market sentiment and trends"""
        try:
            # Get processed posts
            processed_posts = state["processed_posts"]
            
            # Convert to format expected by market sentiment agent
            posts_for_analysis = []
            for post in processed_posts:
                # Get full post from database
                session = self.db_manager.get_session()
                try:
                    db_post = session.query(SocialMediaPost).get(post["post_id"])
                    if db_post:
                        posts_for_analysis.append({
                            "text": db_post.cleaned_content or db_post.content,
                            "source": db_post.source,
                            "timestamp": db_post.timestamp.isoformat() if db_post.timestamp else datetime.now().isoformat(),
                            "sentiment": db_post.sentiment,
                            "sentiment_score": db_post.sentiment_score
                        })
                finally:
                    session.close()
            
            # Run market sentiment analysis
            market_result = self.market_sentiment_agent.process({
                "posts": posts_for_analysis
            })
            
            return {
                "market_analysis": market_result,
                "market_health": market_result.get("health_indicators", {}).get("market_health", "unknown"),
                "investment_opportunities": market_result.get("investment_opportunities", [])
            }
            
        except Exception as e:
            print(f"Error in market sentiment analysis: {e}")
            return {
                "market_analysis": {},
                "market_health": "unknown",
                "investment_opportunities": []
            }
    
    def analyze_competitors(self, state: AgentState) -> AgentState:
        """Analyze competitor mentions in processed posts"""
        competitor_mentions = []
        
        for post in state["processed_posts"]:
            try:
                # Get full post from database for analysis
                session = self.db_manager.get_session()
                try:
                    db_post = session.query(SocialMediaPost).get(post["post_id"])
                    
                    if db_post and db_post.cleaned_content:
                        insights = self.competitor_agent.extract_competitor_insights(
                            post["post_id"], db_post.cleaned_content
                        )
                        competitor_mentions.extend(insights)
                finally:
                    session.close()
            except Exception as e:
                print(f"Error analyzing competitors for post {post['post_id']}: {e}")
        
        return {"competitor_mentions": competitor_mentions}
    
    def generate_insights(self, state: AgentState) -> AgentState:
        """Generate market insights from processed data"""
        from database.models import SocialMediaPost
        
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
            try:
                db_post = session.query(SocialMediaPost).get(post["post_id"])
                if db_post and db_post.sentiment:
                    if db_post.sentiment == "positive":
                        positive_count += 1
                    elif db_post.sentiment == "negative":
                        negative_count += 1
            finally:
                session.close()
        
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
            state.get("social_insights", []),  # social_insights from social agent
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
                "market_insights": len(state["market_insights"]),
                "social_insights": len(state.get("social_insights", []))
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
            social_insights=[],
            final_report=None,
            errors=[]
        )
        
        return self.workflow.invoke(initial_state)
