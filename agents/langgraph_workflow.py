from langgraph.graph import StateGraph, END
from .state import AgentState
from database.db_manager import DatabaseManager
from database.models import SocialMediaPost
from .competitor_agent import CompetitorAnalysisAgent
from .market_sentiment_agent import MarketSentimentAgent
from .coordinator import CoordinatorAgent
from .social_intel_agent import SocialIntelAgent
from utils.logger import app_logger
import json
from datetime import datetime
from typing import Dict, Any, List


class MultiAgentWorkflow:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.competitor_agent = CompetitorAnalysisAgent()
        self.coordinator_agent = CoordinatorAgent()
        self.social_agent = SocialIntelAgent({})  # Initialize with empty config
        self.market_sentiment_agent = MarketSentimentAgent()
        self.workflow = self._build_workflow()

    # ----------------------
    # Helper utilities
    # ----------------------
    def _add_error(self, state: AgentState, message: str) -> AgentState:
        app_logger.error(message)
        errors = state.get("errors", [])
        errors.append(message)
        # return a shallow copy with updated errors to keep state immutable style
        return {**state, "errors": errors}

    def _log_step(self, step_name: str, input_size: int = 0, output_size: int = 0, duration_ms: int = 0, error: str | None = None):
        log_data = {
            "step": step_name,
            "input_size": input_size,
            "output_size": output_size,
            "duration_ms": duration_ms,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        app_logger.info(f"Workflow step completed: {json.dumps(log_data)}")

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
        try:
            raw_posts = self.db_manager.get_unprocessed_posts(limit=50)

            # Convert SQLAlchemy objects to dictionaries
            posts_data = []
            for post in raw_posts:
                posts_data.append(
                    {
                        "id": post.id,
                        "source": post.source,
                        "content": post.content,
                        "author": post.author,
                        "url": post.url,
                        "timestamp": post.timestamp,
                    }
                )

            # If no unprocessed posts, create some mock data for demonstration
            if not posts_data:
                posts_data = [
                    {
                        "id": 1,
                        "source": "reddit",
                        "content": "MTN Mobile Money fees are getting too high. Thinking of switching to Airtel.",
                        "author": "uganda_user123",
                        "url": "https://reddit.com/r/uganda/123",
                        "timestamp": datetime.now(),
                    },
                    {
                        "id": 2,
                        "source": "twitter",
                        "content": "Airtel Money has better network coverage in my village than MTN. Finally can send money home easily!",
                        "author": "rural_user",
                        "url": "https://twitter.com/user/123",
                        "timestamp": datetime.now(),
                    },
                ]

            self._log_step("fetch_data", output_size=len(posts_data))
            return {
                **state,
                "raw_posts": posts_data,
                "processed_posts": [],
                "social_insights": [],
                "competitor_mentions": [],
                "market_insights": [],
                "market_analysis": None,
                "market_health": None,
                "investment_opportunities": [],
                "final_report": None,
                "errors": state.get("errors", [])
            }
        except Exception as e:
            new_state = self._add_error(state, f"Error in fetch_data: {e}")
            return {
                **new_state,
                "raw_posts": [],
                "processed_posts": [],
                "social_insights": [],
                "competitor_mentions": [],
                "market_insights": [],
                "market_analysis": None,
                "market_health": None,
                "investment_opportunities": [],
                "final_report": None
            }

    def process_posts(self, state: AgentState) -> AgentState:
        """Process raw posts through the data pipeline"""
        from data_processing.processor import DataProcessor

        try:
            raw_posts = state.get("raw_posts", [])
            if not raw_posts:
                app_logger.warning("No raw_posts available for processing")
                return {**state, "processed_posts": []}

            processor = DataProcessor()
            processed_posts = []

            for raw_post in raw_posts:
                try:
                    result = processor.process_post(raw_post)
                    if result:
                        processed_posts.append(result)
                        # Mark post as processed in database
                        self.db_manager.mark_post_processed(raw_post.get("id"))
                except Exception as e:
                    app_logger.error(f"Error processing post {raw_post.get('id')}: {e}")
                    errors = state.get("errors", []) + [f"Error processing post {raw_post.get('id')}: {e}"]
                    state = {**state, "errors": errors}

            self._log_step("process_posts", input_size=len(raw_posts), output_size=len(processed_posts))
            return {**state, "processed_posts": processed_posts}
        except Exception as e:
            new_state = self._add_error(state, f"Error in process_posts: {e}")
            return {**new_state, "processed_posts": []}

    def social_intelligence(self, state: AgentState) -> AgentState:
        """Run social intelligence analysis"""
        try:
            processed_posts = state.get("processed_posts", [])
            posts_for_analysis: List[Dict[str, Any]] = []

            if not processed_posts:
                app_logger.warning("No processed_posts for social intelligence analysis")
                return {**state, "social_insights": []}

            for post in processed_posts:
                # Get full post from database
                session = self.db_manager.get_session()
                try:
                    db_post = session.query(SocialMediaPost).get(post.get("post_id"))
                    if db_post:
                        posts_for_analysis.append(
                            {
                                "text": db_post.cleaned_content or db_post.content,
                                "source": db_post.source,
                                "timestamp": db_post.timestamp.isoformat() if db_post.timestamp else datetime.now().isoformat(),
                            }
                        )
                finally:
                    session.close()

            # Run social intelligence analysis
            social_result = self.social_agent.process({"posts": posts_for_analysis})
            insights = social_result.get("insights", []) if isinstance(social_result, dict) else []

            self._log_step("social_intelligence", input_size=len(posts_for_analysis), output_size=len(insights))
            return {**state, "social_insights": insights}
        except Exception as e:
            new_state = self._add_error(state, f"Error in social intelligence: {e}")
            return {**new_state, "social_insights": []}

    def analyze_market_sentiment(self, state: AgentState) -> AgentState:
        """Analyze market sentiment and trends"""
        try:
            processed_posts = state.get("processed_posts", [])
            if not processed_posts:
                app_logger.warning("No processed_posts found for market sentiment analysis")
                return {**state, "market_analysis": {}, "market_health": "unknown", "investment_opportunities": []}

            # Convert to format expected by market sentiment agent
            posts_for_analysis = []
            for post in processed_posts:
                session = self.db_manager.get_session()
                try:
                    db_post = session.query(SocialMediaPost).get(post.get("post_id"))
                    if db_post:
                        posts_for_analysis.append(
                            {
                                "text": db_post.cleaned_content or db_post.content,
                                "source": db_post.source,
                                "timestamp": db_post.timestamp.isoformat() if db_post.timestamp else datetime.now().isoformat(),
                                "sentiment": db_post.sentiment,
                                "sentiment_score": db_post.sentiment_score,
                            }
                        )
                finally:
                    session.close()

            market_result = self.market_sentiment_agent.process({"posts": posts_for_analysis})
            market_result = market_result if isinstance(market_result, dict) else {}

            output: AgentState = {
                **state,
                "market_analysis": market_result,
                "market_health": market_result.get("health_indicators", {}).get("market_health", "unknown"),
                "investment_opportunities": market_result.get("investment_opportunities", []),
            }
            self._log_step("analyze_market_sentiment", input_size=len(posts_for_analysis), output_size=len(output.get("investment_opportunities", [])))
            return output
        except Exception as e:
            new_state = self._add_error(state, f"Error in market sentiment analysis: {e}")
            return {
                **new_state, 
                "market_analysis": {}, 
                "market_health": "unknown", 
                "investment_opportunities": []
            }

    def analyze_competitors(self, state: AgentState) -> AgentState:
        """Analyze competitor mentions in processed posts"""
        try:
            processed_posts = state.get("processed_posts", [])
            competitor_mentions: List[Dict[str, Any]] = []

            for post in processed_posts:
                try:
                    session = self.db_manager.get_session()
                    try:
                        db_post = session.query(SocialMediaPost).get(post.get("post_id"))
                        if db_post and (db_post.cleaned_content or db_post.content):
                            text = db_post.cleaned_content or db_post.content
                            # Use the process method with appropriate input data
                            result = self.competitor_agent.process({
                                "posts": [{"id": post.get("post_id"), "content": text}]
                            })
                            if "competitor_mentions" in result:
                                competitor_mentions.extend(result["competitor_mentions"])
                    finally:
                        session.close()
                except Exception as inner_e:
                    app_logger.error(f"Error analyzing competitors for post {post.get('post_id')}: {inner_e}")
                    errors = state.get("errors", []) + [f"Error analyzing competitors for post {post.get('post_id')}: {inner_e}"]
                    state = {**state, "errors": errors}

            self._log_step("analyze_competitors", input_size=len(processed_posts), output_size=len(competitor_mentions))
            return {**state, "competitor_mentions": competitor_mentions}
        except Exception as e:
            new_state = self._add_error(state, f"Error in analyze_competitors: {e}")
            return {**new_state, "competitor_mentions": []}

    def generate_insights(self, state: AgentState) -> AgentState:
        """Generate market insights from processed data"""
        try:
            market_insights: List[Dict[str, Any]] = []

            # Generate insights from competitor data
            if state.get("competitor_mentions"):
                # Use the process method for competitive intelligence
                competitor_result = self.competitor_agent.process({"hours": 24})
                if isinstance(competitor_result, dict) and competitor_result.get("summary"):
                    for insight in competitor_result["summary"]:
                        market_insights.append(
                            {
                                "type": "competitor_intelligence",
                                "content": insight.get("text", ""),
                                "confidence": insight.get("confidence", 0.5),
                            }
                        )

            # Generate basic sentiment insights
            positive_count = 0
            negative_count = 0
            processed_posts = state.get("processed_posts", [])
            total = len(processed_posts)

            for post in processed_posts:
                session = self.db_manager.get_session()
                try:
                    db_post = session.query(SocialMediaPost).get(post.get("post_id"))
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

                market_insights.append(
                    {
                        "type": "market_sentiment",
                        "content": f"Market sentiment: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative",
                        "confidence": min(0.9, max(0.5, (abs(positive_pct - negative_pct) / 100) + 0.5)),
                    }
                )

            self._log_step("generate_insights", input_size=total, output_size=len(market_insights))
            return {**state, "market_insights": market_insights}
        except Exception as e:
            new_state = self._add_error(state, f"Error in generate_insights: {e}")
            return {**new_state, "market_insights": []}

    def compile_report(self, state: AgentState) -> AgentState:
        """Compile final intelligence report"""
        try:
            # Use coordinator to synthesize all insights
            synthesized = self.coordinator_agent.synthesize_insights(
                state.get("social_insights", []),
                state.get("competitor_mentions", []),
                state.get("market_insights", []),
                state.get("market_analysis", {}),
            )

            # Generate daily briefing
            briefing = self.coordinator_agent.generate_daily_briefing(synthesized)

            final_report = {
                "timestamp": datetime.now().isoformat(),
                "synthesized_insights": synthesized,
                "daily_briefing": briefing,
                "metrics": {
                    "posts_processed": len(state.get("processed_posts", [])),
                    "competitor_mentions": len(state.get("competitor_mentions", [])),
                    "market_insights": len(state.get("market_insights", [])),
                    "social_insights": len(state.get("social_insights", [])),
                },
                "errors": state.get("errors", []),
            }

            # Save to database (best-effort)
            try:
                self.db_manager.add_insight(
                    {
                        "type": "workflow_report",
                        "content": json.dumps(final_report),
                        "confidence": synthesized.get("confidence", 0.7) if isinstance(synthesized, dict) else 0.7,
                        "source_data": [post.get("post_id") for post in state.get("processed_posts", [])],
                    }
                )
            except Exception as db_err:
                app_logger.error(f"Failed to persist workflow_report: {db_err}")
                errors = state.get("errors", []) + [f"DB persist error in compile_report: {db_err}"]
                state = {**state, "errors": errors}

            self._log_step("compile_report")
            return {**state, "final_report": final_report}
        except Exception as e:
            new_state = self._add_error(state, f"Error in compile_report: {e}")
            # Provide a minimal report to keep continuity
            minimal_report = {
                "timestamp": datetime.now().isoformat(),
                "synthesized_insights": {},
                "daily_briefing": {},
                "metrics": {},
                "errors": new_state.get("errors", []),
            }
            return {**new_state, "final_report": minimal_report}

    def run(self):
        """Run the complete workflow"""
        initial_state: AgentState = {
            "messages": [],
            "raw_posts": [],
            "processed_posts": [],
            "competitor_mentions": [],
            "market_insights": [],
            "social_insights": [],
            "market_analysis": None,
            "market_health": None,
            "investment_opportunities": [],
            "final_report": None,
            "errors": [],
        }

        return self.workflow.invoke(initial_state)