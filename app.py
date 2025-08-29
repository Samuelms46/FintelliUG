import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from database.db_manager import DatabaseManager
from data_processing.processor import DataProcessor
from agents.social_intel_agent import SocialIntelAgent
from agents.competitor_agent import CompetitorAnalysisAgent
from agents.market_sentiment_agent import MarketSentimentAgent
from agents.langgraph_workflow import MultiAgentWorkflow
from utils.helpers import format_timestamp, time_ago, safe_json_loads
from config import Config
import json


# Social Intelligence section
st.subheader("Social Intelligence")

social_query = st.text_input(
    "Social search query:",
    value="Uganda fintech",
    key="social_query"
)
social_max = st.slider("Max results", 5, 50, 20, key="social_max")

if st.button("Run Social Intelligence", key="run_social"):
    with st.spinner("Running social intelligence..."):
        try:
            agent = SocialIntelAgent(config={})  # config not used, but required by signature
            result = agent.process({"query": social_query, "max_results": social_max})

            if result.get("error"):
                st.error(result["error"])
            else:
                # High-level metrics
                cols = st.columns(3)
                cols[0].metric("Posts Processed", result.get("posts_processed", 0))
                cols[1].metric("Relevant Posts", result.get("relevant_posts", 0))
                dq = result.get("data_quality_score", 0)
                cols[2].metric("Data Quality", f"{dq*100:.1f}%")

                # Sentiment summary
                sa = result.get("sentiment_analysis", {}) or {}
                st.markdown("### Sentiment Overview")
                st.write(f"Overall: {sa.get('overall_sentiment', 'unknown').title()}")
                st.write(f"Score: {sa.get('sentiment_score', 0):.2f}")

                # Trending topics
                topics = result.get("trending_topics", []) or []
                if topics:
                    st.markdown("### Trending Topics")
                    for t in topics[:5]:
                        st.write(f"- {t.get('topic', 'topic')} ({t.get('mention_count', 0)} mentions)")

                # Insights
                insights = result.get("insights", []) or []
                if insights:
                    st.markdown("### Insights")
                    for i, ins in enumerate(insights[:5], start=1):
                        text = ins.get("insight") or ins.get("text") or str(ins)
                        st.write(f"{i}. {text}")

                # Optional: inspect raw payload
                with st.expander("Show raw result"):
                    st.json(result)

        except Exception as e:
            st.error(f"Failed to run SocialIntelAgent: {e}")
from data_collection.reddit_collector import RedditDataCollector
from database.models import Insight

# Set page config
st.set_page_config(
    page_title="FintelliUG - Uganda Fintech Intelligence",
    page_icon="💹",
    layout="wide"
)

# Initialize components
try:
    db_manager = DatabaseManager()
    data_processor = DataProcessor()
    competitor_agent = CompetitorAnalysisAgent()
    reddit_collector = RedditDataCollector()
except Exception as e:
    st.error(f"Failed to initialize components: {str(e)}")
    st.info("Please check your environment configuration and API keys")
    st.stop()

# App title
st.title("💹 FintelliUG - Uganda Fintech Intelligence Platform")
st.markdown("""
Agentic audience insights platform for Uganda's fintech ecosystem.
Monitor social conversations, analyze sentiment, and generate competitive intelligence.
""")

# Sidebar
st.sidebar.header("Controls")

# ADDING REDDIT COLLECTION TO SIDEBAR
st.sidebar.header("Data Collection")

if st.sidebar.button("🌐 Collect from Reddit"):
    with st.spinner("Collecting data from Reddit..."):
        try:
            results = data_processor.collect_reddit_data(limit=15)
            st.sidebar.success(f"Collected and processed {len(results)} posts from Reddit")
        except Exception as e:
            st.sidebar.error(f"Error collecting from Reddit: {e}")

# ADD REDDIT-SPECIFIC SEARCH
reddit_query = st.sidebar.text_input("Reddit Search Query:", value="Uganda fintech")
reddit_limit = st.sidebar.slider("Number of posts:", 5, 50, 15)

if st.sidebar.button("🔍 Custom Reddit Search"):
    with st.spinner(f"Searching Reddit for '{reddit_query}'..."):
        try:
            results = data_processor.collect_reddit_data(query=reddit_query, limit=reddit_limit)
            st.sidebar.success(f"Found and processed {len(results)} posts")
        except Exception as e:
            st.sidebar.error(f"Search failed: {e}")

# SIDEBAR CONTROLS
if st.sidebar.button("🔄 Collect & Process New Data"):
    with st.spinner("Collecting and processing data..."):
        # This would fetch real data in production
        # For demonstrating this with a mock data/information
        mock_posts = [
            {
                "source": "reddit",
                "content": "MTN Mobile Money fees are getting too high. Thinking of switching to Airtel.",
                "author": "uganda_user123",
                "url": "https://reddit.com/r/uganda/123",
                "timestamp": datetime.utcnow() - timedelta(hours=2)
            },
            {
                "source": "twitter",
                "content": "Airtel Money has better network coverage in my village than MTN. Finally can send money home easily!",
                "author": "rural_user",
                "url": "https://twitter.com/user/123",
                "timestamp": datetime.utcnow() - timedelta(hours=5)
            },
            {
                "source": "news",
                "content": "Bank of Uganda announces new regulations for mobile money providers. Will this increase costs for consumers?",
                "author": "BusinessDaily",
                "url": "https://businessdaily.ug/123",
                "timestamp": datetime.utcnow() - timedelta(days=1)
            }
        ]
        
        results = data_processor.batch_process(mock_posts)
        st.sidebar.success(f"Processed {len(results)} posts")

if st.sidebar.button("🤖 Run Intelligence Workflow"):
    with st.spinner("Running multi-agent workflow..."):
        try:
            workflow = MultiAgentWorkflow()
            result = workflow.run()
            st.sidebar.success("Workflow completed!")
            
            # Show workflow results
            if result and 'final_report' in result:
                st.sidebar.info(f"Processed {result['final_report'].get('metrics', {}).get('posts_processed', 0)} posts")
                st.sidebar.info(f"Generated {result['final_report'].get('metrics', {}).get('market_insights', 0)} insights")
        except Exception as e:
            st.sidebar.error(f"Workflow failed: {str(e)}")
            st.sidebar.info("Check your API keys and environment configuration")

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Dashboard", 
    "💬 Social Posts", 
    "🏢 Competitor Analysis",
    "🔍 Vector Search",
    "📋 Insights"
    "📊 Market Health"
])

with tab1:
    st.header("Market Overview")
    
    # Get recent posts for analytics
    recent_posts = db_manager.get_recent_posts(hours=48)
    
    if recent_posts:
        # Sentiment distribution
        sentiment_counts = {}
        for post in recent_posts:
            sentiment = post.sentiment or "unknown"
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        if sentiment_counts:
            fig1 = px.pie(
                values=list(sentiment_counts.values()),
                names=list(sentiment_counts.keys()),
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # Topic frequency
        topic_counts = {}
        for post in recent_posts:
            if post.topics:
                topics = safe_json_loads(post.topics)
                for topic in topics:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        if topic_counts:
            topics_df = pd.DataFrame({
                "Topic": list(topic_counts.keys()),
                "Count": list(topic_counts.values())
            }).sort_values("Count", ascending=False)
            
            fig2 = px.bar(
                topics_df.head(10),
                x="Topic",
                y="Count",
                title="Top Topics (Last 48 hours)"
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data available. Click 'Collect & Process New Data' to get started.")

with tab2:
    st.header("Recent Social Media Posts")
    
    posts = db_manager.get_recent_posts(limit=50)
    if posts:
        for post in posts:
            with st.expander(f"{post.source.title()} post by {post.author} - {time_ago(post.timestamp)}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(post.cleaned_content or post.content)
                    if post.url:
                        st.caption(f"[Source]({post.url})")
                
                with col2:
                    if post.sentiment:
                        sentiment_color = {
                            "positive": "green",
                            "negative": "red",
                            "neutral": "gray"
                        }.get(post.sentiment, "gray")
                        
                        st.markdown(
                            f"**Sentiment**: :{sentiment_color}[{post.sentiment}] "
                            f"({(post.sentiment_score or 0) * 100:.1f}%)"
                        )
                    
                    if post.topics:
                        topics = safe_json_loads(post.topics)
                        st.markdown("**Topics**: " + ", ".join(topics))
                    
                    st.markdown(f"**Relevance**: {post.relevance_score or 0:.2f}")
    else:
        st.info("No posts found. Collect some data first.")

with tab3:
    st.header("Competitor Analysis")
    
    if st.button("Analyze Competitor Mentions"):
        with st.spinner("Analyzing competitor data..."):
            report = competitor_agent.generate_competitive_intelligence(hours=24)
            
            st.subheader("Competitive Intelligence Summary")
            st.write(f"Period: Last {report['time_period_hours']} hours")
            st.write(f"Total competitor mentions: {report['total_mentions']}")
            
            competitor_items = report.get('competitor_insights') or report.get('competitor_mentions', [])
            if competitor_items:
                # Competitor sentiment
                competitor_sentiment = {}
                for insight in competitor_items:
                    competitor = insight['competitor']
                    sentiment = insight['sentiment']
                    
                    if competitor not in competitor_sentiment:
                        competitor_sentiment[competitor] = {'positive': 0, 'negative': 0, 'neutral': 0}
                    
                    competitor_sentiment[competitor][sentiment] += 1
                
                # Display sentiment by competitor
                for competitor, sentiments in competitor_sentiment.items():
                    total = sum(sentiments.values())
                    st.subheader(competitor)
                    
                    cols = st.columns(3)
                    cols[0].metric("Positive", sentiments['positive'], 
                                  f"{(sentiments['positive']/total*100):.1f}%")
                    cols[1].metric("Negative", sentiments['negative'],
                                  f"{(sentiments['negative']/total*100):.1f}%")
                    cols[2].metric("Neutral", sentiments['neutral'],
                                  f"{(sentiments['neutral']/total*100):.1f}%")
            
            # Summary insights
            if report.get('summary'):
                st.subheader("Key Insights")
                for insight in report['summary']:
                    st.info(f"{insight.get('text', '')} (Confidence: {insight.get('confidence', 0)*100:.1f}%)")
            else:
                st.info("No insights generated from competitor analysis")
    else:
        st.info("Click the button to analyze competitor mentions")

# VECTOR SEARCH TAB 
with tab4:
    st.header("Vector Search")
    
    st.subheader("Search Similar Content")
    search_query = st.text_input("Enter search query:", placeholder="e.g., mobile money fees", key="vector_search")
    
    if st.button("Search", key="search_btn") and search_query:
        with st.spinner("Searching similar content..."):
            results = db_manager.search_similar_posts(search_query, n_results=5)
            
            if results:
                st.success(f"Found {len(results)} similar posts")
                for result in results:
                    with st.expander(f"Similarity: {1 - result['distance']:.3f}"):
                        st.write(result["content"])
                        st.caption(f"Source: {result['metadata']['source']} | "
                                  f"Topics: {result['metadata']['topics']} | "
                                  f"Sentiment: {result['metadata']['sentiment']}")
            else:
                st.info("No similar content found")
    
    st.subheader("Search by Topic")
    topic_options = list(Config.FINTECH_TOPICS.keys()) + ["All Topics"]
    selected_topic = st.selectbox("Select topic:", topic_options, key="topic_select")
    
    if st.button("Search by Topic", key="topic_btn") and selected_topic != "All Topics":
        with st.spinner(f"Searching for {selected_topic} content..."):
            results = db_manager.search_posts_by_topic(selected_topic, n_results=8)
            
            if results:
                st.success(f"Found {len(results)} posts about {selected_topic}")
                for result in results:
                    with st.expander(f"Topic relevance: {1 - result['distance']:.3f}"):
                        st.write(result["content"])
                        st.caption(f"Source: {result['metadata']['source']} | "
                                  f"Sentiment: {result['metadata']['sentiment']}")
            else:
                st.info(f"No content found about {selected_topic}")
    
    st.subheader("Vector Database Statistics")
    if st.button("Show Stats", key="stats_btn"):
        stats = db_manager.get_vector_db_stats()
        st.metric("Total Documents in Vector DB", stats["total_documents"])
        st.write(f"Collection: {stats['collection_name']}")

# INSIGHTS TAB
with tab5:
    st.header("AI-Generated Insights")
    
    insights = db_manager.get_session().query(Insight).order_by(Insight.created_at.desc()).limit(10).all()
    
    if insights:
        for insight in insights:
            with st.expander(f"{insight.type} - {time_ago(insight.created_at)}"):
                st.write(insight.content)
                st.caption(f"Confidence: {insight.confidence * 100:.1f}%")
                
                if insight.source_data:
                    st.caption(f"Based on {len(safe_json_loads(insight.source_data))} data points")
    else:
        st.info("No insights yet. Run the intelligence workflow to generate insights.")

with tab6:
    st.header("Market Health Analysis")

    time_period = st.selectbox(
        "Analysis Period:",
        ["Last 24 hours", "Last 7 days", "Last 30 days"]
    )

    if st.button("Analyze Market Health"):
        with st.spinner("Analyzing market health..."):
            # Convert time period to hours
            hours = 24
            if time_period == "Last 7 days":
                hours = 24 * 7
            elif time_period == "Last 30 days":
                hours = 24 * 30

            # Run market sentiment analysis
            market_agent = MarketSentimentAgent()
            result = market_agent.process({"hours": hours})

            if "error" in result and result["error"]:
                st.error(result["error"])
            else:
                health = result["health_indicators"]

                # Create columns for metrics
                col1, col2, col3 = st.columns(3)

                # Market health status with color (reserved for future styling)
                health_color = {
                    "strong": "green",
                    "stable": "blue",
                    "caution": "orange",
                    "weak": "red"
                }.get(health.get("market_health", "unknown"), "gray")

                # Market Health metric
                col1.metric(
                    "Market Health",
                    (health.get("market_health") or "unknown").title(),
                    f"{health.get('health_score', 0)*100:.1f}%"
                )

                # Opportunity Score metric
                col2.metric(
                    "Opportunity Score",
                    f"{health.get('opportunity_score', 0)*10:.1f}/10",
                    None
                )

                # Risk Level metric
                risk_val = health.get("risk_level", 0.5)
                risk_level = "Low" if risk_val < 0.3 else "Medium" if risk_val < 0.7 else "High"
                col3.metric(
                    "Risk Level",
                    risk_level,
                    None
                )

                # Optional: show growth segments if present
                if health.get("growth_segments"):
                    st.subheader("Growth Segments")
                    st.write(", ".join(health["growth_segments"]))

# Footer
st.markdown("---")
st.caption("FintelliUG MVP - Uganda Fintech Intelligence Platform")