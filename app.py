import json
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
from config import Config
from database.db_manager import DatabaseManager
from database.models import Insight
from data_processing.processor import DataProcessor
from data_collection.reddit_collector import RedditDataCollector
from agents.social_intel_agent import SocialIntelAgent
from agents.competitor_agent import CompetitorAnalysisAgent
from agents.market_sentiment_agent import MarketSentimentAgent
from agents.langgraph_workflow import MultiAgentWorkflow
from utils.helpers import format_timestamp, time_ago, safe_json_loads


# ===== Helper Functions =====

class LoadingStateManager:
    """Manages loading states for different operations."""
    def __init__(self):
        if "loading_states" not in st.session_state:
            st.session_state.loading_states = {}
    
    def start_loading(self, key, message="Loading..."):
        st.session_state.loading_states[key] = {"active": True, "message": message}
        return st.spinner(message)
    
    def stop_loading(self, key):
        if key in st.session_state.loading_states:
            st.session_state.loading_states[key]["active"] = False
    
    def is_loading(self, key):
        return key in st.session_state.loading_states and st.session_state.loading_states[key]["active"]


def init_session_state():
    """Initialize session state variables."""
    if 'components' not in st.session_state:
        st.session_state.components = None
    if 'recent_posts' not in st.session_state:
        st.session_state.recent_posts = None
    if 'last_analysis_time' not in st.session_state:
        st.session_state.last_analysis_time = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'use_mock_data' not in st.session_state:
        st.session_state.use_mock_data = True
    if 'social_last_run' not in st.session_state:
        st.session_state.social_last_run = None
    if 'market_last_run' not in st.session_state:
        st.session_state.market_last_run = None
    if 'competitor_last_run' not in st.session_state:
        st.session_state.competitor_last_run = None
    if 'social_last_result' not in st.session_state:
        st.session_state.social_last_result = None
    if 'workflow_last_result' not in st.session_state:
        st.session_state.workflow_last_result = None
    if 'workflow_last_run' not in st.session_state:
        st.session_state.workflow_last_run = None


def display_metrics(title, metrics_dict):
    """Display a set of metrics in columns."""
    st.subheader(title)
    cols = st.columns(len(metrics_dict))
    for i, (label, value) in enumerate(metrics_dict.items()):
        cols[i].metric(label, value)


def display_insights(insights_list, max_items=5):
    """Display a list of insights with consistent formatting."""
    if not insights_list:
        st.info("No insights available.")
        return
        
    for i, insight in enumerate(insights_list[:max_items], start=1):
        text = insight.get("insight") or insight.get("text") or str(insight)
        st.write(f"{i}. {text}")


def safe_agent_call(agent_func, error_message, *args, **kwargs):
    """Safely call an agent function with proper error handling."""
    try:
        return agent_func(*args, **kwargs)
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        st.info("Check logs for more details.")
        return {"error": str(e)}


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_recent_posts(_db_manager, hours=48, limit=None):
    """Get recent posts with caching.
    
    Note: _db_manager has a leading underscore to tell Streamlit not to hash it.
    """
    if limit:
        return _db_manager.get_recent_posts(limit=limit)
    return _db_manager.get_recent_posts(hours=hours)


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_social_intelligence(query, max_results, use_mock=False):
    """Run social intelligence analysis with caching."""
    agent = SocialIntelAgent(config={})
    if use_mock:
        # Import and use the enhanced mock data
        from utils.mock_data import get_enhanced_mock_data
        return agent.process({"posts": get_enhanced_mock_data()})
    else:
        return agent.process({"query": query, "max_results": max_results})


def get_mock_posts():
    """Return mock posts for testing."""
    return [
        {
            "source": "reddit",
            "content": "MTN Mobile Money fees are getting too high. Thinking of switching to Airtel.",
            "author": "uganda_user123",
            "url": "https://reddit.com/r/uganda/123",
            "timestamp": datetime.now(tz=None) - timedelta(hours=2)
        },
        {
            "source": "twitter",
            "content": "Airtel Money has better network coverage in my village than MTN. Finally can send money home easily!",
            "author": "rural_user",
            "url": "https://twitter.com/user/123",
            "timestamp": datetime.now(tz=None) - timedelta(hours=5)
        },
        {
            "source": "news",
            "content": "Bank of Uganda announces new regulations for mobile money providers. Will this increase costs for consumers?",
            "author": "BusinessDaily",
            "url": "https://businessdaily.ug/123",
            "timestamp": datetime.now(tz=None) - timedelta(days=1)
        }
    ]


def run_intelligence_workflow(progress_callback=None):
    """Run the multi-agent workflow with optional progress indicators."""
    workflow = MultiAgentWorkflow()
    
    if progress_callback:
        # Run with progress updates
        return workflow.run()
    else:
        # Run without progress updates
        return workflow.run()


# ===== Component Initialization =====

def initialize_components():
    """Initialize and return all required components."""
    try:
        components = {
            "db_manager": DatabaseManager(),
            "data_processor": DataProcessor(),
            "competitor_agent": CompetitorAnalysisAgent(),
            "reddit_collector": RedditDataCollector()
        }
        return components
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.info("Please check your environment configuration and API keys")
        st.stop()


# ===== UI Rendering Functions =====

def render_header():
    """Render the app header and description."""
    st.title("💹 FintelliUG - Uganda Fintech Intelligence Platform")
    st.markdown("""
    Agentic audience insights platform for Uganda's fintech ecosystem.
    Monitor social conversations, analyze sentiment, and generate competitive intelligence.
    """)
    # Last-run and cache info
    statuses = []
    if st.session_state.get('social_last_run'):
        statuses.append(f"Social: {time_ago(st.session_state.social_last_run)}")
    if st.session_state.get('competitor_last_run'):
        statuses.append(f"Competitor: {time_ago(st.session_state.competitor_last_run)}")
    if st.session_state.get('market_last_run'):
        statuses.append(f"Market: {time_ago(st.session_state.market_last_run)}")
    if st.session_state.get('workflow_last_run'):
        statuses.append(f"Workflow: {time_ago(st.session_state.workflow_last_run)}")
    if statuses:
        st.caption("Last runs — " + " | ".join(statuses))
    st.caption("Cache TTLs — Recent posts: 1h, Social intelligence: 30m")


def render_social_intelligence_section(components):
    """Render the social intelligence section."""
    st.subheader("Social Intelligence")

    use_mock_data = st.checkbox("Use enhanced mock data", value=False, key="use_enhanced_mock")
    
    col1, col2 = st.columns(2)

    with col1:
        social_query = st.text_input(
            "Social search query:",
            value="Uganda fintech",
            key="social_query",
            disabled=use_mock_data  # Disable when using mock data
        )
    
    with col2:
        social_max = st.slider(
            "Max results", 
            5, 50, 20, 
            key="social_max",
            disabled=use_mock_data  # Disable when using mock data
        ) 
    
    if st.button("Run Social Intelligence", key="run_social"):
        with st.spinner("Running social intelligence..."):
            result = safe_agent_call(
                run_social_intelligence,
                "Failed to run social intelligence",
                social_query,
                social_max, 
                use_mock_data
            )
            
            if result.get("error"):
                st.error(result["error"])
            else:
                # High-level metrics
                metrics = {
                    "Posts Processed": result.get("posts_processed", 0),
                    "Relevant Posts": result.get("relevant_posts", 0),
                    "Data Quality": f"{result.get('data_quality_score', 0)*100:.1f}%"
                }
                display_metrics("Overview", metrics)
                
                # Sentiment summary
                sa = result.get("sentiment_analysis", {}) or {}
                if isinstance(sa, str):
                    import json
                    try:
                        sa = json.loads(sa)
                    except:
                        sa = {}
                st.markdown("### Sentiment Overview")
                st.write(f"Overall: {sa.get('overall_sentiment', 'unknown').title()}")
                st.write(f"Score: {sa.get('sentiment_score', 0):.2f}")
                
                # Trending topics
                topics = result.get("trending_topics", []) or []
                if topics:
                    st.markdown("### Trending Topics")
                    for t in topics[:5]:
                        if isinstance(t, str):
                            try:
                                t = json.loads(t)
                            except:
                                t = {'topic': 'topic', 'mention_count': 0}
                        st.write(f"- {t.get('topic', 'topic')} ({t.get('mention_count', 0)} mentions)")
                
                # Insights
                insights = result.get("insights", []) or []
                if insights:
                    st.markdown("### Insights")
                    display_insights(insights)
                
                # Optional: inspect raw payload
                with st.expander("Show raw result"):
                    st.json(result)


def render_sidebar(components):
    """Render the sidebar with controls."""
    st.sidebar.header("Controls")
    
    # Configuration panel
    with st.sidebar.expander("⚙️ Configuration"):
        st.checkbox("Use mock data", value=True, key="use_mock_data")
        st.number_input("Default analysis period (hours)", 1, 720, 48, key="default_analysis_period")
        st.selectbox("Default sentiment model", ["Basic", "Advanced"], key="sentiment_model")
        st.checkbox("Auto-refresh data", value=False, key="auto_refresh")
        st.slider("Auto-refresh interval (minutes)", 5, 60, 15, key="refresh_interval")
    
    # Data Collection section
    st.sidebar.header("Data Collection")
    
    # Reddit collection
    if st.sidebar.button("🌐 Collect from Reddit"):
        with st.spinner("Collecting data from Reddit..."):
            try:
                results = components["data_processor"].collect_reddit_data(limit=15)
                st.sidebar.success(f"Collected and processed {len(results)} posts from Reddit")
            except Exception as e:
                st.sidebar.error(f"Error collecting from Reddit: {e}")
    
    # Reddit search
    reddit_query = st.sidebar.text_input("Reddit Search Query:", value="Uganda fintech")
    reddit_limit = st.sidebar.slider("Number of posts:", 5, 50, 15)
    
    if st.sidebar.button("🔍 Custom Reddit Search"):
        with st.spinner(f"Searching Reddit for '{reddit_query}'..."):
            try:
                results = components["data_processor"].collect_reddit_data(query=reddit_query, limit=reddit_limit)
                st.sidebar.success(f"Found and processed {len(results)} posts")
            except Exception as e:
                st.sidebar.error(f"Search failed: {e}")
    
    # General data collection
    if st.sidebar.button("🔄 Collect & Process New Data"):
        with st.spinner("Collecting and processing data..."):
            if st.session_state.use_mock_data:
                results = components["data_processor"].batch_process(get_mock_posts())
            else:
                # This would fetch real data in production
                results = components["data_processor"].collect_real_data()
            st.sidebar.success(f"Processed {len(results)} posts")
    
    # Intelligence workflow
    if st.sidebar.button("🤖 Run Intelligence Workflow"):
        with st.spinner("Running multi-agent workflow..."):
            # Create a progress bar
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            # Define a callback for progress updates
            def progress_callback(step, total_steps, message):
                progress = int(step / total_steps * 100)
                progress_bar.progress(progress)
                status_text.text(message)
            
            try:
                result = run_intelligence_workflow(progress_callback)
                
                # Clear the progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.sidebar.success("Workflow completed!")
                
                # Show workflow results
                if result and 'final_report' in result:
                    st.session_state.workflow_last_result = result
                    st.session_state.workflow_last_run = datetime.now(tz=None)
                    metrics = {
                        "Posts Processed": result['final_report'].get('metrics', {}).get('posts_processed', 0),
                        "Insights Generated": result['final_report'].get('metrics', {}).get('market_insights', 0)
                    }
                    st.sidebar.write("Workflow Results:")
                    for key, value in metrics.items():
                        st.sidebar.info(f"{key}: {value}")
            except Exception as e:
                # Clear the progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.sidebar.error(f"Workflow failed: {str(e)}")
                st.sidebar.info("Check your API keys and environment configuration")


def render_dashboard_tab(components):
    """Render the Dashboard tab."""
    st.header("Market Overview")
    
    # Get recent posts for analytics
    recent_posts = get_recent_posts(components["db_manager"], hours=48)
    
    # KPI metrics
    try:
        posts_processed = len(recent_posts) if recent_posts else 0
        relevant_posts = sum(1 for p in recent_posts if (p.relevance_score or 0) >= 0.5) if recent_posts else 0
    except Exception:
        posts_processed, relevant_posts = 0, 0

    data_quality_pct = None
    if st.session_state.get('social_last_result'):
        dq = st.session_state.social_last_result.get('data_quality_score') or 0
        data_quality_pct = f"{dq*100:.1f}%"

    top_competitor = "N/A"
    try:
        latest_top = components["db_manager"].get_latest_top_competitor()
        if latest_top:
            top_competitor = latest_top
    except Exception:
        pass

    kpi_metrics = {
        "Posts Processed": posts_processed,
        "Relevant Posts": relevant_posts,
        "Data Quality": data_quality_pct or "-",
        "Top Competitor by SOV": top_competitor
    }
    display_metrics("KPIs", kpi_metrics)
    
    # Show workflow results panel if available
    if st.session_state.get('workflow_last_result'):
        with st.expander("Latest Workflow Summary"):
            final_report = st.session_state.workflow_last_result.get('final_report', {})
            st.write(final_report.get('summary') or "No summary available")
            metrics_wrk = final_report.get('metrics') or {}
            if metrics_wrk:
                st.write("Metrics:")
                st.json(metrics_wrk)
    
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


def render_social_posts_tab(components):
    """Render the Social Posts tab with Social Intelligence tools."""
    st.header("Recent Social Media Posts")

    # Social Intelligence controls
    st.subheader("Run Social Intelligence")
    use_mock_data = st.checkbox("Use enhanced mock data", value=False, key="use_enhanced_mock_tab")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        social_query = st.text_input(
            "Social search query:",
            value="Uganda fintech",
            key="social_query_tab",
            disabled=use_mock_data
        )
    with col2:
        social_max = st.slider(
            "Max results",
            5, 50, 20,
            key="social_max_tab",
            disabled=use_mock_data
        )
    with col3:
        if st.session_state.social_last_run:
            st.caption(f"Last run: {time_ago(st.session_state.social_last_run)}")
        st.caption("Using cache (30 min TTL)")

    if st.button("Run Social Intelligence", key="run_social_tab"):
        with st.spinner("Running social intelligence..."):
            result = safe_agent_call(
                run_social_intelligence,
                "Failed to run social intelligence",
                social_query,
                social_max,
                use_mock_data
            )
            if result.get("error"):
                st.error(result["error"])
            else:
                st.session_state.social_last_run = datetime.now(tz=None)
                st.session_state.social_last_result = result

    # Render last result if present
    if st.session_state.get('social_last_result'):
        result = st.session_state.social_last_result
        metrics = {
            "Posts Processed": result.get("posts_processed", 0),
            "Relevant Posts": result.get("relevant_posts", 0),
            "Data Quality": f"{result.get('data_quality_score', 0)*100:.1f}%"
        }
        display_metrics("Overview", metrics)

        sa = result.get("sentiment_analysis", {}) or {}
        if isinstance(sa, str):
            try:
                sa = json.loads(sa)
            except:
                sa = {}
        st.markdown("### Sentiment Overview")
        st.write(f"Overall: {sa.get('overall_sentiment', 'unknown').title()}")
        st.write(f"Score: {sa.get('sentiment_score', 0):.2f}")

        topics = result.get("trending_topics", []) or []
        if topics:
            st.markdown("### Trending Topics")
            for t in topics[:5]:
                if isinstance(t, str):
                    try:
                        t = json.loads(t)
                    except:
                        t = {"topic": "topic", "mention_count": 0}
                st.write(f"- {t.get('topic', 'topic')} ({t.get('mention_count', 0)} mentions)")

        insights = result.get("insights", []) or []
        if insights:
            st.markdown("### Insights")
            display_insights(insights)

        evidence = result.get("evidence_posts") or result.get("evidence") or []
        if evidence:
            st.markdown("### Evidence: Recent Posts")
            for ev in evidence[:10]:
                try:
                    source = ev.get("source")
                    content = ev.get("content")
                    topics_ev = ev.get("topics")
                    sent_ev = ev.get("sentiment")
                    ts = ev.get("timestamp")
                    url = ev.get("url")
                    with st.expander(f"{source} - {sent_ev} - {str(ts)}"):
                        st.write(content)
                        if url:
                            st.caption(f"[Source]({url})")
                        if topics_ev:
                            if isinstance(topics_ev, list):
                                st.caption("Topics: " + ", ".join(topics_ev))
                            else:
                                st.caption(f"Topics: {topics_ev}")
                except Exception:
                    continue

        with st.expander("Show raw result"):
            st.json(result)

    st.markdown("---")

    posts = get_recent_posts(components["db_manager"], limit=50)
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


def render_competitor_analysis_tab(components):
    """Render the Competitor Analysis tab."""
    st.header("Competitor Analysis")
    
    if st.button("Analyze Competitor Mentions"):
        with st.spinner("Analyzing competitor data..."):
            report = safe_agent_call(
                components["competitor_agent"].process,
                "Failed to analyze competitor data",
                {"hours": 24}
            )
            
            if report.get("error"):
                st.error(report["error"])
            else:
                st.session_state.competitor_last_run = datetime.now(tz=None)
                st.subheader("Competitive Intelligence Summary")
                st.write(f"Period: Last {report['time_period_hours']} hours")
                st.write(f"Total competitor mentions: {report['total_mentions']}")
                
                competitor_items = report.get('competitor_insights') or report.get('competitor_mentions', [])
                if competitor_items:
                    # Competitor sentiment
                    competitor_sentiment = {}
                    competitor_totals = {}
                    for insight in competitor_items:
                        if isinstance(insight, str):
                            try:
                                insight = json.loads(insight)
                            except:
                                insight = {'competitor': 'unknown', 'sentiment': 'neutral'}
                        competitor = insight.get('competitor')
                        sentiment = insight.get('sentiment')
                        
                        if competitor not in competitor_sentiment:
                            competitor_sentiment[competitor] = {'positive': 0, 'negative': 0, 'neutral': 0}
                        
                        competitor_sentiment[competitor][sentiment] += 1
                        competitor_totals[competitor] = competitor_totals.get(competitor, 0) + 1

                    # Share of Voice donut
                    if competitor_totals:
                        sov_df = pd.DataFrame({
                            "Competitor": list(competitor_totals.keys()),
                            "Mentions": list(competitor_totals.values())
                        })
                        fig_sov = px.pie(sov_df, values="Mentions", names="Competitor", title="Share of Voice (Mentions)")
                        st.plotly_chart(fig_sov, use_container_width=True)
                        # Persist SOV snapshot to DB
                        try:
                            components["db_manager"].save_competitor_sov(competitor_totals, report.get('time_period_hours', 24))
                        except Exception:
                            pass
                    
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
                        if isinstance(insight, str):
                            try:
                                insight = json.loads(insight)
                            except:
                                insight = {'text': 'No insight available', 'confidence': 0}
                        st.info(f"{insight.get('text', '')} (Confidence: {insight.get('confidence', 0)*100:.1f}%)")
                else:
                    st.info("No insights generated from competitor analysis")
    else:
        if st.session_state.competitor_last_run:
            st.caption(f"Last run: {time_ago(st.session_state.competitor_last_run)}")
        st.info("Click the button to analyze competitor mentions")


def render_vector_search_tab(components):
    """Render the Vector Search tab."""
    st.header("Vector Search")
    
    st.subheader("Search Similar Content")
    search_query = st.text_input("Enter search query:", placeholder="e.g., mobile money fees", key="vector_search")
    
    if st.button("Search", key="search_btn") and search_query:
        with st.spinner("Searching similar content..."):
            results = components["db_manager"].search_similar_posts(search_query, n_results=5)
            
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
            results = components["db_manager"].search_posts_by_topic(selected_topic, n_results=8)
            
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
        stats = components["db_manager"].get_vector_db_stats()
        st.metric("Total Documents in Vector DB", stats["total_documents"])
        st.write(f"Collection: {stats['collection_name']}")


def render_insights_tab(components):
    """Render the Insights tab."""
    st.header("AI-Generated Insights")
    
    insights = components["db_manager"].get_session().query(Insight).order_by(Insight.created_at.desc()).limit(10).all()
    
    if insights:
        for insight in insights:
            with st.expander(f"{insight.type} - {time_ago(insight.created_at)}"):
                st.write(insight.content)
                st.caption(f"Confidence: {insight.confidence * 100:.1f}%")
                
                if insight.source_data:
                    st.caption(f"Based on {len(safe_json_loads(insight.source_data))} data points")
    else:
        st.info("No insights yet. Run the intelligence workflow to generate insights.")


def render_market_health_tab(components):
    """Render the Market Health tab."""
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
            result = safe_agent_call(
                market_agent.process,
                "Failed to analyze market health",
                {"hours": hours}
            )

            if result.get("error"):
                st.error(result["error"])
            else:
                st.session_state.market_last_run = datetime.now(tz=None)
                health = result["health_indicators"]
                if isinstance(health, str):
                    try:
                        health = json.loads(health)
                    except:
                        health = {"market_health": "unknown", "health_score": 0, "opportunity_score": 0, "risk_level": 0.5}

                # Create columns for metrics
                col1, col2, col3 = st.columns(3)

                # Market health status with color
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
                    if isinstance(health.get("growth_segments"), list):
                        st.write(", ".join(health["growth_segments"]))
                    else:
                        st.write("No growth segments available")

                with st.expander("Show raw result"):
                    st.json(result)
    else:
        if st.session_state.market_last_run:
            st.caption(f"Last run: {time_ago(st.session_state.market_last_run)}")
    
    # Show workflow panel on Dashboard if available


def render_footer():
    """Render the app footer."""
    st.markdown("---")
    st.caption("FintelliUG MVP - Uganda Fintech Intelligence Platform")


# ===== Main Application =====

def main():
    """Main application entry point."""
    # Set page config
    st.set_page_config(
        page_title="FintelliUG - Uganda Fintech Intelligence",
        page_icon="💹",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Initialize loading state manager
    loading_manager = LoadingStateManager()
    
    # Initialize components if not already done
    if not st.session_state.components:
        st.session_state.components = initialize_components()
    
    components = st.session_state.components
    
    # Render header
    render_header()
    
    # Social Intelligence controls are rendered inside the Social Posts tab
    
    # Render sidebar
    render_sidebar(components)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Dashboard", 
        "💬 Social Posts", 
        "🏢 Competitor Analysis",
        "🔍 Vector Search",
        "📋 Insights",
        "📊 Market Health"
    ])
    
    # Render tab content
    with tab1:
        render_dashboard_tab(components)
    
    with tab2:
        render_social_posts_tab(components)
    
    with tab3:
        render_competitor_analysis_tab(components)
    
    with tab4:
        render_vector_search_tab(components)
    
    with tab5:
        render_insights_tab(components)
    
    with tab6:
        render_market_health_tab(components)
    
    # Render footer
    render_footer()


# Run the application
if __name__ == "__main__":
    main()
