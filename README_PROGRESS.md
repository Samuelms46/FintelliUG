# FintelliUG MVP Progress Tracker

This document tracks the minimum viable product (MVP) scope for the next week, focusing on a usable research dashboard for Uganda fintech insights.

## Week Goal
- Ship a working dashboard with end-to-end agentic flow and clear insights for a live demo.
- Avoid crashes; display helpful errors and fallbacks; show raw data for transparency.

## Shipping Criteria (Definition of Done)
- No UI crashes; all buttons handle errors gracefully.
- Each tab renders with real or mocked data.
- At least 5 KPIs + 2 charts + insights list + vector search results.
- Cache usage visible in UI (e.g., "Using cached result from X min ago").
- "Last run" timestamp shown for major sections.
- README_PROGRESS up to date.

---

## 1) UI Features Checklist

### Overview (Executive)
- [ ] KPIs: Posts Processed, Relevant Posts, Market Health, Data Quality, Top Competitor by SOV
- [ ] Sentiment distribution (pie)
- [ ] Top topics (bar)
- [ ] Sentiment trend (24h/7d line) [optional if data available]

### Social Posts (with Social Intelligence)
- [ ] Inputs: Social query + Max results + Run button
- [ ] Run SocialIntelAgent and render
  - [ ] Sentiment overview (overall + score)
  - [ ] Top 5 topics (list)
  - [ ] 3–5 insights (bullets)
  - [ ] Evidence: recent posts list (source, sentiment, topics, timestamp, link)
  - [ ] Raw result expander (JSON)
- [ ] Communicate cache hits (e.g., cached 30 min)

### Competitor Analysis
- [ ] Analyze button (24h default)
- [ ] Share of Voice (donut) by competitor
- [ ] Sentiment by competitor (stacked bars)
- [ ] 3–5 competitor insights (bullets)

### Market Health
- [ ] New tab with period select (24h/7d/30d) and Analyze button
- [ ] Metrics: Market Health, Opportunity Score, Risk Level
- [ ] Growth segments (list/chips)
- [ ] Raw result expander (JSON)

### Vector Search (Minimal)
- [ ] Query input → top 5 similar posts
- [ ] Show content + source + topics + sentiment + similarity

### Insights
- [ ] Last 10 AI-generated insights list (type, content, confidence, created_at)
- [ ] Show daily briefing/executive summary if available

### Sidebar (Data Collection + Workflow)
- [ ] Reddit collection quick action
- [ ] Custom Reddit search (query + limit)
- [ ] Collect & Process New Data (mock mode OK)
- [ ] Run MultiAgentWorkflow and acknowledge completion; surface key metrics

---

## 2) Agentic Flow Alignment
Pipeline: fetch (Reddit/X) → process_post → social_intelligence → market_sentiment → competitor_analysis → compile_report

- [ ] SocialIntelAgent: wired into Social Posts tab with query mode
- [ ] MarketSentimentAgent: wired into Market Health tab (hours → process)
- [ ] CompetitorAnalysisAgent: produces SOV + sentiment + summary bullets
- [ ] MultiAgentWorkflow: run from sidebar; display metrics and provide link/panel to view compiled report
- [ ] Consistent keys across agents/UI: sentiment_analysis, trending_topics, insights, health_indicators, competitor_mentions
- [ ] Fallbacks: Coordinator fallback insights surface in UI if orchestration parsing fails

---

## 3) Data & Persistence
- [ ] Database seeded and reachable (fintelliug.db)
- [ ] Vector DB path configured (CHROMA_PERSIST_DIR) and writable
- [ ] Posts stored in vector DB by SocialIntelAgent (doc ids + metadata)
- [ ] Insights persistence: add to DB where appropriate (for Insights tab)
- [ ] Caching: Redis reachable; cache keys used; TTL communicated in UI
- [ ] Data retention (optional): delete old vector DB records (BaseAgent.delete_old_records)

---

## 4) Environment & Config
- [x] .env contains required keys: GROQ_API_KEY, AZURE_OPENAI_API_KEY, AZURE_EMBEDDING_ENDPOINT, AZURE_EMBEDDING_BASE, GROQ_MODEL, CHROMA_PERSIST_DIR, REDIS_HOST/PORT
- [ ] Safe defaults + friendly error messages when missing
- [ ] Mock mode toggle (optional) for offline demos

---

## 5) Logging & Observability
- [ ] logs/app.log monitored; surface key messages in UI (last run time, cache hits)
- [ ] Per-stage statuses for workflow (even minimal text)
- [ ] Raw JSON expanders in Social Intelligence and Market Health

---

## 6) Code Quality & Cleanup
- [ ] app.py: Add Social Intelligence section in Social Posts tab
- [ ] app.py: Add Market Health tab and metrics
- [ ] Ensure tab setup has all six tabs (Overview/Social/Competitor/Vector/Insights/Market Health)
- [ ] Remove or fix stray files named "_ _init_ _.py" (rename to __init__.py if needed)
- [ ] Standardize imports and unused variables
- [ ] Coordinator: fallback method signature aligned (done)

---

## 7) Tests & Demo Data
- [ ] Run tests in tests/ and note failures
- [ ] Provide a small demo dataset or "Collect & Process" mock for consistent demos
- [ ] Add smoke test: invoke each agent with minimal input and assert key fields exist

---

## 8) Known Gaps (Repo-specific Findings)
- [ ] Social Intelligence UI not yet wired into Social Posts tab
- [ ] Market Health tab not yet present in app.py (add per MVP plan)
- [ ] MultiAgentWorkflow outputs are acknowledged in sidebar but not visualized in main tabs (consider a small panel in Overview/Insights)
- [ ] Check for stray files with spaces in name (e.g., agents/_ _init_ _.py, utils/_ _init_ _.py, etc.) and clean up
- [ ] Ensure .env is present and valid; BaseAgent hard-requires env keys

---

## 9) One-Week Execution Plan
Day 1
- Align agent outputs (keys) and UI data contracts; add mock mode

Day 2
- Wire Social Intelligence into Social Posts (inputs + results + raw JSON)

Day 3
- Overview KPIs + 2 charts (sentiment pie + topics bar)

Day 4
- Competitor charts (SOV + sentiment by competitor) + insights

Day 5
- Market Health tab metrics + growth segments + raw JSON

Day 6
- Vector Search minimal + Insights list; polish errors/caching/messages

Day 7
- Buffer, walkthrough, demo script, reliability checks

---

## 10) Demo Script (What to Show)
1) Overview: KPIs + two charts. “Last run X minutes ago.”
2) Social Posts → Run Social Intelligence: show sentiment, topics, insights, and a few evidence posts
3) Competitor Analysis → Run: SOV + sentiment by competitor, 3 insights
4) Market Health → Analyze 7 days: 3 metrics + growth segments
5) Vector Search → query “mobile money fees”: show 5 similar posts
6) Insights → show last 10; open any daily briefing if present

---

## Notes
- Keep UI simple and fast; handle errors visibly; prefer clarity over complexity
