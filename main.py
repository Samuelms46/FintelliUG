from agents.social_intel_agent import SocialIntelAgent

agent = SocialIntelAgent()
input_data = {
    "text": "MTN MoMo's new fees are too high, users are frustrated!",
    "source": "twitter",
    "timestamp": "2025-08-20T16:30:00+03:00"
}
result = agent.process(input_data)
print(result)