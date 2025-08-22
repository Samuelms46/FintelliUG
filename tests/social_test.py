import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.social_intel_agent import SocialIntelAgent
from datetime import datetime

# Example input
input_data = {
    "posts": [
        {"text": "MTN MoMo fees are too high!", "timestamp": datetime.now().isoformat()},
        {"text": "Fintech is growing fast in Uganda.", "timestamp": datetime.now().isoformat()},
        {"text": "I love using digital payments.", "timestamp": datetime.now().isoformat()}
    ]
}

agent = SocialIntelAgent(config={})
result = agent.process(input_data)
print(result)