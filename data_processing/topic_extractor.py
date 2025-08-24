import openai
import json
from config import Config
from typing import List, Dict, Any
import re

class TopicExtractor:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
        self.fintech_topics = Config.FINTECH_TOPICS
    
    def extract_topics_keywords(self, text):
        """Extract topics using keyword matching (fast and cheap)"""
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in self.fintech_topics.items():
            # Count how many keywords appear in the text
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Simple confidence based on match count
                confidence = min(matches / 3, 1.0)
                detected_topics.append({"topic": topic, "confidence": confidence})
        
        # Sort by confidence descending
        detected_topics.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Return top 3 topics
        return [t["topic"] for t in detected_topics[:3]], detected_topics[0]["confidence"] if detected_topics else 0.0
    
    def extract_topics_llm(self, text):
        """Use LLM for more accurate topic extraction"""
        try:
            prompt = f"""
            Analyze this text from Uganda's fintech ecosystem and identify the relevant topics.
            Available categories: {list(self.fintech_topics.keys())}
            
            Text: {text}
            
            Return a JSON response with:
            - topics: list of relevant topic names
            - confidence: overall confidence score (0-1)
            
            Only return the JSON object, no other text.
            """
            
            response = openai.ChatCompletion.create(
                model=Config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                return result_data.get("topics", []), result_data.get("confidence", 0.5)
            else:
                return [], 0.5
                
        except Exception as e:
            print(f"Error in LLM topic extraction: {e}")
            return [], 0.5
    
    def extract_topics(self, text, use_llm_fallback=True):
        """
        Extract topics from text using keyword matching first,
        with LLM fallback for ambiguous cases
        """
        topics, confidence = self.extract_topics_keywords(text)
        
        # If low confidence from keywords or ambiguous, use LLM
        if confidence < 0.3 and use_llm_fallback:
            llm_topics, llm_confidence = self.extract_topics_llm(text)
            if llm_topics:
                return llm_topics, llm_confidence
        
        return topics, confidence