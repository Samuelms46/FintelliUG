import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate


load_dotenv("../.env")
# Initialize Groq LLM
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
    temperature=0.7, 
    #max_tokens=512,   
    api_key=os.getenv("GROQ_API_KEY")
)

#a prompt for sentiment analysis
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant analyzing social media for fintech sentiment in Uganda."
    "Classify the sentiment as positive, negative, or neutral and provide a brief explanation."),
    ("human", "{text}")
])

# Create a chain
chain = prompt | llm

# Example usage
social_post = """Today we're announcing an exciting partnership with MoneyGram. Chipper Cash customers in 
Uganda can now seamlessly cash in and cash out in real time at all MoneyGram agent locations countrywide.
This expansion is another step towards bridging the gap between people and first class 
financial services at scale. Special thanks to The Bank of Uganda for working closely 
with our teams on this, and providing the necessary approvals. """
result = chain.invoke({"text": social_post})
print(result.content)