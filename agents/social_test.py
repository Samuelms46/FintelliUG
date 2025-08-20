import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate


load_dotenv(".env")
# Initialize Groq LLM
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL"),
    temperature=0.7, 
    max_tokens=512,   
    groq_api_key=os.getenv("GROQ_API_KEY")
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
social_post = "Chipper Cash allow making a transaction of UGX 1000 at zero charges"
result = chain.invoke({"text": social_post})
print(result.content)