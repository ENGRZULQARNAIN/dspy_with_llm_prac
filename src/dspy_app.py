import dspy
from core.config import ANTHROPIC_API_KEY
import os
from dotenv import load_dotenv



# Model configuration
lm = dspy.LM('claude-3-7-sonnet-20250219',api_key=ANTHROPIC_API_KEY)
dspy.configure(lm=lm)

#simple signature testing question
qa = dspy.Predict('question: str -> response: str')
response = qa(question="what are high memory and low memory on linux?")

print(response.response)

