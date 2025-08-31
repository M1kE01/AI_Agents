import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google import genai
from google.genai.types import Content, Part
from google.adk.models.lite_llm import LiteLlm
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.ERROR)
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Gemini API Key (Get from Google AI Studio: https://aistudio.google.com/app/apikey)
os.environ["GOOGLE_API_KEY"] = '' #INSERT YOUR GEMINI API KEY

# --- Verify Keys (Optional Check) ---
print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")

# Configure ADK to use API keys directly (not Vertex AI for this multi-model setup)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# --- Define Model Constants for easier use ---

MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

print("\nEnvironment configured.")


# Simulated tool functions
def style_analysis(text: str) -> str:
    "This function analyses the style of the text and returns a detailed report."
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
    response = client.models.generate_content(
        model = MODEL_GEMINI_2_0_FLASH,
        contents = "Within the user request analyse the style of the text and provide a detailed report about it. \n" +
                   "Your answer should be one paragraph of plain text. " +
                   "This is the user's request:\n" + text
    )
    return response.text#f"[STYLE ANALYSIS]\n{response.text.strip()}"

def text_summarizer(text: str) -> str:
    "This function reads the input text and summarizes it."
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
    response = client.models.generate_content(
        model = MODEL_GEMINI_2_0_FLASH,
        contents = "Within the user request read the text and summarize it. \n" +
                   "Your answer should be one paragraph of plain text. " +
                   "This is the user's request:\n" + text
    )
    return response.text#f"[SUMMARY]\n{response.text.strip()}"

def continuation_suggester(text: str) -> str:
    "This function reads the input text and suggests how to continue it."
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
    response = client.models.generate_content(
        model = MODEL_GEMINI_2_0_FLASH,
        contents = "Within the user request read the text and continue it with a paragraph in a similar stile.\n" +
                   "Your answer should be one paragraph of plain text. " +
                   "This is the user's request:\n" + text
    )
    return response.text#f"[CONTINUATION]\n{response.text.strip()}"


AGENT_MODEL = MODEL_GEMINI_2_0_FLASH 

text_analysis_agent = Agent(
    name="text_analysis_agent",
    model=AGENT_MODEL,
    description="You are a text analysis assistant. Your job is to work with text based on user's descriptions.",
    instruction=(
        "Use the tools provided to perform style analysis, summarization, and continuation. "
        "Always include the outputs from these tools directly in your response. "
        #"Format each section with clear headers: [SUMMARY], [STYLE ANALYSIS], [CONTINUATION]. "
        "Do not simply say that the task was completed. Include the actual tool outputs verbatim."
    ),
    tools=[style_analysis, text_summarizer, continuation_suggester]
)

# --- Session Management ---
session_service = InMemorySessionService()

APP_NAME = "software_team_app"
USER_ID = "dev_user_001"
SESSION_ID = "dev_session_001"

#session = session_service.create_session(
#    app_name=APP_NAME,
#    user_id=USER_ID,
#    session_id=SESSION_ID
#)

# --- Runner ---
runner = Runner(
    agent=text_analysis_agent,
    app_name=APP_NAME,
    session_service=session_service
)

async def call_agent_async(query: str, runner, user_id, session_id=None):
    print(f"\n>>> User Query: {query}")
    content = Content(role='user', parts=[Part(text=query)])
    final_response_text = "No response generated."
    print_log = []

    # Pass session_id=None so Runner auto-creates a session
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,   # None on first call; reuse later if you store it
        new_message=content
    ):
        if hasattr(event, 'tool_request') and event.tool_request:
            print_log.append(f"\n[TOOL REQUEST] {event.tool_request.tool_name}")
            print_log.append(f"[INPUT] {event.tool_request.input}")

        if hasattr(event, 'tool_response') and event.tool_response:
            print_log.append(f"[OUTPUT] {event.tool_response.output}")

        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
                print_log.append(f"\n[AGENT SUMMARY] {final_response_text}")
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                print_log.append(final_response_text)
            break

    for entry in print_log:
        print(entry)

async def run_conversation():
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    await call_agent_async("Do with this text all you can do:\n" + 
                           "Just then another visitor entered the drawing room: Prince Andrew Bolkónski, " +
                           "the little princess’ husband. He was a very handsome young man, of medium height, " +
                           "with firm, clearcut features. Everything about him, from his weary, bored " +
                           "expression to his quiet, measured step, offered a most striking contrast to " + 
                           "his quiet, little wife. It was evident that he not only knew everyone in the " +
                           "drawing room, but had found them to be so tiresome that it wearied him to look " + 
                           "at or listen to them. And among all these faces that he found so tedious, none " +
                           "seemed to bore him so much as that of his pretty wife. He turned away from her " + 
                           "with a grimace that distorted his handsome face, kissed Anna Pávlovna’s hand, and " +
                           "screwing up his eyes scanned the whole company.", runner, USER_ID, SESSION_ID)

if __name__ == '__main__':
    asyncio.run(run_conversation())