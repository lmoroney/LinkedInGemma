# concierge_agent.py
# A terminal-based AI agent that acts as a local concierge.
# It uses a local Gemma model served by Ollama for reasoning and external tools for web search and browsing.
# This version includes conversation history, robust multi-site browsing, and an email tool.

import os
import requests
from bs4 import BeautifulSoup
import json
import smtplib
from email.message import EmailMessage

# --- Configuration ---
# It's highly recommended to set these as environment variables for security.
# You can get a free Serper API key from https://serper.dev
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

# Ollama configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:latest") # Assumes you have pulled a gemma3 model

# SMTP Configuration for the email tool
SMTP_SERVER = os.environ.get("SMTP_SERVER")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 465)) # Default to 465 for SSL
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")


# --- Part 1: Defining the Agent's Tools ---

def search_web(query: str) -> str:
    """
    Uses the Serper.dev API to perform a web search.
    Returns a formatted string of search results.
    """
    print(f"--- Tool: Searching web for '{query}' ---")
    if not SERPER_API_KEY:
        print("--- DEBUG: SERPER_API_KEY is not set. ---")
        return "Error: SERPER_API_KEY is not set. Cannot perform web search."
    
    print(f"--- DEBUG: Using SERPER_API_KEY ending in '...{SERPER_API_KEY[-4:]}' ---")

    payload = json.dumps({"q": query})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    
    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, data=payload)
        print(f"--- DEBUG: Serper API response status code: {response.status_code} ---")
        print(f"--- DEBUG: Serper API response text: {response.text[:500]} ... ---")
        response.raise_for_status()
        results = response.json()
        
        if not results.get("organic"):
            return "No good search results found."
            
        output = "Search Results:\n"
        for item in results["organic"][:5]: # Get top 5 results
            output += f"- Title: {item.get('title', 'N/A')}\n"
            output += f"  Link: {item.get('link', 'N/A')}\n"
            output += f"  Snippet: {item.get('snippet', 'N/A')}\n\n"
        return output
        
    except requests.exceptions.RequestException as e:
        return f"Error during web search: {e}"

def browse_website(url: str) -> str:
    """
    Scrapes the text content of a given URL.
    Returns the cleaned text content or an error message if it fails.
    """
    print(f"--- Tool: Attempting to browse website '{url}' ---")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'DNT': '1' 
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
            
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        if not text:
            return f"Error: No text content found at {url}"

        print(f"--- Successfully browsed {url} ---")
        return text[:8000]

    except requests.exceptions.RequestException as e:
        return f"Error browsing website {url}: {e}"


def send_email(to_address: str, subject: str, body: str) -> str:
    """
    Sends an email using the configured SMTP settings.
    """
    print(f"--- Tool: Sending email to '{to_address}' ---")
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD]):
        return "Error: SMTP settings are not fully configured. Cannot send email."

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = SMTP_USERNAME
    msg['To'] = to_address

    try:
        # Use SMTP_SSL for port 465
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        return f"Email sent successfully to {to_address}."
    except Exception as e:
        return f"Error sending email: {e}"

# --- Part 2: The Agent's "Brain" (Ollama Interaction) ---

def call_gemma_ollama(prompt: str, output_format: str = "json") -> str:
    """
    A helper function to call the local Ollama API and get a response.
    """
    print(f"--- Thinking with local Gemma ({OLLAMA_MODEL})... ---")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if output_format == "json":
        payload["format"] = "json"
    
    try:
        # Added a 60-second timeout to prevent indefinite hanging
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        # The actual response from Ollama is a JSON string in the 'response' field
        return result.get("response", "{}")

    except requests.exceptions.Timeout:
        return "Error: Ollama API request timed out. The model might be taking too long to respond."
    except requests.exceptions.RequestException as e:
        return f"Error calling Ollama API: {e}. Is Ollama running?"
    except (KeyError, IndexError) as e:
        return f"Error parsing Ollama response: {e}. Response: {response.text}"


# --- Part 3: The Agentic Chain Logic with Memory and Robustness ---

def run_concierge_agent(goal: str, history: list) -> str:
    # Step -1: Extract email address from the goal if it exists
    # 1. Decide what to search for
    # 2. Search the web
    # 3. Choose which sites to browse
    # 4. Browse the websites and collect information
    # 5. Summarize everything for the user
    # 6. Decide if an email should be sent and generate its content
    pass


# --- Part 4: The Terminal Interface ---

def main():
    pass

if __name__ == "__main__":
    main()

# Force a new commit by adding a comment
