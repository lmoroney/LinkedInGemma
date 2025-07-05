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
import base64

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


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

def call_gemma_ollama(prompt: str, output_format: str = "json", image_path: str = None) -> str:
    """
    A helper function to call the local Ollama API and get a response.
    """
    print(f"--- Thinking with local Gemma ({OLLAMA_MODEL})... ---")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if image_path:
        payload["images"] = [encode_image(image_path)]
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
    """
    Runs the main logic of the concierge agent, now with conversation history and robust multi-site browsing.
    Returns the final summary to be added to the history.
    """
    # Step -1: Extract email address from the goal if it exists
    prompt_extract_email = f"""
    You are an expert at finding email addresses in text.
    Analyze the following user request and extract the email address if one is present.
    If you find an email address, respond with ONLY the email address.
    If you do not find an email address, respond with the word "none".

    User request: "{goal}"
    """
    recipient_email_from_goal = call_gemma_ollama(prompt_extract_email, output_format="text").strip()
    if "@" not in recipient_email_from_goal:
        recipient_email_from_goal = "none"


    print(f"\n🎯 Goal: {goal}\n")

    formatted_history = "\n".join(history)

    # 1. Decide what to search for
    prompt1 = f"""
You are a helpful concierge agent. Your task is to understand a user's request and generate a concise, effective search query to find the information they need.

Conversation history:
---
{formatted_history}
---
User's latest request: "{goal}"

Based on the request, what is the best, simple search query for Google?
The query should be 3-5 words.
Respond with ONLY the search query itself.
"""
    search_query = call_gemma_ollama(prompt1, output_format="text").strip().replace('"', '')
    
    # 2. Search the web
    search_results = search_web(search_query)
    print(search_results) # Print search results for debugging


    # 3. Choose which sites to browse
    prompt2 = f"""
You are a smart web navigator. Your task is to analyze Google search results and select the most promising URLs to find the answer to a user's goal. Avoid generic homepages (like yelp.com or google.com) and prefer specific articles, lists, or maps.

User's goal: "{goal}"

Search Results:
---
{search_results}
---

Based on the user's goal and the search results, which are the top 2-3 most promising and specific URLs to browse for details?
Respond with ONLY a list of URLs, one per line.
"""
    browse_urls_str = call_gemma_ollama(prompt2, output_format="text").strip()
    browse_urls = [url.strip() for url in browse_urls_str.split('\n') if url.strip().startswith('http')]

    if not browse_urls:
        print("--- Could not identify promising URLs to browse. Trying to summarize from search results directly. ---")
        # If no URLs are chosen, try to summarize from the snippets
        prompt_summarize_snippets = f"""
        You are a helpful concierge agent. The web browser is not working, but you have search result snippets.
        User's goal: "{goal}"
        Search Results:
        ---
        {search_results}
        ---
        Please provide a summary based *only* on the search result snippets. Do not suggest browsing URLs.
        """
        final_summary = call_gemma_ollama(prompt_summarize_snippets, output_format="text")
        print("\n--- Here is your summary ---\n")
        print(final_summary)
        print("\n--------------------------\n")
        return final_summary


    # 4. Browse the websites and collect information
    all_website_texts = []
    for url in browse_urls:
        text = browse_website(url)
        if not text.startswith("Error"):
            all_website_texts.append(f"Content from {url}:\n{text}")
        else:
            print(f"--- Skipping {url} due to an error. ---")
    
    if not all_website_texts:
        return "I tried to browse several websites but was blocked or couldn't find any information. Please try again."

    aggregated_text = "\n\n---\n\n".join(all_website_texts)

    # 5. Summarize everything for the user
    prompt3 = f"""
You are a meticulous and trustworthy concierge agent. Your primary goal is to provide a clear, concise, and, above all, ACCURATE answer to the user's request by synthesizing information from multiple sources.

User's latest request: "{goal}"

You have gathered the following text from one or more websites:
---
{aggregated_text}
---

Fact-Check and Synthesize:
Based on the information above, provide a comprehensive summary that directly answers the user's request.
Before including any business or item in your summary, you MUST verify that it meets ALL the specific criteria from the user's request (e.g., hours of operation, location, specific features).
If you cannot find explicit confirmation that a business meets a criterion, DO NOT include it in the summary. It is better to provide fewer, accurate results than more, inaccurate ones.

Format your response clearly for the user. If listing places, use bullet points.
"""
    final_summary = call_gemma_ollama(prompt3, output_format="text")

    print("\n--- Here is your summary ---\n")
    print(final_summary)
    print("\n--------------------------\n")

    # 6. Decide if an email should be sent and generate its content
    prompt4 = f"""
You are a highly capable assistant responsible for drafting clear and detailed emails based on a research summary.

User's original request: "{goal}"

Here is the final summary of the research, which has been fact-checked to meet the user's criteria:
---
{final_summary}
---

Here is a reminder of the raw text gathered from the websites, which you can use to find details like reservation links:
---
{aggregated_text}
---

Your task is to decide if an email is appropriate to send to the user with this information. If it is, you must draft the email.

- If the summary contains useful, actionable information (like a list of places, contact info, etc.), then an email should be sent.
- If the summary is short, conversational, or indicates no results were found, an email is not needed.

Instructions for the email draft:
1.  Create a clear subject line that summarizes the content.
2.  The email body should be a list of the places mentioned in the final summary.
3.  For each place, provide a brief summary of what it offers and, if you can find one in the raw text, the direct link for reservations.
4.  Ensure that ONLY information that strictly matches the user's request (e.g., open on a specific day) is included.

Respond in JSON format.
If sending, the JSON should be: {{\"send_email\": true, \"subject\": \"Your requested information\", \"body\": \"...\"}}
If not sending, the JSON should be: {{\"send_email\": false}}

Example for sending:
{{
  "send_email": true,
  "subject": "Your requested list of Sushi Restaurants in Seattle",
  "body": "Hello,\n\nHere are the sushi restaurants that match your criteria:\n\n*   **Shiro's Sushi:** A classic spot known for its traditional edomae sushi. Reservations: [https://www.shiros.com/reservations](https://www.shiros.com/reservations)\n\n*   **Sushi Kashiba:** A high-end sushi experience. Reservations: [https://www.sushikashiba.com/](https://www.sushikashiba.com/)"
}}
"""
    email_decision_str = call_gemma_ollama(prompt4, output_format="json")
    try:
        email_decision = json.loads(email_decision_str)
        if email_decision.get("send_email"):
            subject = email_decision.get("subject")
            body = email_decision.get("body")
            if all([subject, body]):
                print("\n--- I have drafted the following email summary for you ---\n")
                print(f"Subject: {subject}\n\nBody:\n{body}\n")
                print("--------------------------------------------------------")
                
                recipient_email = "none"
                if recipient_email_from_goal != "none":
                    confirm = input(f"Should I send this to the address you provided ({recipient_email_from_goal})? (y/n): ").lower()
                    if confirm == 'y':
                        recipient_email = recipient_email_from_goal
                else:
                    confirm = input("Would you like me to email this summary to you? (y/n): ").lower()
                    if confirm == 'y':
                        recipient_email = input("Please enter your email address: ")

                if recipient_email and recipient_email != "none":
                    result = send_email(recipient_email, subject, body)
                    print(result)
                else:
                    print("--- Okay, I will not send the email. ---")

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"--- Could not determine if an email should be sent due to an error: {e} ---")

    return final_summary


# --- Part 4: The Terminal Interface ---

def main():
    """
    The main function that runs the terminal application loop.
    """
    if not SERPER_API_KEY:
        print("🔴 FATAL ERROR: SERPER_API_KEY environment variable not set.")
        print("Please get a free key from https://serper.dev and set the variable.")
        return

    print(f"🤖 Hello! I am your Local Concierge Agent, powered by a local {OLLAMA_MODEL} model.")
    print("   I can remember our conversation and browse multiple sites for you.")
    print("   If you configure your SMTP settings, I can also send emails.")
    print("   You can also drag and drop an image to find out more about it.")
    print("   Make sure Ollama is running in the background.")
    print('   Type "quit" or "exit" to end the session.')
    
    conversation_history = []
    
    while True:
        user_input = input("\nWhat would you like to find? (or drop an image here)\n> ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("🤖 Goodbye!")
            break
        
        if os.path.isfile(user_input):
            print(f"--- Analyzing image at '{user_input}' ---")
            image_description = call_gemma_ollama("Describe this image", output_format="text", image_path=user_input)
            user_goal = f"tell me places where I can find {image_description}"
        else:
            user_goal = user_input

        agent_summary = run_concierge_agent(user_goal, conversation_history)
        
        conversation_history.append(f"User: {user_goal}")
        conversation_history.append(f"Agent: {agent_summary}")

if __name__ == "__main__":
    main()

# Force a new commit by adding a comment
