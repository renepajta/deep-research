"""
Test script for the deep research MCP server tools.
Tests the helper functions directly (bypassing MCP context).

Uses the same unified OpenAI SDK pattern as the main server.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Import prompts from the package
from deep_research.server import (
    TRIAGE_PROMPT,
    CLARIFYING_PROMPT,
    INSTRUCTION_PROMPT,
    WEB_SEARCH_PROMPT,
    ASK_PROMPT,
    REASON_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    SYNTHESIS_PROMPT,
)


# ─────────────────────────────────────────────────────────────
#  Test clients (initialized directly, not via lifespan)
# ─────────────────────────────────────────────────────────────

_credential = None
_token_provider = None
_chat_client = None
_research_client = None


def get_credential():
    global _credential
    if _credential is None:
        _credential = DefaultAzureCredential()
    return _credential


def get_token_provider():
    global _token_provider
    if _token_provider is None:
        _token_provider = get_bearer_token_provider(
            get_credential(),
            "https://cognitiveservices.azure.com/.default"
        )
    return _token_provider


def get_chat_client():
    """Get chat client for gpt-5.2 (chat, web search, reasoning)."""
    global _chat_client
    if _chat_client is None:
        endpoint = os.environ["AI_FOUNDRY_ENDPOINT"]
        _chat_client = OpenAI(
            base_url=endpoint,
            api_key=get_token_provider(),
        )
    return _chat_client


def get_research_client():
    """Get research client for o3-deep-research."""
    global _research_client
    if _research_client is None:
        endpoint = os.environ["AI_FOUNDRY_DEEP_RESEARCH_ENDPOINT"]
        _research_client = OpenAI(
            base_url=endpoint,
            api_key=get_token_provider(),
        )
    return _research_client


def call_chat(system_prompt: str, user_message: str, use_web_search: bool = False) -> str:
    """Test version of call_chat using direct clients."""
    model = os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-5.2")
    client = get_chat_client()
    
    request_params = {
        "model": model,
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_message}]}
        ],
    }
    
    if use_web_search:
        request_params["tools"] = [{"type": "web_search_preview"}]
    
    response = client.responses.create(**request_params)
    return response.output_text or ""


# ─────────────────────────────────────────────────────────────
#  Test functions
# ─────────────────────────────────────────────────────────────

def test_web_search(query: str = "What is the current price of Bitcoin?"):
    """Test web search capability."""
    print("\n" + "="*60)
    print("🔍 Testing: web_search")
    print("="*60)
    print(f"Query: {query}\n")
    
    result = call_chat(WEB_SEARCH_PROMPT, query, use_web_search=True)
    print(result)
    return result


def test_ask(question: str = "What are the main differences between Python and JavaScript?"):
    """Test ask capability."""
    print("\n" + "="*60)
    print("💬 Testing: ask")
    print("="*60)
    print(f"Question: {question}\n")
    
    result = call_chat(ASK_PROMPT, question, use_web_search=True)
    print(result)
    return result


def test_reason(problem: str = "Should I learn Rust or Go in 2026?"):
    """Test reasoning capability."""
    print("\n" + "="*60)
    print("🧠 Testing: reason")
    print("="*60)
    print(f"Problem: {problem}\n")
    
    result = call_chat(REASON_PROMPT, problem, use_web_search=False)
    print(result)
    return result


def test_triage(topic: str = "Research AI"):
    """Test triage phase."""
    print("\n" + "="*60)
    print("📋 Testing: Triage Phase")
    print("="*60)
    print(f"Topic: {topic}\n")
    
    result = call_chat(TRIAGE_PROMPT, topic, use_web_search=False)
    print(f"Triage Result: {result}")
    return result


def test_clarify(topic: str = "Climate change"):
    """Test clarification phase."""
    print("\n" + "="*60)
    print("❓ Testing: Clarification Phase")
    print("="*60)
    print(f"Topic: {topic}\n")
    
    result = call_chat(CLARIFYING_PROMPT, topic, use_web_search=False)
    print(f"Clarifying Questions:\n{result}")
    return result


def test_instruction(topic: str = "AI in healthcare"):
    """Test instruction phase."""
    print("\n" + "="*60)
    print("📝 Testing: Instruction Phase")
    print("="*60)
    
    context = f"Original query: {topic}\n\nResearch scope: broad scope"
    print(f"Context:\n{context}\n")
    
    result = call_chat(INSTRUCTION_PROMPT, context, use_web_search=False)
    print(f"Research Brief:\n{result}")
    return result


def interactive_test():
    """Interactive testing mode."""
    print("\n" + "="*60)
    print("🎮 Interactive Testing Mode")
    print("="*60)
    
    while True:
        print("\nChoose a test:")
        print("1. web_search - Quick web search")
        print("2. ask - Conversational AI")
        print("3. reason - Logical reasoning")
        print("4. triage - Test triage phase")
        print("5. clarify - Test clarification phase")
        print("6. instruct - Test instruction phase")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-6): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            query = input("Enter search query: ").strip() or "Latest news on OpenAI"
            test_web_search(query)
        elif choice == "2":
            question = input("Enter question: ").strip() or "What is quantum computing?"
            test_ask(question)
        elif choice == "3":
            problem = input("Enter problem: ").strip() or "Should I learn Rust or Go in 2026?"
            test_reason(problem)
        elif choice == "4":
            topic = input("Enter topic to triage: ").strip() or "Research AI"
            test_triage(topic)
        elif choice == "5":
            topic = input("Enter topic for clarification: ").strip() or "Climate change"
            test_clarify(topic)
        elif choice == "6":
            topic = input("Enter topic: ").strip() or "AI in healthcare"
            test_instruction(topic)
        else:
            print("Invalid choice")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        arg = sys.argv[2] if len(sys.argv) > 2 else None
        
        if test_name == "web_search":
            test_web_search(arg or "What is the current price of Bitcoin?")
        elif test_name == "ask":
            test_ask(arg or "What is quantum computing?")
        elif test_name == "reason":
            test_reason(arg or "Should I learn Rust or Go in 2026?")
        elif test_name == "triage":
            test_triage(arg or "Research AI")
        elif test_name == "clarify":
            test_clarify(arg or "Climate change")
        elif test_name == "instruct":
            test_instruction(arg or "AI in healthcare")
        elif test_name == "interactive":
            interactive_test()
        else:
            print(f"Unknown test: {test_name}")
            print("Available: web_search, ask, reason, triage, clarify, instruct, interactive")
    else:
        interactive_test()
