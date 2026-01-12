"""
Deep Research MCP Server using Azure AI Foundry

Exposes three tools - choose based on query complexity:

1. ask: Quick Q&A with web search (seconds)
   → "What is X?", "How does Y work?", factual lookups

2. web_research: Multi-search research crawling 10-25 sources (2-9 min)
   → Market analysis, technical deep-dives, "tell me everything about X"
   → DEFAULT CHOICE for most research questions

3. deep_research: Exhaustive o3-powered research (5-30 min)
   → High-stakes decisions, due diligence, "leave no stone unturned"
   → Use sparingly - very slow and expensive

Uses Azure AI Foundry with two endpoints:
- gpt-5.2: Powers ask and web_research
- o3-deep-research: Powers deep_research only
"""

import asyncio
import logging
import os
import secrets
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from azure.core.exceptions import ClientAuthenticationError
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DefaultAzureCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from mcp.types import CallToolResult, TextContent
from openai import AsyncOpenAI
from pydantic import Field

load_dotenv(override=True)

# ─────────────────────────────────────────────────────────────
#  Logging Configuration
# ─────────────────────────────────────────────────────────────

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Configure our logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deep_research")

# Query truncation for GDPR compliance
QUERY_LOG_MAX_CHARS = 50


def gen_request_id() -> str:
    """Generate a short request ID for log correlation."""
    return secrets.token_hex(3)  # 6 hex chars, e.g., "a1b2c3"


def truncate_query(query: str) -> str:
    """Truncate query for logging (GDPR compliance)."""
    if len(query) <= QUERY_LOG_MAX_CHARS:
        return query
    return query[:QUERY_LOG_MAX_CHARS] + "..."


# ─────────────────────────────────────────────────────────────
#  Authentication Helpers
# ─────────────────────────────────────────────────────────────


def verify_azure_cli_auth() -> bool:
    """Verify Azure CLI authentication by attempting to get a token."""
    try:
        cred = AzureCliCredential()
        # Try to get a token - this will fail if not logged in
        cred.get_token("https://management.azure.com/.default")
        return True
    except ClientAuthenticationError:
        return False
    except Exception as e:
        logger.warning(f"Azure CLI auth check failed: {e}")
        return False


def has_spn_credentials() -> bool:
    """Check if Service Principal credentials are configured."""
    return all(
        [
            os.environ.get("AZURE_CLIENT_ID"),
            os.environ.get("AZURE_CLIENT_SECRET"),
            os.environ.get("AZURE_TENANT_ID"),
        ]
    )


def get_azure_credential():
    """Get Azure credential optimized for the current environment.

    Priority order:
    1. Service Principal (if AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID set)
    2. Managed Identity (if running in Azure)
    3. Azure CLI credential (local development)
    """
    # Option 1: Service Principal credentials
    if has_spn_credentials():
        logger.info("Using Service Principal authentication")
        from azure.identity import ClientSecretCredential

        return ClientSecretCredential(
            tenant_id=os.environ["AZURE_TENANT_ID"],
            client_id=os.environ["AZURE_CLIENT_ID"],
            client_secret=os.environ["AZURE_CLIENT_SECRET"],
        )

    # Option 2: Managed Identity (Azure Container Apps, VMs, etc.)
    if os.environ.get("IDENTITY_ENDPOINT"):
        logger.info("Using Managed Identity authentication")
        client_id = os.environ.get("AZURE_CLIENT_ID")  # For user-assigned MI
        return ManagedIdentityCredential(client_id=client_id)

    # Option 3: Azure CLI credential (local development)
    if not verify_azure_cli_auth():
        print("=" * 60, file=sys.stderr)
        print("ERROR: No Azure credentials found!", file=sys.stderr)
        print("", file=sys.stderr)
        print("Options:", file=sys.stderr)
        print("  1. Run: az login", file=sys.stderr)
        print("  2. Set environment variables:", file=sys.stderr)
        print(
            "     AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID",
            file=sys.stderr,
        )
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

    logger.info("Using Azure CLI credential")
    return ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(
            exclude_shared_token_cache_credential=True,
            exclude_powershell_credential=True,
        ),
    )


# ─────────────────────────────────────────────────────────────
#  Application Context & Lifespan
# ─────────────────────────────────────────────────────────────


@dataclass
class AppContext:
    """Shared application context with initialized clients."""

    chat_client: AsyncOpenAI  # For chat, web search, reasoning (gpt-5.2)
    research_client: AsyncOpenAI  # For deep research (o3-deep-research)


@dataclass
class UsageStats:
    """Token usage and latency statistics for a request."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    model: str = ""

    def __str__(self) -> str:
        return f"{self.total_tokens:,} tokens ({self.input_tokens:,}→{self.output_tokens:,}) │ {self.latency_ms:,}ms │ {self.model}"


def extract_usage(response) -> UsageStats:
    """Extract usage statistics from OpenAI response."""
    stats = UsageStats()
    if hasattr(response, "usage") and response.usage:
        stats.input_tokens = getattr(response.usage, "input_tokens", 0) or 0
        stats.output_tokens = getattr(response.usage, "output_tokens", 0) or 0
        stats.total_tokens = stats.input_tokens + stats.output_tokens
    if hasattr(response, "model"):
        stats.model = response.model or ""
    return stats


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize clients on startup, cleanup on shutdown."""
    # Get credential (validates az login for local dev)
    credential = get_azure_credential()

    # Token provider for Entra ID auth (sync version)
    sync_token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

    # AsyncOpenAI expects api_key to be either str or Callable[[], Awaitable[str]]
    # Wrap the sync token provider in an async function
    async def async_token_provider() -> str:
        return sync_token_provider()

    # Chat client (gpt-5.2) - for chat, web search, reasoning
    chat_endpoint = os.environ["AI_FOUNDRY_ENDPOINT"]
    chat_client = AsyncOpenAI(
        base_url=chat_endpoint,
        api_key=async_token_provider,
    )

    # Research client (o3-deep-research) - for comprehensive research
    research_endpoint = os.environ["AI_FOUNDRY_DEEP_RESEARCH_ENDPOINT"]
    research_client = AsyncOpenAI(
        base_url=research_endpoint,
        api_key=async_token_provider,
    )

    logger.info(
        "Server ready │ chat: %s │ research: %s",
        os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-5.2"),
        os.environ.get("DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME", "o3-deep-research"),
    )

    try:
        yield AppContext(chat_client=chat_client, research_client=research_client)
    finally:
        await chat_client.close()
        await research_client.close()
        logger.info("Server shutdown")


# Initialize MCP server with lifespan
# Use 0.0.0.0 to allow connections from outside container/devcontainer
mcp = FastMCP("Azure Deep Research", lifespan=app_lifespan, host="0.0.0.0")


# ─────────────────────────────────────────────────────────────
#  Health Check & Info Endpoints
# ─────────────────────────────────────────────────────────────


@mcp.custom_route("/", methods=["GET"])
async def root_info(request):
    """Root endpoint with service info."""
    from starlette.responses import JSONResponse

    return JSONResponse(
        {
            "service": "deep-research-mcp",
            "status": "running",
            "mcp_endpoint": "/mcp",
            "health_endpoint": "/health",
            "description": "Deep Research MCP Server - connect MCP clients to /mcp",
        }
    )


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint for Container Apps probes."""
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "healthy", "service": "deep-research-mcp"})


# ─────────────────────────────────────────────────────────────
#  Prompts for multi-agent deep research
# ─────────────────────────────────────────────────────────────

TRIAGE_PROMPT = """# Identity
You are a research triage specialist who evaluates query specificity and research readiness.
Your role is to determine whether a research query can proceed directly or needs clarification first.

# Instructions
<evaluation_criteria>
Analyze the query against these dimensions:

1. **Scope Definition**: Is the topic bounded or open-ended?
   - Clear scope: "AI in healthcare diagnostics" vs. Unclear: "AI"

2. **Temporal Context**: Is the time period specified or implied?
   - Clear: "in 2024-2025" or "current state" vs. Unclear: no time reference for evolving topics

3. **Geographic/Domain Scope**: Is the region or domain defined?
   - Clear: "in the EU" or "in enterprise software" vs. Unclear: "in the economy"

4. **Measurable Outcomes**: Are success criteria or metrics identifiable?
   - Clear: "adoption rates", "economic impact", "market share" vs. Unclear: "tell me about"

5. **Actionable Focus**: Can a researcher know what to prioritize?
   - Clear: specific comparison, analysis type, or question vs. Unclear: general exploration
</evaluation_criteria>

<decision_rules>
- If 3+ dimensions are unclear AND clarification would significantly improve research quality → NEEDS_CLARIFICATION
- If the query has a clear focus even with some open dimensions → READY_FOR_RESEARCH
- When in doubt, bias toward READY_FOR_RESEARCH (researcher can adapt)
- Current events queries with clear topics are READY (time is implicit)
</decision_rules>

<examples>
NEEDS_CLARIFICATION:
- "Research AI" → Missing: scope, focus area, time period, metrics
- "What's happening in the economy?" → Missing: which economy, what aspects, what metrics
- "Tell me about climate change" → Missing: angle (science/policy/impacts), geography, recency

READY_FOR_RESEARCH:
- "Economic impact of semaglutide on global healthcare systems in 2024-2025" → Clear scope, time, metric
- "Compare renewable energy adoption rates between EU and US in the last 5 years" → Clear comparison, regions, time
- "Current state of quantum computing in financial services" → Clear domain intersection, implicit recency
- "How is NVIDIA positioned in the AI chip market?" → Clear company, market, implicit current
</examples>

# Output
Respond with ONLY one of these exact strings:
- NEEDS_CLARIFICATION
- READY_FOR_RESEARCH

No explanation, no additional text.
"""

CLARIFYING_PROMPT = """# Identity
You are a research intake specialist who helps users refine their research queries.
You ask targeted questions that will meaningfully improve research quality without creating friction.

# Instructions
<question_philosophy>
- Questions should UNLOCK better research, not gatekeep it
- Only ask about dimensions that would SIGNIFICANTLY change the research approach
- Respect the user's time—be concise and purposeful
- Frame questions as collaborative refinement, not interrogation
</question_philosophy>

<high_value_dimensions>
Prioritize clarifying these (in order of typical impact):

1. **Focus/Angle** - What specific aspect matters most?
   - "Are you most interested in the technical, economic, or regulatory aspects?"

2. **Temporal Scope** - What time period is relevant?
   - "Should this focus on current state, historical trends, or future projections?"

3. **Geographic/Market Scope** - What regions or markets matter?
   - "Any specific regions or markets to prioritize (e.g., US, EU, global)?"

4. **Depth vs. Breadth** - Comprehensive overview or deep dive?
   - "Would you prefer a broad landscape or a deep analysis of specific areas?"

5. **Intended Use** (only if it would change approach significantly)
   - "Will this inform a specific decision (investment, policy, strategy)?"
</high_value_dimensions>

<question_format>
- Ask 2-4 questions maximum (fewer is better)
- Use bullet points for clarity
- Lead with the highest-impact question
- Provide example options when helpful: "e.g., technical feasibility, market adoption, regulatory hurdles"
- End with an opt-out: "Or I can proceed with a broad scope if you'd prefer."
</question_format>

<tone_guidelines>
- Warm and collaborative: "To make this research most useful for you..."
- Offer choices rather than open-ended questions when possible
- Never condescending or gatekeeping
- Acknowledge what's already clear in their query
</tone_guidelines>

# Output Format
Brief acknowledgment of the topic (1 sentence), then bulleted questions:

"Great topic! To focus the research effectively:

• [Question 1 - highest impact]
• [Question 2]
• [Optional: Question 3]

Or let me know if you'd like me to proceed with a comprehensive overview."
"""

INSTRUCTION_PROMPT = """# Identity
You are a research brief specialist who transforms user queries into comprehensive, 
actionable research instructions optimized for deep research execution.

# Instructions
<transformation_goals>
- Convert vague requests into specific, measurable research objectives
- Preserve ALL user-provided context and preferences
- Fill gaps with reasonable defaults (clearly marked as flexible)
- Structure the brief to enable systematic, thorough research
</transformation_goals>

<brief_components>
Include these sections in the research brief:

1. **Research Objective** (1-2 sentences)
   - Clear statement of what the research should accomplish
   - Written in first person from user's perspective

2. **Scope Definition**
   - Time period: [specified or "current/recent, with historical context as relevant"]
   - Geography/Market: [specified or "global with regional breakdowns if significant"]
   - Domain boundaries: What's in scope vs. explicitly out of scope

3. **Key Questions to Answer** (3-7 questions)
   - Prioritized list of specific questions the research should address
   - Include both primary questions and supporting sub-questions

4. **Desired Outputs**
   - Data tables for comparisons (specify what to compare)
   - Specific metrics or KPIs to include
   - Any visualizations that would add value

5. **Source Preferences**
   - Priority sources: official reports, peer-reviewed research, government data, reputable industry analysis
   - Recency requirements: prefer sources from [timeframe]
   - For contentious topics: include multiple perspectives

6. **Report Structure**
   - Request executive summary
   - Logical section organization
   - Inline citations with full source metadata
   - Conclusions with confidence levels
</brief_components>

<writing_style>
- First person perspective: "I want to understand...", "Please analyze..."
- Specific and measurable: "Compare X and Y across metrics A, B, C"
- Action-oriented: "Identify", "Analyze", "Compare", "Evaluate"
- Acknowledge flexibility: "If data is unavailable for X, alternatives include Y"
</writing_style>

# Output Format
Provide ONLY the research brief, starting with:

"**Research Brief**

**Objective:** [Clear 1-2 sentence objective]

**Scope:**
..."

Do not include any meta-commentary or explanation—just the brief itself.
"""

RESEARCH_SYSTEM_PROMPT = """# Identity
You are an expert research analyst producing publication-quality reports.
You combine rigorous methodology with clear communication to deliver actionable intelligence.

# Instructions
<research_standards>
## Source Quality Hierarchy (prioritize in order)
1. Primary sources: Official reports, government data, regulatory filings, peer-reviewed research
2. Authoritative secondary: Major consulting firms, established research institutions, central banks
3. Quality journalism: Reuters, Bloomberg, Financial Times, domain-specific trade publications
4. Industry sources: Company reports, earnings calls, official announcements
5. Use with caution: Blogs, opinion pieces, unverified claims (cite skeptically if used)
</research_standards>

<analytical_approach>
- Lead with data: Every major claim should have supporting figures, statistics, or concrete evidence
- Quantify when possible: "grew significantly" → "grew 23% YoY from $X to $Y"
- Show trends: Include historical context and trajectory, not just snapshots
- Compare meaningfully: Benchmarks, peer comparisons, market context
- Acknowledge limitations: Note data gaps, conflicting sources, or uncertainty
- Synthesize, don't just summarize: Draw insights and implications from the data
</analytical_approach>

<report_structure>
## Required Sections
1. **Executive Summary** (3-5 key findings with headline metrics)
2. **Background/Context** (why this matters, current landscape)
3. **Analysis Sections** (organized by theme or question)
4. **Data & Evidence** (tables, key statistics, supporting data)
5. **Implications & Outlook** (what this means, future trajectory)
6. **Sources** (complete bibliography with URLs and access dates)

## Formatting Requirements
- Use headers (##, ###) to create clear hierarchy
- Include data tables for comparisons (markdown format)
- Bullet points for key findings and lists
- **Bold** for emphasis on critical metrics or findings
- Inline citations: [Source Name](URL) immediately after claims
</report_structure>

<citation_requirements>
- Every factual claim needs a citation
- Include publication/access date for time-sensitive data
- For statistics: cite the original source, not secondary reports
- When sources conflict: present both with explanation
- Format: "According to [Source](url), the market reached $X billion in 2025."
</citation_requirements>

<quality_checklist>
Before completing, verify:
□ Executive summary captures the 3-5 most important findings
□ All major claims have citations
□ Data is specific (numbers, dates, percentages) not vague
□ Multiple source types used (not over-reliant on one source)
□ Limitations and uncertainties acknowledged
□ Actionable insights or implications included
□ Sources section is complete with URLs
</quality_checklist>
"""

SYNTHESIS_PROMPT = """# Identity
You are an expert research synthesizer and editor who transforms raw research findings 
into polished, executive-ready reports. You excel at identifying key insights, improving 
clarity, and ensuring reports are actionable and well-structured.

# Instructions
<synthesis_objectives>
- Transform raw research output into a coherent, publication-quality report
- Ensure logical flow and narrative structure throughout
- Highlight the most important findings prominently
- Verify citation consistency and completeness
- Add strategic context and actionable recommendations where appropriate
- Improve readability without losing depth or nuance
</synthesis_objectives>

<report_structure>
Reorganize and enhance the research into this structure:

## Executive Summary
- 3-5 bullet points capturing the most critical findings
- Lead with the single most important insight
- Include key metrics/numbers that matter most
- End with the primary implication or recommendation

## Key Findings
- Organized thematically (not by source)
- Each finding supported by specific evidence with citations
- Quantified wherever possible
- Ordered by importance/relevance

## Detailed Analysis
- Logical sections addressing each major aspect of the research
- Smooth transitions between sections
- Data tables and comparisons where they add value
- Nuanced discussion of complexities and trade-offs

## Implications & Recommendations
- What this means for the user/stakeholder
- Specific, actionable recommendations (if applicable)
- Future trends or developments to watch
- Confidence levels for forward-looking statements

## Methodology & Limitations
- Brief note on research approach and sources used
- Acknowledge data gaps or areas of uncertainty
- Suggestions for further research if relevant

## Sources
- Complete bibliography with all sources cited
- URLs included for verification
- Organized alphabetically or by section
</report_structure>

<editorial_standards>
- **Clarity**: Every sentence should be immediately understandable
- **Precision**: Replace vague language with specific facts and figures
- **Flow**: Each paragraph should connect logically to the next
- **Balance**: Present multiple perspectives on contested points
- **Actionability**: Readers should know what to do with this information
- **Citation integrity**: Every factual claim links to its source
</editorial_standards>

<enhancement_checklist>
As you synthesize, ensure:
□ Executive summary can stand alone as a complete briefing
□ No redundancy or repetition across sections
□ Technical terms are explained or contextualized
□ Data is presented in the most impactful format (tables, bullets, etc.)
□ Conclusions follow logically from evidence presented
□ The report answers the original research question directly
□ All sources from the research are preserved in the bibliography
</enhancement_checklist>
"""

WEB_SEARCH_PROMPT = """# Identity
You are a precise web search assistant that synthesizes information from multiple sources 
into accurate, well-cited answers. You prioritize factual accuracy and source transparency.

# Instructions
<response_guidelines>
- Lead with a direct, definitive answer to the query in the first sentence
- Support claims with specific facts, figures, and data points from sources
- Always include inline citations using [Source Title](URL) format
- When sources conflict, acknowledge the discrepancy and present multiple perspectives
- Distinguish clearly between facts (with citations) and any inference or synthesis
- For time-sensitive queries, prioritize the most recent sources and note publication dates
- If information is unavailable or uncertain, explicitly state the limitation
</response_guidelines>

<output_format>
Structure your response as:
1. **Direct Answer**: Concise answer to the query (1-2 sentences)
2. **Key Findings**: Bullet points with specific facts and inline citations
3. **Sources**: Numbered list of all sources with titles and URLs

Example citation format: "According to [Reuters](https://reuters.com/...), the GDP grew 2.3%..."
</output_format>

<quality_standards>
- Prefer primary sources: official reports, peer-reviewed research, government data
- For news, prefer established outlets with editorial standards
- Include publication dates for time-sensitive information
- Never fabricate sources or citations
- If a claim cannot be verified, omit it rather than present it as fact
</quality_standards>
"""

ASK_PROMPT = """# Identity
You are a knowledgeable, conversational AI assistant with access to real-time web information.
You communicate like a trusted colleague—helpful, clear, and personable while maintaining accuracy.

# Instructions
<conversation_style>
- Be warm and approachable while remaining informative
- Match the user's level of formality and expertise
- Use natural language, avoiding robotic or overly formal phrasing
- When appropriate, anticipate follow-up questions and address them proactively
- Keep responses focused and digestible—expand only when depth adds value
</conversation_style>

<information_handling>
- Seamlessly integrate web information into conversational responses
- Cite sources naturally within the flow of conversation: "According to [Source]..."
- For factual claims, always ground them in retrieved information
- When web results are limited or conflicting, be transparent about uncertainty
- Distinguish between established facts, recent developments, and your synthesis
</information_handling>

<response_structure>
- Start with the most important/relevant information
- Use short paragraphs and bullet points for complex topics
- For multi-part questions, address each component clearly
- End with an invitation for follow-up if the topic warrants deeper exploration
</response_structure>

<citation_format>
Include sources inline when making factual claims:
- "The study found that... ([Harvard Business Review](url))"
- Collect unique sources at the end for reference when multiple are used
</citation_format>
"""

RESEARCH_PROMPT = """# Identity
You are a **thorough research analyst** who conducts iterative web research to build comprehensive understanding.
Your goal: **Find 10-25 quality sources through multiple targeted searches.**

You answer complex questions by searching, reading, refining your understanding, and searching again.

# Research Process

## Phase 1: Decompose the Query
- Break the topic into 3-6 distinct research angles
- Each angle should require different search terms
- Example: "biotech catalyst investing" → FDA process, PDUFA dates, binary events, risk management, valuation methods, tracking tools

## Phase 2: Iterative Search Loop
For EACH research angle:
1. **Search** with targeted keywords
2. **Read** promising results to extract facts
3. **Assess** what's still missing or unclear
4. **Refine** search terms based on what you learned
5. **Repeat** until that angle is covered

## Phase 3: Cross-Reference & Validate
- When sources conflict, search for clarification
- Look for primary sources when secondary sources cite them
- Verify key claims with multiple sources

## Phase 4: Synthesize
- Integrate findings across all angles
- Identify patterns and connections
- Note gaps and limitations

# Search Behavior Guidelines

## DO:
- Conduct 8-15 separate web searches minimum
- Read deeply into promising sources
- Follow citation trails to primary sources
- Search for specific terms you encounter (e.g., "72-hour rule PDUFA")
- Vary search terms: try synonyms, specific names, technical terms
- When a search fails, reformulate and try again

## DON'T:
- Stop after 2-3 searches
- Accept first results without verification
- Skip reading sources that look relevant
- Give up when initial searches are unproductive

# Target Metrics
- **Sources**: 10-25 quality sources
- **Searches**: 8-15+ distinct searches
- **Time**: 2-9 minutes (take the time needed)
- **Coverage**: Multiple angles, cross-referenced

# Output Structure

## Executive Summary
- 3-5 key findings with the most important insights
- Lead with actionable conclusions

## Detailed Findings
- Organized by theme/angle (not by source)
- Specific facts, figures, and data points
- Inline citations: [Source Name](url)

## Key Definitions & Concepts
- Define technical terms encountered
- Explain industry-specific concepts

## Data & Comparisons
- Tables where appropriate
- Quantified metrics when available

## Confidence Assessment
- **High**: Multiple authoritative sources agree
- **Medium**: Limited sources or some ambiguity
- **Low**: Sparse data or conflicting information
- Note specific gaps in coverage

## Sources
- Complete list of all sources consulted
- Include URLs for verification

# Quality Standards
- Every major claim needs a citation
- Prefer primary sources: official reports, regulatory documents, peer-reviewed research
- For news: prefer established outlets (Reuters, Bloomberg, industry publications)
- When sources conflict: present both perspectives
- Be explicit about what you couldn't verify

# When to Recommend deep_research Instead
- Topic requires 30+ sources or multi-day research
- Need exhaustive verification with adversarial cross-checking
- High-stakes decision requiring "defensible" output
- Complex topic requiring code analysis or document processing
"""


# ─────────────────────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────────────────────


async def call_chat(
    ctx: Context[ServerSession, AppContext],
    system_prompt: str,
    user_message: str,
    use_web_search: bool = False,
    req_id: str = "",
    phase: str = "",
) -> tuple[str, UsageStats]:
    """Call GPT model using Responses API with optional web search.

    Returns tuple of (response_text, usage_stats).
    """
    app = ctx.request_context.lifespan_context
    model = os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-5.2")

    request_params = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": user_message}]},
        ],
    }

    if use_web_search:
        request_params["tools"] = [{"type": "web_search_preview"}]

    start = time.perf_counter()
    response = await app.chat_client.responses.create(**request_params)
    latency_ms = int((time.perf_counter() - start) * 1000)

    stats = extract_usage(response)
    stats.latency_ms = latency_ms
    stats.model = model

    if phase:
        logger.info("[%s] %s │ %s", req_id, phase, stats)

    return response.output_text or "", stats


async def perform_deep_research(
    ctx: Context[ServerSession, AppContext], instructions: str, req_id: str = ""
) -> tuple[str, UsageStats]:
    """Execute deep research using o3-deep-research with streaming and retry logic.

    Returns tuple of (response_text, usage_stats).
    Logs progress updates during long-running research.
    Automatically retries on transient connection failures (e.g., laptop sleep).
    """
    app = ctx.request_context.lifespan_context
    model = os.environ.get("DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME", "o3-deep-research")
    full_input = f"{RESEARCH_SYSTEM_PROMPT}\n\n{instructions}"
    stats = UsageStats(model=model)

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAYS = [5, 15, 30]  # seconds between retries

    # Progress tracking
    LOG_INTERVAL_SECS = 30  # Log progress every 30 seconds

    start = time.perf_counter()
    logger.info("[%s] research │ starting %s", req_id, model)

    for attempt in range(MAX_RETRIES + 1):
        full_text = ""
        last_log_time = time.perf_counter()
        last_log_chars = 0
        web_search_count = 0
        code_interpreter_count = 0

        try:
            stream = await app.research_client.responses.create(
                model=model,
                input=full_input,
                stream=True,
                tools=[
                    {"type": "web_search_preview"},
                    {"type": "code_interpreter", "container": {"type": "auto"}},
                ],
            )

            async for event in stream:
                if event.type == "response.output_text.delta":
                    full_text += event.delta

                    # Log progress periodically
                    now = time.perf_counter()
                    if now - last_log_time >= LOG_INTERVAL_SECS:
                        elapsed_secs = now - start
                        elapsed_str = _format_elapsed(elapsed_secs)
                        chars_since = len(full_text) - last_log_chars
                        tools_used = []
                        if web_search_count:
                            tools_used.append(f"{web_search_count} searches")
                        if code_interpreter_count:
                            tools_used.append(f"{code_interpreter_count} code runs")
                        tools_str = f" │ {', '.join(tools_used)}" if tools_used else ""
                        logger.info(
                            "[%s] research │ %s elapsed │ %d chars (+%d)%s",
                            req_id,
                            elapsed_str,
                            len(full_text),
                            chars_since,
                            tools_str,
                        )
                        last_log_time = now
                        last_log_chars = len(full_text)

                        # Update MCP progress (10-85% range for research phase)
                        # Estimate based on typical research taking 5-20 mins
                        elapsed_mins = elapsed_secs / 60
                        progress_pct = min(10 + int(elapsed_mins * 5), 80)
                        await ctx.report_progress(
                            progress_pct, 100, f"Researching... {elapsed_str} elapsed"
                        )

                elif event.type == "response.web_search_call.in_progress":
                    web_search_count += 1
                    logger.info(
                        "[%s] research │ web search #%d", req_id, web_search_count
                    )

                elif event.type == "response.code_interpreter_call.in_progress":
                    code_interpreter_count += 1
                    logger.info(
                        "[%s] research │ code interpreter #%d",
                        req_id,
                        code_interpreter_count,
                    )

                elif event.type == "response.completed":
                    if event.response.output_text:
                        full_text = event.response.output_text
                    # Extract usage from completed response
                    if hasattr(event.response, "usage") and event.response.usage:
                        stats.input_tokens = (
                            getattr(event.response.usage, "input_tokens", 0) or 0
                        )
                        stats.output_tokens = (
                            getattr(event.response.usage, "output_tokens", 0) or 0
                        )
                        stats.total_tokens = stats.input_tokens + stats.output_tokens

            # Success - break out of retry loop
            break

        except Exception as e:
            error_str = str(e)
            is_transient = any(
                phrase in error_str.lower()
                for phrase in [
                    "incomplete chunked read",
                    "connection reset",
                    "connection closed",
                    "peer closed",
                    "timeout",
                    "temporary failure",
                    "service unavailable",
                ]
            )

            if is_transient and attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                elapsed_str = _format_elapsed(time.perf_counter() - start)
                logger.warning(
                    "[%s] research │ connection lost after %s │ retry %d/%d in %ds │ %s",
                    req_id,
                    elapsed_str,
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                    error_str[:100],
                )
                await asyncio.sleep(delay)
                # Note: We restart the full request - o3-deep-research doesn't support resumption
                # The model will re-do the work, but at least we don't fail completely
                continue
            else:
                # Non-transient error or exhausted retries
                raise

    stats.latency_ms = int((time.perf_counter() - start) * 1000)
    elapsed_str = _format_elapsed(stats.latency_ms / 1000)
    tools_summary = []
    if web_search_count:
        tools_summary.append(f"{web_search_count} searches")
    if code_interpreter_count:
        tools_summary.append(f"{code_interpreter_count} code runs")
    tools_str = f" │ {', '.join(tools_summary)}" if tools_summary else ""
    logger.info(
        "[%s] research │ done │ %s │ %d chars%s │ %s",
        req_id,
        elapsed_str,
        len(full_text),
        tools_str,
        stats,
    )

    return full_text, stats


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable string (e.g., '5m 23s' or '45s')."""
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


# ─────────────────────────────────────────────────────────────
#  MCP Tools
# ─────────────────────────────────────────────────────────────


@mcp.tool()
async def ask(
    question: str = Field(
        description="Question or topic to get a quick, conversational answer about"
    ),
    ctx: Context[ServerSession, AppContext] = None,
) -> CallToolResult:
    """Answer questions conversationally with real-time web search. Fast responses for everyday queries."""
    req_id = gen_request_id()
    try:
        logger.info("[%s] ask │ %s", req_id, truncate_query(question))
        result, stats = await call_chat(
            ctx,
            ASK_PROMPT,
            question,
            use_web_search=True,
            req_id=req_id,
            phase="answer",
        )
        return CallToolResult(content=[TextContent(type="text", text=result)])
    except Exception as e:
        logger.error("[%s] ask failed │ %s", req_id, e)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Request failed: {str(e)}")],
            isError=True,
        )


@mcp.tool()
async def deep_research(
    topic: str = Field(
        description="High-stakes research requiring exhaustive verification (use sparingly)"
    ),
    ctx: Context[ServerSession, AppContext] = None,
    skip_synthesis: bool = Field(
        default=False, description="Return raw research without final polish (faster)"
    ),
) -> CallToolResult:
    """Exhaustive research using o3 model. VERY SLOW (5-30 min) and EXPENSIVE. Use ONLY for: due diligence, high-stakes decisions, adversarial verification, when web_research isn't thorough enough. For most research, use web_research instead."""
    req_id = gen_request_id()
    total_start = time.perf_counter()
    total_tokens = 0

    try:
        logger.info("[%s] deep_research │ %s", req_id, truncate_query(topic))

        # Phase 1: TRIAGE (5%)
        await ctx.report_progress(0, 100, "Analyzing query scope...")
        triage_result, stats = await call_chat(
            ctx,
            TRIAGE_PROMPT,
            topic,
            use_web_search=False,
            req_id=req_id,
            phase="triage",
        )
        total_tokens += stats.total_tokens

        scope_context = (
            "broad scope - cover key aspects comprehensively since query is general"
            if "NEEDS_CLARIFICATION" in triage_result.upper()
            else "focused scope - query is specific, address it directly"
        )

        # Phase 2: INSTRUCTION (10%)
        await ctx.report_progress(5, 100, "Creating research brief...")
        instruction_input = f"Original query: {topic}\n\nResearch scope: {scope_context}\n\nCreate comprehensive research instructions."
        enhanced_instructions, stats = await call_chat(
            ctx,
            INSTRUCTION_PROMPT,
            instruction_input,
            use_web_search=False,
            req_id=req_id,
            phase="instruct",
        )
        total_tokens += stats.total_tokens

        # Phase 3: RESEARCH (10-85%)
        await ctx.report_progress(
            10, 100, "Deep research in progress (this takes 5-30 minutes)..."
        )
        raw_research, stats = await perform_deep_research(
            ctx, enhanced_instructions, req_id=req_id
        )
        total_tokens += stats.total_tokens

        if not raw_research:
            logger.error("[%s] deep_research │ no output generated", req_id)
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="Research completed but no output was generated.",
                    )
                ],
                isError=True,
            )

        # Phase 4: SYNTHESIS (85-100%)
        if skip_synthesis:
            await ctx.report_progress(100, 100, "Complete (synthesis skipped)")
            total_ms = int((time.perf_counter() - total_start) * 1000)
            logger.info(
                "[%s] deep_research │ done │ %d tokens │ %.1fs │ skipped synthesis",
                req_id,
                total_tokens,
                total_ms / 1000,
            )
            return CallToolResult(content=[TextContent(type="text", text=raw_research)])

        await ctx.report_progress(85, 100, "Synthesizing final report...")
        synthesis_input = f"# Original Research Query\n{topic}\n\n# Research Scope\n{scope_context}\n\n# Raw Research Findings\n{raw_research}\n\n# Your Task\nSynthesize and enhance the above research into a polished, executive-ready report."
        final_report, stats = await call_chat(
            ctx,
            SYNTHESIS_PROMPT,
            synthesis_input,
            use_web_search=False,
            req_id=req_id,
            phase="synthesize",
        )
        total_tokens += stats.total_tokens

        if not final_report:
            final_report = raw_research

        await ctx.report_progress(100, 100, "Research complete")
        total_ms = int((time.perf_counter() - total_start) * 1000)
        logger.info(
            "[%s] deep_research │ done │ %d tokens │ %.1fs",
            req_id,
            total_tokens,
            total_ms / 1000,
        )
        return CallToolResult(content=[TextContent(type="text", text=final_report)])

    except Exception as e:
        logger.error("[%s] deep_research failed │ %s", req_id, e)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Research failed: {str(e)}")],
            isError=True,
        )


@mcp.tool()
async def web_research(
    topic: str = Field(
        description="Research topic - crawls 10-25 web sources (2-9 min)"
    ),
    ctx: Context[ServerSession, AppContext] = None,
) -> CallToolResult:
    """Search the web iteratively, crawling 10-25 sources (2-9 min). DEFAULT for research questions. Searches, reads, refines, repeats until comprehensive. Use for: market analysis, technical topics, competitive research, "tell me about X". Only escalate to deep_research for high-stakes verification."""
    req_id = gen_request_id()
    start = time.perf_counter()

    try:
        logger.info("[%s] web_research │ %s", req_id, truncate_query(topic))

        # Iterative multi-search research
        result, stats = await call_chat(
            ctx,
            RESEARCH_PROMPT,
            topic,
            use_web_search=True,
            req_id=req_id,
            phase="web_research",
        )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "[%s] web_research │ done │ %s │ %.1fs",
            req_id,
            stats,
            elapsed_ms / 1000,
        )

        return CallToolResult(content=[TextContent(type="text", text=result)])

    except Exception as e:
        logger.error("[%s] web_research failed │ %s", req_id, e)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Web research failed: {str(e)}")],
            isError=True,
        )


# ─────────────────────────────────────────────────────────────
#  Server entry point
# ─────────────────────────────────────────────────────────────


def main():
    """Run the MCP server.

    Supports two transport modes:
    - stdio (default): For VS Code MCP integration, started via mcp.json
    - http: For standalone server mode, useful for dev/debugging

    Usage:
        python -m deep_research.server          # stdio mode (VS Code)
        python -m deep_research.server --http   # HTTP mode on port 8000
    """
    import argparse

    parser = argparse.ArgumentParser(description="Deep Research MCP Server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run as HTTP server on port 8000 (default: stdio for VS Code)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP mode (default: 8000)"
    )
    args = parser.parse_args()

    if args.http:
        # Port is set via mcp.settings since run() doesn't accept port argument
        mcp.settings.port = args.port
        logger.info("Starting HTTP server on port %d", args.port)
        mcp.run(transport="streamable-http", mount_path="/mcp")
    else:
        # stdio mode for VS Code MCP integration
        logger.info("Starting stdio server for VS Code MCP")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
