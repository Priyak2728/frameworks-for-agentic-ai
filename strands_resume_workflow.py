from strands import Agent, tool
from strands.models import BedrockModel


# =========================================================
# 1. CUSTOM TOOLS
# =========================================================

@tool
def check_server_status(server: str) -> str:
    """Check the status of a server."""

    servers = {
        "server-1": "Running - CPU 42%, Memory 65%",
        "server-2": "Down - No response",
        "server-3": "Running - CPU 28%, Memory 51%"
    }

    return servers.get(
        server.lower(),
        "Server not found"
    )


@tool
def restart_server(server: str) -> str:
    """Restart a server."""

    return f"{server} has been restarted successfully."


@tool
def create_ticket(issue: str) -> str:
    """Create an IT support ticket."""

    return f"Support ticket created for: {issue}"


# =========================================================
# 2. AMAZON BEDROCK MODEL
# =========================================================

model = BedrockModel(
    model_id="global.anthropic.claude-sonnet-4-6",
    region_name="us-east-1",
    temperature=0.2
)


# =========================================================
# 3. CREATE STRANDS AGENT
# =========================================================

agent = Agent(
    model=model,
    tools=[
        check_server_status,
        restart_server,
        create_ticket
    ],
    system_prompt="""
    You are an IT support agent.

    When a user reports a server problem:

    1. Check the server status.
    2. If the server is down, restart it.
    3. Check whether the restart solved the issue.
    4. If the problem continues, create a support ticket.
    5. Clearly explain what action you took.
    """
)


# =========================================================
# 4. USER REQUEST
# =========================================================

response = agent(
    "Server-2 is not responding. Please fix the problem."
)


# =========================================================
# 5. OUTPUT
# =========================================================

print(response)