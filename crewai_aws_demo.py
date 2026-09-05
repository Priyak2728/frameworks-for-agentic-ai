# file: crewai_aws_demo.py

from crewai import Agent, Task, Crew

# 🔹 Define Agents (use string LLM instead of object)
researcher = Agent(
    role="Researcher",
    goal="Find key points about Agentic AI",
    backstory="Expert in AI research",
    llm="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Write a simple explanation for beginners",
    backstory="Expert content creator",
    llm="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    verbose=True
)

# 🔹 Tasks
task1 = Task(
    description="Research Agentic AI and list key concepts",
    expected_output="Bullet points of key concepts",
    agent=researcher
)

task2 = Task(
    description="Write a simple and clear explanation based on the research",
    expected_output="Beginner-friendly explanation",
    agent=writer
)

# 🔹 Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True
)

# 🔹 Run
result = crew.kickoff()

print("\n FINAL OUTPUT:\n")
print(result)