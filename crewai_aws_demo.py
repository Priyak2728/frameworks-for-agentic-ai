from crewai import Agent, Task, Crew

# 🔹 Define Agents
researcher = Agent(
    role="Researcher",
    goal="Find key points about Agentic AI",
    backstory="Expert in AI research",
    llm="bedrock/amazon.nova-lite-v1:0",
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Write a simple explanation for beginners",
    backstory="Expert content creator",
    llm="bedrock/amazon.nova-lite-v1:0",
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

print("\nFINAL OUTPUT:\n")
print(result)