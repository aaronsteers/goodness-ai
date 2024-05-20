"""The Crew module provides a high-level interface for defining and running"""

import yaml
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI

# from langchain_groq import ChatGroq

VERBOSE = 2
from langchain_community.llms import Ollama


@CrewBase
class GoodnessCrew:
    """The GoodnessCrew class defines a crew of agents that work together to
    determine the goodness of a given input."""

    agents_config = yaml.safe_load("config/agents.yaml")
    tasks_config = yaml.safe_load("config/tasks.yaml")

    @property
    def default_llm(self) -> ChatOpenAI:
        return Ollama(model="llama3")

    def __init__(self) -> None:
        pass

    @property
    @agent
    def moral_advisor_agent(self) -> Agent:
        """The ChatAgent class defines an agent that interacts with the user
        to get input."""
        return Agent(
            config=self.agents_config["moral_advisor"],
            llm=self.default_llm,
        )

    @property
    @agent
    def moral_computer_agent(self) -> Agent:
        """The ChatAgent class defines an agent that interacts with the user
        to get input."""
        return Agent(
            config=self.agents_config["moral_computer"],
            llm=self.default_llm,
        )

    @property
    @task
    def refine_question_task(self) -> Task:
        """Refine the input question."""
        return Task(
            config=self.tasks_config["refine_moral_question"],
            agent=self.moral_advisor_agent,
        )

    @property
    @task
    def answer_question_task(self) -> Task:
        """Refine the input question."""
        return Task(
            config=self.tasks_config["answer_moral_question"],
            agent=self.moral_computer_agent,
        )

    @property
    @task
    def judge_correctness_task(self) -> Task:
        """Refine the input question."""
        return Task(
            config=self.tasks_config["judge_correctness"],
            agent=self.moral_computer_agent,
        )

    @crew
    def crew(self) -> Crew:
        """The GoodnessCrew class defines a crew of agents that work together to
        determine the goodness of a given input."""
        return Crew(
            agents=[self.moral_advisor_agent, self.moral_computer_agent],
            tasks=[
                self.refine_question_task,
                self.answer_question_task,
                self.judge_correctness_task,
            ],
            process=Process.sequential,
            verbose=VERBOSE,
        )
