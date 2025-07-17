#!/usr/bin/env python
# coding: utf-8

# # LangChain: Agents
# 
# ## Outline:
# 
# * Using built in LangChain tools: DuckDuckGo search and Wikipedia
# * Defining your own tools

# In[ ]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings("ignore")
import datetime
current_date = datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature=0, model=llm_model)

tools = load_tools(["llm-math","wikipedia"], llm=llm)




agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)



agent("What is the 25% of 300?")


question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 


agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)


customer_list = [["aarshi", "verma"], 
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"], 
                 ["Geoff","Fusion"], 
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]



agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 


import langchain
langchain.debug=True
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""") 
langchain.debug=False


from langchain.agents import tool
from datetime import date

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())


agent= initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)


try:
    result = agent("whats the date today?") 
except: 
    print("exception on external access")