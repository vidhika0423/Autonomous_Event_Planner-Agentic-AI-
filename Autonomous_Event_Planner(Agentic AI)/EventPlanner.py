from typing import Sequence, Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, ToolCall
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
# for email sending
import os
import re
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()
# for email through smtp
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    req_location: str
    final_venue: str
    events_planned: str
    travel_options: str
    recipient_emails: List[str]
    event_date: Optional[str]
    budget: Optional[str]
    event: str
    tool_calls: List[ToolCall]
    
document_content = ''
document_event=''
document_travel=''
email_content=''
final_fielname=''

@tool
def update_venue(content: str) -> str:
    """
    Updates the venue description according to user preference.
    """
    global document_content
    document_content = content
    return f"Venue description updated:\n{document_content}\nIs this okay, or do you want to modify it further?"

@tool
def save_venue(final_venue: str) -> str:
    """
    Save the approved venue and proceed to next process.
    Args:
        finalized venue
    """
    global document_content
    document_content = final_venue
    return f"Venue finalized: {final_venue}\nLet's proceed to event planning."

@tool
def update_events(content: str) -> str:
    """
    Updates the events description according to user preference.

    """
    global document_event
    document_event = content
    return f"Event description updated! Current preferences:\n{document_event}\n\nIs this okay, or do you want to modify it further?"

@tool
def save_event_plan(events_planned:str)->str:
    """
    save the approved events anf proceed to next process
    Args:
        events_planned: finalized events 
    """
    global document_event
    document_event=events_planned
    return f"Events has been finalized:\n{events_planned}"



@tool
def update_travel_opts(content:str)->str:
    """updates the travel options if user is not happy with that. Find a travel options according to user requirement"""
    global document_travel
    document_travel=content
    return f"Travel options has been successfully updated! Current options are:\n{document_travel}\n\nIs this okay, or do you want to modify it further?"

@tool
def save_travel_opts(travel_opts:str)->str:
    """
    Save the approved travel options and proceed to next process.
    """
    global document_travel
    document_travel=travel_opts
    return f"Travel options has been finalized:\n{travel_opts}\n\nNow let's proceed to the next step of finalizing guest list."

@tool
def update_email_content(content: str) -> str:
    """Updates the email content with the provided content."""
    global email_content
    email_content = content
    return f"Email has been updated successfully! The current content is:\n{content}"


@tool
def save_email_content(filename: str) -> str:
    """Save the current email content to a text file .
    
    Args:
        filename: Name for the text file.
    """

    global email_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    global final_filename
    final_filename=filename
    try:
        with open(filename, 'w') as file:
            file.write(email_content)
        print(f"\nEmail has been saved to: {filename}")
        return f"Email has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"



tools = [update_venue,save_venue,update_events,save_event_plan,update_travel_opts,save_travel_opts]
venue_tools = [update_venue, save_venue]
event_tools = [update_events, save_event_plan]
travel_tools=[update_travel_opts,save_travel_opts]
email_tools=[update_email_content, save_email_content]

def get_initial_input(state: AgentState) -> AgentState:
    state['messages'].append(HumanMessage(content=input("Describe your event: ")))
    state['event'] = input("Event name/type: ")
    state['event_date'] = input("Event date: ")
    state['budget'] = input("Budget: ")
    state['req_location'] = input("Preferred location: ")
    state['tool_calls'] = []
    return state

model_venue = ChatGroq(model="gemma2-9b-it").bind_tools(tools)

def search_venue(state: AgentState) -> AgentState:
    global document_content
    messages = list(state['messages'])
    additional_info = document_content.strip() if document_content.strip() else messages[-1].content

    sys_msg = SystemMessage(content=f"""
    You are the venue assistant.
    You can use the following tools when needed:
    - 'update_venue' to update the venue description
    - 'save_venue' to finalize the venue

    When the user confirms the venue selection (using words like 'finalize', 'confirm', 'save'), call the 'save_venue' tool with the venue name as argument.

    Event details:
    - Event: {state['event']}
    - Budget: {state['budget']}
    - Date: {state['event_date']}
    - Location: {state['req_location']}
    - Info: {additional_info}

    Suggest 3 venues first. Then use tools appropriately depending on user responses.
    """)
    
    user_input = input("\nWhat would you like to do with the venue? ")
    print(f"\nUSER: {user_input}")
    
    all_messages = [sys_msg] + messages + [HumanMessage(content=user_input)]

    
    response = model_venue.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
        state['tool_calls'] = response.tool_calls

    messages = list(state["messages"]) + [HumanMessage(content=user_input), response]
    state['messages'] = messages
    state['final_venue'] = document_content
    return state

# arranging events

model_event = ChatGroq(model="gemma2-9b-it").bind_tools(tools)

def arrange_events(state: AgentState) -> AgentState:
    global document_event
    messages = list(state['messages'])
    additional_info = document_event.strip() if document_event.strip() else messages[-1].content

    sys_msg = SystemMessage(content=f"""
    You are an event planning assistant specializing in suggesting creative and engaging activities for different kinds of celebrations.
    You can use the following tools when needed:
    - 'update_events' → to modify event description.
    - 'save_event_plan' → to finalize the events.

    When the user provides final event details or says things like 'finalize', 'save', 'proceed', or 'done',
    use the 'save_event_plan' tool with the finalized events description.

    Venue: {state.get('final_venue', 'Not selected')}
    Event: {state.get('event', 'Not specified')}
    Date: {state.get('event_date', 'Not specified')}
    Info so far: {additional_info}

    Example to finalize:
    {{ "tool_call": "save_event_plan", "arguments": {{ "events_planned": "[finalized events]" }} }}

    Start by asking: "What kind of celebration are you planning (e.g., birthday, anniversary, farewell, corporate)?"
    """)

    user_input = input("\nDescribe your thoughts on event planning: ").strip()
    print(f"\nUSER: {user_input}")

    all_messages = [sys_msg] + messages + [HumanMessage(content=user_input)]

    response = model_event.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
        state['tool_calls'] = response.tool_calls

    messages = list(state['messages']) + [HumanMessage(content=user_input), response]
    state['messages'] = messages
    state['events_planned'] = document_event
    return state



model_travel = ChatGroq(model="gemma2-9b-it").bind_tools(tools)

def find_travel(state: AgentState) -> AgentState:
    global document_travel
    messages = list(state['messages'])
    venue_info = state.get('final_venue', 'Not selected')
    additional_info = document_travel.strip() if document_travel.strip() else messages[-1].content

    sys_msg = SystemMessage(content=f"""
    You are a travel planning assistant. Your task is to help the user plan their travel to the event venue.

    Responsibilities:
    - Suggest the best travel options to **{venue_info}** from nearby airports and railway stations.
    - Recommend local commute options to reach **{venue_info}** from key locations.
    - Highlight popular landmarks near **{venue_info}**.

    Tools you can use:
    - 'update_travel_opts' → to modify the travel plan.
    - 'save_travel_opts' → to finalize the travel plan.

    When the user provides complete travel details or says things like 'finalize', 'save', 'proceed', or 'done',
    use the 'save_travel_opts' tool with the finalized travel description.

    Venue: {venue_info}
    Date: {state.get('event_date', 'Not specified')}
    Info so far: {additional_info}

    Example to finalize:
    {{ "tool_call": "save_travel_opts", "arguments": {{ "travel_opts": "[finalized travel plan]" }} }}

    Start by asking: "Do you have any specific travel preferences (mode of transport, preferred station/airport, etc.)?"
    """)

    user_input = input("\nDescribe your travel preferences: ").strip()
    print(f"\nUSER: {user_input}")

    all_messages = [sys_msg] + messages + [HumanMessage(content=user_input)]

    response = model_travel.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
        state['tool_calls'] = response.tool_calls

    messages = list(state['messages']) + [HumanMessage(content=user_input), response]
    state['messages'] = messages
    document_travel = response.content
    state['travel_options'] = document_travel
    return state



def should_continue_venue(state: AgentState) -> str:
    messages=state["messages"]
    last_message=messages[-1]
    if isinstance(last_message,AIMessage):
        for tool_call in last_message.tool_calls:
            if tool_call.name=='save_venue':
                return 'event_edge'
    if isinstance(last_message, ToolMessage):
        if "Venue finalized" in last_message.content:
            return 'event_edge'
    return 'continue_venue'

def should_continue_events(state: AgentState) -> str:
    messages=state['messages']
    last_message=messages[-1]
    if isinstance(last_message,AIMessage):
        for tool_call in last_message.tool_calls:
            if tool_call.name=='save_event_plan':
                return 'travel_edge'
    if isinstance(last_message,ToolMessage):
        if 'Events has been finalized' in last_message.content:
            return 'travel_edge'
    return 'continue_events'


def should_continue_travel(state: AgentState) -> str:
    messages=state['messages']
    last_message=messages[-1]
    if isinstance(last_message,AIMessage):
        for tool_call in last_message.tool_calls:
            if tool_call.get('name') == 'save_travel_opts':
                return 'email_edge'
    if isinstance(last_message,ToolMessage):
        if 'Travel options has been finalized' in last_message.content:
            return 'email_edge'
    return "continue_travels"



def print_messages(messages):
    if not messages:
        return
    message = messages[-1]
    if isinstance(message, ToolMessage):
        print(f"\nTOOL RESULT: {message.content}")

# for drafting email
model_email = ChatGroq(model="gemma2-9b-it").bind_tools(email_tools)

def email_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a professional writing assistant specializing in drafting, editing, and refining emails.
    Your responsibilities:
    Assist the user in writing clear, concise, and effective emails.
    Help rewrite, polish, or modify email drafts upon request.
    Always reflect the latest version of the email after changes.
    
    IMPORTANT: You must ONLY use these tools for email drafting:
    - update_email_content: to modify the email content
    - save_email_content: to save the final email
    
    DO NOT use any other tools like venue, travel, or event tools.
    Event details to include in the email:
    Venue: {state.get('final_venue', 'Not selected')}
    Event: {state.get('event', 'Not specified')}
    Date: {state.get('event_date', 'Not specified')}
    Travel options: {state.get('travel_options', 'Not specified')}
    
    Current email draft:
    {email_content}
    """)

    print("Let's start drafting your email.")
    user_input = input("\nWhat would you like to do with the email? ")
    print(f"\nUSER: {user_input} ")
    combined_input = f"{user_input}\n\nCurrent draft is:\n{email_content}"
    user_message = HumanMessage(content=combined_input)
    
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model_email.invoke(all_messages)

    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue_drafting_email(state: AgentState) -> str:
    """Determine if we should continue drafting email or end the conversation."""
    messages = state["messages"]
    if not messages:
        return "continue_email"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "email has been saved successfully" in message.content.lower()):
            return "end"
    return "continue_email"


# sending email functions
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)


def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        return True, "Email sent"
    except Exception as e:
        return False, f"Error in sending email: {str(e)}"



def load_email_list(filename='email.json'):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data.get('recipients', [])
    except Exception as e:
        print(f"❗ Error loading email list: {e}")
        return []


def load_email_body(final_filename):
    try:
        with open(final_filename, 'r') as file:
            return file.read()
    except Exception as e:
        print(f"❗ Error reading email body: {e}")
        return ""

def send_bulk_email(subject, body_filename, recipients_filename='emails.json'):
    recipients = load_email_list(recipients_filename)
    if not recipients:
        print("❗ No recipients found in email.json.")
        return

    body = load_email_body(body_filename)
    if not body:
        print("❗ Email body is empty.")
        return

    for email in recipients:
        if is_valid_email(email):
            success, message = send_email(email, subject, body)
            print(f"📧 Sending to {email}: {message}")
        else:
            print(f"⚠️ Invalid email format: {email}")


def send_email_node(state: AgentState) -> AgentState:
    subject = "invitation"
    global final_filename
    if not final_filename:
        print("❗ Final filename not set. Cannot send email.")
        return state
    send_bulk_email(subject, body_filename=final_filename)
    return state

graph = StateGraph(AgentState)
graph.add_node("get_data", get_initial_input)
graph.add_node("get_venue", search_venue)
graph.add_node("venue_tools", ToolNode(tools=tools))

graph.add_node('get_events',arrange_events)         #for event scheduling

# ToolNode for events
graph.add_node('event_tools',ToolNode(tools=tools))


graph.add_node('find_travel_opts',find_travel)      #for searching travel options

# ToolNode for travel options
graph.add_node('travel_tools',ToolNode(tools=travel_tools))
# node for email sending
graph.add_node("email_agent", email_agent)
graph.add_node("email_tools", ToolNode(tools=tools))
graph.add_node("send_email", send_email_node)

graph.add_edge(START, "get_data")
graph.add_edge("get_data", "get_venue")
graph.add_edge("get_venue", "venue_tools")

graph.add_conditional_edges(
    "venue_tools",
    should_continue_venue,
    {
        "continue_venue": "get_venue",
        "event_edge": 'get_events',
    }
)

# reached get_events node
graph.add_edge('get_events','event_tools')
graph.add_conditional_edges(
    'event_tools',
    should_continue_events,
    {
        'continue_events':'get_events',
        'travel_edge':'find_travel_opts'
    }
)
# reached to find travel node
graph.add_edge('find_travel_opts','travel_tools')
graph.add_conditional_edges(
    'travel_tools',
    should_continue_travel,
    {
        'continue_travels':'find_travel_opts',
        'email_edge':'email_agent'
    }
)
graph.add_edge("email_agent", "email_tools")
graph.add_conditional_edges(
    'email_tools',
    should_continue_drafting_email,
    {
        'continue_email': 'email_agent',
        'end': 'send_email'
    }
)
graph.add_edge("send_email", END)

app = graph.compile()

def run_eventPlanner():
    print("\n------- Event Planner --------")
    state = {
        "messages": [],
        "tool_calls": [],
        "req_location": "",
        "final_venue": "",
        "events_planned": "",
        "travel_options": "",
        "recipient_emails": [],
        "event_date": "",
        "budget": "",
        "event": "",
        "email_messages":[]
    }

    for step in app.stream(state, stream_mode="values", config={"recursion_limit": 100}):
        if "messages" in step:
            print_messages(step["messages"])

    print("\nPlanning complete!")

if __name__ == "__main__":
    run_eventPlanner()
