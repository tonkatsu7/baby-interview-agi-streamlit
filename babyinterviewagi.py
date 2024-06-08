from typing import Dict, List
from collections import deque
from enum import Enum
import streamlit as st
import pandas as pd
import json
from langchain.chains.llm import LLMChain
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.chat_models.azure_openai import ChatOpenAI

LLM_VERBOSE = False
DEFAULT_FIRST_QUESTION = "Why are you looking for a new job?"

class InterviewState(Enum):
    NOT_STARTED = 0
    STARTED = 1
    COMPLETED = 2

class MessageParticipant(Enum):
    USER = 0
    ASSISTANT = 1

class MessageType(Enum):
    QUESTION = 0
    ANSWER = 1
    EVALUATION = 2
    OTHER = 3

class AgentType(Enum):
    EVALUATION = 1
    CREATION = 2
    PRIORITISATION = 3
    ASSESSMENT = 4

class AnswerEvaluationChain(LLMChain):
    """Chain to evaluate the answer to a particular question."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        evaluation_template = (
            # "You are an AI who evaluates the answer to a job interview question based on the following position description: {position_description}"
            "You are a hiring manager and your task is to evaluate the answer to a job interview question based on the following position description: {position_description}"
            " and applicant resume: {applicant_resume}."
            "Take into account these previous answers: {context}."
            "The latest question: {question}."
            "The latest answer: {answer}."
            "Based on the answer, provide a short evaluation of how well the applicant answered the question"
            " so that it can be used yowards the final assessment of the interview."
            "Evaluation:"
        )
        prompt = PromptTemplate(
            template=evaluation_template,
            input_variables=["position_description", "applicant_resume", "context", "question", "answer"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
    def evaluate_answer(self, position_description: str, applicant_resume: str, context: List[str], question: str, answer: str) -> str:
        """Evaluate the answer."""
        return self.run(position_description=position_description, applicant_resume=applicant_resume, context=context, question=question, answer=answer)
    
class QuestionCreationChain(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, position_description: str, applicant_resume: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        question_creation_template = (
            ### TODO - add the previously asked questions to the prompt and ask not to repeat them
            "You are a job interview question creation AI that uses the results of a job interview Q&A execution agent to create new questions for the following job description: {position_description}"
            " and job applicant resume: {resume}."
            " The last question asked of the applicant was: \"{question_description}\","
            " with applicant giving the answer: \"{answer}\"."
            " The evauation of the applicant's answer was: {evaluation}."
            " This is the list of unanswered questions: {incomplete_questions}."
            " This is the list of previously asked questions: {previous_questions}."
            " Based on the answer, either insert new questions into the unanswered questions list or modify existing questions in the unanswered questions list,"
            " for the execution agent to execute."
            " Don't ask questions that have already been asked."
            " If the answer has also completely answered an unanswered question, remove that question from the unanswered questions list."
            " If the answer has partially answered an unanswered question, modify that question in the unanswered questions list."
            " The new questions should not overlap with existing unanswered questions."
            " Return the consolidated single list of questions as an index numbered list."
            " Only output the list items, don't output a label."
            " Consolidated list of questions:"
        )
        prompt = PromptTemplate(
            template=question_creation_template,
            partial_variables={"position_description": position_description, "resume": applicant_resume},
            input_variables=["evaluation", "answer", "question_description", "incomplete_questions"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
    def get_next_question(self, evaluation: str, answer: str, question_description: str, question_list: List[str], previous_questions: List[str]) -> List[Dict]:
        """Get the next question."""
        incomplete_questions = question_list #", ".join(question_list)
        response = self.run(evaluation=evaluation, answer=answer, question_description=question_description, incomplete_questions=incomplete_questions, previous_questions=previous_questions)
        new_questions = response.split('\n')
        return [{"question_title": question_title} for question_title in new_questions if question_title.strip(" '\"")]
    
class QuestionPrioritizationChain(LLMChain):
    """Chain to prioritize questions."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, position_description: str, applicant_resume: str, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        question_prioritization_template = (
            "You are an job interview question prioritization AI tasked with cleaning the formatting of and reprioritizing the following job interview questions: {question_titles}."
            "Consider the job description: {position_description}"
            " and job applicant resume: {resume}."
            "Do not remove any existing questions. Do not modify any existing questions. Return the reprioritized questions as an index numbered list."
            "Only output the list items, don't output a label."
            "Reprioritised list of questions:"
        )
        prompt = PromptTemplate(
            template=question_prioritization_template,
            partial_variables={"position_description": position_description, "resume": applicant_resume},
            input_variables=["question_titles"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

    def prioritize_questions(self, this_question_id: int, question_list: List[Dict]) -> List[Dict]:
        """Prioritize questions."""
        question_titles = [t["question_title"] for t in question_list]
        response = self.run(question_titles=question_titles)
        new_questions = response.split('\n')
        prioritized_question_list = []
        for question_string in new_questions:
            if not question_string.strip():
                continue
            question_parts = question_string.strip().split(".", 1)
            if len(question_parts) == 2:
                question_id = int(question_parts[0].strip()) + this_question_id
                question_title = question_parts[1].strip(" '\"")
                prioritized_question_list.append({"question_id": question_id, "question_title": question_title})
        return prioritized_question_list
    
class InterviewAssessmentChain(LLMChain):
    """Chain to assess all the interview answers and decide whether to proceed with the applicant."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        evaluation_template = (
            # "You are an AI who assesses all the answers from the job interview questions based on the following position description: {position_description}"
            "You are a hiring manager and your task is to assesses all the answers from the job interview questions based on the following position description: {position_description}"
            " and applicant resume: {applicant_resume}."
            # " Your are to respond very obvioulsly in the persona of {persona}."
            " Take into account these previously answered questions: {context}."
            " The answers: {answers}."
            " Based on the answers, decide whether to proceed with the applicant and provide a justification."
            " Response:"
        )
        prompt = PromptTemplate(
            template=evaluation_template,
            input_variables=["position_description", "applicant_resume", "context", "answers"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    
    def assess_interview(self, position_description: str, applicant_resume: str, context: List[str], answers: List[str]) -> str:
        """Evaluate the interview."""
        return self.run(position_description=position_description, applicant_resume=applicant_resume, context=context, answers=answers)

def ready_for_interview():
    # Check if the user has entered the required fields
    # - OpenAI API key
    # - Position Description
    # - Applicant Resume
    # - First Question
    # Validate Max questions is a positive integer greater than 0
    if not st.session_state.openai_api_key:
        st.error("Please enter your OpenAI API key to start the interview.")
        return False
    if not st.session_state.position_description:
        st.error("Please enter the Position Description to start the interview.")
        return False
    if not st.session_state.applicant_resume:
        st.error("Please enter the Applicant Resume to start the interview.")
        return False
    if not st.session_state.max_questions or st.session_state.max_questions <= 0:
        st.error("Please enter a positive integer greater than 0 for the Max Questions to ask.")
        return False
    if not st.session_state.first_question:
        st.error("Please enter the First Question to ask the applicant.")
        return False
    return True

def start_interview():
    if not ready_for_interview():
        return

    st.session_state.interview_state["interview_state"] = InterviewState.STARTED
    st.session_state.interview_state["number_of_questions_evaluated"] = 0
    st.session_state.interview_state["chat_input_message"] = "Answer the questions to continue the interview"
    st.session_state.interview_state["current_question"] = {"question_id": 0, "question_title": st.session_state.first_question}
    st.session_state.interview_state["conversation"] = [{"role": MessageParticipant.ASSISTANT, 
                                                             "content": st.session_state.first_question,
                                                             "type": MessageType.QUESTION,
                                                             "question_id": 0,
                                                             }]
    st.session_state.interview_state["number_of_questions_asked"] += 1
    st.session_state.agents["question_id_counter"] = 0
    st.session_state.agents["question_list"] = deque()

    chat = ChatOpenAI(openai_api_key=st.session_state.openai_api_key)
    st.session_state.agents["evaluation"] = AnswerEvaluationChain.from_llm(llm=chat, 
                                                                          verbose=LLM_VERBOSE)
    st.session_state.agents["creation"] = QuestionCreationChain.from_llm(llm=chat, 
                                                                         position_description=st.session_state.position_description, 
                                                                         applicant_resume=st.session_state.applicant_resume, 
                                                                         verbose=LLM_VERBOSE)
    st.session_state.agents["prioritisation"] = QuestionPrioritizationChain.from_llm(llm=chat, 
                                                                                     position_description=st.session_state.position_description, 
                                                                                     applicant_resume=st.session_state.applicant_resume, 
                                                                                     verbose=LLM_VERBOSE)
    st.session_state.agents["assessment"] = InterviewAssessmentChain.from_llm(llm=chat, 
                                                                              verbose=LLM_VERBOSE)

def end_interview():
    assessment = st.session_state.agents["assessment"].assess_interview(st.session_state.position_description,
                                                                        st.session_state.applicant_resume, 
                                                                        get_all_questions(), 
                                                                        get_all_answers())
    st.session_state.interview_state["conversation"].append({"role": MessageParticipant.ASSISTANT, 
                                                             "content": assessment,
                                                             "type": MessageType.OTHER,
                                                             })

    st.session_state.interview_state["interview_state"] = InterviewState.COMPLETED
    st.session_state.interview_state["chat_input_message"] = "Press the button to reset the interview"
    st.session_state.interview_state["conversation"].append({"role": MessageParticipant.ASSISTANT, 
                                                             "content": "Thank you for your time. The interview is now complete.",
                                                             "type": MessageType.OTHER,
                                                             })

def get_all_questions():
    return [message["content"] for message in st.session_state.interview_state["conversation"] if message["type"] == MessageType.QUESTION]

def get_all_answers():
    return [message["content"] for message in st.session_state.interview_state["conversation"] if message["type"] == MessageType.ANSWER]

def init_state():
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api = None

    if 'position_description' not in st.session_state:
        st.session_state.position_description = None

    if 'applicant_resume' not in st.session_state:
        st.session_state.applicant_resume = None

    if 'max_questions' not in st.session_state:
        st.session_state.max_questions = 3

    if 'first_question' not in st.session_state:
        st.session_state.first_question = DEFAULT_FIRST_QUESTION

    if ('interview_state' not in st.session_state) or ('agents' not in st.session_state):
        reset_interview()

def reset_interview():
    st.session_state.interview_state = {
        "interview_state": InterviewState.NOT_STARTED, # InterviewState
        "current_question": { "question_id": 0, "question_title": st.session_state.first_question}, # {"question_id": int, "question_title": "content"}
        "number_of_questions_asked": 0, # int
        "number_of_questions_evaluated": 0, # int
        "chat_input_message": "Press the button to start a new interview", # "content"
        "conversation": [], # {"role": MessageParticipant, "content": "content", "type": MessageType}
        "evaluation_agent_log": [], # "question_id": int, "question": "content", "answer": "content", "evaluation": "content"
        "creation_agent_log": [], # "question_id": int, "previous_questions": "content", "previous_questions_count": int, "new_questions": "content", "new_questions_count": int
        "proiritisation_agent_log": [], # "question_id": int, "prioritised_questions": "content"
    } 
    st.session_state.agents = {
        "question_id_counter": 0,
        "question_list": [],
        "creation": None,
        "prioritisation": None,
        "evaluation": None,
        "assessment": None
    }

def interview_not_started():
    return st.session_state.interview_state["interview_state"].value == InterviewState.NOT_STARTED.value

def interview_started():
    return st.session_state.interview_state["interview_state"].value == InterviewState.STARTED.value

def interview_completed():
    return st.session_state.interview_state["interview_state"].value == InterviewState.COMPLETED.value

def submit_answer_handler():
    answer = st.session_state.chat_dialog;

    # Add user input to the conversation and process it
    st.session_state.interview_state["conversation"].append({"role": MessageParticipant.USER, 
                                                             "content": answer,
                                                             "type": MessageType.ANSWER,
                                                             "question_id": st.session_state.interview_state["current_question"]["question_id"],
                                                             })

    # Evaluate the answer
    all_answers = [message["content"] for message in st.session_state.interview_state["conversation"] if message["type"] == MessageType.ANSWER]
    all_questions = [message["content"] for message in st.session_state.interview_state["conversation"] if message["type"].value == MessageType.QUESTION.value]
    previous_questions = all_questions[:-1]
    question = st.session_state.interview_state["current_question"]
    answer_evaluation = st.session_state.agents["evaluation"].evaluate_answer(st.session_state.position_description, 
                                                                             st.session_state.applicant_resume, 
                                                                             all_answers[:-1], # all previous answers
                                                                             question["question_title"], # the current question
                                                                             answer)

    if answer_evaluation:
        # Add the response to the conversation
        st.session_state.interview_state["conversation"].append({"role": MessageParticipant.ASSISTANT, 
                                                                 "content": answer_evaluation,
                                                                 "type": MessageType.EVALUATION,
                                                                 "question_id": question["question_id"],
                                                                 })
        # Update the state for the next question/interaction
        st.session_state.interview_state["number_of_questions_evaluated"] += 1
        # Update agent log
        st.session_state.interview_state["evaluation_agent_log"].append({"question_id": question["question_id"],
                                                                         "question":   question["question_title"], 
                                                                         "answer":     answer, 
                                                                         "evaluation": answer_evaluation
                                                                         })

    if st.session_state.interview_state["number_of_questions_evaluated"] >= st.session_state.max_questions:
        end_interview()
    else:
        # Get the new questions based on last answer
        new_questions = st.session_state.agents["creation"].get_next_question(answer_evaluation, 
                                                                              answer, 
                                                                              question["question_title"], 
                                                                              list(st.session_state.agents["question_list"]),
                                                                              previous_questions)
        # Update agent log
        st.session_state.interview_state["creation_agent_log"].append({"question_id": question["question_id"],
                                                                       "previous_questions": list(st.session_state.agents["question_list"]),
                                                                       "previous_questions_count": len(list(st.session_state.agents["question_list"])), 
                                                                       "new_questions": new_questions,
                                                                       "new_questions_count": len(new_questions)
                                                                       })

        # Add new questions to the list
        for new_question in new_questions:
            # Update the question id index to continue from the last question id of the list
            st.session_state.agents["question_id_counter"] += 1
            new_question.update({"question_id": st.session_state.agents["question_id_counter"]})
            st.session_state.agents["question_list"].append(new_question)
        # Reprioritise the list
        st.session_state.agents["question_list"] = deque(st.session_state.agents["prioritisation"].prioritize_questions(int(question["question_id"]), 
                                                                                                                        list(st.session_state.agents["question_list"])))
        # Update agent log
        st.session_state.interview_state["proiritisation_agent_log"].append({"question_id": question["question_id"],
                                                                             "prioritised_questions": list(st.session_state.agents["question_list"])
                                                                             })
        # Ask the top question after reprioritisation
        next_question = st.session_state.agents["question_list"].popleft()
        st.session_state.interview_state["conversation"].append({"role": MessageParticipant.ASSISTANT,
                                                                 "content": next_question["question_title"],
                                                                 "type": MessageType.QUESTION,
                                                                 "question_id": next_question["question_id"],
                                                                 })
        st.session_state.interview_state["current_question"] = next_question
        st.session_state.interview_state["number_of_questions_asked"] += 1


def main():
    """Main function to set up and manage the Streamlit interface."""
    if 'interview_state' not in st.session_state:
        init_state()
    
    st.set_page_config(
        # Expand sidebar if not interview_started
        initial_sidebar_state="expanded" if not interview_started() else "collapsed",
        page_title="Virtual Job Interview",
        # Wide if interview_started
        layout="wide" if not interview_not_started() else "centered"
    )
    with st.sidebar:
        st.write(st.session_state)
    setup_ui()

def setup_ui():
    """Set up the user interface for the interview."""
    # Maximise col1 if not interview_started
    col1, col2 = st.columns([1, 1])
    with col1:
        render_header()
        render_conversation()
        render_chat_input()
    
    with col2:
        render_management_info()

    render_sidebar_controls()

def format_nested_list(data):
    return json.dumps(data, indent=2)

def render_header():
    """Render the appropriate header based on interview state."""
    if interview_completed():
        st.header("Virtual Job Interview Complete!")
    elif interview_started():
        st.header("Virtual Job Interview Underway...")
    else:
        st.header("Configure the Virtual Job Interview using the sidebar 🧑‍💼👩‍💼👨‍💼")

def render_conversation():
    """Display the conversation history."""
    for message in st.session_state.interview_state["conversation"]:
        with st.chat_message(message["role"].name):
            # Display evluation in caption
            if message["type"].value == MessageType.EVALUATION.value:
                st.markdown(f"EVALUATION {message["question_id"]+1}:")
                st.markdown(message['content'])
            elif message["type"].value == MessageType.QUESTION.value:
                st.markdown(f"QUESTION {message["question_id"]+1}:")
                st.markdown(message["content"])
            elif message["type"].value == MessageType.ANSWER.value:
                st.markdown(f"ANSWER {message["question_id"]+1}:")
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])

def render_chat_input():
    """Render the chat input box."""
    st.chat_input(st.session_state.interview_state["chat_input_message"], key="chat_dialog", on_submit=submit_answer_handler, disabled=not interview_started())

def render_management_info():
    if st.session_state.agents["question_list"]:
        tab1, tab2 = st.tabs(["Question List", "Agent Logs"])
        with tab1:
            render_question_list()

        with tab2:                
            render_agent_logs()

def render_question_list():
    # Tab 1 displays the question list and analytics about the question list as it changes progressively
    st.header("Question List:")
    st.subheader("Questions Asked:")
    for question in st.session_state.interview_state["conversation"][:-1]:
        if question["type"].value == MessageType.QUESTION.value:
            st.write(f"{question['content']}")
    st.subheader("Current Question:")
    st.write(f"{st.session_state.interview_state['current_question']['question_title']}")
    st.subheader("Questions Remaining:")
    next_question = True
    for question in st.session_state.agents["question_list"]:
        if next_question:
            next_question = False
            # Display bold for the next question
            st.write(f"**{question['question_id']+1}. {question['question_title']}**")
        else:
            st.write(f"{question['question_id']+1}. {question['question_title']}")
    # Display charts of question list analytics
    st.subheader("Question List Stats:")
    st.metric("Questions Asked", st.session_state.interview_state["number_of_questions_asked"])
    st.metric("Questions Evaluated", st.session_state.interview_state["number_of_questions_evaluated"])
    st.metric("Questions Remaining", len(st.session_state.agents["question_list"]))

def render_agent_logs():
    # Display the Evaluation agent log in a dataframe
    if st.session_state.interview_state["evaluation_agent_log"]:
        st.subheader("Evaluation Agent Log:")
        df = pd.DataFrame(st.session_state.interview_state["evaluation_agent_log"])
        df.set_index("question_id", inplace=True)
        st.write(df)

    # Display the Creation agent log in a dataframe
    if st.session_state.interview_state["creation_agent_log"]:
        st.subheader("Creation Agent Log:")
        
        # Process the data for display
        processed_data = []
        for entry in st.session_state.interview_state["creation_agent_log"]:
            processed_entry = {
                "question_id": entry["question_id"],
                "previous_questions": format_nested_list(entry.get("previous_questions", [])),
                "previous_questions_count": entry["previous_questions_count"],
                "new_questions": format_nested_list(entry.get("new_questions", [])),
                "new_questions_count": entry["new_questions_count"]
            }
            processed_data.append(processed_entry)
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        df.set_index("question_id", inplace=True)
        
        # Display DataFrame
        st.dataframe(df)  # Using st.dataframe to display the table

    # Display the Prioritisation agent log in a dataframe
    if st.session_state.interview_state["proiritisation_agent_log"]:
        st.subheader("Prioritisation Agent Log:")

        # Process the data for display
        processed_data = []
        for entry in st.session_state.interview_state["proiritisation_agent_log"]:
            processed_entry = {
                "question_id": entry["question_id"],
                "prioritised_questions": format_nested_list(entry.get("prioritised_questions", "[]"))
            }
            processed_data.append(processed_entry)
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        df.set_index("question_id", inplace=True)
        
        # Display DataFrame
        st.dataframe(df)  # Using st.dataframe to display the table

def render_sidebar_controls():
    """Render controls in the sidebar to configure the interview."""
    with st.sidebar:
        st.session_state.openai_api_key = st.text_input("Your OpenAI API Key", type="password", disabled=not interview_not_started())    
        configure_interview_details()
        manage_interview_controls()
        st.write(st.session_state)

def configure_interview_details():
    """Allow the user to set interview details in the sidebar."""
    st.session_state.position_description = st.text_area("Position Description", value=st.session_state.position_description, disabled=not interview_not_started())
    st.session_state.applicant_resume = st.text_area("Applicant Resume", value=st.session_state.applicant_resume, disabled=not interview_not_started())
    st.session_state.max_questions = st.number_input("Max Questions", value=st.session_state.max_questions, disabled=not interview_not_started())
    st.session_state.first_question = st.text_area("First Question", value=st.session_state.first_question, disabled=not interview_not_started())    

def manage_interview_controls():
    """Provide buttons to start or reset the interview."""
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Start Interview", on_click=start_interview, disabled=not interview_not_started())
    with col2:
        st.button("Reset Interview", on_click=reset_interview, disabled=interview_not_started())


if __name__ == "__main__":
    main()