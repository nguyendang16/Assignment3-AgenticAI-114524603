import os
import json
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from termcolor import colored
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_embeddings, get_llm, DATA_FOLDER, DB_FOLDER, FILES


# Generic Retry Logic (Provider agnostic)
retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)


def initialize_vector_dbs():
    embeddings = get_embeddings()
    retrievers = {}
    
    for key in FILES.keys():
        persist_dir = os.path.join(DB_FOLDER, key)

        if os.path.exists(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            retrievers[key] = vectorstore.as_retriever(search_kwargs={"k": 5})
        else:
            print(colored(f"❌ Error: Database for '{key}' not found!", "red"))
            print(colored(f"⚠️ Please run 'python build_rag.py' first.", "yellow"))
            continue
    
    return retrievers

RETRIEVERS = initialize_vector_dbs()


class AgentState(TypedDict):
    question: str
    documents: str
    generation: str
    search_count: int
    needs_rewrite: str


@retry_logic
def retrieve_node(state: AgentState):
    print(colored("--- 🔍 RETRIEVING ---", "blue"))
    question = state["question"]
    llm = get_llm()
    
    # --- [START] Improved Routing Logic ---
    options = list(FILES.keys()) + ["both", "none"]
    router_prompt = f"""You are a query router for a financial document retrieval system.
Your task is to classify the user's question and route it to the correct data source.

AVAILABLE DATA SOURCES:
- "apple": Apple Inc. financial reports (includes iPhone, iPad, Mac, Services, App Store, Apple Watch, etc.)
- "tesla": Tesla Inc. financial reports (includes Model S/3/X/Y, Cybertruck, energy storage, Supercharger, Autopilot, etc.)
- "both": Questions comparing Apple AND Tesla, or questions about both companies
- "none": Questions unrelated to Apple or Tesla financials

ROUTING RULES:
1. If the question mentions Apple, iPhone, iPad, Mac, App Store, or Apple-specific products → "apple"
2. If the question mentions Tesla, Model 3, Model Y, Cybertruck, Supercharger, or Tesla-specific products → "tesla"
3. If the question asks to COMPARE both companies or mentions BOTH Apple AND Tesla → "both"
4. If the question is unrelated to either company's financials → "none"

EXAMPLES:
- "What was Apple's revenue in 2024?" → {{"datasource": "apple"}}
- "Tesla 的研發費用是多少?" → {{"datasource": "tesla"}}
- "Compare R&D expenses of Apple and Tesla" → {{"datasource": "both"}}
- "What is the weather today?" → {{"datasource": "none"}}

Output ONLY valid JSON with no additional text: {{"datasource": "..."}}

User Question: {question}
"""
    
    try:
        response = llm.invoke(router_prompt)
        content = response.content.strip()
        # Handle cases where LLM might wrap JSON in backticks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        res_json = json.loads(content)
        target = res_json.get("datasource", "both")
    except Exception as e:
        print(colored(f"⚠️ Error parsing router output: {e}. Defaulting to 'both'.", "yellow"))
        target = "both"
    
    print(colored(f"🎯 Routing to: {target}", "cyan"))
    # --- [END] ---

    docs_content = ""
    targets_to_search = []
    if target == "both":
        targets_to_search = list(FILES.keys())
    elif target in FILES:
        targets_to_search = [target]
    
    for t in targets_to_search:
        if t in RETRIEVERS:
            docs = RETRIEVERS[t].invoke(question)
            source_name = t.capitalize()
            docs_content += f"\n\n[Source: {source_name}]\n" + "\n".join([d.page_content for d in docs])

    return {"documents": docs_content, "search_count": state["search_count"] + 1}

@retry_logic
def grade_documents_node(state: AgentState): 
    print(colored("--- ⚖️ GRADING ---", "yellow"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    system_prompt = """You are a relevance grader for a financial document retrieval system.
Your task is to assess whether the retrieved documents contain USEFUL information for answering the user's question.

GRADING CRITERIA - Be LENIENT:
1. If documents contain financial data from the company mentioned in the question → "yes"
2. If documents contain ANY relevant numbers, even if not the exact metric asked → "yes"
3. For comparison questions: if documents contain data for AT LEAST ONE company → "yes"
4. If the question asks about a specific year but documents have data for nearby years → "yes"

ONLY output "no" if:
- Documents are completely unrelated to the question topic
- Documents are about a different company entirely
- Documents contain no financial data at all

DEFAULT TO "yes" if uncertain - it's better to attempt an answer than to rewrite unnecessarily.

CRITICAL: Output ONLY one word: "yes" or "no"."""
    
    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Retrieved document context: \n\n {documents} \n\n User question: {question}")
    ]
    
    response = llm.invoke(msg)
    content = response.content.strip().lower()
    
    grade = "yes" if "yes" in content else "no"
    print(f"   Relevance Grade: {grade}")
    return {"needs_rewrite": grade}

@retry_logic
def generate_node(state: AgentState):
    print(colored("--- ✍️ GENERATING ---", "green"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm() 
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert financial analyst. Answer the user's question using the provided context.

RULES:
1. LANGUAGE: Always respond in English.

2. EXTRACT NUMBERS CAREFULLY:
   - Look for the EXACT metric mentioned in the question
   - Financial tables show: 2024, 2023, 2022 columns - use the CORRECT year
   - Common metrics: Total net sales, Revenue, R&D expenses, Cost of sales, Net income, Gross margin
   - Numbers are typically in MILLIONS unless stated otherwise

3. CITATIONS: Include source after each fact:
   - Apple data: [Source: Apple 10-K]
   - Tesla data: [Source: Tesla 10-K]

4. FOR COMPARISON QUESTIONS:
   - Extract data for EACH company from the context
   - If one company's data is missing, still provide what you found
   - State which company has higher/lower values

5. YEARLY DATA (IMPORTANT):
   - "Twelve Months Ended" = Full year data (use this for yearly questions)
   - "Three Months Ended" = Quarterly data (don't confuse with yearly)
   - Apple fiscal year ends in September, Tesla in December

6. IF DATA NOT FOUND: Say "The specific information requested is not available in the provided documents."

Context:
{context}"""),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": documents, "question": question})
    return {"generation": response.content}

@retry_logic
def rewrite_node(state: AgentState): 
    print(colored("--- 🔄 REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()
    
    rewrite_prompt = f"""Rewrite this financial question to be more searchable.

ORIGINAL: {question}

TRANSFORMATION RULES:
1. Use exact financial terms from 10-K reports:
   - Revenue/Sales → "Total net sales" or "Total revenues"
   - Profit → "Net income" or "Operating income"  
   - R&D → "Research and development"
   - Costs → "Cost of sales" or "Cost of revenues"
   - Assets → "Total assets"
   - CapEx → "Capital expenditures" or "Payments for property, plant and equipment"
   - Energy revenue → "Energy generation and storage"

2. Keep the company name (Apple/Tesla) and year (2024)

3. Make it a simple, direct question in English

Output ONLY the rewritten question, nothing else."""
    
    msg = [HumanMessage(content=rewrite_prompt)]
    response = llm.invoke(msg)
    new_query = response.content.strip()
    print(f"   New Question: {new_query}")
    return {"question": new_query}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    def decide_to_generate(state):
        if state["needs_rewrite"] == "yes":
            return "generate"
        else:
            if state["search_count"] > 2: 
                print("   (Max retries reached, generating anyway...)")
                return "generate"
            return "rewrite"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()

def run_graph_agent(question: str):
    app = build_graph()
    inputs = {"question": question, "search_count": 0, "needs_rewrite": "no", "documents": "", "generation": ""}
    # Using stream to see progress if needed, but invoke is fine for simple return
    result = app.invoke(inputs)
    return result["generation"]

# --- Legacy ReAct Agent ---
def run_legacy_agent(question: str):
    print(colored("--- 🤖 RUNNING LEGACY AGENT (ReAct) ---", "magenta"))
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools.retriever import create_retriever_tool
    from langchain.tools.render import render_text_description

    tools = []
    for key, retriever in RETRIEVERS.items():
        tools.append(create_retriever_tool(
            retriever, 
            f"search_{key}_financials", 
            f"Searches {key.capitalize()}'s financial data."
        ))

    if not tools:
        return "System Error: No tools available."

    llm = get_llm()

    template = """You are a financial analyst with access to company financial reports.

AVAILABLE TOOLS:
{tools}

CRITICAL RULES:
1. ENGLISH ONLY: Final Answer MUST be in English.
2. YEARLY vs QUARTERLY DATA:
   - "Twelve Months Ended" = FULL YEAR data (use this for yearly questions)
   - "Three Months Ended" = QUARTERLY data (smaller numbers, don't use for yearly)
   - Apple fiscal year ends September, Tesla ends December
3. YEAR COLUMNS: Tables show 2024, 2023, 2022 - read the CORRECT column
4. NUMBERS: Usually in MILLIONS (e.g., $391,035 million = $391 billion)
5. If data not found: Say "I don't know" - don't guess

FORMAT:
Question: the input question
Thought: what info do I need and which tool to use
Action: one of [{tool_names}]
Action Input: specific search query with financial terms
Observation: tool result (provided by system)
... (repeat if needed)
Thought: I now know the final answer
Final Answer: answer in English with specific numbers

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )

    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Legacy Agent Error: {e}"