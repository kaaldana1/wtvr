from collections import Counter
import os, json, textwrap, re, time
import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")  
MODEL    = os.getenv("MODEL_NAME", "bens_model")              

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int, 'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 128,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            # try best-effort to surface error text
            err_text = None
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}

ACTION_RE = re.compile(r"^\s*(CALCULATE|FINAL)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)

# --- PROVIDED: prompts ---
SYSTEM_AGENT = """
You are a reasoning agent that has to solve various different tasks.
Here are the list of tasks:
1) common sense question and answering
2) future prediction (even if it's impossible, just follow the question's instructions), 
3) coding tasks (output only valid code when asked)
4) math problems
5) planning tasks (output plans in the required format)

You have access to one tool:
1) CALCULATE: <arithmetic expression>
   - use only numbers, + - * / **, parentheses, and round(x, ndigits)
   - example: CALCULATE: round((3*2.49)*1.07, 2)

Or you can give the final answer in this format:
2) FINAL: <answer>

When solving a task, make sure to follow these guidelines:
1) Read the user question carefully and follow formatting instructions in it
2) When you choose FINAL, <answer> should exactly be the final expected output (for example: include \\boxed{...}, or only code, or only the plan lines) and nothing else.
3) Return ONE line only: either
  CALCULATE: <expression>
  or
  FINAL: <answer>
  Make sure not to include further explanations, reasoning steps, or extra text.
"""

#===========================================================================
# Techniques to implement:
# 1. tool reasoning with calculate (done)
# 2. classify domain type: 
#   math, common sense, coding, planning, future prediction
# 3. CoT for these domains: math and planning
# 4. self-verify
#===========================================================================

#===========================================================================
# Domain classifier: math, common sense, coding, planning or future preducition
#===========================================================================

def classify_domain(question:str) -> str:
    question = question.lower()

    coding_task_phrases = ['import', 'def', 'write a function', 'implement', 'class', 'script', 'program', 'python', 'write code']
    
    planning_task_phrases = ['[plan]', 'my plan is as follows']
    
    # dev prompt is in the style "you are an agenet that can predict future events"
    future_prediction_phrase = "you are an agent that can predict future events"

    math_task_phrases = ['24-game', '24 game', 'total', 'calculate', 'solve', 'compute', 'what is', 'determine', 'evaluate', 
                         'find the value', 'how much', 'how many', 'estimate', 
                         'total', 'sum', 'difference', 'product', 'quotient', 'percentage', 'increase', 'decrease',
                         'percentage', 'area', 'volume', 'length', 'distance', 'angle', 'equation', 'formula', 
                         'coordinates', 'expression', 'integral', 'derivative', 'function', 'graph', 'plot',
                         'log_', 'ln(', 'e^', 'pi', 'sqrt(']

    if any(phrase in question for phrase in coding_task_phrases):
        return "coding"
    elif any(phrase in question for phrase in planning_task_phrases):
        return "planning"
    elif future_prediction_phrase in question:
        return "future prediction"
    elif any(phrase in question for phrase in math_task_phrases):
        return "math"
    else:
        return "common sense"
    

def make_first_prompt(question: str, domain: str) -> str:
    header = (
        "Global Output Guidelines:\n"
        " 1) You must return exactly one line only \n"
        " 2) It must start with either 'CALCULATE: ' or 'FINAL: ' \n"
        " 3) Do NOT INCLUDE any extra text, reasoning steps, or explanations \n"
        "Priority of instructions:\n"
        " 1) ALWAYS follow the explicit output instructions in the task\n"
        " 2) If the task is silent on output format, follow the domain-specific guidelines below.\n"
    )
    if domain == "math":
        domain_guidelines = (
            "Math-specific Guidelines:\n"
            " - Ensure that the answer respects any required formatting instructions in the task\n"
            " - Include no explanations, reasoning steps, numbering or extra text in your final answer\n"
        )
    elif domain == "planning":
        domain_guidelines = (
            "Planning-specific Guidelines:\n"
            " - This is a STRIPS planning task\n" 
            " - The FINAL answer should only be a plan in the required format specified in the task\n"
            " - If the task requires specific syntax (for example, (action arg1 arg2 ...)), make sure to follow it exactly\n"
            " - Include no explanations, reasoning steps, numbering or extra text in your final answer\n"
        )
    elif domain == "coding":
        domain_guidelines = (
            "Coding-specifc Guidelines:\n"
            " - This is a coding task\n"
            " - The FINAL answer should ONLY be the code, unless the task explicitly allows otherwise\n"
            " - If the task provides a required signature or skeleton code, you have to match it verbatim\n"
            " - Include no comments, explanations, reasoning steps, numbering or extra text in your final answer\n"
        )
    elif domain == "future prediction":
        domain_guidelines = (
            "Future prediction-specific Guidelines:\n"
            " - You must make a prediction; even if it is not possible to make a prediction, do NOT refuse\n"
            " - If the task requires a particular format\n"
            "   your <answer> after \"FINAL:\" must follow that exactly.\n"
            " - Include no explanations, reasoning steps, numbering or extra text in your final answer\n"
        )
    elif domain == "common sense":
        domain_guidelines = (
            "Common sense-specific Guidelines:\n"
            " - Provide a concise answer. Be straight to the point.\n"
            " - If the task specifies an option answer format (for example: A/B/C), or a phrase like 'YES' or 'NO',\n"
            "  your <answer> needs to follow that exactly.\n"
            " - Include no explanations, reasoning steps, numbering or extra text in your final answer\n"
        )

    return f"""{header}{domain_guidelines}
Task:
{question}

If you need to do any arithmetic calculations to solve the task, use the CALCULATE tool and reply as:
CALCULATE: <expression>
Otherwise, provide the final answer as:
FINAL: <answer>"""

# Some none-math domains may still require calculation. The second prompt 
# should remind the model of the domain and the format expected with that domain
def make_second_prompt(question: str, result: str, domain: str) -> str:
    
    if domain == "coding":
        reminder = ( "REMEMBER that you are solving a coding task. ")
    elif domain == "planning":
        reminder = ( "REMEMBER that you are solving a STRIPS planning task. ")
    elif domain == "future prediction":
        reminder = ( "REMEMBER that you are solving a future prediction task. ")
    elif domain == "common sense":
        reminder = ( "REMEMBER that you are solving a common sense question and answering task. ")
    else: 
        reminder = ( "REMEMBER that you are solving a math problem. If the task doesn't explicitly mention any boxed notation (\\\\boxed), then <answer> should just be the plain number or alebraic expression WITHOUT \\\\boxed{...}. ")
    return f"""The global output guidelines and {domain}-specific guidelines provided earlier still apply.
For reference, here is the task:
{question}

CALCULATE tool results:
{result}

{reminder} 
If the task instructions contradict any of these guidelines, ALWAYS prioritize the task instructions.

Now provide the final answer. Reply exactly as: 
FINAL: <answer>"""



def parse_action(text: str):
    """
    Returns ("CALCULATE", expr) or ("FINAL", answer); raises ValueError on bad format.
    """
    m = ACTION_RE.match(text.strip())
    if not m:
        raise ValueError(f"Unrecognized action format: {text!r}")
    action = m.group(1).upper()
    payload = m.group(2).strip()
    return action, payload


def calculator_tool(expression: str):
    allowed_names = {"round": round}

    if re.search(r"[A-Za-z]", expression):
        raise ValueError(f"Inalid calculate")
    return eval(expression, {"__builtins__": {}}, allowed_names)


#===========================================================================
# Chain of thought: 
# Runs only for math and planning domains
# We make three separate passes
# each with prompts of temp 0.2 to produce more diverse reasoning
# Model needs to think silently about whether the previous answer is logically correct
#===========================================================================

def single_pass_cot(question: str, previous_answer: str, system: str, domain: str, temperature: float = 0.2, verbose: bool = True) -> str:
    cot_prompt = f"""
MANDATORY OUTPUT REQUIREMENTS:
    1) You must return only one line in this exact format:
    FINAL: <answer>
    2) <answer> HAS TO be the task's direct final answer (for example: if the answer to a task is 5, then <answer> must be 5 and NOT a meta-statement about whether or not the answer follows the format)
    3) Do not include reasoning steps, explanations, or extra text
    Carefully think silently about whether the proprosed answer is correct and follows ALL instructions. 
If the proposed answer is correct, keep the output as-is.
But if you find any mistakes, your next step is to re-solve and minimally change the answer. Ensure that the answer is in the required format that is specified in the task -- if the task is silent on format, follow the {domain}-specific guidelines provided earlier.


Question:
{question}
Proposed Final Answer:
{previous_answer}
"""

    
    cot = call_model_chat_completions(prompt=cot_prompt, system=system, temperature=temperature)
    if not cot["ok"]:
        raise RuntimeError(f"API error: {cot['error']}")

    if verbose: print("CoT →", cot["text"])
    action, payload = parse_action(cot["text"])
    if action != "FINAL":
        return previous_answer.strip()
    return payload.strip()
    
def chain_of_thought(question: str, previous_answer: str, domain: str) -> str:
    if domain == "math":
        system = "You are a correctness verifier for math problems."
    else: #planning
        system = "You are a correctness verifier for STRIPS planning tasks."
    
    cot_answers = []
    for i in range(3):
        cot_answer = single_pass_cot(question, previous_answer, system, domain, temperature=0.2, verbose=False)
        cot_answers.append(cot_answer)

    counts = Counter(cot_answers)
    top_answer, top_count = counts.most_common(1)[0]
    return top_answer.strip()


#===========================================================================
# Self-verification: Tasks for any domain should require the model to self-verify
# that its final answer meets the task requirements and formatting instructions
# It does so by reading the task and the model's previous final answer, and
# asking the model to verify that the answer meets all requirements
#===========================================================================

def self_verification(question: str, previous_answer: str, domain: str, verbose: bool = True) -> str:
    system = "You are a strict answer-format validator."
    prompt = f"""
MANDATORY OUTPUT REQUIREMENTS:
    1) You must return only one line in this exact format:
    FINAL: <answer>
    2) <answer> HAS TO be the task's direct final answer (for example, "YES", "67", "\\boxed{{5}}"), NOT a meta-statement about whether or not the answer follows the format
    3) Do not include reasoning steps, explanations, or extra text

You will receive a question and a proposed final answer. Your job is to:
1) Read the task and take note any explicit output formatting instructions
   (for example: output must be in \\boxed{{...}}, or YES/NO answers only,
   or the output fraction must be simplified, or only Python code that builds upon (not remove) the base code, etc...)
2) Use  the {domain}-specific guidelines if and ONLY IF the taks itself if silent about the formatting
3) If the proposed answer is already correctly-formatted, keep <answer> EXACTLY AS-IS


Task:
{question}
Previous Final Answer:
{previous_answer}

"""
    format_verification = call_model_chat_completions(prompt=prompt, system=system, temperature=0.0,)
    if not format_verification["ok"] or not format_verification["text"]:
        return previous_answer.strip()
        

    if verbose: print("Verifier →", format_verification["text"])
    try:
        action, payload = parse_action(format_verification["text"])
    except ValueError:
        return previous_answer.strip()
    
    if action != "FINAL":
        return previous_answer.strip()
    return payload.strip()


def answer_normalizer(question: str, answer: str, domain: str):

    # strip the final if there are any
    if domain in ["math"]:
        if answer.lower().startswith("final:"):
            answer = answer.split(":",1)[1].strip()

         # some outputs are in latex
        if answer.startswith("$") and answer.endswith("$"):
            answer = answer[1:-1].strip()

        if not "\\boxed" in question and "\\boxed" in answer:
            left_brak = answer.find("{")
            right_brak = answer.find("}")

            if right_brak > left_brak:
                return answer[left_brak+1:right_brak].strip()
    
    return  answer


# ============================ MAIN AGENT LOOP ============================

def run_agent(question: str, max_tool_uses: int = 3, verbose: bool = True):
    # classify the domain
    domain = classify_domain(question)

    r1 = call_model_chat_completions(prompt=make_first_prompt(question, domain), system=SYSTEM_AGENT, temperature=0.0,)
    if not r1["ok"]:
        raise RuntimeError(f"API error: {r1['error']}")

    if verbose: print("LLM →", r1["text"])

    # adding fallbacks when the LLM generates free form text
    try: 
        action, payload = parse_action(r1["text"])
    except ValueError:
        proposed_answer = (r1["text"] or "").strip()
        if domain in ["math", "planning"]:
            proposed_answer = chain_of_thought(question, proposed_answer, domain)
        
        final_answer = self_verification(question, proposed_answer, domain, verbose=verbose)
        return answer_normalizer(question, final_answer, domain)

    tool_uses = 0

    while action == "CALCULATE":
        if tool_uses >= max_tool_uses:
            raise RuntimeError("Exceeded tool-use limit.")
        tool_uses += 1

        # if LLM payload to calculator tool is wrong, ask the LLM again. This time, it cannot use calculator tool:
        try:
            calc_value = calculator_tool(payload)
        except Exception as e:
            error_handler_prompt = f""" Your previous CALCULATE expression ({payload}) was invalid because it was not a pure arithmetic expression.
            Moving forward DO NOT use CALCULATE atl all. Skip straight to giving your final answer. 
            MANDATORY OUTPUT REQUIREMENTS:
                1) You must return only one line in this exact format:
                    FINAL: <answer>
                2) <answer> HAS TO be the task's direct final answer (for example: if the answer to a task is 5, then <answer> must be 5 and NOT a meta-statement about whether or not the answer follows the format)
                3) Do not include reasoning steps, explanations, or extra text
            Task:
            {question}
            """

            error_response = call_model_chat_completions(prompt=error_handler_prompt, system=SYSTEM_AGENT, temperature=0.0,)
            if not error_response["text"] or not error_response["ok"]:
                return (payload or "").strip()

            if verbose: print("LLM →", error_response["text"])

            try: 
                action, payload = parse_action(error_response["text"])
            except ValueError:
                proposed_answer = (error_response["text"] or "").strip()
                if domain in ["math", "planning"]:
                    proposed_answer = chain_of_thought(question, proposed_answer, domain)
                final_answer = self_verification(question, proposed_answer, domain, verbose=verbose)
                
                return answer_normalizer(question, final_answer, domain)
            
            # the LLM is still trying to access the calculate tool, then just output the previous one
            if action != "FINAL":
                proposed_answer = payload.strip()
                final_answer = self_verification(question, proposed_answer, domain, verbose=verbose)
                return answer_normalizer(question, final_answer, domain)
            
            proposed_answer = payload.strip()
            if domain in ["math", "planning"]:
                proposed_answer = chain_of_thought(question, proposed_answer, domain)
            final_answer = self_verification(question, proposed_answer, domain, verbose=verbose)
            return answer_normalizer(question, final_answer, domain)

        if verbose: print("CALC =", calc_value)

        #ask model again with calculator result
        rN = call_model_chat_completions(prompt=make_second_prompt(question,str(calc_value), domain), system=SYSTEM_AGENT, temperature=0.0,)
        if not rN["ok"]:
            raise RuntimeError(f"API error: {rN['error']}")
        if verbose: print("LLM →", rN["text"])

        try: 
            action, payload = parse_action(rN["text"])
        except ValueError:
            proposed_answer = (rN["text"] or "").strip()
            if domain in ["math", "planning"]:
                proposed_answer = chain_of_thought(question, proposed_answer, domain)
        
            final_answer = self_verification(question, proposed_answer, domain, verbose=verbose)
            return answer_normalizer(question, final_answer, domain)
    
    proposed_answer = payload

    if domain in ["math", "planning"]:
        # we do chain of thought for these two domains
        proposed_answer = chain_of_thought(question, proposed_answer, domain)
    
    final_answer = self_verification(question, proposed_answer, domain, verbose=verbose)
    # action must be FINAL here


    final_answer = answer_normalizer(question, final_answer, domain)

    return final_answer

if __name__ == "__main__":
    # Example usage
    question = "The AIME Triathlon consists of a half-mile swim, a 30-mile bicycle ride, and an eight-mile run. Tom swims, bicycles, and runs at constant rates. He runs fives times as fast as he swims, and he bicycles twice as fast as he runs. Tom completes the AIME Triathlon in four and a quarter hours. How many minutes does he spend bicycling?"
    answer = run_agent(question, verbose=True)
    print("Final Answer:", answer)