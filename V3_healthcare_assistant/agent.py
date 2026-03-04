# =============================================================
# Anil Pathak's Agentic Healthcare Assistant
# Capstone Project — Agentic Healthcare Assistant for Medical
#                    Task Automation
# =============================================================
# Submitted by  : Anil Pathak
# Generated with: Claude (Anthropic) AI Coding Assistant
# File          : agent.py
# Purpose       : Core Healthcare Agent orchestrator.
#                 Implements a ReAct-style agentic loop using
#                 GPT-4o-mini function calling. Manages 13 tools
#                 across patient DB, appointments, RAG, and medical
#                 search. Integrates conversation memory, patient
#                 context tracking, and LLMOps interaction logging.
#                 Compatible with openai 2.x (uses client.chat
#                 .completions.create directly — no LangChain agent).
# =============================================================

import json
import os
import time
import sys

from openai import OpenAI
from dotenv import load_dotenv

# Ensure project root is on sys.path for config and tool imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tools.appointment_tool import (
    find_doctors_by_specialty,
    get_available_slots_for_doctor,
    book_appointment,
    cancel_appointment,
    get_patient_appointments,
    get_all_specialties
)
from tools.patient_db_tool import (
    get_patient_by_name,
    get_patient_by_id,
    update_patient_record,
    list_all_patients,
    add_patient_record
)
from tools.rag_tool import retrieve_patient_history, search_across_all_patients
from tools.medical_search_tool import (
    search_medical_information,
    get_disease_overview,
    get_drug_information
)
from memory.memory_module import HealthcareMemory
from prompts.system_prompts import MAIN_SYSTEM_PROMPT

load_dotenv()


# ── Tool schema definitions ───────────────────────────────────
# OpenAI function-calling schemas for all 13 agent tools.
# The agent selects tools automatically based on the user query.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_doctors_by_specialty",
            "description": "Find available doctors by medical specialty (e.g. Cardiologist, Nephrologist, General Physician, Diabetologist)",
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty": {
                        "type": "string",
                        "description": "The medical specialty to search for"
                    }
                },
                "required": ["specialty"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_slots_for_doctor",
            "description": "Get available appointment slots for a specific doctor by their doctor_id",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "string",
                        "description": "The doctor's ID (e.g. D001)"
                    },
                    "from_date": {
                        "type": "string",
                        "description": "Start date YYYY-MM-DD. Leave empty for today."
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "Number of days ahead to check. Default 7."
                    }
                },
                "required": ["doctor_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book a specific appointment slot for a patient",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id":      {"type": "string", "description": "The slot ID to book"},
                    "patient_id":   {"type": "string", "description": "The patient's ID"},
                    "patient_name": {"type": "string", "description": "The patient's full name"}
                },
                "required": ["slot_id", "patient_id", "patient_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an existing appointment using slot ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "string", "description": "The slot ID to cancel"}
                },
                "required": ["slot_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_appointments",
            "description": "Get all booked appointments for a patient",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "The patient's ID"}
                },
                "required": ["patient_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_by_name",
            "description": "Search for a patient record by full or partial name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Patient name or partial name"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_by_id",
            "description": "Retrieve a patient record by their unique Patient_ID (e.g. P001)",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "The patient's unique ID"}
                },
                "required": ["patient_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_patient_record",
            "description": "Update a specific field in a patient's record",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "The patient's ID"},
                    "field":      {"type": "string", "description": "The field/column name to update"},
                    "value":      {"type": "string", "description": "The new value to set"}
                },
                "required": ["patient_id", "field", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_patients",
            "description": "List all patients registered in the healthcare system",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_patient_history",
            "description": "Retrieve and summarise a patient's medical history from PDF reports using RAG (FAISS vector store + GPT-4o-mini)",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "The patient's full name"
                    },
                    "query": {
                        "type": "string",
                        "description": "Specific question about the patient. Leave empty for full summary."
                    }
                },
                "required": ["patient_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_medical_information",
            "description": "Search MedlinePlus (NLM/NIH) and WHO for trusted, up-to-date medical information on any health topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Medical topic or question to search for"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_disease_overview",
            "description": "Get a comprehensive medical overview of a specific disease or condition",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_name": {"type": "string", "description": "The disease or condition name"}
                },
                "required": ["disease_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_drug_information",
            "description": "Get detailed information about a specific drug or medication including uses, dosage, and side effects",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "Name of the drug or medication"}
                },
                "required": ["drug_name"]
            }
        }
    }
]


# ── Tool dispatcher map ───────────────────────────────────────
# Maps tool name strings (from GPT function calls) to actual
# Python functions. Add new tools here and in TOOLS above.

TOOL_MAP = {
    "find_doctors_by_specialty":      find_doctors_by_specialty,
    "get_available_slots_for_doctor": get_available_slots_for_doctor,
    "book_appointment":               book_appointment,
    "cancel_appointment":             cancel_appointment,
    "get_patient_appointments":       get_patient_appointments,
    "get_patient_by_name":            get_patient_by_name,
    "get_patient_by_id":              get_patient_by_id,
    "update_patient_record":          update_patient_record,
    "list_all_patients":              list_all_patients,
    "retrieve_patient_history":       retrieve_patient_history,
    "search_medical_information":     search_medical_information,
    "get_disease_overview":           get_disease_overview,
    "get_drug_information":           get_drug_information,
}


def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """
    Dispatch a tool call by name, passing the provided arguments.

    Args:
        tool_name (str):  Name of the tool to invoke.
        tool_args (dict): Keyword arguments to pass to the tool.

    Returns:
        str: JSON string result from the tool, or error JSON if
             the tool is unknown or raises an exception.
    """
    fn = TOOL_MAP.get(tool_name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return fn(**tool_args)
    except Exception as e:
        return json.dumps({"error": f"Tool '{tool_name}' failed: {str(e)}"})


# ── Healthcare Agent ──────────────────────────────────────────

class HealthcareAgent:
    """
    Agentic Healthcare Assistant orchestrator.

    Implements a ReAct-style agentic loop:
    1. Receives user message
    2. Calls GPT-4o-mini with full tool schema
    3. If GPT returns tool_calls, executes each tool and feeds
       results back to GPT as tool messages
    4. Repeats until GPT returns finish_reason='stop'
    5. Returns final text response

    Maintains conversation memory (sliding window) and active
    patient context across multi-turn sessions. Logs all
    interactions and tool calls for LLMOps analytics.

    Attributes:
        client          (OpenAI):          OpenAI API client.
        model           (str):             Model name to use.
        memory          (HealthcareMemory):Conversation + patient context.
        messages        (list):            Full message history for API.
        session_id      (str):             Unique session identifier.
        enable_logging  (bool):            Whether to log to interaction_logs.json.
        last_tools_used (list):            Tools used in last chat() call.
    """

    def __init__(self, session_id: str = "default", enable_logging: bool = True):
        """
        Initialise the Healthcare Agent.

        Args:
            session_id     (str):  Identifier for this session.
                                   Used to group logs in analytics.
            enable_logging (bool): Enable LLMOps interaction logging.
                                   Set False during evaluation runs to
                                   avoid polluting production logs.
        """
        self.client         = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model          = "gpt-4o-mini"
        self.memory         = HealthcareMemory(window_size=10)
        self.messages       = []
        self.session_id     = session_id
        self.enable_logging = enable_logging
        self.last_tools_used = []

        # Bind logger functions at init time to avoid repeated imports
        if self.enable_logging:
            try:
                from evaluation.logger import log_tool_call, log_interaction
                self._log_tool_call  = log_tool_call
                self._log_interaction = log_interaction
            except ImportError:
                self.enable_logging = False

    def _build_system_message(self) -> str:
        """
        Build the system prompt, injecting the current patient context
        from memory into the {patient_context} placeholder.

        Returns:
            str: Fully populated system prompt string.
        """
        patient_ctx = self.memory.get_patient_context_string()
        return MAIN_SYSTEM_PROMPT.format(patient_context=patient_ctx)

    def _log_tool(self, tool_name: str, tool_args: dict,
                  tool_result: str, elapsed_ms: float):
        """
        Log a tool call to the LLMOps store.
        Skips book_appointment to avoid double-logging
        (appointment_tool.py already logs bookings directly).

        Args:
            tool_name  (str):   Tool that was called.
            tool_args  (dict):  Arguments passed.
            tool_result(str):   JSON result string.
            elapsed_ms (float): Execution time in milliseconds.
        """
        if not self.enable_logging:
            return
        # book_appointment is logged in appointment_tool.py — skip here
        if tool_name == "book_appointment":
            return
        try:
            self._log_tool_call(
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                success=True,
                execution_time_ms=elapsed_ms,
                session_id=self.session_id
            )
        except Exception:
            pass

    def chat(self, user_input: str, verbose: bool = True) -> str:
        """
        Process a user message through the agentic loop and return
        the final text response.

        The loop:
        1. Prepends/updates the system message with current patient context
        2. Appends user message to history
        3. Calls GPT-4o-mini with function-calling enabled
        4. If tool_calls returned: executes tools, appends results, loops
        5. If finish_reason='stop': returns final assistant response

        Args:
            user_input (str):  The user's message.
            verbose    (bool): Print tool call traces to stdout.
                               Set False in UI/evaluation contexts.

        Returns:
            str: The agent's final natural language response.
        """
        interaction_start  = time.time()
        self.last_tools_used = []

        # Update system message with latest patient context on each turn
        if not self.messages:
            self.messages.append({
                "role":    "system",
                "content": self._build_system_message()
            })
        else:
            self.messages[0]["content"] = self._build_system_message()

        self.messages.append({"role": "user", "content": user_input})

        if verbose:
            print(f"\n{'─'*60}\n🤔 Agent Processing...")

        # Agentic loop — max 10 iterations guards against infinite loops
        max_iterations = 10
        for iteration in range(max_iterations):

            response      = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=2000
            )
            message       = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Serialise assistant message for history
            msg_dict = {
                "role":       "assistant",
                "content":    message.content,
                "tool_calls": [
                    {
                        "id":   tc.id,
                        "type": "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in (message.tool_calls or [])
                ] if message.tool_calls else None
            }
            self.messages.append(msg_dict)

            # ── Final response (no tool calls) ────────────────
            if finish_reason == "stop" or not message.tool_calls:
                final_response = message.content or "I couldn't generate a response."
                self.memory.add_interaction(user_input, final_response)

                if self.enable_logging:
                    elapsed_ms = (time.time() - interaction_start) * 1000
                    try:
                        self._log_interaction(
                            user_input=user_input,
                            agent_response=final_response,
                            tools_used=self.last_tools_used,
                            response_time_ms=elapsed_ms,
                            session_id=self.session_id
                        )
                    except Exception:
                        pass

                return final_response

            # ── Tool execution ────────────────────────────────
            for tool_call in message.tool_calls:
                tool_name  = tool_call.function.name
                tool_start = time.time()

                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tool_result   = dispatch_tool(tool_name, tool_args)
                tool_elapsed  = (time.time() - tool_start) * 1000

                # Track unique tools used in this turn
                if tool_name not in self.last_tools_used:
                    self.last_tools_used.append(tool_name)

                if verbose:
                    preview = tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                    print(f"\n  🔧 Tool: {tool_name}")
                    print(f"     Args: {json.dumps(tool_args)}")
                    print(f"     Result: {preview}")

                # Log tool call (skips book_appointment — logged elsewhere)
                self._log_tool(tool_name, tool_args, tool_result, tool_elapsed)

                # Feed tool result back into the conversation
                self.messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      tool_result
                })

                # Update patient context when a patient is successfully looked up
                if tool_name in ["get_patient_by_name", "get_patient_by_id"]:
                    try:
                        result_data = json.loads(tool_result)
                        if result_data.get("status") == "success":
                            patient = result_data.get("patient") or (
                                result_data.get("patients", [{}])[0]
                            )
                            if patient:
                                self.memory.set_patient_context(patient)
                    except Exception:
                        pass

        return "I've completed processing your request. Let me know if you need anything else."

    def reset_session(self):
        """
        Reset the agent to a clean state.
        Clears message history, tool traces, and memory.
        Called between evaluation test cases and on UI session reset.
        """
        self.messages        = []
        self.last_tools_used = []
        self.memory.clear_conversation()
        self.memory.clear_patient_context()
        if hasattr(self, '_verbose') and self._verbose:
            print("🔄 Session reset.")
