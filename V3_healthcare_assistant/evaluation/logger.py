# =============================================================
# Anil Pathak's Agentic Healthcare Assistant
# Capstone Project — Agentic Healthcare Assistant for Medical
#                    Task Automation
# =============================================================
# Submitted by  : Anil Pathak
# Generated with: Claude (Anthropic) AI Coding Assistant
# File          : evaluation/logger.py
# Purpose       : LLMOps interaction and tool usage logger.
#                 Persists all agent interactions, tool calls,
#                 and evaluation results to a JSON log file.
#                 Provides analytics summary for the dashboard.
# =============================================================

import json
import os
from datetime import datetime

# Import absolute log path from central config
# This ensures the same file is read/written regardless of
# which module calls the logger or what the CWD is
try:
    from config import LOG_FILE
except ImportError:
    # Fallback: resolve relative to this file's location
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_FILE = os.path.join(BASE_DIR, "data", "interaction_logs.json")


# ── Internal helpers ─────────────────────────────────────────

def _load_logs() -> list:
    """
    Load all log entries from the JSON log file.

    Returns:
        list: All log entries, or empty list if file doesn't exist
              or cannot be parsed.
    """
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_logs(logs: list):
    """
    Persist log entries to the JSON log file.

    Args:
        logs (list): Full list of log entries to write.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2, default=str)


# ── Public logging functions ─────────────────────────────────

def log_interaction(user_input, agent_response, tools_used,
                    response_time_ms, session_id="default"):
    """
    Log a complete agent interaction (user query + final response).

    Args:
        user_input       (str):   The user's original message.
        agent_response   (str):   The agent's final response text.
        tools_used       (list):  Names of tools invoked during this turn.
        response_time_ms (float): Total wall-clock time in milliseconds.
        session_id       (str):   Identifier for the current session.

    Returns:
        dict: The log entry that was saved.
    """
    logs = _load_logs()
    entry = {
        "id": len(logs) + 1,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "type": "interaction",
        "user_input": user_input,
        "agent_response": agent_response[:500],   # truncate for storage
        "tools_used": tools_used,
        "tool_count": len(tools_used),
        "response_time_ms": round(float(response_time_ms), 2)
    }
    logs.append(entry)
    _save_logs(logs)
    return entry


def log_tool_call(tool_name, tool_args, tool_result, success,
                  execution_time_ms, session_id="default"):
    """
    Log an individual tool call made by the agent or UI.

    Args:
        tool_name        (str):   Name of the tool that was called.
        tool_args        (dict):  Arguments passed to the tool.
        tool_result      (str):   JSON string result returned by tool.
        success          (bool):  Whether the call succeeded.
        execution_time_ms(float): Time taken by the tool in milliseconds.
        session_id       (str):   Identifier for the current session.

    Returns:
        dict: The log entry that was saved.
    """
    logs = _load_logs()

    # Parse result JSON to determine actual success from status field
    try:
        rd = json.loads(tool_result)
        actual_success = rd.get("status") in ["success", "ok"] or success
    except Exception:
        actual_success = success

    entry = {
        "id": len(logs) + 1,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "type": "tool_call",
        "tool_name": tool_name,
        "tool_args": tool_args,
        "result_preview": str(tool_result)[:300],
        "success": bool(actual_success),
        "execution_time_ms": round(float(execution_time_ms), 2)
    }
    logs.append(entry)
    _save_logs(logs)
    return entry


def log_evaluation(query, response, scores, session_id="default"):
    """
    Log a model evaluation result (LLM-as-judge scoring).

    Args:
        query      (str):  The query that was evaluated.
        response   (str):  The agent response that was scored.
        scores     (dict): Dimension scores (relevance, accuracy, etc.).
        session_id (str):  Identifier for the current session.

    Returns:
        dict: The log entry that was saved.
    """
    logs = _load_logs()

    # Ensure all score values are stored as float for correct averaging
    clean_scores = {}
    for k, v in scores.items():
        try:
            clean_scores[k] = float(v)
        except (ValueError, TypeError):
            clean_scores[k] = v

    entry = {
        "id": len(logs) + 1,
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "type": "evaluation",
        "query": query,
        "response_preview": response[:300],
        "scores": clean_scores,
        "overall_score": float(clean_scores.get("overall", 0))
    }
    logs.append(entry)
    _save_logs(logs)
    return entry


# ── Analytics ────────────────────────────────────────────────

def get_all_logs() -> list:
    """Return all log entries across all types."""
    return _load_logs()


def get_logs_by_type(log_type: str) -> list:
    """
    Filter logs by entry type.

    Args:
        log_type (str): One of 'interaction', 'tool_call', 'evaluation'.

    Returns:
        list: Filtered log entries.
    """
    return [l for l in _load_logs() if l.get("type") == log_type]


def get_analytics_summary() -> dict:
    """
    Compute aggregated analytics from all log entries.
    Used by the Logs & Analytics dashboard page.

    Returns:
        dict: Summary metrics including tool frequency, success rates,
              average response time, evaluation scores, and booking KPIs.
    """
    logs = _load_logs()
    interactions = [l for l in logs if l.get("type") == "interaction"]
    tool_calls   = [l for l in logs if l.get("type") == "tool_call"]
    evaluations  = [l for l in logs if l.get("type") == "evaluation"]

    # Build per-tool frequency and success counts
    tool_freq    = {}
    tool_success = {}
    tool_total   = {}
    for t in tool_calls:
        name = t.get("tool_name", "unknown")
        tool_freq[name]  = tool_freq.get(name, 0) + 1
        tool_total[name] = tool_total.get(name, 0) + 1
        if t.get("success"):
            tool_success[name] = tool_success.get(name, 0) + 1

    tool_success_rate = {
        name: round((tool_success.get(name, 0) / tool_total[name]) * 100, 1)
        for name in tool_total
    }

    avg_rt = round(
        sum(float(i.get("response_time_ms", 0)) for i in interactions) / len(interactions), 2
    ) if interactions else 0

    avg_eval = round(
        sum(float(e.get("overall_score", 0)) for e in evaluations) / len(evaluations), 2
    ) if evaluations else 0

    # Booking-specific KPIs — critical for assignment requirement:
    # "Log and analyse agent performance per module (success rate of bookings)"
    booking_calls   = [t for t in tool_calls if t.get("tool_name") == "book_appointment"]
    booking_success = [t for t in booking_calls if t.get("success")]
    booking_rate    = round(
        len(booking_success) / len(booking_calls) * 100, 1
    ) if booking_calls else 0

    return {
        "total_interactions":       len(interactions),
        "total_tool_calls":         len(tool_calls),
        "total_evaluations":        len(evaluations),
        "tool_usage_frequency":     tool_freq,
        "tool_success_rates":       tool_success_rate,
        "avg_response_time_ms":     avg_rt,
        "avg_evaluation_score":     avg_eval,
        "most_used_tool":           max(tool_freq, key=tool_freq.get) if tool_freq else "N/A",
        "booking_total":            len(booking_calls),
        "booking_success":          len(booking_success),
        "booking_success_rate_pct": booking_rate
    }


def clear_logs():
    """Clear all log entries. Use with caution."""
    _save_logs([])
