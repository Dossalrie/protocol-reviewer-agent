from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

DATA_PATH = Path(__file__).with_name("opportunities.json")
OUTPUT_PATH = Path(__file__).with_name("opportunities.csv")

FOUNDER_PROFILE = {
    "founders": ["Claudio", "Markus"],
    "claudio": [
        "physics",
        "ai leadership",
        "enterprise software",
        "fundraising",
        "strategy",
        "management",
    ],
    "markus": [
        "physics",
        "spectroscopy",
        "hardware",
        "successful exit",
        "deep-tech commercialization",
    ],
}

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)


def call_llm(prompt: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "Fallback venture memo: start with one orchestrator agent with a small set of specialized tools. "
            "The best first opportunities are the ones with strong founder fit, clear European market pull, and realistic capital needs."
        )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as exc:
        return f"Fallback venture memo due to LLM error: {exc}"


def load_opportunities(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["opportunities"]


def score_opportunity(opportunity: dict[str, Any]) -> dict[str, Any]:
    description = f"{opportunity['description']} {opportunity['market']}".lower()
    founder_keywords = " ".join(FOUNDER_PROFILE["claudio"] + FOUNDER_PROFILE["markus"]).lower()

    claudio_score = 0
    markus_score = 0

    for keyword in FOUNDER_PROFILE["claudio"]:
        if keyword.lower() in description:
            claudio_score += 2

    for keyword in FOUNDER_PROFILE["markus"]:
        if keyword.lower() in description:
            markus_score += 2

    if any(term in description for term in ["sensor", "spectroscopy", "optical", "imaging", "diagnostic"]):
        markus_score += 2

    if any(term in description for term in ["ai", "optimization", "software", "digital", "control"]):
        claudio_score += 2

    if any(term in description for term in ["electro", "heat", "chem", "catal", "materials", "reactor"]):
        claudio_score += 1
        markus_score += 1

    founder_fit = round(min(10, 4 + claudio_score + markus_score / 2), 1)
    impact_score = round(min(10, opportunity.get("impact_score", 6)), 1)
    market_score = round(min(10, opportunity.get("market_score", 6)), 1)
    europe_score = round(min(10, opportunity.get("europe_score", 6)), 1)
    capital_score = round(min(10, opportunity.get("capital_score", 6)), 1)

    overall_score = round(
        (founder_fit * 0.3) + (market_score * 0.25) + (impact_score * 0.2) + (europe_score * 0.15) + (capital_score * 0.1),
        1,
    )

    return {
        "technology": opportunity["name"],
        "institution": opportunity.get("institution", "Unknown"),
        "trl": opportunity.get("trl", "Unknown"),
        "licensing": opportunity.get("licensing_status", "Unknown"),
        "market": opportunity.get("market", "Unknown"),
        "founder_fit": founder_fit,
        "climate_impact": opportunity.get("climate_impact", "Medium"),
        "impact_score": impact_score,
        "market_score": market_score,
        "europe_score": europe_score,
        "capital_score": capital_score,
        "overall_score": overall_score,
        "notes": opportunity.get("notes", "Needs further validation"),
        "founder_keywords": founder_keywords,
    }


def discover_opportunities(state: MessagesState) -> dict[str, list[SystemMessage]]:
    user_prompt = state["messages"][-1].content
    opportunities = load_opportunities()
    payload = {
        "user_prompt": user_prompt,
        "opportunities": opportunities,
    }
    return {
        "messages": state["messages"] + [SystemMessage(content=json.dumps(payload, indent=2))],
    }


def score_opportunities(state: MessagesState) -> dict[str, list[SystemMessage]]:
    payload = json.loads(state["messages"][-1].content)
    opportunities = payload.get("opportunities", [])
    scored = [score_opportunity(opportunity) for opportunity in opportunities]
    scored.sort(key=lambda item: item["overall_score"], reverse=True)
    return {
        "messages": state["messages"] + [SystemMessage(content=json.dumps(scored, indent=2))],
    }


def draft_report(state: MessagesState) -> dict[str, list[HumanMessage]]:
    scored = json.loads(state["messages"][-1].content)
    prompt = f"""
You are a deep-tech venture scout. The founders are Claudio and Markus.
Review the scored opportunities below and produce a concise investment memo.

Requirements:
- Recommend whether to build one orchestrator agent or a multi-agent system first.
- Start with one orchestrator agent with a few specialized tools rather than a fully autonomous multi-agent system.
- For each opportunity, include a one-sentence why-it-matters and a one-sentence risk.
- Highlight the top 3 opportunities for Europe.

Scored opportunities:
{json.dumps(scored, indent=2)}
"""
    report_text = call_llm(prompt)
    return {"messages": state["messages"] + [HumanMessage(content=report_text)]}


def export_csv(state: MessagesState) -> dict[str, list[SystemMessage]]:
    scored = json.loads(state["messages"][-2].content)
    output_rows = []
    for item in scored:
        output_rows.append(
            {
                "Technology": item["technology"],
                "Institution": item["institution"],
                "TRL": item["trl"],
                "Licensing": item["licensing"],
                "Market": item["market"],
                "Founder Fit": item["founder_fit"],
                "Climate Impact": item["climate_impact"],
                "Impact Score": item["impact_score"],
                "Market Score": item["market_score"],
                "Europe Score": item["europe_score"],
                "Capital Score": item["capital_score"],
                "Overall Score": item["overall_score"],
                "Notes": item["notes"],
            }
        )

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Technology",
                "Institution",
                "TRL",
                "Licensing",
                "Market",
                "Founder Fit",
                "Climate Impact",
                "Impact Score",
                "Market Score",
                "Europe Score",
                "Capital Score",
                "Overall Score",
                "Notes",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "messages": state["messages"] + [SystemMessage(content=f"Saved opportunities to {OUTPUT_PATH}")],
    }


def build_graph() -> Any:
    graph = StateGraph(MessagesState)
    graph.add_node("discover", discover_opportunities)
    graph.add_node("score", score_opportunities)
    graph.add_node("report", draft_report)
    graph.add_node("export", export_csv)
    graph.add_edge(START, "discover")
    graph.add_edge("discover", "score")
    graph.add_edge("score", "report")
    graph.add_edge("report", "export")
    graph.add_edge("export", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    prompt = """
    Scout sustainability and decarbonization technologies in Europe that could become a startup opportunity for Claudio and Markus.
    Prioritize opportunities that fit their physics and deep-tech backgrounds and could be commercialized in Europe.
    """

    result = app.invoke({"messages": [HumanMessage(content=prompt)]})
    print("\n=== Venture scout report ===")
    print(result["messages"][-1].content)
    print(f"\nSaved ranked opportunities to {OUTPUT_PATH}")
