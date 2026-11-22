#!/usr/bin/env python3
"""End-to-End TMDB 5000 Knowledge Graph Question Answering Demo (Phases 1-4B).

Executes the complete pipeline:
User Question -> NaturalLanguageQueryParser -> GraphQueryEngine -> NaturalLanguageAnswerGenerator -> Conversational Answer
"""

from __future__ import annotations
import argparse
import sys
import time

from src.gemini_client import GeminiClient
from src.query_engine import load_query_engine, GraphQueryEngine
from src.query_parser import NaturalLanguageQueryParser
from src.answer_generator import NaturalLanguageAnswerGenerator


def run_pipeline(
    question: str,
    parser: NaturalLanguageQueryParser,
    engine: GraphQueryEngine,
    generator: NaturalLanguageAnswerGenerator,
):
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("-" * 80)

    t0 = time.perf_counter()

    # Step 1: Parse Natural Language into QuerySpecification
    parsed_resp = parser.parse_query(question)
    parse_time = (time.perf_counter() - t0) * 1000

    print(f"[PARSER] Status: {parsed_resp.status} | Provider: {parsed_resp.provider} | Latency: {parse_time:.1f}ms")

    # Handle Clarification or Unsupported
    if parsed_resp.clarification_required or parsed_resp.is_unsupported or not parsed_resp.query_specification:
        ans_resp = generator.generate_answer(
            question=question,
            parser_response=parsed_resp,
        )
        print(f"\n[RESPONSE - {ans_resp.status}]:")
        print(ans_resp.answer)
        print("=" * 80 + "\n")
        return

    spec = parsed_resp.query_specification
    print(f"[SPECIFICATION] Operation: '{spec.operation}' | Parameters: {spec.parameters}")

    # Step 2: Execute in Knowledge Graph Query Engine
    t1 = time.perf_counter()
    retrieval_result = engine.execute_query(spec)
    exec_time = (time.perf_counter() - t1) * 1000

    count = retrieval_result.get("result_count", 0)
    print(f"[GRAPH ENGINE] Success: {retrieval_result.get('success')} | Records: {count} | Latency: {exec_time:.1f}ms")

    # Step 3: Generate Natural Language Answer Grounded on Graph Data
    t2 = time.perf_counter()
    ans_resp = generator.generate_answer(
        question=question,
        query_specification=spec,
        retrieval_result=retrieval_result,
        parser_response=parsed_resp,
    )
    gen_time = (time.perf_counter() - t2) * 1000

    total_time = (time.perf_counter() - t0) * 1000
    print(f"[ANSWER GEN] Status: {ans_resp.status} | Provider: {ans_resp.provider} | Grounding Passed: {ans_resp.grounding_passed} | Latency: {gen_time:.1f}ms")
    if ans_resp.grounding_violations:
        print(f"[GROUNDING ALERT] Violations: {ans_resp.grounding_violations}")
        print(f"[GROUNDING REPAIR] Deterministic fallback used: {ans_resp.grounding_repaired}")

    print(f"\n[FINAL ANSWER]:")
    print(ans_resp.answer)

    if ans_resp.telemetry and ans_resp.telemetry.cost_available:
        print(f"\n[TELEMETRY] Tokens: {ans_resp.telemetry.total_tokens} (prompt: {ans_resp.telemetry.prompt_tokens}, completion: {ans_resp.telemetry.completion_tokens}) | Est. Cost: ${ans_resp.telemetry.estimated_cost_usd:.6f}")
    print(f"[TOTAL PIPELINE LATENCY]: {total_time:.1f}ms")
    print("=" * 80 + "\n")


DEMO_QUESTIONS = [
    "Tell me about The Dark Knight",
    "Show me the top 5 highest-rated Science Fiction movies",
    "Show movies directed by Christopher Nolan",
    "Compare Avatar and Titanic",
    "Which actors appeared in both The Dark Knight and Inception?",
    "What is the average rating for Animation movies?",
    "Show movies released between 2008 and 2010",
    "Who directed Inception?",
    "Will Avatar 3 make two billion dollars?",  # Unsupported
    "Tell me about Batman",                     # Clarification required
]


def main():
    parser_arg = argparse.ArgumentParser(description="Full TMDB Knowledge Graph QA Pipeline")
    parser_arg.add_argument("-q", "--question", type=str, help="Single question to ask")
    parser_arg.add_argument("-i", "--interactive", action="store_true", help="Interactive terminal mode")
    args = parser_arg.parse_args()

    print("\n--- Initializing Knowledge Graph Query Engine ---")
    engine = load_query_engine()

    print("--- Initializing Gemini Client, Parser, and Answer Generator ---")
    client = GeminiClient()
    parser = NaturalLanguageQueryParser(client=client, allow_mock_fallback=True)
    generator = NaturalLanguageAnswerGenerator(client=client)

    if not client.is_configured():
        print("[NOTICE] GEMINI_API_KEY is not set. Running in MOCK / LOCAL FALLBACK mode.")
    else:
        print(f"[INFO] Connected to Gemini API ({client.model}, {client.pricing_tier} tier).")

    if args.question:
        run_pipeline(args.question, parser, engine, generator)
    elif args.interactive:
        print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                user_q = input("Ask a question: ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit"):
                    break
                run_pipeline(user_q, parser, engine, generator)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        print("\n--- Running 10 Representative Knowledge Graph Questions ---\n")
        for q in DEMO_QUESTIONS:
            run_pipeline(q, parser, engine, generator)


if __name__ == "__main__":
    main()
