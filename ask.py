"""Ask the agent questions interactively about the transcript's case.

By default the transcript's customer turns are replayed first (deterministically,
so no LLM calls) to load the case memory, then each question is asked after the
whole conversation. --fresh skips the replay: the case memory starts empty and no
fact established during the chat (outage ticket, dates, statement) is known yet,
only the account header. Type "quit" to exit.

Usage: python ask.py [--fresh] [--no-llm]
"""
import argparse
import sys

from eval_cases import UNSEEN_TIME
from run import Agent, header

FRESH_TIME = "00:00"  # before the first turn: only header facts are known


def load_agent(fresh=False, use_llm=None):
    agent = Agent(use_llm=use_llm)
    if fresh:
        return agent, FRESH_TIME, 0
    # Replay the transcript without LLMs; only the resulting case memory is kept.
    warm = Agent(use_llm=False)
    warm.run()
    agent.memory = warm.memory
    return agent, UNSEEN_TIME, len(warm.memory.decisions)


def format_answer(t):
    r = t["route"]
    return "\n".join([
        f"route:      {r['status']} -> {', '.join(r['categories']) or '-'} ({r['source']})",
        f"governing:  {t['governing_rule'] or '-'}",
        f"cited:      {', '.join(t['cited_articles']) or '-'}",
        f"decision:   {t['decision']}",
        f"reply:      {t['reply']}",
    ])


def main(argv=None, input_fn=input, out=sys.stdout):
    parser = argparse.ArgumentParser(description="Ask the billing-dispute agent questions.")
    parser.add_argument("--fresh", action="store_true", help="start with empty case memory")
    parser.add_argument("--no-llm", action="store_true", help="deterministic only, even if USE_LLM=1")
    args = parser.parse_args(argv)

    agent, time, loaded = load_agent(args.fresh, use_llm=False if args.no_llm else None)
    print("\n".join(header(agent)), file=out)
    print(f"Case memory: {'empty (--fresh)' if args.fresh else f'loaded {loaded} transcript turns'}. "
          f'Type "quit" to exit.', file=out)
    while True:
        try:
            question = input_fn("\nquestion> ").strip()
        except EOFError:
            break
        if question.lower() == "quit":
            break
        if not question:
            continue
        print(format_answer(agent.answer(question, time)), file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
