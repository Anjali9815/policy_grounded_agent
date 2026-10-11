"""Case memory carried in graph state across turns."""
from dataclasses import dataclass, field


def check_lines(resolution):
    """Printable check lines keyed by check id, including rules deliberately not applied."""
    lines = {c["id"]: f"[{c['status']:>10}] {c['id']}: {c['detail']}" for c in resolution["checks"]}
    lines.update({f"not_applied.{n['rule']}": f"[not applied] {n['rule']}: {n['reason']}"
                  for n in resolution["not_applied"]})
    return lines


@dataclass
class CaseMemory:
    facts: dict = field(default_factory=dict)      # facts known as of the latest turn
    checks: dict = field(default_factory=dict)     # check id -> line as last shown
    decisions: list = field(default_factory=list)  # one entry per turn so far

    def update_facts(self, known):
        self.facts = known

    def check_view(self, resolution):
        """Full check block the first time, then only what changed since last shown."""
        lines = check_lines(resolution)
        if not self.checks:
            view = {"full": True, "lines": list(lines.values())}
        else:
            changed = [v for k, v in lines.items() if self.checks.get(k) != v]
            changed += [f"[   removed] {k}" for k in self.checks if k not in lines]
            view = {"full": False, "lines": changed}
        self.checks = lines
        return view

    def record(self, time, categories, intent, decision, governing_rule, verifier_passed):
        self.decisions.append({"time": time, "categories": categories, "intent": intent,
                               "decision": decision, "governing_rule": governing_rule,
                               "verifier_passed": verifier_passed})
