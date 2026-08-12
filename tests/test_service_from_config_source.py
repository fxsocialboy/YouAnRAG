from pathlib import Path


def test_from_config_does_not_reference_self_or_domain_gate_before_assignment():
    source = (Path(__file__).resolve().parents[1] / "src/rag_v2/agent/service.py").read_text(encoding="utf-8")
    from_config_body = source.split("    def from_config(", 1)[1].split("    def answer(", 1)[0]
    assert "self.domain_gate" not in from_config_body
    assert from_config_body.index("domain_gate = DomainGate()") < from_config_body.index("domain_gate=domain_gate")
