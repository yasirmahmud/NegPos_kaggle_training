#!/usr/bin/env python3
"""Generate four structurally valid, formally inequivalent RTL mutants per pair.

This script is intentionally stdlib-only so it can run inside the WSL-hosted
OSS CAD Suite. Each retained negative passes Icarus compilation and Yosys
synthesis, then produces a bounded SAT counterexample against the correct RTL.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PIPELINE_VERSION = "1"
MODULE_BLOCK = re.compile(
    r"\bmodule\s+(?:automatic\s+)?(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\b(?P<body>.*?)\bendmodule\b",
    re.DOTALL,
)
IDENTIFIER = r"[A-Za-z_$][A-Za-z0-9_$]*"


@dataclass(frozen=True)
class Mutation:
    mutation_type: str
    variant: str
    module_name: str
    start: int
    end: int
    before: str
    after: str
    line: int
    column: int

    def apply(self, rtl: str) -> str:
        return rtl[: self.start] + self.after + rtl[self.end :]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def mask_comments_and_strings(text: str) -> str:
    """Replace comment/string contents with spaces while preserving offsets."""
    chars = list(text)
    index = 0
    state = "code"
    while index < len(chars):
        char = chars[index]
        next_char = chars[index + 1] if index + 1 < len(chars) else ""
        if state == "code":
            if char == '"':
                chars[index] = " "
                state = "string"
            elif char == "/" and next_char == "/":
                chars[index] = chars[index + 1] = " "
                state = "line_comment"
                index += 1
        elif state == "string":
            chars[index] = " " if char != "\n" else "\n"
            if char == "\\" and next_char:
                chars[index + 1] = " "
                index += 1
            elif char == '"':
                state = "code"
        elif state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                chars[index] = " "
        if state == "code" and char == "/" and next_char == "*":
            chars[index] = chars[index + 1] = " "
            state = "block_comment"
            index += 1
        elif state == "block_comment":
            if char == "*" and next_char == "/":
                chars[index] = chars[index + 1] = " "
                state = "code"
                index += 1
            elif char != "\n":
                chars[index] = " "
        index += 1
    return "".join(chars)


def line_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    previous_newline = text.rfind("\n", 0, offset)
    return line, offset - previous_newline


def add_mutation(
    target: list[Mutation], seen: set[str], rtl: str, mutation_type: str, variant: str,
    module_name: str, start: int, end: int, after: str,
) -> None:
    before = rtl[start:end]
    if before == after:
        return
    key = sha256_text(rtl[:start] + after + rtl[end:])
    if key in seen:
        return
    seen.add(key)
    line, column = line_column(rtl, start)
    target.append(Mutation(mutation_type, variant, module_name, start, end, before, after, line, column))


def assignment_candidates(
    rtl: str, masked: str, module_name: str, block_start: int, block_end: int,
    target: list[Mutation], seen: set[str],
) -> None:
    block_mask = masked[block_start:block_end]
    pattern = re.compile(
        rf"(?P<lhs>(?:{IDENTIFIER}(?:\s*\[[^;\]]+\])?|\{{[^;\n]+\}}))\s*(?P<op><=|(?<![=!<>])=(?!=))\s*(?P<rhs>[^;]+);"
    )
    for match in pattern.finditer(block_mask):
        absolute_start = block_start + match.start()
        line_start = rtl.rfind("\n", 0, absolute_start) + 1
        prefix = rtl[line_start:absolute_start].strip().casefold()
        is_continuous_assignment = bool(re.search(r"\bassign\s*$", prefix))
        if not is_continuous_assignment and re.search(
            r"\b(?:input|output|inout|wire|reg|logic|parameter|localparam|genvar|integer)\b", prefix
        ):
            continue
        rhs_start = block_start + match.start("rhs")
        rhs_end = block_start + match.end("rhs")
        rhs = rtl[rhs_start:rhs_end].strip()
        if not rhs:
            continue
        replacements = (
            ("arithmetic_operator_mutation", "rhs_plus_one", f"({rhs}) + 1'b1"),
            ("counter_boundary_off_by_one", "rhs_minus_one", f"({rhs}) - 1'b1"),
            ("arithmetic_operator_mutation", "rhs_bitwise_invert", f"~({rhs})"),
            ("arithmetic_operator_mutation", "rhs_xor_one", f"({rhs}) ^ 1'b1"),
            ("missing_or_incorrect_state_output_assignment", "stuck_at_zero", "'0"),
            ("missing_or_incorrect_state_output_assignment", "stuck_at_one", "'1"),
        )
        for mutation_type, variant, replacement in replacements:
            add_mutation(target, seen, rtl, mutation_type, variant, module_name, rhs_start, rhs_end, replacement)
        statement_start = block_start + match.start()
        statement_end = block_start + match.end()
        add_mutation(
            target, seen, rtl, "missing_or_incorrect_state_output_assignment", "delete_assignment",
            module_name, statement_start, statement_end, "",
        )


def enumerate_mutations(rtl: str) -> list[Mutation]:
    masked = mask_comments_and_strings(rtl)
    mutations: list[Mutation] = []
    seen: set[str] = {sha256_text(rtl)}
    for module_match in MODULE_BLOCK.finditer(masked):
        module_name = module_match.group("name")
        start, end = module_match.span()
        block = masked[start:end]

        replacements = (
            ("comparator_mutation", re.compile(r"===|!==|==|!=|<=|>=|(?<!<)<(?![=<])|(?<!>)>(?![=>])"),
             {"===": "!==", "!==": "===", "==": "!=", "!=": "==", "<=": "<", ">=": ">", "<": "<=", ">": ">="}),
            ("arithmetic_operator_mutation", re.compile(r"(?<![+])\+(?![+])|(?<![-])-\s*(?![-:>])|&&|\|\||(?<![&])&(?![&])|(?<![|])\|(?![|])|(?<!\^)[\^](?!\^)|<<|>>"),
             {"+": "-", "-": "+", "&&": "||", "||": "&&", "&": "|", "|": "&", "^": "|", "<<": ">>", ">>": "<<"}),
            ("blocking_nonblocking_assignment", re.compile(r"<="), {"<=": "="}),
            ("one_cycle_timing_or_handshake_error", re.compile(r"\b(?:posedge|negedge)\b"), {"posedge": "negedge", "negedge": "posedge"}),
        )
        for mutation_type, pattern, mapping in replacements:
            for match in pattern.finditer(block):
                token = match.group(0).strip()
                replacement = mapping.get(token)
                if replacement is None:
                    continue
                absolute_start = start + match.start()
                absolute_end = start + match.end()
                # Reset/clock edge mutations get their more specific category.
                local_context = masked[max(start, absolute_start - 40) : min(end, absolute_end + 40)].casefold()
                actual_type = mutation_type
                if mutation_type == "one_cycle_timing_or_handshake_error" and re.search(r"\b(?:rst|reset|reset_n|rst_n)\b", local_context):
                    actual_type = "reset_polarity_or_behavior"
                add_mutation(mutations, seen, rtl, actual_type, f"{token}_to_{replacement}", module_name, absolute_start, absolute_end, replacement)

        for match in re.finditer(r"\bif\s*\(\s*", block):
            insertion = start + match.end()
            add_mutation(
                mutations, seen, rtl, "enable_or_condition_inversion", "invert_if_condition",
                module_name, insertion, insertion, "!",
            )

        for match in re.finditer(r"\bsigned\b", block):
            absolute_start, absolute_end = start + match.start(), start + match.end()
            add_mutation(
                mutations, seen, rtl, "signedness_modification", "remove_signed",
                module_name, absolute_start, absolute_end, "",
            )

        # Primitive gate substitution covers structural RTL with no assignments.
        primitive_alternatives = {
            "and": ("or", "xor", "nand", "nor"),
            "or": ("and", "xor", "nor", "nand"),
            "xor": ("xnor", "or", "and", "nand"),
            "xnor": ("xor", "nor", "nand", "or"),
            "nand": ("and", "or", "nor", "xor"),
            "nor": ("or", "and", "nand", "xnor"),
            "buf": ("not",),
            "not": ("buf",),
        }
        primitive_pattern = re.compile(
            rf"\b(?P<gate>and|or|xor|xnor|nand|nor|buf|not)\b\s+(?:{IDENTIFIER}\s*)?\(", re.I
        )
        for match in primitive_pattern.finditer(block):
            gate = match.group("gate").casefold()
            gate_start = start + match.start("gate")
            gate_end = start + match.end("gate")
            for replacement in primitive_alternatives[gate]:
                add_mutation(
                    mutations, seen, rtl, "arithmetic_operator_mutation",
                    f"primitive_{gate}_to_{replacement}", module_name,
                    gate_start, gate_end, replacement,
                )

        # A wrong initial value is a single reset/state-behavior mutation. This
        # also exposes long-latency counter errors within a bounded SAT horizon.
        init_values = ["'1", "1", "2", "3"]
        comparison_values = list(dict.fromkeys(re.findall(
            r"(?:==|!=|<=|>=|<|>)\s*([0-9][0-9A-Za-z_']*)", block
        )))
        init_values.extend(value for value in comparison_values if value not in init_values)
        declaration_pattern = re.compile(
            rf"\b(?:output\s+)?(?:reg|logic)\s+(?:signed\s+)?(?:\[[^\]]+\]\s*)?(?P<name>{IDENTIFIER})\s*(?P<init>=\s*[^,;\)]+)?"
        )
        for match in declaration_pattern.finditer(block):
            if match.group("init"):
                init_start = start + match.start("init") + match.group("init").index("=") + 1
                init_end = start + match.end("init")
                while init_start < init_end and rtl[init_start].isspace():
                    init_start += 1
                while init_end > init_start and rtl[init_end - 1].isspace():
                    init_end -= 1
                for value in init_values:
                    add_mutation(
                        mutations, seen, rtl, "reset_polarity_or_behavior", "modify_initial_state",
                        module_name, init_start, init_end, value,
                    )
            else:
                insertion = start + match.end("name")
                for value in init_values:
                    add_mutation(
                        mutations, seen, rtl, "reset_polarity_or_behavior", "add_incorrect_initial_state",
                        module_name, insertion, insertion, f" = {value}",
                    )

        for match in re.finditer(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", block):
            for group in (1, 2):
                value = int(match.group(group))
                group_start = start + match.start(group)
                group_end = start + match.end(group)
                for delta in (-1, 1):
                    if value + delta >= 0:
                        add_mutation(
                            mutations, seen, rtl, "bit_width_or_slice_modification", f"bound_{delta:+d}",
                            module_name, group_start, group_end, str(value + delta),
                        )

        # Swap encoded state values or named states in assignments.
        state_names = list(dict.fromkeys(re.findall(rf"\b(?:localparam|parameter)\b[^;]*?\b({IDENTIFIER})\s*=", block)))
        if len(state_names) >= 2:
            for left, right in zip(state_names, state_names[1:] + state_names[:1]):
                for match in re.finditer(rf"(?P<op><=|(?<![=!<>])=(?!=))\s*\b{re.escape(left)}\b", block):
                    name_start = start + match.end() - len(left)
                    add_mutation(
                        mutations, seen, rtl, "fsm_transition_modification", f"{left}_to_{right}",
                        module_name, name_start, name_start + len(left), right,
                    )

        assignment_candidates(rtl, masked, module_name, start, end, mutations, seen)

    # Interleave categories deterministically to encourage four different errors.
    grouped: dict[str, list[Mutation]] = defaultdict(list)
    for mutation in mutations:
        grouped[mutation.mutation_type].append(mutation)
    ordered: list[Mutation] = []
    category_order = [
        "reset_polarity_or_behavior",
        "comparator_mutation",
        "arithmetic_operator_mutation",
        "bit_width_or_slice_modification",
        "signedness_modification",
        "counter_boundary_off_by_one",
        "fsm_transition_modification",
        "enable_or_condition_inversion",
        "missing_or_incorrect_state_output_assignment",
        "one_cycle_timing_or_handshake_error",
        "blocking_nonblocking_assignment",
    ]
    for category in category_order:
        if grouped[category]:
            ordered.append(grouped[category].pop(0))
    while any(grouped.values()):
        for category in category_order:
            if grouped[category]:
                ordered.append(grouped[category].pop(0))
    return ordered


def run_command(command: list[str], timeout_seconds: float) -> tuple[int, str, float, bool]:
    start = time.monotonic()
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", timeout=timeout_seconds,
        )
        return result.returncode, result.stdout, time.monotonic() - start, False
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return 124, output, time.monotonic() - start, True


def yosys_script(gold_path: Path, gate_path: Path, top: str, depth: int) -> str:
    gold = str(gold_path).replace("\\", "/")
    gate = str(gate_path).replace("\\", "/")
    top_escaped = top.replace("\\", "\\\\")
    preparation = (
        f"hierarchy -check -top {top_escaped}; prep -top {top_escaped} -flatten; "
        "async2sync; dffunmap; memory_map; opt; chformal -remove; check"
    )
    return (
        f"read_verilog -sv {gold}; {preparation}; rename {top_escaped} gold; design -stash gold_design; "
        f"read_verilog -sv {gate}; {preparation}; log SYNTHESIS_PASS; rename {top_escaped} gate; design -stash gate_design; "
        "design -reset; design -copy-from gold_design -as gold gold; design -copy-from gate_design -as gate gate; "
        "miter -equiv -flatten -make_assert gold gate miter; hierarchy -top miter; "
        f"sat -seq {depth} -set-init-zero -verify -prove-asserts -show-inputs -show-outputs miter"
    )


def formal_check(gold_path: Path, gate_path: Path, top: str, depth: int, timeout: float) -> dict[str, Any]:
    script = yosys_script(gold_path, gate_path, top, depth)
    code, output, elapsed, timed_out = run_command(["yosys", "-Q", "-p", script], timeout)
    synthesis_pass = "SYNTHESIS_PASS" in output
    counterexample = (
        code != 0 and synthesis_pass and "model found: FAIL!" in output and
        "proof did fail" in output and not timed_out
    )
    equivalent = code == 0 and synthesis_pass and "no model found: SUCCESS!" in output
    witness_lines = []
    if counterexample:
        for line in output.splitlines():
            if re.search(r"^(\s*(init|\d+)\s+\\|\s*Signal Name|\s*-{3,})", line):
                witness_lines.append(line.rstrip())
    return {
        "exit_code": code,
        "synthesis_pass": synthesis_pass,
        "counterexample_found": counterexample,
        "equivalent_within_bound": equivalent,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 6),
        "log_sha256": sha256_text(output),
        "witness_excerpt": witness_lines[-40:],
        "failure_tail": "\n".join(output.splitlines()[-8:]) if not counterexample and not equivalent else "",
    }


def compile_iverilog(path: Path, top: str, output_path: Path, timeout: float) -> dict[str, Any]:
    code, output, elapsed, timed_out = run_command(
        ["iverilog", "-g2012", "-s", top, "-o", str(output_path), str(path)], timeout
    )
    return {
        "pass": code == 0 and not timed_out,
        "exit_code": code,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 6),
        "log_sha256": sha256_text(output),
        "failure_tail": "\n".join(output.splitlines()[-8:]) if code != 0 else "",
    }


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def process_record(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    record, config = payload
    curation = record["curation"]
    record_id = curation["record_id"]
    shard_path = Path(config["shard_dir"]) / f"{curation['rank']:04d}_{record_id[:16]}.jsonl"
    if config["resume"] and shard_path.exists():
        existing = [json.loads(line) for line in shard_path.open("r", encoding="utf-8") if line.strip()]
        if len(existing) == config["negatives_per_record"]:
            attempted = max(item.get("generation", {}).get("attempt_number", 0) for item in existing)
            return {
                "rank": curation["rank"], "record_id": record_id, "status": "resumed",
                "accepted": len(existing), "attempted": attempted, "shard": str(shard_path),
            }

    rtl = record["ground_truth"]
    mutations = enumerate_mutations(rtl)[: config["max_candidates"]]
    if not mutations:
        return {"rank": curation["rank"], "record_id": record_id, "status": "failed", "accepted": 0, "attempted": 0, "reason": "no_mutation_candidates"}

    accepted: list[dict[str, Any]] = []
    attempted = 0
    rejection_counts: Counter[str] = Counter()
    distinct_types: set[str] = set()
    with tempfile.TemporaryDirectory(prefix=f"neg_{curation['rank']:04d}_") as temporary_name:
        temporary = Path(temporary_name)
        gold_path = temporary / "gold.sv"
        gate_path = temporary / "gate.sv"
        compile_path = temporary / "sim.out"
        write_text(gold_path, rtl)

        # Reference preflight: Icarus compile and self-equivalence must both pass.
        modules = list(dict.fromkeys(mutation.module_name for mutation in mutations))
        viable_modules: set[str] = set()
        baseline_failures: dict[str, Any] = {}
        for module_name in modules:
            compile_result = compile_iverilog(gold_path, module_name, compile_path, config["tool_timeout"])
            if not compile_result["pass"]:
                baseline_failures[module_name] = {"compile": compile_result}
                continue
            formal_result = formal_check(gold_path, gold_path, module_name, config["seq_depth"], config["formal_timeout"])
            if not formal_result["equivalent_within_bound"]:
                baseline_failures[module_name] = {"compile": compile_result, "self_equivalence": formal_result}
                continue
            viable_modules.add(module_name)

        if not viable_modules:
            return {
                "rank": curation["rank"], "record_id": record_id, "status": "failed", "accepted": 0,
                "attempted": 0, "reason": "reference_preflight_failed", "baseline_failures": baseline_failures,
            }

        deferred: list[Mutation] = []
        ordered: list[Mutation] = []
        for mutation in mutations:
            if mutation.mutation_type not in distinct_types:
                ordered.append(mutation)
                distinct_types.add(mutation.mutation_type)
            else:
                deferred.append(mutation)
        ordered.extend(deferred)
        distinct_types.clear()

        for mutation in ordered:
            if len(accepted) >= config["negatives_per_record"]:
                break
            if mutation.module_name not in viable_modules:
                rejection_counts["nonviable_mutated_module"] += 1
                continue
            attempted += 1
            mutant = mutation.apply(rtl)
            write_text(gate_path, mutant)
            compile_result = compile_iverilog(gate_path, mutation.module_name, compile_path, config["tool_timeout"])
            if not compile_result["pass"]:
                rejection_counts["compile_fail"] += 1
                continue
            formal_result = formal_check(gold_path, gate_path, mutation.module_name, config["seq_depth"], config["formal_timeout"])
            if not formal_result["synthesis_pass"]:
                rejection_counts["synthesis_fail"] += 1
                continue
            if not formal_result["counterexample_found"]:
                rejection_counts["no_bounded_counterexample"] += 1
                continue

            negative_index = len(accepted) + 1
            negative_id = sha256_text(record_id + "\0" + sha256_text(mutant))
            accepted.append({
                "id": negative_id,
                "source_id": record_id,
                "source_rank": curation["rank"],
                "negative_index": negative_index,
                "specification": record["question"],
                "correct_rtl": rtl,
                "negative_rtl": mutant,
                "mutation": {
                    "type": mutation.mutation_type,
                    "variant": mutation.variant,
                    "module": mutation.module_name,
                    "location": {"line": mutation.line, "column": mutation.column},
                    "before": mutation.before,
                    "after": mutation.after,
                    "localized_edit_count": 1,
                },
                "verification": {
                    "reference_verified_label": True,
                    "compile": "PASS",
                    "compiler": "Icarus Verilog",
                    "synthesis": "PASS",
                    "synthesizer": "Yosys",
                    "functional_verification": "FAIL",
                    "functional_method": "Yosys bounded SAT equivalence miter",
                    "counterexample_found": True,
                    "bounded_depth": config["seq_depth"],
                    "top_module": mutation.module_name,
                    "iverilog_log_sha256": compile_result["log_sha256"],
                    "yosys_log_sha256": formal_result["log_sha256"],
                    "counterexample_witness_excerpt": formal_result["witness_excerpt"],
                },
                "generation": {"pipeline_version": PIPELINE_VERSION, "attempt_number": attempted},
            })
            distinct_types.add(mutation.mutation_type)

    if len(accepted) == config["negatives_per_record"]:
        atomic_jsonl(shard_path, accepted)
        return {
            "rank": curation["rank"], "record_id": record_id, "status": "generated",
            "accepted": len(accepted), "attempted": attempted, "shard": str(shard_path),
            "mutation_types": sorted(distinct_types), "rejection_counts": dict(rejection_counts),
        }
    return {
        "rank": curation["rank"], "record_id": record_id, "status": "failed",
        "accepted": len(accepted), "attempted": attempted, "reason": "insufficient_valid_mutants",
        "candidate_count": len(mutations), "rejection_counts": dict(rejection_counts),
        "viable_modules": sorted(viable_modules),
    }


def tool_version(command: list[str]) -> str:
    code, output, _, _ = run_command(command, 10)
    if code != 0:
        return "unavailable"
    return next((line.strip() for line in output.splitlines() if line.strip()), "unknown")


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = [json.loads(line) for line in args.input.open("r", encoding="utf-8") if line.strip()]
    if args.limit:
        records = records[: args.limit]
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "shard_dir": str(args.shard_dir),
        "resume": args.resume,
        "negatives_per_record": args.negatives_per_record,
        "max_candidates": args.max_candidates,
        "seq_depth": args.seq_depth,
        "tool_timeout": args.tool_timeout,
        "formal_timeout": args.formal_timeout,
    }
    payloads = [(record, config) for record in records]
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_record, payload) for payload in payloads]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if completed % 25 == 0 or completed == len(futures):
                passed = sum(item["accepted"] == args.negatives_per_record for item in results)
                elapsed = time.monotonic() - started
                print(f"progress={completed}/{len(futures)} complete_records={passed} elapsed_s={elapsed:.1f}", flush=True)
    results.sort(key=lambda item: item["rank"])
    atomic_jsonl(args.manifest, results)

    complete = [item for item in results if item["accepted"] == args.negatives_per_record]
    if len(complete) == len(records):
        merged: list[dict[str, Any]] = []
        for item in complete:
            with Path(item["shard"]).open("r", encoding="utf-8") as handle:
                merged.extend(json.loads(line) for line in handle if line.strip())
        atomic_jsonl(args.output, merged)

    status_counts = Counter(item["status"] for item in results)
    failure_reasons = Counter(item.get("reason") for item in results if item.get("reason"))
    rejection_counts = Counter()
    mutation_types = Counter()
    attempts = 0
    for item in results:
        attempts += item.get("attempted", 0)
        rejection_counts.update(item.get("rejection_counts", {}))
        if item.get("shard") and Path(item["shard"]).exists():
            with Path(item["shard"]).open("r", encoding="utf-8") as handle:
                mutation_types.update(json.loads(line)["mutation"]["type"] for line in handle if line.strip())
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "configuration": config | {"workers": args.workers},
        "input": {"path": str(args.input.resolve()), "sha256": sha256_file(args.input), "records_requested": len(records)},
        "result": {
            "complete_source_records": len(complete),
            "failed_source_records": len(records) - len(complete),
            "negative_records": len(complete) * args.negatives_per_record,
            "attempted_mutations": attempts,
            "rejected_mutations_before_four": max(0, attempts - len(complete) * args.negatives_per_record),
            "status_counts": dict(status_counts),
            "failure_reasons": dict(failure_reasons),
            "current_nonresumed_candidate_rejection_counts": dict(rejection_counts),
            "mutation_type_counts": dict(mutation_types),
        },
        "tools": {
            "iverilog": tool_version(["iverilog", "-V"]),
            "yosys": tool_version(["yosys", "-V"]),
            "python": sys.version.split()[0],
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256_file(args.manifest)},
        "output": None,
        "acceptance_contract": {
            "compile": "Icarus Verilog PASS",
            "synthesis": "Yosys synthesis/check PASS",
            "functional": "Yosys SAT equivalence miter finds a counterexample within configured depth",
            "mutation": "one localized source edit",
        },
    }
    if len(complete) == len(records) and args.output.exists():
        summary["output"] = {"path": str(args.output.resolve()), "sha256": sha256_file(args.output), "bytes": args.output.stat().st_size}
    atomic_json(args.summary, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--negatives-per-record", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--seq-depth", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--tool-timeout", type=float, default=10.0)
    parser.add_argument("--formal-timeout", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["result"]["failed_source_records"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
