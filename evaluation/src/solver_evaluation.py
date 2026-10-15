import ast
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from z3 import *


U = DeclareSort("U")


############ Z3の直接出力　整形コード ############

# ----------------------------
# 1) 文字列中のシンボルを収集
# ----------------------------
def _collect_symbols(expr_src: str) -> Tuple[Set[str], Dict[str, Set[int]]]:
    """
    returns:
      const_names: 裸で現れる識別子（Const候補）
      func_arities: 呼び出し位置に現れる識別子 -> arity集合（Function候補）
    """
    tree = ast.parse(expr_src, mode="eval")

    const_names: Set[str] = set()
    func_arities: Dict[str, Set[int]] = {}

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            # f(a,b) の f が Name の場合だけ拾う（student_of(...) など）
            if isinstance(node.func, ast.Name):
                fname = node.func.id
                func_arities.setdefault(fname, set()).add(len(node.args))
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            const_names.add(node.id)

    V().visit(tree)
    return const_names, func_arities


# ----------------------------
# 2) 安全寄り eval 環境を作る
# ----------------------------
def Iff(a, b):
    return a == b

def _make_z3_env(const_names: Set[str], func_arities: Dict[str, Set[int]]) -> Dict[str, Any]:
    env: Dict[str, Any] = {}

    # Z3 組み込み（必要なものだけ）
    env.update({
        "And": And,
        "Or": Or,
        "Not": Not,
        "Implies": Implies,
        "Xor": Xor,
        "Iff": Iff,
        "BoolVal": BoolVal,
        "True": True,
        "False": False,
    })

    # repr っぽい ForAll(x, body) / Exists(x, body) を吸収するラッパ
    def ForAll_wrap(v, body):
        vs = v if isinstance(v, (list, tuple)) else [v]
        return ForAll(vs, body)

    def Exists_wrap(v, body):
        vs = v if isinstance(v, (list, tuple)) else [v]
        return Exists(vs, body)

    env["ForAll"] = ForAll_wrap
    env["Exists"] = Exists_wrap

    # まず Function 候補を作る（裸名と衝突したら Function 優先）
    for fname, arset in func_arities.items():
        if fname in env:
            continue
        if len(arset) != 1:
            raise ValueError(f"Ambiguous arity for function '{fname}': {sorted(arset)}")
        arity = next(iter(arset))
        env[fname] = Function(fname, *([U] * arity), BoolSort())

    # 次に Const 候補を作る
    # ただし Z3 組み込み名や Function 名は避ける
    for cname in const_names:
        if cname in env:
            continue
        env[cname] = Const(cname, U)

    return env


# ----------------------------
# 3) 構文を制限した上で eval
# ----------------------------
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,   # True/False/数値など（使わないなら削ってOK）
    ast.Tuple,
    ast.List,
)

def _assert_safe_expr(expr_src: str) -> None:
    tree = ast.parse(expr_src, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed syntax in expr: {type(node).__name__}")


def parse_z3py_string(expr_src: str) -> Any:
    """
    Z3Pyっぽい文字列 -> Z3 AST
    """
    _assert_safe_expr(expr_src)

    consts, funcs = _collect_symbols(expr_src)
    env = _make_z3_env(consts, funcs)

    # globals を空にして builtins も消す（最低限のガード）
    return eval(expr_src, {"__builtins__": {}}, env)


# ----------------------------
# 4) record から premises_conj, conclusion を作る
# ----------------------------
def build_argument_from_z3_strings(record: Dict[str, Any]):
    prem_src = record["premises"]
    conc_src = record["conclusion"]

    premises_conj = parse_z3py_string(prem_src)
    conclusion = parse_z3py_string(conc_src)

    return premises_conj, conclusion


############ Z3 entailment classification ############
def classify_from_premises(premises, conclusion):
    # premises が単一の式でもリストでも動くように揃える
    if isinstance(premises, list):
        prem_list = premises
    else:
        prem_list = [premises]

    # Γ ⊨ C ?  （前提と結論の否定は両立しないか？）
    s_true = Solver()
    s_true.add(*prem_list)
    s_true.add(Not(conclusion))
    r_true = s_true.check()

    # Γ ⊨ ¬C ?  （前提と結論は両立しないか？）
    s_false = Solver()
    s_false.add(*prem_list)
    s_false.add(conclusion)
    r_false = s_false.check()

    if r_true == unsat and r_false == unsat:
        return "inconsistent premises"
    elif r_true == unsat:
        return "true"
    elif r_false == unsat:
        return "false"
    else:
        return "unknown"

########### prover9判定コード ###########

Label = Literal["true", "false", "uncertain", "error_parse", "inconsistent premises"]


# ----------------------------
# Prover9 / Mace4 runners
# ----------------------------
@dataclass
class CmdResult:
    ok: bool
    timeout: bool
    returncode: int
    stdout: str
    stderr: str
    kind: str  # "prover9" or "mace4"


def _run_cmd(cmd: List[str], input_str: str, timeout_sec: int, kind: str) -> CmdResult:
    try:
        cp = subprocess.run(
            cmd,
            input=input_str,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        return CmdResult(
            ok=(cp.returncode == 0),
            timeout=False,
            returncode=cp.returncode,
            stdout=cp.stdout or "",
            stderr=cp.stderr or "",
            kind=kind,
        )
    except subprocess.TimeoutExpired as e:
        return CmdResult(
            ok=False,
            timeout=True,
            returncode=-1,
            stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
            stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
            kind=kind,
        )


def prover9_proves(input_in: str, prover9_timeout_sec: int = 5, wall_timeout_sec: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Returns (proved?, error_reason_if_any).
    Prefer exit codes over brittle stdout parsing.
    Prover9 exit codes (manual): 0=max_proofs found, 1=fatal, 2=sos_empty(no proof), 4=time limit, etc.
    """
    r = _run_cmd(["prover9", "-t", str(prover9_timeout_sec)], input_in, wall_timeout_sec, kind="prover9")
    out = (r.stdout + "\n" + r.stderr).lower()

    if r.timeout:
        return False, "timeout"

    # Fatal errors (syntax, etc.)
    if r.returncode == 1 or "fatal error" in out or "parse error" in out or "syntax error" in out:
        return False, "syntax"

    # Exit code 0 => at least one proof found
    if r.returncode == 0:
        return True, None

    # Otherwise: not proved (sos_empty, max_seconds, etc.)
    return False, None


def mace4_has_model(input_in: str, mace4_timeout_sec: int = 5, wall_timeout_sec: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Returns (model_found?, error_reason_if_any).
    Use Mace4 exit codes (manual):
      0: found max_models (default 1) => model exists
      2: search completed, no models within constraints
      3/4: some models found but ended early/time limit => model exists
      1: fatal error
    """
    r = _run_cmd(["mace4", "-t", str(mace4_timeout_sec)], input_in, wall_timeout_sec, kind="mace4")
    out = (r.stdout + "\n" + r.stderr).lower()

    if r.timeout:
        return False, "timeout"

    if r.returncode == 1 or "fatal error" in out or "parse error" in out or "syntax error" in out:
        return False, "syntax"

    if r.returncode in (0, 3, 4):
        return True, None

    if r.returncode == 2:
        return False, None

    # Fallback (unknown return code): keep old heuristic as a last resort
    if "exiting with" in out and "model" in out:
        return True, None
    if "model found" in out:
        return True, None

    return False, None


# ----------------------------
# Prover9 input parsing / rewriting
# ----------------------------
_SECTION_RE = re.compile(
    r"formulas\((assumptions|goals)\)\.\s*(.*?)\s*end_of_list\.\s*",
    re.IGNORECASE | re.DOTALL,
)


def _strip_comments(line: str) -> str:
    # Prover9 supports comments starting with % (and sometimes # in some contexts).
    line = line.split("%", 1)[0]
    line = line.split("#", 1)[0]
    return line.strip()


def parse_prover9_in(input_in: str) -> Tuple[List[str], List[str]]:
    """
    Extract assumptions and goals as lists of formula strings (without trailing '.').
    Ignores blank lines and comments.
    """
    sections = {m.group(1).lower(): m.group(2) for m in _SECTION_RE.finditer(input_in)}
    if "assumptions" not in sections or "goals" not in sections:
        raise ValueError("Input must contain both formulas(assumptions). ... end_of_list. and formulas(goals). ... end_of_list.")

    def parse_block(block: str) -> List[str]:
        formulas = []
        for raw in block.splitlines():
            s = _strip_comments(raw)
            if not s:
                continue
            # allow trailing '.' (optional)
            if s.endswith("."):
                s = s[:-1].strip()
            if s:
                formulas.append(s)
        return formulas

    return parse_block(sections["assumptions"]), parse_block(sections["goals"])


def build_prover9_in(assumptions: List[str], goals: List[str]) -> str:
    """
    Build a canonical prover9 input.
    Ensures each formula ends with '.'.
    """
    def fmt(lines: List[str]) -> str:
        out = []
        for f in lines:
            f = f.strip()
            if not f:
                continue
            if f.endswith("."):
                f = f[:-1].strip()
            out.append(f"  {f}.")
        return "\n".join(out)

    return (
        "formulas(assumptions).\n"
        f"{fmt(assumptions)}\n"
        "end_of_list.\n\n"
        "formulas(goals).\n"
        f"{fmt(goals)}\n"
        "end_of_list.\n"
    )


def negate_formula(f: str) -> str:
    """
    Prover9 negation: -( ... ).
    """
    f = f.strip()
    if f.endswith("."):
        f = f[:-1].strip()
    return f"-({f})"


Label = Literal["true", "false", "uncertain", "error_parse", "inconsistent premises"]


def classify_entailment_prover9(
    input_in: str,
    prover9_timeout_sec: int = 5,
    mace4_timeout_sec: int = 5,
    wall_timeout_sec: int = 10,
) -> Tuple[Label, dict]:
    """
    Given a prover9 input that contains assumptions + goals, classify:

    - "true"                  : assumptions ⊨ goals
    - "false"                 : assumptions ⊨ ¬goals
    - "uncertain"             : goals are true in some models and false in some models
    - "error_parse"           : parse/syntax/runtime/timeout issues so that we cannot decide
    - "inconsistent premises" : assumptions are inconsistent / unsatisfiable

    This version is more conservative than the original:
      - It distinguishes solver parse/syntax errors from actual inconsistency.
      - It only returns "inconsistent premises" when there is strong evidence.
    """

    # ----------------------------
    # internal helper functions
    # ----------------------------
    def _normalize_err(err: Any) -> str:
        if err is None:
            return ""
        if isinstance(err, str):
            return err.strip()
        return str(err).strip()

    def _classify_solver_error(err: Any) -> str | None:
        """
        Return one of:
          - "parse"
          - "timeout"
          - "runtime"
          - None
        based on the error message text.
        """
        s = _normalize_err(err).lower()
        if not s:
            return None

        parse_keywords = [
            "syntax error",
            "parse error",
            "bad character",
            "unexpected token",
            "unexpected end",
            "missing",
            "ill-formed",
            "cannot parse",
            "parse_input_failed",
            "sread_term",
            "read error",
            "fatal error: s",
            "argument expected",
        ]
        timeout_keywords = [
            "timeout",
            "time out",
            "timed out",
            "wall timeout",
            "max_seconds",
            "cpu limit",
        ]
        runtime_keywords = [
            "error",
            "fatal",
            "failed",
            "exception",
            "non-zero exit",
            "segmentation fault",
            "killed",
            "cannot open",
            "no such file",
        ]

        if any(k in s for k in parse_keywords):
            return "parse"
        if any(k in s for k in timeout_keywords):
            return "timeout"
        if any(k in s for k in runtime_keywords):
            return "runtime"
        return "runtime"

    def _run_prover9_single_goal(goal: str) -> dict:
        """
        Returns:
          {
            "status": "proved" | "not_proved" | "error_parse" | "error_timeout" | "error_runtime",
            "err": ...
          }
        """
        p_in = build_prover9_in(assumptions, [goal])
        try:
            proved, err = prover9_proves(
                p_in,
                prover9_timeout_sec=prover9_timeout_sec,
                wall_timeout_sec=wall_timeout_sec,
            )
        except Exception as e:
            return {"status": "error_runtime", "err": str(e)}

        err_kind = _classify_solver_error(err)
        if err_kind == "parse":
            return {"status": "error_parse", "err": err}
        if err_kind == "timeout":
            return {"status": "error_timeout", "err": err}
        if err_kind == "runtime":
            return {"status": "error_runtime", "err": err}

        return {"status": "proved" if proved else "not_proved", "err": err}

    def _run_mace4(formulas: list[str]) -> dict:
        """
        Returns:
          {
            "status": "sat" | "unsat_or_no_model" | "error_parse" | "error_timeout" | "error_runtime",
            "err": ...
          }
        Important:
          Because the legacy helper returns only bool + err, False may mean:
            - truly no model
            - timeout
            - parse error
            - other runtime failure
          So we classify conservatively.
        """
        m_in = build_prover9_in(assumptions, formulas)
        try:
            sat, err = mace4_has_model(
                m_in,
                mace4_timeout_sec=mace4_timeout_sec,
                wall_timeout_sec=wall_timeout_sec,
            )
        except Exception as e:
            return {"status": "error_runtime", "err": str(e)}

        err_kind = _classify_solver_error(err)
        if err_kind == "parse":
            return {"status": "error_parse", "err": err}
        if err_kind == "timeout":
            return {"status": "error_timeout", "err": err}
        if err_kind == "runtime":
            return {"status": "error_runtime", "err": err}

        if sat:
            return {"status": "sat", "err": err}

        # No explicit error text and sat=False.
        # This is the ambiguous legacy case: maybe באמת no model, maybe solver limitation.
        return {"status": "unsat_or_no_model", "err": err}

    # ----------------------------
    # Step 0: parse assumptions/goals blocks
    # ----------------------------
    try:
        assumptions, goals = parse_prover9_in(input_in)
    except Exception as e:
        return "error_parse", {
            "reason": "parse_input_failed",
            "exception": str(e),
        }

    if not goals:
        return "error_parse", {"reason": "no_goals_found"}

    # ----------------------------
    # Step 1: Try proving goals
    # ----------------------------
    goal_results = []
    for g in goals:
        r = _run_prover9_single_goal(g)
        goal_results.append({"goal": g, **r, "where": "prove_goal"})

    # ----------------------------
    # Step 2: Try proving negated goals
    # ----------------------------
    neg_goal_results = []
    for g in goals:
        ng = negate_formula(g)
        r = _run_prover9_single_goal(ng)
        neg_goal_results.append(
            {"goal": g, "negated_goal": ng, **r, "where": "prove_neg_goal"}
        )

    proof_errors = [
        x for x in (goal_results + neg_goal_results)
        if x["status"] in {"error_parse", "error_timeout", "error_runtime"}
    ]

    proved_all = all(r["status"] == "proved" for r in goal_results)
    proved_all_neg = all(r["status"] == "proved" for r in neg_goal_results)

    # Strong evidence of inconsistency: explosion
    if proved_all and proved_all_neg:
        return "inconsistent premises", {
            "reason": "both_goal_and_neg_goal_proved",
            "note": "Assumptions are likely inconsistent (principle of explosion).",
            "goal_results": goal_results,
            "neg_goal_results": neg_goal_results,
        }

    if proved_all:
        return "true", {
            "decided_by": "prover9",
            "goal_results": goal_results,
            "neg_goal_results": neg_goal_results,
        }

    if proved_all_neg:
        return "false", {
            "decided_by": "prover9",
            "goal_results": goal_results,
            "neg_goal_results": neg_goal_results,
        }

    # If any proof attempt had explicit parse/runtime errors, do not collapse into inconsistency.
    # We still allow Mace4 to rescue the case only when errors were NOT parse/runtime.
    any_parse_error_in_proofs = any(r["status"] == "error_parse" for r in proof_errors)
    any_runtime_error_in_proofs = any(r["status"] == "error_runtime" for r in proof_errors)

    if any_parse_error_in_proofs:
        return "error_parse", {
            "reason": "solver_parse_error_in_prover9",
            "goal_results": goal_results,
            "neg_goal_results": neg_goal_results,
        }

    if any_runtime_error_in_proofs:
        return "error_parse", {
            "reason": "solver_runtime_error_in_prover9",
            "goal_results": goal_results,
            "neg_goal_results": neg_goal_results,
        }

    # ----------------------------
    # Step 3: Mace4 satisfiability checks
    # ----------------------------
    # assumptions + goals
    mace_pos = _run_mace4(goals)

    # assumptions + negated goals
    neg_goals = [negate_formula(g) for g in goals]
    mace_neg = _run_mace4(neg_goals)

    details = {
        "decided_by": "mace4",
        "mace4_pos": mace_pos,
        "mace4_neg": mace_neg,
        "goal_results": goal_results,
        "neg_goal_results": neg_goal_results,
        "note_multiple_goals": (
            "For multiple goals, this implementation checks "
            "'assumptions ∧ g1 ∧ ... ∧ gn' versus 'assumptions ∧ ¬g1 ∧ ... ∧ ¬gn'. "
            "This is stronger than checking satisfiability of ¬(g1 ∧ ... ∧ gn)."
        ) if len(goals) > 1 else None,
    }

    # Explicit Mace4 parse/runtime failures => error_parse
    if mace_pos["status"] == "error_parse" or mace_neg["status"] == "error_parse":
        return "error_parse", {
            **details,
            "reason": "solver_parse_error_in_mace4",
        }

    if mace_pos["status"] == "error_runtime" or mace_neg["status"] == "error_runtime":
        return "error_parse", {
            **details,
            "reason": "solver_runtime_error_in_mace4",
        }

    # Timeout means undecidable in current configuration
    if mace_pos["status"] == "error_timeout" or mace_neg["status"] == "error_timeout":
        return "error_parse", {
            **details,
            "reason": "solver_timeout_in_mace4",
        }

    pos_sat = mace_pos["status"] == "sat"
    neg_sat = mace_neg["status"] == "sat"

    if pos_sat and not neg_sat:
        # Strong evidence that assumptions ∧ goals is satisfiable,
        # and no model found for assumptions ∧ ¬goals.
        # We still interpret this as true, as in the original function.
        return "true", details

    if not pos_sat and neg_sat:
        return "false", details

    if pos_sat and neg_sat:
        return "uncertain", details

    # Here both are "unsat_or_no_model".
    # This is NOT enough to conclude inconsistency in general unless we have stronger evidence.
    # But for backward compatibility, we keep the original label only with an explicit caution.
    return "inconsistent premises", {
        **details,
        "reason": "no_model_found_for_both_pos_and_neg",
        "warning": (
            "This likely indicates inconsistent/unsatisfiable assumptions, "
            "but may also be caused by Mace4 search incompleteness or insufficient bounds."
        ),
    }
