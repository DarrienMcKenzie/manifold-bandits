# bmc/process_safe_math_verify.py
import multiprocessing as mp
import traceback
from typing import Optional

# Default config
VERIFY_TIMEOUT_SECONDS = 5.0
VERIFY_CONFIG_MODES = ["expr", "latex"]  # order: expr first, then latex

# ------------------------------
# Child process function (top-level)
# ------------------------------
def proc_target_parse_and_verify(conn, gold_text: str, pred_text: str, modes):
    """
    Runs inside a child process.
    Parses gold/pred texts using Math-Verify and returns score through a pipe.
    Returns (ok: bool, result: float or str)
    """
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        # Build extraction config based on modes
        cfg_map = {
            "expr": ExprExtractionConfig,
            "latex": LatexExtractionConfig,
        }
        def build_cfg(modes_list):
            cfg = []
            for m in modes_list:
                m_lower = m.lower()
                if m_lower.startswith("expr"):
                    cfg.append(ExprExtractionConfig())
                elif m_lower.startswith("latex"):
                    cfg.append(LatexExtractionConfig())
                else:
                    raise ValueError(f"Unknown parse mode: {m}")
            return cfg

        gold_cfg = build_cfg(modes)
        pred_cfg = build_cfg(modes)

        # Parse with parsing_timeout=None (we handle timeout via process)
        gold_parsed = parse(gold_text, extraction_config=gold_cfg, parsing_timeout=None)
        pred_parsed = parse(pred_text, extraction_config=pred_cfg, parsing_timeout=None)

        # Verify with timeout_seconds=None (parent enforces timeout)
        score = verify(gold_parsed, pred_parsed, timeout_seconds=None)

        conn.send((True, float(score)))
        conn.close()
    except Exception:
        conn.send((False, traceback.format_exc()))
        conn.close()

# ------------------------------
# Parent-facing safe wrapper
# ------------------------------
def process_safe_verify(
    gold_text: str,
    pred_text: str,
    modes=VERIFY_CONFIG_MODES,
    timeout_seconds: float = VERIFY_TIMEOUT_SECONDS
) -> Optional[float]:
    """
    Runs Math-Verify in an isolated process with hard timeout.
    Returns float score or None on timeout/error.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    # Spawn a process using the top-level child function
    proc = ctx.Process(
        target=proc_target_parse_and_verify,
        args=(child_conn, gold_text, pred_text, modes)
    )
    proc.daemon = False
    proc.start()
    child_conn.close()

    try:
        if parent_conn.poll(timeout_seconds):
            ok, result = parent_conn.recv()
            proc.join(timeout=1.0)
            if ok:
                return float(result)
            else:
                # Child exception occurred
                return None
        else:
            # Timeout: hard kill
            try:
                proc.terminate()
            except Exception:
                pass
            proc.join(timeout=1.0)
            return None
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass