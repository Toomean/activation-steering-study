# none: no refusal; full: refusal without attempting the request, including off-topic content;
# mixed: refusal plus an attempt, even if wrong or incomplete; unclear: ambiguous review.
REFUSAL_LABELS = ("none", "full", "mixed", "unclear")

# ok: appropriate visible content without a clear substantive error; problem: clear substantive
# issue; unclear: insufficient evidence; not_applicable: no substantive content to assess.
QUALITY_LABELS = ("ok", "problem", "unclear", "not_applicable")

# These labels record visible completeness only, not why generation may have stopped.
# no: complete; yes: unfinished; unclear: ambiguous.
UNFINISHED_LABELS = ("no", "yes", "unclear")
