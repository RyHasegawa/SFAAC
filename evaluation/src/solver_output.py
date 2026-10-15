import re


def parse_premises_conclusion(text: str) -> dict[str, str]:
    """Parse labelled premises and conclusion sections from solver text."""
    pattern = re.compile(
        r"""(?is)
        \bpremises\b
        \s*:?\s*
        (.*?)
        \bconclusion\b
        \s*:?\s*
        (.*)
        """,
        re.VERBOSE,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(
            "Could not find both 'premises' and 'conclusion' sections in the input text."
        )

    premises = re.sub(r"\s*\n\s*", " ", match.group(1)).strip()
    conclusion = re.sub(r"\s*\n\s*", " ", match.group(2)).strip()
    return {"premises": premises, "conclusion": conclusion}
