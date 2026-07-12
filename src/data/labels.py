"""Rhythm label definitions for the two cascade stages.

Stage 1 (binary): normal sinus rhythm (SR) vs. anything else.
Stage 2 (multi-class): only runs on Stage-1 anomalies. The dataset's raw
`event_rhythm` codes are grouped into clinically coherent buckets so that
very-low-support codes (e.g. JTACH, n~5 total) don't get their own class.
"""

NORMAL_RHYTHM = "SR"

# raw event_rhythm code -> Stage 2 group name
RHYTHM_GROUPS = {
    # sinus-node rhythms with abnormal rate/regularity but normal conduction
    "STACH": "sinus_rate",
    "SBRAD": "sinus_rate",
    "SARRH": "sinus_rate",
    # supraventricular tachyarrhythmias (non-sinus)
    "AF": "atrial_tachy",
    "AFLT": "atrial_tachy",
    "SVTACH": "atrial_tachy",
    "MATACH": "atrial_tachy",
    # paced rhythms
    "VPACE": "paced",
    "AVPACE": "paced",
    "APACE": "paced",
    # conduction blocks
    "1AVB": "conduction_block",
    "LBBB": "conduction_block",
    "RBBB": "conduction_block",
    # junctional rhythms
    "JR": "junctional",
    "JTACH": "junctional",
    # ventricular
    "VTACH": "ventricular",
}

STAGE2_CLASSES = sorted(set(RHYTHM_GROUPS.values()))
STAGE2_CLASS_TO_IDX = {name: i for i, name in enumerate(STAGE2_CLASSES)}


def stage2_group_for(event_rhythm: str) -> str | None:
    """Return the Stage 2 group name for a raw rhythm code, or None for SR / unknown codes."""
    return RHYTHM_GROUPS.get(event_rhythm)
