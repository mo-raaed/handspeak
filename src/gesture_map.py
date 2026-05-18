"""
gesture_map.py
Maps each ASL alphabet letter (A–Z) to a medical word
used by non-verbal patients to communicate needs.
"""

GESTURE_MAP = {
    "A": "",        # STOP signal
    "B": "Pain",
    "C": "Water",
    "D": "Food",
    "E": "Medicine",
    "F": "",        # SPEAK signal
    "G": "Doctor",
    "H": "Help",
    "I": "I",
    "J": "Problem",
    "K": "Cold",
    "L": "",        # CLEAR signal
    "M": "Stop",
    "N": "No",
    "O": "Okay",
    "P": "Please",
    "Q": "Quiet",
    "R": "Rest",
    "S": "Sleep",
    "T": "Toilet",
    "U": "Uncomfortable",
    "V": "",        # START signal
    "W": "Want",
    "X": "Anxious",
    "Y": "Yes",
    "Z": "Dizzy",
}


def get_word(letter: str) -> str:
    """Return the medical word mapped to the given letter.

    Parameters
    ----------
    letter : str
        A single uppercase letter A–Z.

    Returns
    -------
    str
        The corresponding medical word, or the letter itself
        if no mapping exists.
    """
    return GESTURE_MAP.get(letter.upper(), letter)


if __name__ == "__main__":
    # Quick sanity check
    for letter, word in GESTURE_MAP.items():
        print(f"{letter} → {word}")
