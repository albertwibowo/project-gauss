import gzip


def calculate_ncd(text1: str, text2: str) -> float:
    """Function to calculate normalised compression distance.

    The distance measure the extra 'effort' needed to change one
    text to another. The lower the value, the easier it
    is to convert text A -> text B. This means text B belong
    to the same class as text A.

    Args:
        text1 (str): A text.
        text2 (str): A text.

    Returns:
        distance (float): Normalised compression distance
    """
    text1 = str(text1)
    text2 = str(text2)
    combined_text = "".join([text1, text2])
    c_text1 = len(gzip.compress(text1.encode()))
    c_text2 = len(gzip.compress(text2.encode()))
    c_combined = len(gzip.compress(combined_text.encode()))

    return (c_combined - min(c_text1, c_text2)) / max(c_text1, c_text2)
