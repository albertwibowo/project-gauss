import gzip 

def calculate_ce(text1:str, text2:str) -> float:
    """Function to calculate cross entropy.

    The distance measure the 'effort' needed to change one
    text to another. The lower the value, the easier it 
    is to convert text A -> text B. This means text B belong
    to the same class as text A.

    Args:
        text1 (str): A text.
        text2 (str): A text.

    Returns:
        distance (float): cross entory distance 
    """
    text1 = str(text1)
    text2 = str(text2)
    combined_text = " ".join([text1, text2])
    c_text1 = len(gzip.compress(text1.encode()))
    c_combined = len(gzip.compress(combined_text.encode()))
    return c_combined - c_text1