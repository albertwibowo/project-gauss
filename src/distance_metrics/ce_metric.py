def calculate_ce(cx1: int, cx1cx2: int):
    """Function to calculate normalised compression distance.

    The distance measure the 'effort' needed to change one
    text to another. The lower the value, the easier it 
    is to convert text A -> text B. This means text B belong
    to the same class as text A.

    Args:
        cx1 (int): Length of compressed text - text to be classifed.
        cx1cx2 (int): Combined length of cx1 and cx2 where cx2 is a document.

    Returns:
        distance (float): cross entory distance 
    """
    return cx1cx2 - cx1