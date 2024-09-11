import re

def session_validator(session: str) -> bool:
    """
    Validate that the session is in the format YYYY/YYYY and the second year is one greater than the first.

    Args:
        session (str): The session string to validate.

    Returns:
        bool: True if the session is valid, False otherwise.
    """
    # Regular expression to match the format 'YYYY/YYYY'
    pattern = re.compile(r'^\d{4}/\d{4}$')

    # Check if the session matches the pattern
    if not pattern.match(session):
        return False

    # Split the session into start year and end year
    start_year, end_year = map(int, session.split('/'))

    # Check if the end year is exactly one greater than the start year
    if end_year != start_year + 1:
        return False

    return True
