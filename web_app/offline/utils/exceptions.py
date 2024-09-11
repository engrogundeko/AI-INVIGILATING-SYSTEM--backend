class FacialVerificationException(Exception):
    def __init__(self, message="Facial verification failed.") -> None:
        super().__init__(message)
