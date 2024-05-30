from fastapi import Depends, HTTPException, status
from ..accounts.model import User
from .auth import get_current_user

# from config import ...


# Define an asynchronous function for checking user permissions
def has_permission(permission_code: str):
    """
    Check if a user has the specified permission.
    This function is a dependency that checks if the current user has a specific permission.
    If the user does not have the permission, it raises an HTTPException with a 403 status code.
    Parameters:
        - permission (str): The permission code to check.
    Returns:
        - function: An asynchoronous function that checks user permissions.
    """

    def _has_permission(user: User = Depends(get_current_user)):
        # Check if the user has the specified permission.
        has_permission = user.has_permission(permission_code, user)
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied"
            )
        return user

    return _has_permission


# Commit changes to the database
