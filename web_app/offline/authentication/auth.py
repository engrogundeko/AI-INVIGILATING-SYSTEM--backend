from datetime import datetime, timedelta
from typing import Annotated
from pydantic import BaseModel
from jose import JWTError, jwt

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

from ..repository import userRespository, userRoleRepository
from ..config import ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY, ALGORITHM


# Defining an OAuth2 password bearer schema for token authentication
oauth2_schema = OAuth2PasswordBearer(tokenUrl="token")

# Creating a password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated=["auto"])


# Defining a Pydantic model for the Token
class Token(BaseModel):
    access_token: str
    token_type: str


# Defining a Pydantic model for the TokenData
class TokenData(BaseModel):
    email: str | None = None


def get_user(email: str):
    return userRespository.find_one({"email": email})


async def get_user_role(token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
    user_email = get_current_user(token)
    user = userRespository.find_one({"email": user_email})
    role = userRoleRepository.find_one({"user_id": user["_id"]})
    return role


def get_current_user(token: Annotated[str, Depends(oauth2_schema)]):
    """
    Get the current user based on the provided JWT token.
    This function decodes the JWT token using the secret key and verifies the user's credentials.
    Parameters:
        token (str): The JWT token for authentication.
    Raises:
        HTTPException:
            - If the token cannot be decoded or is invalid.
            - If the username retrieved from the token is None.
            - If the user does not exist in the system.
    Returns:
        User: An instance of the User model representing the authenticated user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    return token_data.email


def hash_password(password):
    """This hash password into another format"""
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(email: str, password: str):
    """
    Authenticate a user based on the provided username and password.
    This function attempts to authenticate a user by checking if the username exists
    in the database and if the provided password matches the stored password hash.
    Parameters:
        username (str): The username of the user to be authenticated.
        password (str): The password provided for authentication.
    Returns:
        User or bool:
            - If authentication is successful, it returns an instance of the User model
              representing the authenticated user.
            - If authentication fails (due to incorrect username, password, or other errors),
              it returns False.
    """
    try:
        user = get_user(email)
        if not user:
            return False
        if not verify_password(password, user.email):
            return False
        return user
    except Exception as e:
        return False


def _create_access_token(data: dict, expires_delta: timedelta | None = None):
    """
    Create an access token for user authentication.
    This function generates an access token by encoding the provided data dictionary
    (typically containing user information) with an optional expiration time.
    Parameters:
        data (dict): A dictionary containing user data to be encoded into the token.
        expires_delta (timedelta | None): Optional. The time duration for which the token
            should be valid. If not provided, a default expiration time of 15 minutes
            from the current time is used.
    Returns:
        str: The generated access token as a string.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
        # print(expire)
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
        # print(expire)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_access_token(email: str):
    """
    Generate an access token for the specified username.

    This function generates an access token for a user based on their username.
    The token is created with an expiration time determined by the value of
    ACCESS_TOKEN_EXPIRE_MINUTES and is of type "bearer."
    Parameters:
        username (str): The username for which the access token is generated.
    Returns:
        dict: A dictionary containing the generated access token and its type.
            Example:
            {
                "access_token": "your_generated_token_here",
                "token_type": "bearer"
            }
    """
    access_token_expires = timedelta(minutes=int(ACCESS_TOKEN_EXPIRE_MINUTES))
    access_token = _create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
