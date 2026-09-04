"""Authentication and authorization utilities.

This module provides authentication and authorization functionality for the RAG application,
including JWT token validation, user context management, and access control checks.

Validates Requirement 15.3: Users can only access their own conversation history
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import get_settings

logger = logging.getLogger(__name__)

# Security scheme for JWT bearer tokens
security = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when authorization check fails."""
    pass


def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token for a user.
    
    Args:
        user_id: User identifier
        expires_delta: Token expiration time (defaults to 24 hours)
        
    Returns:
        JWT token string
        
    Example:
        token = create_access_token(user_id="user123")
    """
    settings = get_settings()
    
    if expires_delta is None:
        expires_delta = timedelta(hours=24)
    
    expire = datetime.utcnow() + expires_delta
    
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    token = jwt.encode(payload, settings.jwt_secret, algorithm="HS256")
    return token


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        AuthenticationError: If token is invalid or expired
        
    Example:
        payload = decode_access_token(token)
        user_id = payload["sub"]
    """
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"]
        )
        
        # Verify token type
        if payload.get("type") != "access":
            raise AuthenticationError("Invalid token type")
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials]) -> Optional[str]:
    """Extract user ID from JWT token in request.
    
    Args:
        credentials: HTTP authorization credentials from request
        
    Returns:
        User ID if authenticated, None if anonymous
        
    Raises:
        HTTPException: If token is invalid
        
    Example:
        user_id = get_current_user(credentials)
        if user_id:
            print(f"Authenticated as {user_id}")
        else:
            print("Anonymous user")
    """
    if credentials is None:
        # No credentials provided - anonymous user
        return None
    
    try:
        token = credentials.credentials
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID"
            )
        
        return user_id
        
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


def get_user_id_from_request(request: Request) -> str:
    """Extract user ID from request, defaulting to anonymous.
    
    Attempts to extract user ID from JWT token in Authorization header.
    If no token is provided or token is invalid, returns 'anonymous'.
    
    Args:
        request: FastAPI request object
        
    Returns:
        User ID string (or 'anonymous' if not authenticated)
        
    Example:
        user_id = get_user_id_from_request(request)
    """
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        return "anonymous"
    
    try:
        token = auth_header.split(" ")[1]
        payload = decode_access_token(token)
        return payload.get("sub", "anonymous")
    except (AuthenticationError, IndexError):
        return "anonymous"


def check_conversation_access(user_id: str, conversation_user_id: str) -> None:
    """Check if user has access to a conversation.
    
    Validates Requirement 15.3: Users can only access their own conversation history
    
    Args:
        user_id: Current user's ID
        conversation_user_id: User ID associated with the conversation
        
    Raises:
        AuthorizationError: If user does not have access to the conversation
        
    Example:
        check_conversation_access(
            user_id="user123",
            conversation_user_id="user123"
        )  # OK
        
        check_conversation_access(
            user_id="user123",
            conversation_user_id="user456"
        )  # Raises AuthorizationError
    """
    if user_id != conversation_user_id:
        logger.warning(
            f"Access denied: user {user_id} attempted to access "
            f"conversation owned by {conversation_user_id}"
        )
        raise AuthorizationError(
            f"Access denied: you do not have permission to access this conversation"
        )


def require_authentication(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    """Require authentication and return user ID.
    
    Similar to get_current_user but raises an exception if user is not authenticated.
    
    Args:
        credentials: HTTP authorization credentials from request
        
    Returns:
        User ID
        
    Raises:
        HTTPException: If user is not authenticated
        
    Example:
        user_id = require_authentication(credentials)
        # Guaranteed to have a user_id here
    """
    user_id = get_current_user(credentials)
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    return user_id


# Utility function for logging security events
def log_access_attempt(
    user_id: str,
    resource_type: str,
    resource_id: str,
    action: str,
    allowed: bool
) -> None:
    """Log access control attempt for security auditing.
    
    Args:
        user_id: User attempting access
        resource_type: Type of resource (e.g., 'conversation')
        resource_id: ID of the resource
        action: Action being attempted (e.g., 'read', 'write')
        allowed: Whether access was allowed
        
    Example:
        log_access_attempt(
            user_id="user123",
            resource_type="conversation",
            resource_id="conv-456",
            action="read",
            allowed=True
        )
    """
    level = logging.INFO if allowed else logging.WARNING
    logger.log(
        level,
        f"Access {'granted' if allowed else 'denied'}: "
        f"user={user_id}, resource={resource_type}:{resource_id}, action={action}"
    )
