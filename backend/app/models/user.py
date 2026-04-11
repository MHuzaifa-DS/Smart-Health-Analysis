"""models/user.py — User and auth schemas."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator
import re


class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_type: Optional[str] = None

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v and v not in ("male", "female", "other"):
            raise ValueError("Gender must be male, female, or other")
        return v

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if v and not (0 < v < 150):
            raise ValueError("Age must be between 1 and 149")
        return v


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_type: Optional[str] = None


class UserProfile(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_type: Optional[str] = None
    created_at: Optional[datetime] = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserProfile


class RefreshTokenRequest(BaseModel):
    refresh_token: str
