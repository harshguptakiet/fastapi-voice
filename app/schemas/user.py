from pydantic import BaseModel, EmailStr, ConfigDict
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    name: str
    email: EmailStr
    profile_picture: str

class UserCreate(UserBase):
    """Schema used when creating a new user"""
    pass

class UserOut(UserBase):
    """Schema used when returning user data"""
    id: int
    created_at: datetime
    # Pydantic v2: enable ORM mode
    model_config = ConfigDict(from_attributes=True)
