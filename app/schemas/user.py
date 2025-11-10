from pydantic import BaseModel, EmailStr
from datetime import datetime

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

    class Config:
        orm_mode = True

    
