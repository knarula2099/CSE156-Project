# user.py - Handles user authentication & profile management
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import sqlite3

# Create Router
router = APIRouter()

# SQLite Database Connection
conn = sqlite3.connect("climate_app.db", check_same_thread=False)
cursor = conn.cursor()

# Create User Table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        interests TEXT
    )
""")
conn.commit()

# Pydantic Model for User Profile
class UserProfile(BaseModel):
    name: str
    interests: List[str]

# API Endpoints
@router.post("/save_profile/")
def save_profile(profile: UserProfile):
    """Login user if exists, otherwise create a new profile."""
    # Open a new database connection for each request
    conn = sqlite3.connect("climate_app.db")  
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT name FROM users WHERE name = ?", (profile.name,))
        existing_user = cursor.fetchone()

        if existing_user:
            return {"message": "User logged in successfully!", "status": "logged_in"}

        cursor.execute("INSERT INTO users (name, interests) VALUES (?, ?)", 
                       (profile.name, ", ".join(profile.interests)))
        conn.commit()
        return {"message": "Profile created successfully!", "status": "new_user"}
    finally:
        conn.close()  # Always close the connection to avoid memory leaks

@router.get("/get_profile/")
def get_profile(name: Optional[str] = None):
    """Retrieve user profile based on name."""
    if name:
        cursor.execute("SELECT name, interests FROM users WHERE name = ?", (name,))
    else:
        cursor.execute("SELECT name, interests FROM users LIMIT 1")
    
    user = cursor.fetchone()
    if user:
        return {"name": user[0], "interests": user[1].split(", "), "status": "logged_in"}
    
    return {"name": "", "interests": [], "status": "not_found"}