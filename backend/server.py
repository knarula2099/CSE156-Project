from fastapi import FastAPI, APIRouter
from frontend.components.user import router as user_router
from pydantic import BaseModel
import sqlite3

app = FastAPI()
router = APIRouter()

# Register User Routes
app.include_router(user_router, prefix="/user", tags=["User Management"])

# SQLite Connection for Discussions
conn = sqlite3.connect("climate_app.db", check_same_thread=False)
cursor = conn.cursor()


# Create Discussion Table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS discussions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        topic TEXT UNIQUE
    )
""")

# Create Replies Table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS replies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        discussion_id INTEGER,
        username TEXT,
        reply_text TEXT,
        FOREIGN KEY (discussion_id) REFERENCES discussions (id)
    )
""")
conn.commit()

# Pydantic Models
class Discussion(BaseModel):
    username: str
    topic: str

class Reply(BaseModel):
    username: str
    topic: str
    reply_text: str

# Add a New Discussion
@app.post("/add_discussion/")
def add_discussion(discussion: Discussion):
    cursor.execute("INSERT OR IGNORE INTO discussions (username, topic) VALUES (?, ?)", 
                   (discussion.username, discussion.topic))
    conn.commit()
    return {"message": "Discussion added successfully!"}

# Get All Discussions
@app.get("/get_discussions/")
def get_discussions():
    cursor.execute("SELECT username, topic FROM discussions")
    discussions = [{"username": row[0], "topic": row[1]} for row in cursor.fetchall()]
    return {"discussions": discussions}

# Add a Reply
@app.post("/add_reply/")
def add_reply(reply: Reply):
    cursor.execute("SELECT id FROM discussions WHERE topic = ?", (reply.topic,))
    discussion_id = cursor.fetchone()
    
    if discussion_id:
        cursor.execute("INSERT INTO replies (discussion_id, username, reply_text) VALUES (?, ?, ?)", 
                       (discussion_id[0], reply.username, reply.reply_text))
        conn.commit()
        return {"message": "Reply added successfully!"}
    
    return {"error": "Discussion topic not found!"}

# Get Replies for a Topic
@app.get("/get_replies/")
def get_replies(topic: str):
    cursor.execute("SELECT r.username, r.reply_text FROM replies r JOIN discussions d ON r.discussion_id = d.id WHERE d.topic = ?", (topic,))
    replies = [{"username": row[0], "reply": row[1]} for row in cursor.fetchall()]
    return {"topic": topic, "replies": replies}