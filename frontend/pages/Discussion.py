import streamlit as st
from supabase import create_client, Client
import os

# Supabase credentials
SUPABASE_URL = 'https://cmdqgmqilcydypbfmjis.supabase.co'
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNtZHFnbXFpbGN5ZHlwYmZtamlzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI1MTMwMDIsImV4cCI6MjA1ODA4OTAwMn0.PZbgFvtVo2Ef-qZ9NlwO58orKnNOC6QlEs6lEVwNsdo"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_user_profile(username, interests):
    """Save user profile in the 'users' table."""
    response = supabase.table("users").upsert({
        "username": username,
        "interests": interests
    }).execute()
    
    return response

def add_discussion(username, topic):
    """Add a new discussion topic."""
    response = supabase.table("discussions").insert({
        "username": username,
        "topic": topic
    }).execute()
    
    return response

def fetch_discussions():
    """Fetch all discussions from the 'discussions' table."""
    response = supabase.table("discussions").select("*").execute()
    return response.data if response.data else []


def add_reply(discussion_id, username, reply_text):
    """Add a reply to a discussion."""
    response = supabase.table("replies").insert({
        "discussion_id": discussion_id,
        "username": username,
        "reply_text": reply_text
    }).execute()
    
    return response

def fetch_replies(discussion_id):
    """Fetch replies for a specific discussion."""
    response = supabase.table("replies").select("*").eq("discussion_id", discussion_id).execute()
    return response.data if response.data else []


def discussion_ui():
    """Renders the discussion page."""
    st.title("🗣️ Discussion Forum")

    # Initialize session state for login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "user_interests" not in st.session_state:
        st.session_state.user_interests = []

    # User profile section
    st.subheader("👤 User Profile")
    username = st.text_input("Enter your username:", value=st.session_state.username)

    # Save username in session state
    if username:
        st.session_state.username = username

    interests = st.multiselect("Select your research interests", 
                               ["Renewable Energy", "Carbon Pricing", "Climate Adaptation",
                                "Sustainable Agriculture", "Biodiversity", "Sea Level Rise"],
                               default=st.session_state.get("user_interests", []))

    # Save Profile = Login or Create New User
    if st.button("Save Profile (Login)"):
        response = save_user_profile(username, interests)
        
        if response:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_interests = interests
            st.success("Profile saved successfully!")
        else:
            st.error("Failed to save profile.")

    st.markdown("---")

    # Form to add a new discussion topic
    with st.form("add_discussion_form", clear_on_submit=True):
        new_topic = st.text_area("Enter a discussion topic:", key="new_topic_input")
        submitted = st.form_submit_button("Add Discussion")

        if submitted and new_topic and username:
            response = add_discussion(username, new_topic)
            if response:
                st.success("Discussion added successfully!")
            else:
                st.error("Failed to add discussion!")


    # Fetch and display all discussions
    st.subheader("📢 Existing Discussions")
    discussions = fetch_discussions()

    if discussions:
        for discussion in discussions:
            with st.expander(f"🔹 {discussion['topic']} (by {discussion['username']})"):
                discussion_id = discussion["id"]
                
                # Fetch replies
                replies = fetch_replies(discussion_id)
                if replies:
                    for reply in replies:
                        st.write(f"💬 **{reply['username']}:** {reply['reply_text']}")
                else:
                    st.write("No replies yet.")

                # Reply form
                with st.form(f"reply_form_{discussion_id}", clear_on_submit=True):
                    reply_text = st.text_area(f"Reply to {discussion['topic']}:", key=f"reply_{discussion_id}")
                    reply_submitted = st.form_submit_button("Submit Reply")

                    if reply_submitted and reply_text and username:
                        response = add_reply(discussion_id, username, reply_text)
                        if response:
                            st.success("Reply added successfully!")
                            st.session_state.reply_submitted = True
                            st.rerun()
                        else:
                            st.error("Failed to add reply!")

    else:
        st.write("No discussions available yet.")
