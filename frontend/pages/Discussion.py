import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"  # Change this to match your FastAPI URL

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
        response = requests.post(f"{API_URL}/user/save_profile/", json={"name": username, "interests": interests})

        # Check response before parsing JSON
        if response.status_code == 200:
            try:
                data = response.json()
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_interests = interests
                st.success(data["message"])
            except requests.exceptions.JSONDecodeError:
                st.error("Error decoding JSON response from server.")
        else:
            st.error(f"Failed to save profile! Status: {response.status_code}, Response: {response.text}")


    st.markdown("---")

    # Form to add a new discussion topic
    with st.form("add_discussion_form", clear_on_submit=True):
        new_topic = st.text_area("Enter a discussion topic:", key="new_topic_input")
        submitted = st.form_submit_button("Add Discussion")

        if submitted and new_topic and username:
            response = requests.post(f"{API_URL}/add_discussion/", json={"username": username, "topic": new_topic})
            if response.status_code == 200:
                st.success("Discussion added successfully!")
            else:
                st.error("Failed to add discussion!")

    # Fetch and display all discussions
    st.subheader("📢 Existing Discussions")

    try:
        response = requests.get(f"{API_URL}/get_discussions/")
        
        if response.status_code == 200:
            discussions_data = response.json()  # Decode JSON safely
            discussions = discussions_data.get("discussions", [])  # Handle missing key

            if discussions:
                for discussion in discussions:
                    with st.expander(f"🔹 {discussion['topic']} (by {discussion['username']})"):
                        # Fetch replies
                        try:
                            reply_response = requests.get(f"{API_URL}/get_replies/", params={"topic": discussion["topic"]})
                            
                            if reply_response.status_code == 200:
                                replies_data = reply_response.json()
                                replies = replies_data.get("replies", [])  # Handle missing key

                                if replies:
                                    for reply in replies:
                                        st.write(f"💬 **{reply['username']}:** {reply['reply']}")
                                else:
                                    st.write("No replies yet.")

                        except requests.exceptions.RequestException as e:
                            st.error(f"Error fetching replies: {e}")

                        # Reply form
                        with st.form(f"reply_form_{discussion['topic']}", clear_on_submit=True):
                            reply_text = st.text_area(f"Reply to {discussion['topic']}:", key=f"reply_{discussion['topic']}")
                            reply_submitted = st.form_submit_button("Submit Reply")

                            if reply_submitted and reply_text and username:
                                reply_payload = {
                                    "username": username, 
                                    "topic": discussion["topic"], 
                                    "reply_text": reply_text
                                }

                                try:
                                    reply_post_response = requests.post(f"{API_URL}/add_reply/", json=reply_payload)

                                    if reply_post_response.status_code == 200:
                                        st.success("Reply added successfully!")
                                    else:
                                        st.error("Failed to add reply!")

                                except requests.exceptions.RequestException as e:
                                    st.error(f"Error posting reply: {e}")

            else:
                st.write("No discussions available yet.")

        else:
            st.error(f"Failed to fetch discussions! Status Code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching discussions: {e}")