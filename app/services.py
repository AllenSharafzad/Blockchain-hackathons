import json
import os
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger("services")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Path to store history files
HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "history"
HISTORY_DIR.mkdir(exist_ok=True, parents=True)

def get_history_path(user_id):
    """Get the path to the history file for a user"""
    return HISTORY_DIR / f"{user_id}_history.json"

def handle_consent(data):
    """Handle user consent for data collection"""
    user_id = data.get("user_id", data.get("student_id", "default_user"))
    consent = data.get("consent", False)
    
    logger.info(f"Recording consent for user {user_id}: {consent}")
    
    # Create user history file if it doesn't exist
    history_path = get_history_path(user_id)
    if not history_path.exists():
        with open(history_path, "w") as f:
            json.dump({"user_id": user_id, "history": [], "consent": consent}, f)
    else:
        # Update consent if file exists
        try:
            with open(history_path, "r") as f:
                user_data = json.load(f)
            user_data["consent"] = consent
            with open(history_path, "w") as f:
                json.dump(user_data, f)
        except Exception as e:
            logger.error(f"Error updating consent: {str(e)}")
    
    return {"status": "success", "user_id": user_id}

def get_history(user_id):
    """Get the chat history for a user"""
    history_path = get_history_path(user_id)
    
    if not history_path.exists():
        logger.warning(f"History file for user {user_id} does not exist")
        return {"status": "error", "message": "No history found", "history": []}
    
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
        return {"status": "success", "history": data.get("history", [])}
    except Exception as e:
        logger.error(f"Error reading history for user {user_id}: {str(e)}")
        return {"status": "error", "message": str(e), "history": []}

def save_interaction(user_id, query, response):
    """Save an interaction to the user's history"""
    history_path = get_history_path(user_id)
    
    try:
        if history_path.exists():
            with open(history_path, "r") as f:
                data = json.load(f)
        else:
            data = {"user_id": user_id, "history": [], "consent": True}
        
        # Add the new interaction
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response
        }
        
        data["history"].append(interaction)
        
        # Save the updated history
        with open(history_path, "w") as f:
            json.dump(data, f)
            
        logger.info(f"Saved interaction for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving interaction for user {user_id}: {str(e)}")
        return False

def delete_history_item(user_id, index):
    """Delete a single history item"""
    history_path = get_history_path(user_id)
    
    if not history_path.exists():
        return {"status": "error", "message": "No history found"}
    
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
        
        if index < 0 or index >= len(data.get("history", [])):
            return {"status": "error", "message": "Invalid history index"}
        
        data["history"].pop(index)
        
        with open(history_path, "w") as f:
            json.dump(data, f)
            
        logger.info(f"Deleted history item {index} for user {user_id}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error deleting history item: {str(e)}")
        return {"status": "error", "message": str(e)}

def delete_all_history(user_id):
    """Delete all history for a user"""
    history_path = get_history_path(user_id)
    
    if not history_path.exists():
        return {"status": "error", "message": "No history found"}
    
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
        
        data["history"] = []
        
        with open(history_path, "w") as f:
            json.dump(data, f)
            
        logger.info(f"Deleted all history for user {user_id}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error deleting all history: {str(e)}")
        return {"status": "error", "message": str(e)}

def update_rating(user_id, message_id, rating):
    """Update the rating for a specific message"""
    history_path = get_history_path(user_id)
    
    if not history_path.exists():
        logger.warning(f"History file for user {user_id} does not exist")
        return {"status": "error", "message": "No history found"}
    
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
            
        # Find the message by ID or index
        # Since we don't have a direct message ID, we'll use the index as an approximation
        history = data.get("history", [])
        if message_id < 0 or message_id >= len(history):
            logger.warning(f"Invalid message ID {message_id} for user {user_id}")
            return {"status": "error", "message": "Invalid message ID"}
            
        # Add rating to the message
        history[message_id]["rating"] = rating
        
        # Save updated history
        with open(history_path, "w") as f:
            json.dump(data, f)
            
        logger.info(f"Updated rating for message {message_id} to {rating} for user {user_id}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error updating rating: {str(e)}")
        return {"status": "error", "message": str(e)}

def update_feedback(user_id, message_id, feedback):
    """Update the feedback for a specific message"""
    history_path = get_history_path(user_id)
    
    if not history_path.exists():
        logger.warning(f"History file for user {user_id} does not exist")
        return {"status": "error", "message": "No history found"}
    
    try:
        with open(history_path, "r") as f:
            data = json.load(f)
            
        # Find the message by ID or index
        history = data.get("history", [])
        if message_id < 0 or message_id >= len(history):
            logger.warning(f"Invalid message ID {message_id} for user {user_id}")
            return {"status": "error", "message": "Invalid message ID"}
            
        # Add feedback to the message
        history[message_id]["feedback"] = feedback
        
        # Save updated history
        with open(history_path, "w") as f:
            json.dump(data, f)
            
        logger.info(f"Updated feedback for message {message_id} to '{feedback}' for user {user_id}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error updating feedback: {str(e)}")
        return {"status": "error", "message": str(e)}