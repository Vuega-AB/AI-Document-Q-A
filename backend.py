import os
import requests
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import PyPDF2
import faiss
import time # Make sure time is imported
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import json
from dotenv import load_dotenv
from io import BytesIO
from together import Together
import re
from pymongo import MongoClient, server_api
import logging
from openai import OpenAI
import sys
import httpx
from urllib.parse import urljoin
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import dropbox
import hashlib
import io
import tempfile
import bcrypt
import gradio
from datetime import datetime # Make sure datetime is imported
from datetime import datetime, timezone # MODIFIED: Added timezone
import random
import string
from datetime import datetime, timezone, timedelta
import pyotp

def generate_otp(length=6):
    return "".join(random.choices(string.digits, k=length))
# --- Environment Variables & Initializations ---
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MongoDB") 
if not MONGO_URI:
    MONGO_URI = os.getenv("MONGO_URI")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")

CONFIG_FILENAME = "app_config_main.json"
INDEX_FILE_DROPBOX = "/faiss_index.index"
TEXT_FILE_DROPBOX = "/text_store.json"
TOKEN_FILE = "dropbox_token.json"
MONGO_DB_NAME = "IntelLawDB_Gradio"
FAISS_COLLECTION_NAME = "faiss_index_store"
TEXT_STORE_COLLECTION_NAME = "text_content_store"
ADMIN_USERS_COLLECTION_NAME = "admin_users" # User credentials and status

# --- Global Variables for Backend State ---
gemini_model_genai = None
together_client = None
openai_client = None
dbx = None
mongo_client_instance = None
mongo_db_obj = None
auth_mongo_db_obj = None
embedding_model = None
faiss_index = None
text_store = []
BACKEND_INITIAL_LOAD_MSG = "Backend not initialized."

AVAILABLE_MODELS_DICT = {
    "gemini-1.5-flash-latest": {"price": "Custom", "type": "gemini", "name": "Gemini 1.5 Flash"},
    "openai-gpt-4o": {"price": "Custom", "type": "openai", "name": "OpenAI GPT-4o"},
    "meta-llama/Llama-3-70B-Instruct-hf": {"price": "$0.90", "type": "together", "name": "Llama3 70B Instruct (HF)"},
    "meta-llama/Llama-3-8B-Instruct-hf": {"price": "$0.20", "type": "together", "name": "Llama3 8B Instruct (HF)"},
    "microsoft/WizardLM-2-8x22B": {"price": "$1.80", "type": "together", "name": "WizardLM-2 8x22B"},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"price": "$1.20", "type": "together", "name": "Mixtral 8x22B Instruct"},
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"price": "$0.60", "type": "together", "name": "Hermes-2 Mixtral DPO"},
}
AVAILABLE_MODELS_NAMES = [details['name'] for details in AVAILABLE_MODELS_DICT.values()]
MODEL_NAME_TO_ID_MAP = {details['name']: model_id for model_id, details in AVAILABLE_MODELS_DICT.items()}
MODEL_ID_TO_NAME_MAP = {v: k for k, v in MODEL_NAME_TO_ID_MAP.items()}

# --- Status Constants ---
STATUS_PENDING_EMAIL_CONFIRMATION = "pending_email_confirmation"
STATUS_ACTIVE = "active"
STATUS_SUSPENDED = "suspended"

# --- Password Hashing Functions ---
def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(plain_password: str, hashed_password_bytes: bytes) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password_bytes)

# --- MongoDB User Authentication Functions ---
def get_auth_db():
    global auth_mongo_db_obj, mongo_client_instance
    if auth_mongo_db_obj is None:
        if mongo_client_instance is None:
            initialize_mongodb_client() # Ensures client is initialized
        if mongo_client_instance:
            # Use the same DB instance for auth, just different collections
            auth_mongo_db_obj = mongo_client_instance[MONGO_DB_NAME] 
    return auth_mongo_db_obj

def create_admin_user_if_not_exists(email, plain_password, role="admin"):
    db = get_auth_db()
    if db is None:
        print("Error: MongoDB for auth not available. Cannot create/update admin user.")
        return False
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    
    user = users_collection.find_one({"email": email})
    current_dt = datetime.now(timezone.utc) # MODIFIED: Use datetime

    if user:
        print(f"User {email} already exists.")
        if user.get("status") != STATUS_ACTIVE or user.get("role") != "admin":
            users_collection.update_one(
                {"email": email},
                {"$set": {"status": STATUS_ACTIVE, "role": "admin", "updated_at": current_dt}} # MODIFIED
            )
            print(f"Updated user {email} to ensure admin role and active status.")
        return True
    
    hashed_pass = hash_password(plain_password)
    try:
        users_collection.insert_one({
            "email": email,
            "password": hashed_pass,
            "role": role,
            "status": STATUS_ACTIVE, # Admins are active by default
            "created_at": current_dt, # MODIFIED
            "updated_at": current_dt, # MODIFIED
            "full_name": email.split('@')[0] # Added for consistency
        })
        print(f"Admin user {email} created successfully with active status.")
        return True
    except Exception as e:
        print(f"Error creating admin user {email}: {e}")
        return False


def create_user(email, plain_password):
    """
    Creates a new user with 'pending_email_confirmation' status,
    generates an OTP, and stores it.
    Returns: (success_bool, message_str, otp_str_or_None)
    """
    db = get_auth_db()
    if db is None:
        return False, "Database error, please try again later.", None
    
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    email_lower = email.lower()
    existing_user = users_collection.find_one({"email": email_lower})
    current_dt = datetime.now(timezone.utc)
    
    otp = generate_otp()
    otp_expiry = current_dt + timedelta(minutes=10) # OTP valid for 10 minutes

    if existing_user:
        if existing_user.get("status") == STATUS_PENDING_EMAIL_CONFIRMATION:
            # User exists but email not confirmed. Update password, regenerate OTP.
            hashed_pass = hash_password(plain_password)
            users_collection.update_one(
                {"email": email_lower},
                {"$set": {
                    "password": hashed_pass,
                    "updated_at": current_dt,
                    "created_at": current_dt, # Optionally refresh for new OTP window
                    "email_otp": otp,          # Update OTP
                    "otp_expires_at": otp_expiry # Update OTP expiry
                }}
            )
            print(f"User {email_lower} re-attempted signup. OTP regenerated.")
            return True, "An OTP has been re-sent to your email.", otp
        else: # User exists and is in another state (active, suspended)
            return False, "Email address already registered and confirmed or in another state.", None

    hashed_pass = hash_password(plain_password)
    try:
        user_doc_fields = {
            "email": email_lower,
            "password": hashed_pass,
            "role": "user",
            "status": STATUS_PENDING_EMAIL_CONFIRMATION,
            "created_at": current_dt,
            "updated_at": current_dt,
            "full_name": email_lower.split('@')[0],
            "email_otp": otp,
            "otp_expires_at": otp_expiry,
            "has_completed_initial_login": False, # << NEW
            "is_2fa_enabled": False,       # << NEW: 2FA status
            "totp_secret": None
        }
        users_collection.insert_one(user_doc_fields)
        print(f"User {email_lower} registered with {STATUS_PENDING_EMAIL_CONFIRMATION} status. OTP generated.")
        return True, "User record created. An OTP has been sent to your email.", otp
    except Exception as e:
        print(f"Error creating user {email_lower}: {e}")
        return False, "An error occurred during registration.", None

# def mark_initial_login_complete(email):
#     db = get_auth_db()
#     if not db: return False
#     users_collection = db[ADMIN_USERS_COLLECTION_NAME]
#     result = users_collection.update_one(
#         {"email": email.lower()},
#         {"$set": {"has_completed_initial_login": True, "updated_at": datetime.now(timezone.utc)}}
#     )
#     return result.modified_count > 0

def set_user_totp_secret(email, totp_secret):
    """Stores the TOTP secret for a user (usually before it's fully enabled)."""
    db = get_auth_db()
    if db is None:  # <<< CORRECTED CHECK
        print("Error: Auth DB not available in set_user_totp_secret.")
        return False, "Database connection error. Failed to set TOTP secret."
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    try:
        result = users_collection.update_one(
            {"email": email.lower()},
            {"$set": {"totp_secret": totp_secret, "is_2fa_enabled": False, "updated_at": datetime.now(timezone.utc)}}
        )
        if result.modified_count > 0 or result.matched_count > 0 : # matched_count in case it was already set to the same
            return True, "TOTP secret set/updated."
        else: # User not found
            return False, "User not found. Failed to set TOTP secret."
    except Exception as e:
        print(f"Error in set_user_totp_secret for {email}: {e}")
        return False, "Database operation error."
    
def enable_user_2fa(email):
    """Enables 2FA for the user, stores hashed recovery codes, and marks initial login as complete."""
    db = get_auth_db()
    if db is None:  # <<< CORRECTED CHECK
        print("Error: Auth DB not available in enable_user_2fa.")
        return False, "Database connection error. Failed to enable 2FA."
    
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    email_lower = email.lower() # Ensure consistent casing
    user = users_collection.find_one({"email": email_lower})

    if not user: # Added check for user existence
        print(f"Error: User {email_lower} not found in enable_user_2fa.")
        return False, "User not found. Cannot enable 2FA."
        
    if not user.get("totp_secret"): # Prerequisite check
        print(f"Error: TOTP secret not set for {email_lower} before enabling 2FA.")
        return False, "Cannot enable 2FA: Critical setup step (TOTP secret) missing."

    try:
        result = users_collection.update_one(
            {"email": email_lower}, # Use email_lower here too
            {"$set": {
                "is_2fa_enabled": True,
                "has_completed_initial_login": True, 
                "updated_at": datetime.now(timezone.utc)
            }}
        )
        if result.modified_count > 0:
            print(f"2FA enabled for user {email_lower}.")
            return True, "2FA enabled successfully."
        elif result.matched_count > 0 and user.get("is_2fa_enabled") is True: # Already enabled with same codes?
            print(f"2FA was already enabled for {email_lower}, considered success.")
            return True, "2FA was already enabled." # Or a more specific message
        else:
            print(f"Failed to enable 2FA for {email_lower}. Matched: {result.matched_count}, Modified: {result.modified_count}")
            return False, "Failed to update record to enable 2FA."
            
    except Exception as e:
        print(f"Database error in enable_user_2fa for {email_lower}: {e}")
        return False, "Database operation error during 2FA enabling."
    
def disable_user_2fa(email):
    """Disables 2FA for the user."""
    db = get_auth_db()
    if not db: return False, "Database error."
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    result = users_collection.update_one(
        {"email": email.lower()},
        {"$set": {
            "is_2fa_enabled": False,
            "totp_secret": None,
            "updated_at": datetime.now(timezone.utc)
        }}
    )
    return result.modified_count > 0, "2FA disabled." if result.modified_count > 0 else "Failed to disable 2FA."

def verify_totp_code(totp_secret, submitted_code):
    """Verifies a TOTP code against the user's secret."""
    if not totp_secret or not submitted_code:
        return False
    totp = pyotp.TOTP(totp_secret)
    return totp.verify(submitted_code, valid_window=1) # Allow current, previous, and next window (e.g., +/- 30s)

# def generate_recovery_codes(count=10, length=10):
#     """Generates a list of unique recovery codes."""
#     codes = set()
#     characters = string.ascii_uppercase + string.digits
#     while len(codes) < count:
#         code = ''.join(random.choices(characters, k=length // 2)) + '-' + \
#                ''.join(random.choices(characters, k=length // 2))
#         codes.add(code)
#     return list(codes)

# def hash_recovery_code(code):
#     # Use bcrypt or another strong hash for recovery codes if storing them.
#     # For simplicity, if you show them once and don't re-show, you might not hash them
#     # in the DB but rather hash the user's input when they try to use one.
#     # Here, let's assume we store hashes.
#     return bcrypt.hashpw(code.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') # Store as string

# def verify_recovery_code(email, submitted_code):
#     """Verifies a recovery code and marks it as used."""
#     db = get_auth_db()
#     if not db: return False, "Database error."
#     users_collection = db[ADMIN_USERS_COLLECTION_NAME]
#     user = users_collection.find_one({"email": email.lower()})

#     if not user or not user.get("is_2fa_enabled"):
#         return False, "User not found or 2FA not enabled."

#     hashed_recovery_codes = user.get("recovery_codes", [])
#     used_recovery_codes = user.get("used_recovery_codes", [])

#     for hashed_code_str in hashed_recovery_codes:
#         hashed_code_bytes = hashed_code_str.encode('utf-8')
#         if bcrypt.checkpw(submitted_code.encode('utf-8'), hashed_code_bytes):
#             # Code is valid. Check if already used.
#             if hashed_code_str in used_recovery_codes:
#                 return False, "Recovery code already used."
            
#             # Mark as used
#             users_collection.update_one(
#                 {"_id": user["_id"]},
#                 {"$addToSet": {"used_recovery_codes": hashed_code_str}}
#             )
#             return True, "Recovery code accepted."
#     return False, "Invalid recovery code."


def verify_otp_and_activate_user(email, submitted_otp):
    """
    Verifies the submitted OTP for the given email and activates the user if valid.
    Returns: (success_bool, message_str)
    """
    db = get_auth_db()
    if db is None: return False, "Database error."
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    email_lower = email.lower()
    user = users_collection.find_one({"email": email_lower})

    if not user: return False, "User not found."
    if user.get("status") == STATUS_SUSPENDED: return True, "Account already created."

    if user.get("status") != STATUS_PENDING_EMAIL_CONFIRMATION:
        return False, "Account not awaiting email confirmation."

    stored_otp = user.get("email_otp")
    otp_expires_at = user.get("otp_expires_at") 

    if not stored_otp or not otp_expires_at:
        return False, "OTP not found or has an issue. Please request a new one."
    
    # --- FIX STARTS HERE ---
    # Ensure otp_expires_at is a datetime object and make it timezone-aware (assuming UTC)
    if not isinstance(otp_expires_at, datetime):
        # This case should ideally not happen if you store datetime objects.
        # If it's a string or timestamp, you'd need to parse it first.
        # For example, if it was a Unix timestamp (float):
        # otp_expires_at = datetime.fromtimestamp(otp_expires_at, timezone.utc)
        print(f"ERROR: otp_expires_at for {email_lower} is not a datetime object from DB: {type(otp_expires_at)}")
        return False, "Internal error with OTP expiry format. Please try again."

    if otp_expires_at.tzinfo is None:
        # If it's naive, assume it was stored as UTC and make it UTC-aware
        otp_expires_at = otp_expires_at.replace(tzinfo=timezone.utc)
    # --- FIX ENDS HERE ---

    current_time_utc = datetime.now(timezone.utc) # Get current UTC time once

    if otp_expires_at < current_time_utc:
        # Clear expired OTP
        users_collection.update_one({"_id": user["_id"]}, {"$unset": {"email_otp": "", "otp_expires_at": ""}})
        return False, "OTP has expired. Please request a new one."

    if stored_otp == submitted_otp:
        users_collection.update_one(
            {"_id": user["_id"]},
            {
                "$set": {"status": STATUS_SUSPENDED, "updated_at": current_time_utc}, # Use consistent current time
                "$unset": {"email_otp": "", "otp_expires_at": ""} 
            }
        )
        return True, "Email confirmed successfully! Your account is now active."
    else:
        return False, "Invalid OTP entered."

# Also, ensure in create_user and regenerate_otp_for_user, you are consistently using timezone-aware datetimes
# for otp_expires_at when storing them. Your current code `datetime.now(timezone.utc) + timedelta(...)`
# correctly creates timezone-aware datetimes. The issue is likely how PyMongo retrieves them.

# Consider a small helper in get_all_users_from_db and other places fetching datetimes:
def ensure_timezone_aware(dt_obj, default_tz=timezone.utc):
    if isinstance(dt_obj, datetime):
        if dt_obj.tzinfo is None:
            return dt_obj.replace(tzinfo=default_tz)
        return dt_obj # Already aware
    return dt_obj # Not a datetime, return as is

def regenerate_otp_for_user(email):
    """
    Regenerates an OTP for a user whose status is 'pending_email_confirmation'.
    Returns: (new_otp_str_or_None, message_str)
    """
    db = get_auth_db()
    if db is None: return None, "Database error."
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    email_lower = email.lower()
    
    # Find user and ensure they are in the correct state to receive a new OTP
    user = users_collection.find_one({"email": email_lower})
    if not user:
        return None, "User not found."
    if user.get("status") != STATUS_PENDING_EMAIL_CONFIRMATION:
        return None, "Account is not awaiting email confirmation (e.g., already active or suspended)."

    new_otp = generate_otp()
    new_otp_expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
    
    try:
        users_collection.update_one(
            {"_id": user["_id"]}, # Use _id for precision
            {"$set": {
                "email_otp": new_otp,
                "otp_expires_at": new_otp_expiry,
                "updated_at": datetime.now(timezone.utc)
            }}
        )
        print(f"OTP regenerated for {email_lower}.")
        return new_otp, "A new OTP has been generated."
    except Exception as e:
        print(f"Error regenerating OTP for {email_lower}: {e}")
        return None, "Failed to regenerate OTP."

# def confirm_user_email_in_db(email):
#     """Confirms a user's email and sets status to 'pending_admin_approval'."""
#     db = get_auth_db()
#     if db is None: return False, "Database error."
#     users_collection = db[ADMIN_USERS_COLLECTION_NAME]
#     user = users_collection.find_one({"email": email})
#     if not user: return False, "User not found."
    
#     # Check current status for idempotency or incorrect state
#     if user.get("status") == STATUS_SUSPENDED:
#         return True, "Email already confirmed and pending admin approval."
#     if user.get("status") == STATUS_ACTIVE:
#         return True, "Email already confirmed and account is active."
#     if user.get("status") != STATUS_PENDING_EMAIL_CONFIRMATION:
#         return False, f"Account is not awaiting email confirmation (current status: {user.get('status')})."

#     try:
#         result = users_collection.update_one(
#             {"email": email, "status": STATUS_PENDING_EMAIL_CONFIRMATION},
#             {"$set": {"status": STATUS_SUSPENDED, "updated_at": datetime.now(timezone.utc)}}
#         )
#         if result.modified_count > 0:
#             return True, "Email confirmed successfully. Your account is now pending admin approval."
#         return False, "Email confirmation failed or user not in correct state."
#     except Exception as e:
#         print(f"Error confirming email for {email}: {e}")
#         return False, "An error occurred during email confirmation."
def get_full_user_for_auth(email):
    """
    Fetches the full user document, including sensitive fields like password
    and TOTP secret, for authentication purposes.
    USE WITH CAUTION and only in authentication flows.
    """
    db = get_auth_db()
    if db is None:
        print(f"Auth DB not available when fetching full user for {email}")
        return None
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    return users_collection.find_one({"email": email.lower()})

def get_user_by_email(email):
    db = get_auth_db()
    if db is None:
        print("Error: MongoDB for auth not available. Cannot get user.")
        return None
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    return users_collection.find_one({"email": email})

def get_all_users_from_db():
    # ... (ensure this function correctly fetches all users and converts timestamps to datetime) ...
    # Make sure it returns the 'status' field as is from the DB.
    # The renaming of 'pending' to 'pending_admin_approval' will be reflected here if the DB stores it that way.
    db = get_auth_db()
    if db is None: return []
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    try:
        users_cursor = users_collection.find({})
        users_list = []
        for user_doc in users_cursor:
            user_doc['_id'] = str(user_doc['_id'])
            if 'password' in user_doc: del user_doc['password']
            for ts_field in ["created_at", "updated_at", "last_login_at"]: # Add deleted_at
                if ts_field in user_doc:
                    if isinstance(user_doc[ts_field], (int, float)):
                        user_doc[ts_field] = datetime.fromtimestamp(user_doc[ts_field], timezone.utc)
                    elif isinstance(user_doc[ts_field], datetime) and user_doc[ts_field].tzinfo is None:
                         user_doc[ts_field] = user_doc[ts_field].replace(tzinfo=timezone.utc)
            if 'full_name' not in user_doc or not user_doc['full_name']:
                user_doc['full_name'] = user_doc.get('email', "N/A").split('@')[0]
            if 'totp_secret' in user_doc: del user_doc['totp_secret']
            if 'recovery_codes' in user_doc: del user_doc['recovery_codes']
            if 'used_recovery_codes' in user_doc: del user_doc['used_recovery_codes']
            # You can include 'is_2fa_enabled'
            user_doc['is_2fa_enabled'] = user_doc.get('is_2fa_enabled', False)
            users_list.append(user_doc)
        return users_list
    except Exception as e:
        print(f"Error fetching all users: {e}"); return []

def update_user_status_in_db(user_email, new_status):
    """Updates the status of a user in the database."""
    db = get_auth_db()
    if db is None: return False, "Database not connected."
    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    
    # Admin should only be able to set these statuses via the dropdown
    admin_allowed_statuses = [STATUS_ACTIVE, STATUS_SUSPENDED]
    if new_status not in admin_allowed_statuses:
        return False, f"Invalid target status '{new_status}' for admin action."

    current_user_state = users_collection.find_one({"email": user_email})
    if not current_user_state:
        return False, f"User with email '{user_email}' not found."

    # Prevent setting to PENDING_EMAIL_CONFIRMATION or other non-admin-settable states
    if new_status == STATUS_PENDING_EMAIL_CONFIRMATION: # Double check
        return False, "Admin cannot set status to 'Pending Email Confirmation'."

    try:
        current_dt = datetime.now(timezone.utc)
        update_fields = {"status": new_status, "updated_at": current_dt}

        result = users_collection.update_one({"email": user_email}, {"$set": update_fields})
        
        if result.modified_count > 0:
            return True, f"User '{user_email}' status updated to '{new_status}'."
        elif result.matched_count == 1 and current_user_state.get("status") == new_status:
             return True, f"User status for '{user_email}' was already '{new_status}'. No change made."
        return False, f"User status for '{user_email}' could not be updated."
    except Exception as e:
        print(f"Error updating user status for {user_email}: {e}")
        return False, "An error occurred while updating user status."

def hard_delete_user_from_db(user_email): # <<< NEW FUNCTION
    """Permanently deletes a user from the database by their email."""
    db = get_auth_db()
    if db is None:
        return False, "Database not connected. User not deleted."

    users_collection = db[ADMIN_USERS_COLLECTION_NAME]
    user_to_delete = users_collection.find_one({"email": user_email})

    if not user_to_delete:
        return False, f"User with email '{user_email}' not found. No deletion performed."

    # IMPORTANT: Add checks here if there are users that should NEVER be hard-deleted
    # e.g., if user_email is the APP_ADMIN_EMAIL. This check is also done in Flask.
    # if user_email == "your_super_admin@example.com":
    # return False, "This critical admin account cannot be hard deleted."

    try:
        result = users_collection.delete_one({"email": user_email})
        if result.deleted_count == 1:
            print(f"User '{user_email}' HARD DELETED successfully from the database.")
            return True, f"User '{user_email}' has been permanently deleted."
        else:
            # This case means find_one found it, but delete_one didn't delete it.
            # Highly unlikely if find_one worked, unless there's a race condition or replica set issue.
            print(f"Warning: User '{user_email}' found but not hard deleted. Deleted count: {result.deleted_count}")
            return False, f"User '{user_email}' was found but could not be deleted. Please check logs."
    except Exception as e:
        print(f"Error hard deleting user {user_email} from database: {e}")
        return False, f"An error occurred while trying to permanently delete user '{user_email}'."

# --- Dropbox Functions ---
# ... (existing Dropbox functions: load_access_token, save_access_token, get_dropbox_access_token, get_valid_access_token, initialize_dropbox_client) ...
def load_access_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as file: data = json.load(file); return data.get("access_token"), data.get("expires_at")
    return None, None

def save_access_token(access_token, expires_in):
    expires_at = int(time.time()) + expires_in - 300
    with open(TOKEN_FILE, "w") as file: json.dump({"access_token": access_token, "expires_at": expires_at}, file)

def get_dropbox_access_token():
    if not (DROPBOX_APP_KEY and DROPBOX_APP_SECRET and DROPBOX_REFRESH_TOKEN):
        print("Error: Dropbox credentials (APP_KEY, APP_SECRET, REFRESH_TOKEN) not fully configured.")
        return None
    response = requests.post("https://api.dropbox.com/oauth2/token", data={"grant_type": "refresh_token", "refresh_token": DROPBOX_REFRESH_TOKEN}, auth=(DROPBOX_APP_KEY, DROPBOX_APP_SECRET))
    if response.status_code == 200:
        data = response.json()
        save_access_token(data["access_token"], data.get("expires_in", 14400))
        return data["access_token"]
    else:
        print(f"Failed to refresh Dropbox token: {response.text}")
        return None

def get_valid_access_token():
    access_token, expires_at = load_access_token()
    if access_token and expires_at and int(time.time()) < expires_at: return access_token
    return get_dropbox_access_token()

def initialize_dropbox_client():
    global dbx
    if not (DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET):
        print("Warning: Dropbox credentials not fully set. Dropbox features disabled.")
        dbx = None; return
    try:
        access_token = get_valid_access_token()
        if access_token:
            dbx = dropbox.Dropbox(access_token)
            print("Dropbox client initialized successfully.")
        else:
            print("Failed to obtain Dropbox access token. Dropbox client not initialized.")
            dbx = None
    except Exception as e:
        print(f"Error connecting to Dropbox: {e}")
        dbx = None

# --- MongoDB Functions (for app data) ---
def initialize_mongodb_client():
    global mongo_client_instance, mongo_db_obj, auth_mongo_db_obj
    if MONGO_URI and mongo_client_instance is None:
        try:
            mongo_client_instance = MongoClient(MONGO_URI, server_api=server_api.ServerApi('1'))
            mongo_client_instance.admin.command('ping') # Verify connection
            mongo_db_obj = mongo_client_instance[MONGO_DB_NAME] # For app data (FAISS, text_store)
            auth_mongo_db_obj = mongo_client_instance[MONGO_DB_NAME] # For auth data (admin_users)
                                                                  # Using same DB, but conceptually could be different
            print("MongoDB client initialized successfully.")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            mongo_client_instance = None; mongo_db_obj = None; auth_mongo_db_obj = None
    elif not MONGO_URI:
        print("Warning: MONGO_URI not set. MongoDB features disabled.")
    # else:
        # print("MongoDB client already initialized or MONGO_URI not set.")


# --- Data Persistence Functions (FAISS, text_store) ---
# ... (existing: save_data_to_selected_db, load_data_from_selected_db) ...
def save_data_to_selected_db(selected_db):
    global faiss_index, text_store, dbx, mongo_db_obj

    if selected_db == "Dropbox" and dbx is None: initialize_dropbox_client()
    if selected_db == "MongoDB" and mongo_db_obj is None: initialize_mongodb_client() 

    index_to_save = faiss_index
    if index_to_save is None or embedding_model is None:
        print("Warning: FAISS index or embedding model is None. Cannot save empty or uninitialized index.")
        if embedding_model and index_to_save is None:
            index_to_save = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
            print("Created new empty FAISS index for saving.")
        else:
            return

    if selected_db == "Dropbox":
        if dbx is None: print("Dropbox not initialized for saving."); return
        try:
            if index_to_save.ntotal == 0:
                print("Skipping FAISS index save to Dropbox as it's empty.")
            else:
                temp_idx_file = "temp_faiss_to_dropbox.index"
                faiss.write_index(index_to_save, temp_idx_file)
                with open(temp_idx_file, "rb") as f: dbx.files_upload(f.read(), INDEX_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
                os.remove(temp_idx_file)
            
            text_json = json.dumps(text_store, ensure_ascii=False, indent=4).encode('utf-8')
            dbx.files_upload(text_json, TEXT_FILE_DROPBOX, mode=dropbox.files.WriteMode.overwrite)
            print(f"Data saved to Dropbox. Index size: {index_to_save.ntotal}, Text items: {len(text_store)}")
        except Exception as e: print(f"Error saving to Dropbox: {e}")
    elif selected_db == "MongoDB":
        if mongo_db_obj is None: print("MongoDB not initialized for saving app data."); return
        try:
            if index_to_save.ntotal == 0:
                 print("Skipping FAISS index save to MongoDB as it's empty.")
                 mongo_db_obj[FAISS_COLLECTION_NAME].delete_one({"_id": "main_faiss_index"})
            else:
                temp_idx_file = "temp_faiss_to_mongo.idx"
                faiss.write_index(index_to_save, temp_idx_file)
                with open(temp_idx_file, "rb") as f: index_bytes = f.read()
                os.remove(temp_idx_file)
                mongo_db_obj[FAISS_COLLECTION_NAME].update_one({"_id": "main_faiss_index"}, {"$set": {"index_data": index_bytes}}, upsert=True)

            mongo_db_obj[TEXT_STORE_COLLECTION_NAME].delete_many({})
            if text_store: mongo_db_obj[TEXT_STORE_COLLECTION_NAME].insert_many(text_store)
            print(f"Data saved to MongoDB. Index size: {index_to_save.ntotal}, Text items: {len(text_store)}")
        except Exception as e: print(f"Error saving to MongoDB: {e}")


def load_data_from_selected_db(selected_db):
    global faiss_index, text_store, embedding_model, dbx, mongo_db_obj
    if embedding_model is None:
        print("Error: Embedding model not initialized. Cannot load data.")
        return "Embedding model not initialized. Load failed."

    # Initialize fresh, empty structures
    # Ensure embedding_model.get_sentence_embedding_dimension() returns a valid integer
    try:
        dimension = embedding_model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"Invalid embedding dimension: {dimension}")
        faiss_index = faiss.IndexFlatL2(dimension)
    except Exception as e:
        print(f"Critical error initializing FAISS index with dimension: {e}")
        return f"Critical error initializing FAISS index: {e}. Load failed."

    text_store = []
    alert_msg = ""
    temp_idx_file_path = None # To ensure it's defined for finally block

    if selected_db == "Dropbox":
        # ... (your existing Dropbox code - ensure it also handles faiss.read_index errors robustly) ...
        if dbx is None: initialize_dropbox_client()
        if dbx is None: alert_msg = "Dropbox not initialized. Cannot load."; return alert_msg
        try:
            temp_idx_file_path = "temp_faiss_from_dropbox.index"
            _, res_index = dbx.files_download(path=INDEX_FILE_DROPBOX)
            with open(temp_idx_file_path, "wb") as f: f.write(res_index.content)
            loaded_index = faiss.read_index(temp_idx_file_path)
            if loaded_index.d == faiss_index.d:
                faiss_index = loaded_index
            else:
                alert_msg += f"Warning: Dropbox index dimension mismatch ({loaded_index.d} vs {faiss_index.d}). Not loading. "
            # os.remove(temp_idx_file_path) # Moved to finally

            _, res_text = dbx.files_download(path=TEXT_FILE_DROPBOX)
            text_store = json.loads(res_text.content.decode('utf-8'))
            alert_msg += f"Data loaded from Dropbox. {len(text_store)} text items, index has {faiss_index.ntotal} vectors."
        except dropbox.exceptions.ApiError as e:
            if isinstance(e.error, dropbox.files.DownloadError) and e.error.is_path() and e.error.get_path().is_not_found():
                alert_msg += "No existing data on Dropbox. Initialized empty store."
            else: alert_msg += f"Dropbox API error: {e}"
        except RuntimeError as faiss_error: # Catch FAISS read errors
            alert_msg += f"Error reading FAISS index from Dropbox: {faiss_error}. Index likely corrupt. Re-initializing. "
            # faiss_index remains the newly initialized empty one
        except Exception as e: alert_msg += f"Error loading from Dropbox: {e}"
        finally:
            if temp_idx_file_path and os.path.exists(temp_idx_file_path):
                os.remove(temp_idx_file_path)


    elif selected_db == "MongoDB":
        if mongo_db_obj is None: initialize_mongodb_client()
        if mongo_db_obj is None: alert_msg = "MongoDB not initialized for app data. Cannot load."; return alert_msg
        
        index_doc = None # Define before try block
        try:
            index_doc = mongo_db_obj[FAISS_COLLECTION_NAME].find_one({"_id": "main_faiss_index"})
            if index_doc and "index_data" in index_doc and index_doc["index_data"]: # Check if index_data exists and is not empty
                temp_idx_file_path = "temp_faiss_from_mongo.idx"
                with open(temp_idx_file_path, "wb") as f: f.write(index_doc["index_data"])
                
                # Before attempting to read, check if file has content
                if os.path.getsize(temp_idx_file_path) == 0:
                    raise RuntimeError("Temporary FAISS index file is empty after writing from MongoDB data.")

                loaded_index = faiss.read_index(temp_idx_file_path) # This is where the error occurs
                if loaded_index.d == faiss_index.d: # faiss_index is the newly initialized one
                    faiss_index = loaded_index
                    # alert_msg += f"FAISS index loaded from MongoDB. Dim: {faiss_index.d}, Vectors: {faiss_index.ntotal}. " # Becomes too verbose
                else:
                    alert_msg += f"Warning: MongoDB index dimension mismatch ({loaded_index.d} vs {faiss_index.d}). Not loading. "
            else:
                 alert_msg += "No FAISS index data found or data is empty in MongoDB. "
        
        except RuntimeError as faiss_error: # Catch FAISS-specific runtime errors
            alert_msg += f"Error reading FAISS index from MongoDB data: {faiss_error}. Index is likely corrupt. "
            if index_doc and "_id" in index_doc: # If we know the document ID that caused the error
                try:
                    delete_result = mongo_db_obj[FAISS_COLLECTION_NAME].delete_one({"_id": index_doc["_id"]})
                    if delete_result.deleted_count > 0:
                        alert_msg += "Corrupted FAISS index document deleted from MongoDB. "
                    else:
                        alert_msg += "Attempted to delete corrupted index, but document not found (or already deleted). "
                except Exception as db_del_err:
                    alert_msg += f"Error trying to delete corrupted index from MongoDB: {db_del_err}. "
            else:
                alert_msg += "Cannot identify specific MongoDB document for corrupted index to delete. "
            # faiss_index remains the newly initialized empty one
        except Exception as e: # Catch other general errors during index loading phase
            alert_msg += f"General error during MongoDB FAISS index loading: {e}. "
        finally:
            if temp_idx_file_path and os.path.exists(temp_idx_file_path):
                os.remove(temp_idx_file_path)

        # Load text_store regardless of index success/failure
        try:
            text_docs = list(mongo_db_obj[TEXT_STORE_COLLECTION_NAME].find({}))
            text_store = [{k: v for k, v in doc.items() if k != '_id'} for doc in text_docs]
            alert_msg += f"Loaded {len(text_store)} text items from MongoDB. "
        except Exception as e:
            alert_msg += f"Error loading text_store from MongoDB: {e}. "
            text_store = [] # Ensure text_store is empty on error

        # Final status message construction
        if faiss_index.ntotal > 0:
            alert_msg += f"Final Index: {faiss_index.ntotal} vectors. "
        else:
            alert_msg += f"Final Index: Empty. "

        if not (index_doc and "index_data" in index_doc and index_doc["index_data"]) and not text_docs:
             if "No FAISS index data found" in alert_msg and "0 text items" in alert_msg: # Check if already handled
                pass # Message likely already comprehensive
             else:
                alert_msg += "Initialized empty store as no data found in MongoDB. "


    # Consistency checks (these are good to keep)
    if faiss_index.ntotal > 0 and not text_store and "corrupt" not in alert_msg.lower():
        alert_msg += " Warning: Index has vectors but text store is empty. Data might be inconsistent. Clearing index to be safe."
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    elif faiss_index.ntotal == 0 and text_store and ("corrupt" not in alert_msg.lower() and "mismatch" not in alert_msg.lower()):
        alert_msg += " Warning: Text store loaded but index is empty/failed to load. Consider re-indexing. "

    final_status_message = f"DB Load Status ({selected_db}): {alert_msg.strip()}"
    print(final_status_message)
    return final_status_message

# --- PDF Processing ---
# ... (existing: chunk_text, extract_text_from_pdf_bytes, process_and_add_pdf_core) ...
def chunk_text(text, chunk_size=400, min_chunk_length=20):
    paragraphs = re.split(r'\n{2,}', text); chunks = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para); temp_chunk = ""
        for sentence in sentences:
            if len(temp_chunk) + len(sentence) < chunk_size: temp_chunk += sentence + " "
            else:
                cleaned_chunk = temp_chunk.strip()
                if len(cleaned_chunk) >= min_chunk_length: chunks.append(cleaned_chunk)
                temp_chunk = sentence + " "
        cleaned_chunk = temp_chunk.strip()
        if len(cleaned_chunk) >= min_chunk_length: chunks.append(cleaned_chunk)
    return chunks

def extract_text_from_pdf_bytes(pdf_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes)); text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        text += page_text + "\n" if page_text else ""
    return text

def process_and_add_pdf_core(pdf_bytes, file_name, selected_db):
    global text_store, faiss_index, embedding_model
    if embedding_model is None: return "Embedding model not initialized.", False
    if faiss_index is None: 
        if embedding_model: faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        else: return "FAISS index not initialized and embedding model missing.", False

    file_hash = hashlib.md5(pdf_bytes).hexdigest()
    if any(item.get('file_hash') == file_hash for item in text_store if isinstance(item, dict)):
        return f"File '{file_name}' (hash: {file_hash[:7]}) seems to already exist based on hash.", False
    
    raw_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not raw_text.strip():
        return f"No text could be extracted from '{file_name}'.", False
        
    chunks = chunk_text(raw_text)
    if not chunks:
        return f"Could not break '{file_name}' into processable chunks.", False

    embeddings = embedding_model.encode(chunks)
    embeddings_np = np.array(embeddings).astype("float32")
    if embeddings_np.shape[0] > 0: faiss_index.add(embeddings_np)
    
    for chunk_text_content in chunks:
        text_store.append({"text": chunk_text_content, "file_name": file_name, "file_hash": file_hash})
    return f"Processed and added '{file_name}'. Chunks: {len(chunks)}, Index size: {faiss_index.ntotal}", True

# --- AI Response Generation ---
# ... (existing: generate_response_gemini, generate_response_together_ai, generate_response_openai_api) ...
def generate_response_gemini(prompt, context, temp, top_p, system_prompt):
    if not gemini_model_genai: return "Gemini client not initialized."
    input_parts = [system_prompt + "\nContext: " + context, "Question: " + prompt]
    config = genai.GenerationConfig(max_output_tokens=2048, temperature=temp, top_p=top_p)
    try: response = gemini_model_genai.generate_content(input_parts, generation_config=config); return response.text
    except Exception as e: return f"Gemini Error: {e}"

def generate_response_together_ai(prompt, context, model_id, temp, top_p, system_prompt):
    if not together_client: return "TogetherAI client not initialized."
    try:
        response = together_client.chat.completions.create(
            model=model_id, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}],
            temperature=temp, top_p=top_p
        )
        return response.choices[0].message.content.strip()
    except Exception as e: return f"TogetherAI Error ({model_id}): {e}"

def generate_response_openai_api(prompt, context, temp, top_p, system_prompt): 
    if not openai_client: return "OpenAI client not initialized."
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}],
            temperature=temp, top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e: return f"OpenAI Error: {e}"

# --- RAG ---
# ... (existing: retrieve_context_from_db) ...
def retrieve_context_from_db(query, top_k=5):
    global text_store, faiss_index, embedding_model
    if embedding_model is None: return "Embedding model not initialized for RAG."
    if faiss_index is None or faiss_index.ntotal == 0: return "No documents indexed."
    
    query_embedding = embedding_model.encode([query])
    query_embedding_np = np.array(query_embedding).astype("float32")
    
    if query_embedding_np.shape[1] != faiss_index.d:
        return f"Query embedding dimension ({query_embedding_np.shape[1]}) does not match FAISS index dimension ({faiss_index.d})."

    distances, indices = faiss_index.search(query_embedding_np, top_k)
    
    valid_indices = [i for i in indices[0] if 0 <= i < len(text_store)]
    retrieved_texts = [text_store[idx]["text"] for idx in valid_indices if isinstance(text_store[idx], dict) and "text" in text_store[idx]]
    return "\n\n".join(retrieved_texts) if retrieved_texts else "No relevant context found."


# --- Web Scraping ---
# ... (existing web scraping functions) ...
async def fetch_page_async(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status() 
        return response.text, str(response.url)

async def extract_pdf_links_from_url_async(url):
    try:
        html, base_url_resolved = await fetch_page_async(url)
        soup = BeautifulSoup(html, "html.parser")
        return [urljoin(base_url_resolved, a["href"]) for a in soup.find_all("a", href=True) if ".pdf" in a["href"].lower()]
    except Exception as e:
        print(f"Error scraping {url} for PDF links: {e}")
        return []

async def process_scraped_pdf_links_async(urls):
    all_pdf_links = set()
    tasks = [extract_pdf_links_from_url_async(u) for u in urls]
    for url_group in await asyncio.gather(*tasks):
        all_pdf_links.update(url_group)
    return list(all_pdf_links)

async def download_and_process_scraped_pdf(session, pdf_link, selected_db):
    try:
        async with session.get(pdf_link, timeout=60) as response:
            if response.status == 200:
                pdf_bytes = await response.read()
                filename = os.path.basename(pdf_link)
                status_msg, success = process_and_add_pdf_core(pdf_bytes, filename, selected_db) 
                return status_msg, success, filename
            return f"Failed to download {pdf_link} (status: {response.status})", False, os.path.basename(pdf_link)
    except Exception as e:
        return f"Error processing {pdf_link}: {e}", False, os.path.basename(pdf_link)

async def batch_download_and_process_pdfs(pdf_links, selected_db):
    results = []
    connector = aiohttp.TCPConnector(ssl=False) 
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_and_process_scraped_pdf(session, link, selected_db) for link in pdf_links]
        for result in await asyncio.gather(*tasks):
            results.append(result)
    return results

def get_page_items_sync(url, base_url_val, listing_endpoint_val):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser"); items = set()
        for item in soup.find_all("a"):
            link = item.get("href")
            if link and f"/{listing_endpoint_val}/" in link and link != url and not link.endswith("/rss"):
                if not link.startswith("http"): link = urljoin(base_url_val, link)
                items.add(link)
        return list(items)
    except Exception as e: print(f"Error scraping page items {url}: {e}"); return []

def get_all_page_urls_for_scraping(base_url_val, listing_endpoint_val, pagination_format_val, num_pages_val):
    all_page_urls_to_scrape = set()
    for page_num in range(1, int(num_pages_val) + 1):
        url_path = f"{listing_endpoint_val}/{pagination_format_val}{page_num}"
        url = urljoin(base_url_val + ("/" if not base_url_val.endswith("/") else ""), url_path)
        page_items = get_page_items_sync(url, base_url_val, listing_endpoint_val)
        if not page_items and page_num > 1 : 
            print(f"No items found on page {page_num} ({url}), stopping pagination.")
            break
        all_page_urls_to_scrape.update(page_items)
        if not page_items and page_num == 1: 
             print(f"No items found on the first page ({url}). Check scraper settings.")
             break
    return list(all_page_urls_to_scrape)

# --- UI Callable Backend Functions ---
# ... (existing: get_unique_filenames_from_text_store, _build_file_list_updates, etc.) ...
def get_unique_filenames_from_text_store():
    global text_store
    if not text_store: return []
    unique_files = {} 
    for item in text_store:
        if isinstance(item, dict) and 'file_hash' in item and 'file_name' in item:
            if item["file_hash"] not in unique_files:
                unique_files[item["file_hash"]] = item["file_name"]
    return sorted(list(unique_files.values()))

def _build_file_list_updates():
    filenames = get_unique_filenames_from_text_store()
    md_output = "#### Current Files in Database\n---\n"
    if filenames:
        for f_name in filenames: md_output += f"- {f_name}\n"
    else: md_output += "_No files currently in this database._\n"
    checkbox_group_update = gradio.update(choices=filenames, value=[]) 
    markdown_update = gradio.update(value=md_output)
    return checkbox_group_update, markdown_update

def get_current_file_list_md_backend():
    _ , md_update = _build_file_list_updates()
    return md_update.get('value', "_Error generating file list._")

def update_app_config_backend(models_names, vary_t, temp, vary_p, top_p, sys_prompt, current_app_config_state):
    selected_model_ids = [MODEL_NAME_TO_ID_MAP[name] for name in models_names if name in MODEL_NAME_TO_ID_MAP]
    if len(selected_model_ids) > 3: selected_model_ids = selected_model_ids[:3]
    
    current_app_config_state.update({
        "selected_models": selected_model_ids,
        "vary_temperature": vary_t, "temperature": temp,
        "vary_top_p": vary_p, "top_p": top_p,
        "system_prompt": sys_prompt
    })
    return "Configuration updated.", current_app_config_state

def switch_db_backend(selected_db_val, current_selected_db_state_value): 
    if selected_db_val == current_selected_db_state_value:
        db_status_msg = f"Already using {selected_db_val}. No change."
    else:
        db_status_msg = load_data_from_selected_db(selected_db_val)
    cb_update, md_update = _build_file_list_updates()
    return selected_db_val, db_status_msg, cb_update, md_update

def handle_pdf_upload_backend(files_obj_list, selected_db_from_state):
    if files_obj_list is None:
        cb_update, md_update = _build_file_list_updates()
        return "No files uploaded.", cb_update, md_update
    alerts = []
    any_successful_upload = False
    for file_obj in files_obj_list:
        file_path = file_obj.name 
        file_display_name = os.path.basename(getattr(file_obj, 'orig_name', file_path))
        with open(file_path, 'rb') as f: pdf_bytes = f.read()
        status_msg, success = process_and_add_pdf_core(pdf_bytes, file_display_name, selected_db_from_state)
        alerts.append(status_msg)
        if success: any_successful_upload = True
    if any_successful_upload: save_data_to_selected_db(selected_db_from_state)
    status_summary = "\n".join(alerts)
    cb_update, md_update = _build_file_list_updates()
    return status_summary, cb_update, md_update


def delete_files_backend(filenames_to_delete, selected_db):
    global text_store, faiss_index, embedding_model
    if not filenames_to_delete:
        cb_update, md_update = _build_file_list_updates()
        return "No files selected for deletion.", cb_update, md_update
    if embedding_model is None or faiss_index is None:
        cb_update, md_update = _build_file_list_updates()
        return "Error: Core components not ready. Deletion aborted.", cb_update, md_update
    kept_text_store_entries_with_original_indices = []
    for i, item in enumerate(text_store):
        if isinstance(item, dict) and item.get("file_name") not in filenames_to_delete:
            kept_text_store_entries_with_original_indices.append((i, item))
    if len(kept_text_store_entries_with_original_indices) == len(text_store):
        cb_update, md_update = _build_file_list_updates()
        return "Selected files not found or no changes made.", cb_update, md_update
    new_text_store = [item for _, item in kept_text_store_entries_with_original_indices]
    new_faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    if kept_text_store_entries_with_original_indices:
        original_indices_to_keep = [original_idx for original_idx, _ in kept_text_store_entries_with_original_indices]
        valid_original_indices_to_keep = [idx for idx in original_indices_to_keep if idx < faiss_index.ntotal]
        if valid_original_indices_to_keep:
            vectors_to_keep = faiss_index.reconstruct_n(0, faiss_index.ntotal)
            kept_vectors = vectors_to_keep[valid_original_indices_to_keep, :]
            if kept_vectors.shape[0] > 0: new_faiss_index.add(kept_vectors.astype("float32"))
        else: print("Warning: No valid vectors to keep after filtering indices for deletion.")
    text_store = new_text_store
    faiss_index = new_faiss_index
    save_data_to_selected_db(selected_db)
    num_deleted = len(filenames_to_delete)
    status_msg = f"Successfully deleted {num_deleted} file(s) and their associated data. Index rebuilt."
    cb_update, md_update = _build_file_list_updates()
    return status_msg, cb_update, md_update

def apply_uploaded_config_backend(config_file_obj, current_app_config_state_dict):
    if config_file_obj is None:
        return "No config file uploaded.", False, current_app_config_state_dict, *[gradio.update()]*6
    try:
        with open(config_file_obj.name, 'r') as f: new_config = json.load(f)
        current_app_config_state_dict.update(new_config)
        sel_model_ids = current_app_config_state_dict.get("selected_models", [])
        model_names_for_ui = [MODEL_ID_TO_NAME_MAP[mid] for mid in sel_model_ids if mid in MODEL_ID_TO_NAME_MAP]
        return (
            "Configuration loaded successfully from file.", True,
            current_app_config_state_dict,
            gradio.update(value=model_names_for_ui),
            gradio.update(value=current_app_config_state_dict.get("vary_temperature", True)),
            gradio.update(value=current_app_config_state_dict.get("temperature", 0.7)),
            gradio.update(value=current_app_config_state_dict.get("vary_top_p", False)),
            gradio.update(value=current_app_config_state_dict.get("top_p", 0.9)),
            gradio.update(value=current_app_config_state_dict.get("system_prompt", ""))
        )
    except Exception as e:
        error_msg = f"Error loading config: {e}"
        return error_msg, False, current_app_config_state_dict, *[gradio.update()]*6

def generate_config_for_download_backend(current_app_config_state_dict):
    try:
        config_to_download = { # Only saving a subset, not the full model list.
            "vary_temperature": current_app_config_state_dict.get("vary_temperature", True),
            "temperature": current_app_config_state_dict.get("temperature", 0.7),
            "vary_top_p": current_app_config_state_dict.get("vary_top_p", False),
            "top_p": current_app_config_state_dict.get("top_p", 0.9),
            "system_prompt": current_app_config_state_dict.get("system_prompt", "You are a helpful assistant."),
            "selected_models": current_app_config_state_dict.get("selected_models", []) # Saving selected model IDs
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding='utf-8') as tmp_file:
            json.dump(config_to_download, tmp_file, indent=2)
            tmp_file_path = tmp_file.name
        return tmp_file_path, "Config ready for download.", True
    except Exception as e:
        return None, f"Error generating config file for download: {e}", False

# In backend.py

def chat_interface_backend(
    user_input: str,
    chat_history: list[list[str | None]] | None,
    selected_db_state_val: str,
    model_config_dict: dict
) -> list[list[str | None]]:
    """
    Handles the chat logic by retrieving context, generating responses from selected AI models,
    and updating the chat history.

    Args:
        user_input: The question asked by the user.
        chat_history: The existing conversation history.
        selected_db_state_val: The currently active database ('MongoDB' or 'Dropbox').
        model_config_dict: The 'model_config' portion of the main application configuration,
                           containing temperature, top_p, selected models, etc.

    Returns:
        The updated chat history.
    """
    if not user_input or not user_input.strip():
        return chat_history if chat_history is not None else []

    current_chat_history = chat_history if chat_history is not None else []

    # 1. Retrieve context from the vector database using RAG
    print(f"Retrieving context for query: '{user_input[:50]}...'")
    context_text = retrieve_context_from_db(user_input, top_k=5)

    # 2. Extract model parameters from the passed-in dictionary
    temp_config = model_config_dict.get('temperature', 0.7)
    top_p_config = model_config_dict.get('top_p', 0.9)
    system_prompt_config = model_config_dict.get('system_prompt', "You are a helpful assistant.")
    selected_ai_model_ids = model_config_dict.get('selected_models', [])

    # 3. Determine parameter variations based on checkboxes
    temp_values_to_run = [temp_config]
    if model_config_dict.get('vary_temperature', False) and temp_config > 0.01:
        # Create a list of varied temperatures, clamping values between 0.01 and 1.0
        temp_values_to_run = sorted(list(set([
            round(max(0.01, temp_config * 0.5), 2), 
            temp_config, 
            round(min(1.0, temp_config * 1.5), 2)
        ])))

    top_p_values_to_run = [top_p_config]
    if model_config_dict.get('vary_top_p', False) and top_p_config > 0.01:
        # Create a list of varied top_p values, clamping values between 0.01 and 1.0
        top_p_values_to_run = sorted(list(set([
            round(max(0.01, top_p_config * 0.5), 2), 
            top_p_config,
            round(min(1.0, top_p_config * 1.5), 2)
        ])))

    # 4. Generate responses from all selected models and parameter combinations
    bot_response_content_parts = []
    if not selected_ai_model_ids:
        final_bot_response = "System: No AI model has been selected in the configuration. Please ask an admin to configure one."
    else:
        for model_id in selected_ai_model_ids:
            model_detail = AVAILABLE_MODELS_DICT.get(model_id, {})
            model_type = model_detail.get("type")
            model_display_name = model_detail.get("name", model_id)

            for temp_val in temp_values_to_run:
                for top_p_val in top_p_values_to_run:
                    print(f"Generating response from {model_display_name} (Temp: {temp_val}, Top-P: {top_p_val})")
                    response_content = f"Error: Model type '{model_type}' for '{model_display_name}' is not configured correctly."
                    
                    # Dynamically call the appropriate generation function
                    if model_type == "gemini" and gemini_model_genai:
                        response_content = generate_response_gemini(user_input, context_text, temp_val, top_p_val, system_prompt_config)
                    elif model_type == "together" and together_client:
                        response_content = generate_response_together_ai(user_input, context_text, model_id, temp_val, top_p_val, system_prompt_config)
                    elif model_type == "openai" and openai_client:
                        response_content = generate_response_openai_api(user_input, context_text, temp_val, top_p_val, system_prompt_config)
                    
                    # Format the response with model details
                    model_info_str = f"{model_display_name} (T:{temp_val}, P:{top_p_val})"
                    bot_response_content_parts.append(f"--- {model_info_str} ---\n{response_content}")

                    if not model_config_dict.get('vary_top_p', False):
                        break  # Exit inner loop if not varying top_p
                if not model_config_dict.get('vary_temperature', False):
                    break  # Exit outer loop if not varying temperature

        final_bot_response = "\n\n".join(bot_response_content_parts)
        if not final_bot_response:
            final_bot_response = "System: No responses were generated. This could be due to an issue with the selected AI models or their configurations."

    # 5. Append the user message and final bot response to history
    current_chat_history.append([user_input, final_bot_response])
    return current_chat_history

# --- End of backend.py relevant section ---

def run_scraper_backend(base_url, endpoint, pagination, num_pages, selected_db_val):
    if sys.platform == "win32" and isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    status_updates = ["Starting scraping..."]
    page_urls_to_scan = get_all_page_urls_for_scraping(base_url, endpoint, pagination, num_pages)
    status_updates.append(f"Found {len(page_urls_to_scan)} site pages to scan for PDF links.")
    any_successful_scrape_process = False
    if page_urls_to_scan:
        pdf_links_found = asyncio.run(process_scraped_pdf_links_async(page_urls_to_scan))
        status_updates.append(f"Found {len(pdf_links_found)} unique PDF links.")
        if pdf_links_found:
            status_updates.append(f"Starting PDF download and processing for {len(pdf_links_found)} links...")
            processing_results = asyncio.run(batch_download_and_process_pdfs(pdf_links_found, selected_db_val))
            success_count = 0; processed_files_messages = []
            for msg, success, fname in processing_results:
                processed_files_messages.append(f"{fname}: {msg} ({'Success' if success else 'Failed'})")
                if success: success_count += 1; any_successful_scrape_process = True
            status_updates.append(f"\n--- PDF Processing Results ---"); status_updates.extend(processed_files_messages)
            status_updates.append(f"\nScraping finished. Processed {success_count} new PDFs out of {len(pdf_links_found)} found.")
    else: status_updates.append("No site pages found to scan based on current settings.")
    if any_successful_scrape_process: save_data_to_selected_db(selected_db_val)
    cb_update, md_update = _build_file_list_updates()
    return "\n".join(status_updates), cb_update, md_update


# In backend.py

# --- CORRECTED initialize_all_components ---
def initialize_all_components(default_db="MongoDB"):
    """Initializes all backend components in the correct order."""
    global gemini_model_genai, together_client, openai_client, embedding_model, faiss_index, BACKEND_INITIAL_LOAD_MSG, mongo_client_instance, mongo_db_obj

    print("--- Backend Initialization Started ---")
    if mongo_client_instance is None:
        initialize_mongodb_client()
    
    # CORRECTED CHECK: Use 'is not None'
    if mongo_db_obj is not None:
        print("Verifying main application configuration in DB...")
        backend_get_initial_main_config()
        if not load_scraper_config_from_db():
             print("No scraper source config found, initializing with defaults.")
             default_sources = get_predefined_scraper_sources_list()
             if default_sources:
                 save_scraper_config_to_db([default_sources[0]])
                 print(f"Saved default scraper source '{default_sources[0]['display_name']}' to DB.")
    else:
        print("WARNING: MongoDB not available, cannot initialize main config in DB.")

    print("Initializing AI service clients...")
    if GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            gemini_model_genai = genai.GenerativeModel("gemini-1.5-flash-latest")
            print("  - Google Gemini client configured.")
        except Exception as e:
            print(f"  - Google Gemini client init failed: {e}")
    else:
        print("  - Google Gemini client: SKIPPED (GOOGLE_API_KEY not set).")

    # ... (other AI client inits)

    print("Initializing local RAG components...")
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"  - SentenceTransformer model loaded (Dimension: {embedding_model.get_sentence_embedding_dimension()}).")
        faiss_index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        print("  - FAISS index initialized.")
    except Exception as e:
        print(f"  - CRITICAL ERROR loading SentenceTransformer model: {e}. RAG features disabled.")
        BACKEND_INITIAL_LOAD_MSG = "Critical: Embedding model failed. RAG non-functional."
        print("--- Backend Initialization Halted ---")
        return

    print("Initializing Dropbox client...")
    initialize_dropbox_client()
    print(f"Loading initial data from default database: '{default_db}'...")
    BACKEND_INITIAL_LOAD_MSG = load_data_from_selected_db(default_db)
    
    print(f"Final DB Load Status: {BACKEND_INITIAL_LOAD_MSG}")
    print("--- Backend Initialization Complete ---")




def get_backend_initial_load_message():
    global BACKEND_INITIAL_LOAD_MSG
    return BACKEND_INITIAL_LOAD_MSG


SCRAPER_CONFIG_COLLECTION_NAME = "scraper_configurations" 
DEFAULT_SCRAPER_PAUSE_SECONDS = 1 #

# --- NEW: Scraper Configuration Backend Functions ---

PREDEFINED_SCRAPER_SOURCES = [
    {
        "source_key": "imy_se",
        "display_name": "IMY (Sweden)",
        "parameters": {
            "base_url_root": "https://www.imy.se",
            "listing_path_segment": "tillsyner",
            "pagination_query_format": "?query=&page=",
            "max_pages": 5, # Default max_pages
            "pause_seconds": DEFAULT_SCRAPER_PAUSE_SECONDS
        }
    },
    {
        "source_key": "ico_uk",
        "display_name": "ICO (UK) - Enforcement",
        "parameters": {
            "base_url_root": "https://ico.org.uk",
            "listing_path_segment": "action-weve-taken/enforcement/",
            "pagination_query_format": "?page_num=", # Example, needs verification
            "max_pages": 3,
            "pause_seconds": DEFAULT_SCRAPER_PAUSE_SECONDS
        }
    },
    {
        "source_key": "dummy_source_1",
        "display_name": "Dummy News Site",
        "parameters": {
            "base_url_root": "https://www.example-news.com",
            "listing_path_segment": "articles/data-privacy",
            "pagination_query_format": "/page/", # Example
            "max_pages": 2,
            "pause_seconds": DEFAULT_SCRAPER_PAUSE_SECONDS
        }
    }
]

def get_predefined_scraper_sources_list():
    """Returns a deep copy of the predefined scraper sources."""
    return json.loads(json.dumps(PREDEFINED_SCRAPER_SOURCES)) # Simple deep copy

def load_scraper_config_from_db():
    """Loads the scraper configuration list from MongoDB."""
    global mongo_db_obj
    if mongo_db_obj is None:
        print("MongoDB not initialized. Cannot load scraper config.")
        return [] # Return empty list, UI can show default message
    try:
        config_doc = mongo_db_obj[SCRAPER_CONFIG_COLLECTION_NAME].find_one({"_id": "active_scraper_config"})
        if config_doc and "sources" in config_doc and isinstance(config_doc["sources"], list):
            return config_doc["sources"]
        return [] # No config found or malformed
    except Exception as e:
        print(f"Error loading scraper config from DB: {e}")
        return []

def save_scraper_config_to_db(config_sources_list: list):
    """Saves the entire scraper configuration list to MongoDB."""
    global mongo_db_obj
    if mongo_db_obj is None:
        print("MongoDB not initialized. Cannot save scraper config.")
        return False, "Database not connected."
    try:
        mongo_db_obj[SCRAPER_CONFIG_COLLECTION_NAME].update_one(
            {"_id": "active_scraper_config"},
            {"$set": {"sources": config_sources_list, "updated_at": time.time()}},
            upsert=True
        )
        return True, "Scraper configuration saved successfully."
    except Exception as e:
        print(f"Error saving scraper config to DB: {e}")
        return False, f"Error saving scraper config: {e}"

def get_scraper_ui_initial_data():
    """Called by Gradio UI on load to get initial scraper tab data."""
    predefined = get_predefined_scraper_sources_list()
    db_config = load_scraper_config_from_db()
    return predefined, db_config

def backend_update_scraper_source_in_db(source_key_to_update: str, display_name: str, new_parameters: dict):
    """Adds or updates a source in the DB config and saves it."""
    current_db_config = load_scraper_config_from_db()
    
    found = False
    for i, source_conf in enumerate(current_db_config):
        if source_conf.get("source_key") == source_key_to_update:
            current_db_config[i]["parameters"] = new_parameters
            current_db_config[i]["display_name"] = display_name # Ensure display name is also updated
            found = True
            break
    
    if not found:
        current_db_config.append({
            "source_key": source_key_to_update,
            "display_name": display_name,
            "parameters": new_parameters
        })
        
    success, msg = save_scraper_config_to_db(current_db_config)
    if success:
        return current_db_config, f"Source '{display_name}' updated/added: {msg}"
    else:
        # If save failed, return the original config from before the attempt
        return load_scraper_config_from_db(), f"Failed to update/add '{display_name}': {msg}"


def backend_remove_scraper_source_from_db(source_key_to_remove: str):
    """Removes a source from the DB config and saves it."""
    current_db_config = load_scraper_config_from_db()
    
    initial_len = len(current_db_config)
    updated_db_config = [s for s in current_db_config if s.get("source_key") != source_key_to_remove]
    
    if len(updated_db_config) == initial_len:
        return updated_db_config, f"Source key '{source_key_to_remove}' not found in DB config. No changes made."

    success, msg = save_scraper_config_to_db(updated_db_config)
    if success:
        return updated_db_config, f"Source '{source_key_to_remove}' removed: {msg}"
    else:
        return load_scraper_config_from_db(), f"Failed to remove '{source_key_to_remove}': {msg}"


#Config
# --- Add this new constant near the top ---
MAIN_CONFIG_COLLECTION_NAME = "main_app_config"

# --- Add this new group of functions for Main Configuration Management ---

def get_default_main_config():
    """Returns the default structure for the main application config."""
    # Find a default model ID if possible
    default_model_id = None
    if MODEL_NAME_TO_ID_MAP:
        first_model_name = AVAILABLE_MODELS_NAMES[0]
        # Look up its corresponding ID
        default_model_id = MODEL_NAME_TO_ID_MAP.get(first_model_name)

    return {
        "model_config": {
            "temperature": 0.7,
            "top_p": 0.9,
            "system_prompt": "You are a helpful assistant. Answer questions based on the provided context.",
            "vary_temperature": True,
            "vary_top_p": False,
            "selected_models": [default_model_id] if default_model_id else []
        },
        "cron_schedule": "0 2 * * *", # Default: Run at 2:00 AM every day
    }

def backend_get_initial_main_config():
    """
    Loads the main config from the DB. If it doesn't exist, creates, saves,
    and returns the default config. This is the primary function for loading config.
    """
    global mongo_db_obj
    if mongo_db_obj is None:
        print("MongoDB not initialized. Returning default config without saving.")
        return get_default_main_config()
    
    try:
        config_doc = mongo_db_obj[MAIN_CONFIG_COLLECTION_NAME].find_one({"_id": "singleton_config"})
        if config_doc:
            # Remove MongoDB's '_id' before returning to the UI state
            config_doc.pop('_id', None)
            return config_doc
        else:
            # Config doesn't exist, so create and save the default one
            print("No main config found in DB. Creating and saving default config.")
            default_config = get_default_main_config()
            # The spread operator `**` unpacks the dictionary
            mongo_db_obj[MAIN_CONFIG_COLLECTION_NAME].insert_one({
                "_id": "singleton_config",
                **default_config
            })
            return default_config
    except Exception as e:
        print(f"Error loading main config from DB: {e}. Returning default.")
        return get_default_main_config()
def backend_update_main_config(updated_config: dict):
    """Saves the provided main configuration object to the database."""
    global mongo_db_obj
    if mongo_db_obj is None:
        return False, "Database not connected. Config not saved.", updated_config
    
    try:
        # Use update_one with upsert=True to either update the existing doc or create it
        mongo_db_obj[MAIN_CONFIG_COLLECTION_NAME].update_one(
            {"_id": "singleton_config"},
            {"$set": updated_config},
            upsert=True
        )
        return True, "Configuration saved successfully!", updated_config
    except Exception as e:
        print(f"Error saving main config to DB: {e}")
        return False, f"Error saving configuration: {e}", updated_config

def backend_apply_uploaded_main_config(config_file_obj):
    """
    Replaces the entire main config with the content of an uploaded JSON file.
    Returns: (status_msg, success_bool, final_config_dict)
    """
    if config_file_obj is None:
        return "No config file uploaded.", False, backend_get_initial_main_config()
    try:
        with open(config_file_obj.name, 'r', encoding='utf-8') as f:
            new_config = json.load(f)
        
        # Here you could add validation to ensure the uploaded config has the right structure
        if "model_config" not in new_config or "cron_schedule" not in new_config:
            raise ValueError("Uploaded config is missing required keys like 'model_config' or 'cron_schedule'.")
            
        success, msg, final_config = backend_update_main_config(new_config)
        if success:
            msg = "Configuration successfully applied from uploaded file."
        return msg, success, final_config
    except Exception as e:
        error_msg = f"Error applying uploaded config: {e}"
        return error_msg, False, backend_get_initial_main_config()

def backend_generate_main_config_for_download(current_main_config: dict):
    """
    Generates a temporary file containing the current main config for download.
    Returns: Path to the temporary file or None on error.
    """
    try:
        # Use a temporary file that Gradio can serve
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding='utf-8') as tmp_file:
            # The current_main_config from the UI state is already a Python dict
            json.dump(current_main_config, tmp_file, indent=2)
            tmp_file_path = tmp_file.name
        # Gradio's DownloadButton component needs the file path to be returned
        return tmp_file_path
    except Exception as e:
        print(f"Error generating config file for download: {e}")
        # Return None to indicate failure; the UI won't trigger a download
        return None