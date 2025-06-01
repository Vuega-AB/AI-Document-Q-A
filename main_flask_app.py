import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
from dotenv import load_dotenv
from datetime import datetime # Ensure datetime is imported

# Corrected backend imports
from backend import (
    initialize_all_components,
    create_admin_user_if_not_exists,
    create_user,
    get_user_by_email,
    verify_password,
    MONGO_URI, # If used directly in Flask app, else remove
    get_all_users_from_db,     # <<< ADDED
    update_user_status_in_db # <<< ADDED
)

load_dotenv()

APP_ADMIN_EMAIL = "saragaballa2002@gmail.com"
APP_ADMIN_PASSWORD = "11112002"
GRADIO_APP_URL = os.getenv("GRADIO_APP_URL", "http://localhost:7860")


FLASK_APP_INITIALIZED = False
def run_flask_app_initializations():
    global FLASK_APP_INITIALIZED
    if not FLASK_APP_INITIALIZED:
        print("Running initializations for Flask App process...")
        # MONGO_URI check can be here if needed, but backend.py handles its own connection
        # if not MONGO_URI: 
        #     print("CRITICAL: MONGO_URI is not set in .env (for Flask App).")
        # else:
        #     print(f"MongoDB URI found for Flask App: {MONGO_URI[:20]}...")
        initialize_all_components(default_db="MongoDB") # Or "MongoDB"
        create_admin_user_if_not_exists(APP_ADMIN_EMAIL, APP_ADMIN_PASSWORD, role="admin")
        FLASK_APP_INITIALIZED = True
    else:
        print("Flask App initializations already run.")

app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_very_secure_default_secret_key_CHANGE_ME_TOO")

run_flask_app_initializations()

# --- Helper Functions ---
def get_name_initials(name_str):
    if not name_str or not isinstance(name_str, str):
        return "N/A"
    parts = name_str.split()
    if len(parts) > 1:
        return (parts[0][0] + parts[-1][0]).upper()
    elif parts:
        return parts[0][:2].upper()
    return "N/A"

def format_datetime_for_display(dt_obj):
    if isinstance(dt_obj, datetime):
        return dt_obj.strftime("%b %d, %Y, %I:%M %p") # Example: May 16, 2024, 03:45 PM
    elif dt_obj is None:
        return "N/A" # Or an empty string, or "Never"
    return str(dt_obj) # Fallback for other types or if already string

# --- Routes ---
@app.route("/")
def index():
    if "user_email" in session:
        user = get_user_by_email(session["user_email"])
        if user and user.get("status") == "pending":
            session["user_status"] = "pending"
            return redirect(url_for("pending_activation"))
        return redirect(url_for("app_frame"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_email" in session:
        user_check = get_user_by_email(session["user_email"])
        if user_check and user_check.get("status") == "pending":
            session["user_status"] = "pending"
            return redirect(url_for("pending_activation"))
        return redirect(url_for("app_frame"))

    if request.method == "POST":
        email = request.form.get("email")
        pwd   = request.form.get("password")
        if not email or not pwd:
            flash("Email and password are required.", "error")
            return render_template("login.html")

        user = get_user_by_email(email)
        # Password from DB is bytes, ensure verify_password handles this
        if user and user.get("password") and verify_password(pwd, user.get("password")):
            if user.get("status") == "active":
                session["user_email"] = email
                session["user_role"] = user.get("role", "user")
                session.pop("user_status", None)
                # Optionally update last_login_at here in the DB
                # backend.update_user_last_login(email) # You'd need this backend function
                flash("Logged in successfully!", "success")
                return redirect(url_for("app_frame"))
            elif user.get("status") == "pending":
                session["user_email"] = email
                session["user_status"] = "pending"
                return redirect(url_for("pending_activation"))
            else:
                flash(f"Your account is currently {user.get('status', 'unavailable')}. Please contact support.", "error")
        else:
            flash("Invalid credentials.", "error")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not email or not password or not confirm_password:
            flash("All fields are required.", "error")
            return render_template("signup.html", email=email), 400
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html", email=email), 400
        
        if "@" not in email or "." not in email.split("@")[-1]:
            flash("Invalid email format.", "error")
            return render_template("signup.html", email=email), 400

        success, message = create_user(email, password)
        if success:
            session["user_email"] = email
            session["user_status"] = "pending"
            flash(message, "info")
            return redirect(url_for("pending_activation"))
        else:
            flash(message, "error")
            return render_template("signup.html", email=email), 400
            
    return render_template("signup.html")

@app.route("/pending_activation")
def pending_activation():
    if "user_email" not in session or session.get("user_status") != "pending":
        flash("Please log in to check your account status.", "info")
        return redirect(url_for("login"))
    return render_template("pending_activation.html")

@app.route("/settings")
def settings():
    if "user_email" not in session or session.get("user_role") != "admin":
        flash("Access denied. Admins only.", "error")
        if "user_email" in session:
            return redirect(url_for("app_frame"))
        return redirect(url_for("login"))

    db_users = get_all_users_from_db() # Fetch real users from backend
    
    processed_users = []
    for user_data in db_users:
        name_to_use = user_data.get("full_name", user_data.get("email", "N/A").split('@')[0])
        
        processed_users.append({
            "id": user_data.get("_id"),
            "role": user_data.get("role", "user"),
            "name": name_to_use,
            "email": user_data.get("email"),
            "status": user_data.get("status", "unknown"),
            "last_active": format_datetime_for_display(user_data.get("last_login_at")), # From DB if exists
            "created_at": format_datetime_for_display(user_data.get("created_at")), # From DB
            "name_initials": get_name_initials(name_to_use),
            "avatar_url": user_data.get("avatar_url")
        })
    
    # Sort users: pending first, then by creation date (newest first if created_at is datetime)
    # Note: format_datetime_for_display returns string, so sort on original datetime from db_users
    processed_users.sort(key=lambda u: (
        u["status"] != "pending", # False (0) for pending, True (1) for others, so pending comes first
        # Find original datetime for sorting, default to very old time if not found
        next((item.get("created_at") for item in db_users if item.get("email") == u["email"]), datetime.min)
    ), reverse=False) # False for created_at makes newest come last if using reverse=False
                       # to make newest pending come first, you might need specific logic or sort descending for date

    return render_template("settings.html", users=processed_users)


@app.route("/admin/update_user_status", methods=["POST"])
def admin_update_user_status():
    if "user_email" not in session or session.get("user_role") != "admin":
        return {"success": False, "message": "Unauthorized"}, 403

    data = request.get_json()
    target_user_email = data.get("email")
    new_status = data.get("new_status")

    if not target_user_email or not new_status:
        return {"success": False, "message": "Email and new status are required."}, 400

    success, message = update_user_status_in_db(target_user_email, new_status)
    
    if success:
        return {"success": True, "message": message, "email": target_user_email, "new_status": new_status}, 200
    else:
        return {"success": False, "message": message}, 400


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/app")
def app_frame():
    if "user_email" not in session:
        return redirect(url_for("login"))
    
    user = get_user_by_email(session["user_email"])
    if not user or user.get("status") != "active":
        if user and user.get("status") == "pending":
            session["user_status"] = "pending"
            return redirect(url_for("pending_activation"))
        
        session.clear()
        flash("Your account is not active or an error occurred. Please log in.", "error")
        return redirect(url_for("login"))
        
    user_role = session.get("user_role", "user")
    return render_template("gradio_iframe.html", user_role=user_role, gradio_app_url=GRADIO_APP_URL)

if __name__ == "__main__":
    # For Render, Gunicorn will run the app. This block is for local development.
    # Render provides the PORT environment variable.
    port = int(os.environ.get("PORT", 5000)) # Use PORT set by Render or 5000 locally
    print(f"Launching Flask server for authentication and iframe on 0.0.0.0:{port}...")
    print(f"Gradio app is expected to be running at: {GRADIO_APP_URL}")
    # When run with 'python main_flask_server.py', debug=True is fine.
    # Gunicorn will handle production, so debug=True here is less of a concern for deployment.
    app.run(host="0.0.0.0", port=port, debug=True)