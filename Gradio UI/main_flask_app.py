# main_flask_app.py
import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
from dotenv import load_dotenv

# Corrected backend imports
from backend import (
    initialize_all_components,
    create_admin_user_if_not_exists,
    create_user,
    get_user_by_email,
    verify_password,
    MONGO_URI
)

load_dotenv()

APP_ADMIN_EMAIL = "saragaballa2002@gmail.com"
APP_ADMIN_PASSWORD = "11112002"

FLASK_APP_INITIALIZED = False
def run_flask_app_initializations():
    global FLASK_APP_INITIALIZED
    if not FLASK_APP_INITIALIZED:
        print("Running initializations for Flask App process...")
        if not MONGO_URI:
            print("CRITICAL: MONGO_URI is not set in .env (for Flask App).")
        else:
            print(f"MongoDB URI found for Flask App: {MONGO_URI[:20]}...")
        initialize_all_components(default_db="Dropbox")
        create_admin_user_if_not_exists(APP_ADMIN_EMAIL, APP_ADMIN_PASSWORD, role="admin")
        FLASK_APP_INITIALIZED = True
    else:
        print("Flask App initializations already run.")

app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "a_very_secure_default_secret_key_CHANGE_ME_TOO")

run_flask_app_initializations()

@app.route("/")
def index():
    if "user_email" in session:
        user = get_user_by_email(session["user_email"]) # Fetch user to check status
        if user and user.get("status") == "pending":
            # If user is in session but status is pending, ensure they see pending page
            session["user_status"] = "pending" # Ensure status is in session
            return redirect(url_for("pending_activation"))
        # If active or other logic, proceed to app_frame
        return redirect(url_for("app_frame"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_email" in session:
        user_check = get_user_by_email(session["user_email"])
        if user_check and user_check.get("status") == "pending":
            session["user_status"] = "pending" # Ensure status is in session
            return redirect(url_for("pending_activation"))
        return redirect(url_for("app_frame"))

    if request.method == "POST":
        email = request.form.get("email")
        pwd   = request.form.get("password")
        if not email or not pwd:
            flash("Email and password are required.", "error")
            return render_template("login.html")

        user = get_user_by_email(email)
        if user and verify_password(pwd, user.get("password")):
            if user.get("status") == "active":
                session["user_email"] = email
                session["user_role"] = user.get("role", "user")
                session.pop("user_status", None) # Clear pending status if any
                flash("Logged in successfully!", "success")
                return redirect(url_for("app_frame"))
            elif user.get("status") == "pending":
                session["user_email"] = email # Log them in
                session["user_status"] = "pending" # Mark status in session
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
            return render_template("signup.html", email=email), 400 # Pass email back
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html", email=email), 400 # Pass email back
        
        if "@" not in email or "." not in email.split("@")[-1]: # Basic email check
            flash("Invalid email format.", "error")
            return render_template("signup.html", email=email), 400

        success, message = create_user(email, password) # backend.create_user sets status to "pending"
        if success:
            # User created successfully, status is 'pending'
            # Set session variables and redirect to pending_activation page
            session["user_email"] = email
            session["user_status"] = "pending"
            flash(message, "info") # Use 'info' or 'success' for this message
            return redirect(url_for("pending_activation")) # <<<< CHANGED REDIRECT
        else:
            flash(message, "error")
            return render_template("signup.html", email=email), 400 # Or 500
            
    return render_template("signup.html")

@app.route("/pending_activation")
def pending_activation():
    # Ensure user is logged in and their status is indeed pending
    if "user_email" not in session or session.get("user_status") != "pending":
        # If they land here without proper session state, send to login
        flash("Please log in to check your account status.", "info")
        return redirect(url_for("login"))
    
    # Optionally, you can pass the email to the template if you want to display it
    # email = session.get("user_email")
    # return render_template("pending_activation.html", user_email=email)
    return render_template("pending_activation.html")


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
    # Stricter check: only active users get to the app frame
    if not user or user.get("status") != "active":
        # If user status changed or something went wrong, clear session and send to login/pending
        if user and user.get("status") == "pending":
            session["user_status"] = "pending"
            return redirect(url_for("pending_activation"))
        
        session.clear()
        flash("Your account is not active or an error occurred. Please log in.", "error")
        return redirect(url_for("login"))
        
    return render_template("gradio_iframe.html")

if __name__ == "__main__":
    print("Launching Flask server for authentication and iframe on 0.0.0.0:5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)