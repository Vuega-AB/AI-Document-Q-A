import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta # Ensure datetime, timezone, timedelta are imported
from flask_mail import Mail, Message # ADDED
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature # ADDED
import time # Add this if not present

# Corrected backend imports
from backend import (
    initialize_all_components,
    create_admin_user_if_not_exists,
    create_user,
    get_user_by_email,
    verify_password,
    hard_delete_user_from_db,
    verify_otp_and_activate_user,
    regenerate_otp_for_user,
    # MONGO_URI, # If used directly in Flask app, else remove (backend handles its own)
    get_all_users_from_db,
    update_user_status_in_db,
    confirm_user_email_in_db, # <<< ADDED
    STATUS_PENDING_EMAIL_CONFIRMATION,
    STATUS_ACTIVE,
    STATUS_SUSPENDED, # Ensure this is defined in backend if used in dropdown
    STATUS_DEACTIVATED  # Ensure this is defined in backend
)

load_dotenv()

APP_ADMIN_EMAIL = os.getenv("APP_ADMIN_EMAIL", "saragaballa2002@gmail.com")
APP_ADMIN_PASSWORD = os.getenv("APP_ADMIN_PASSWORD", "11112002")
GRADIO_APP_URL = os.getenv("GRADIO_APP_URL", "http://localhost:7860")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

if not FLASK_SECRET_KEY:
    # For development, we can use a hardcoded default, but WARN LOUDLY.
    # In a real production environment, this should cause the app to fail to start
    # or use a securely generated key.
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! WARNING: FLASK_SECRET_KEY is not set in .env.          !!!")
    print("!!! Using a default, insecure key for development.         !!!")
    print("!!! THIS IS NOT SAFE FOR PRODUCTION. Set a strong secret key!  !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    FLASK_SECRET_KEY = "dev_default_unsafe_secret_key_CHANGE_ME_IMMEDIATELY"
    # raise ValueError("CRITICAL: FLASK_SECRET_KEY is not set in the environment. This is required for security.")


app = Flask(__name__, template_folder="templates")
app.secret_key = FLASK_SECRET_KEY # CRITICAL for session security and token signing

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.googlemail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() in ['true', '1', 't']
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'False').lower() in ['true', '1', 't']
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])
app.config['MAIL_MAX_EMAILS'] = None # Default, can be set to limit mails per connection

mail = Mail(app)
# itsdangerous Serializer for tokens. Uses app.secret_key.
ts = URLSafeTimedSerializer(app.secret_key)


FLASK_APP_INITIALIZED = False
def run_flask_app_initializations():
    global FLASK_APP_INITIALIZED
    if not FLASK_APP_INITIALIZED:
        print("Running initializations for Flask App process...")
        initialize_all_components(default_db="MongoDB") # Or "MongoDB"
        create_admin_user_if_not_exists(APP_ADMIN_EMAIL, APP_ADMIN_PASSWORD, role="admin")
        FLASK_APP_INITIALIZED = True
    else:
        print("Flask App initializations already run.")

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
        # Ensure datetime is timezone-aware or consistently handled before display
        # If dt_obj might be naive, you might want to localize it or assume UTC
        # For simplicity, assuming backend provides timezone-aware or consistently naive UTC datetimes
        return dt_obj.strftime("%b %d, %Y, %I:%M %p")
    elif dt_obj is None:
        return "N/A"
    return str(dt_obj)

def send_system_email(to_email, subject, template_name_no_ext, **kwargs):
    """Helper to send emails using HTML and TXT templates."""
    if not app.config.get('MAIL_USERNAME') or not app.config.get('MAIL_PASSWORD'):
        app.logger.error(f"Mail not configured. Cannot send '{subject}' to {to_email}.")
        print(f"MAIL ERROR: Mail server not configured. Cannot send '{subject}' to {to_email}.")
        return False
    try:
        # Ensure app_name or similar is passed for templates if they use it
        kwargs.setdefault('app_name', 'IntelLaw') # Default app_name if not provided
        html_body = render_template(f"email/{template_name_no_ext}.html", **kwargs)
        text_body = render_template(f"email/{template_name_no_ext}.txt", **kwargs)
        msg = Message(subject, recipients=[to_email], html=html_body, body=text_body)
        mail.send(msg)
        app.logger.info(f"Email '{subject}' sent successfully to {to_email}.")
        return True
    except Exception as e:
        app.logger.error(f"Failed to send email '{subject}' to {to_email}: {e}")
        print(f"MAIL ERROR: Failed to send email '{subject}' to {to_email}: {e}")
        return False

# --- Routes ---
@app.route("/")
def index():
    if "user_email" in session:
        user = get_user_by_email(session["user_email"])
        if not user: session.clear(); flash("Session invalid.", "error"); return redirect(url_for("login"))

        # Use the precise status constants
        if user.get("status") == STATUS_SUSPENDED: # <<< CHANGED
            return redirect(url_for("pending_activation"))
        elif user.get("status") == STATUS_ACTIVE:
            return redirect(url_for("app_frame"))
        else: # pending_email_confirmation, suspended, deactivated
            session.clear()
            flash("Your account requires attention. Please log in.", "info")
            return redirect(url_for("login"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_email" in session: # Already fully logged in (implies active user from previous logic)
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pwd   = request.form.get("password", "")

        if not email or not pwd:
            flash("Email and password are required.", "error")
            return render_template("login.html", email=email)

        user = get_user_by_email(email)
        if user and user.get("password") and verify_password(pwd, user.get("password")):
            if user.get("status") == STATUS_ACTIVE:
                session["user_email"] = email
                session["user_role"] = user.get("role", "user")
                session["user_full_name"] = user.get("full_name", email.split('@')[0])
                # backend.update_user_last_login(email) # You'd need this backend function
                flash("Logged in successfully!", "success")
                return redirect(url_for("app_frame"))
            elif user.get("status") == STATUS_SUSPENDED: # <<< CHANGED
                session["pending_email_for_status_check"] = email
                flash("Your account is awaiting admin approval.", "info")
                return redirect(url_for("pending_activation"))
            elif user.get("status") == STATUS_PENDING_EMAIL_CONFIRMATION:
                flash("Your email address has not been confirmed. Please check your inbox.", "warning")
                return render_template("login.html", email=email)
            elif user.get("status") == STATUS_DEACTIVATED:
                flash("Your account has been deactivated. Please contact support.", "error")
                return render_template("login.html", email=email)
            else: # Suspended, etc.
                flash(f"Your account is currently {user.get('status', 'unavailable')}. Please contact support.", "error")

        else:
            flash("Invalid credentials.", "error")
    return render_template("login.html")

MAX_EMAIL_RETRIES = 3
EMAIL_RETRY_DELAY_SECONDS = 5 # Wait 5 seconds between retries

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_email" in session: # If already logged in, redirect
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # --- Validations ---
        if not email or not password or not confirm_password:
            flash("All fields are required.", "error")
            return render_template("signup.html", email=email), 400
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html", email=email), 400
        if "@" not in email or "." not in email.split("@")[-1]: # Basic email check
            flash("Invalid email format.", "error")
            return render_template("signup.html", email=email), 400
        if len(password) < 6: # Basic password length check
            flash("Password must be at least 6 characters long.", "error")
            return render_template("signup.html", email=email), 400
        # --- End Validations ---

        # create_user in backend now sets status to STATUS_PENDING_EMAIL_CONFIRMATION
        success, message_from_backend, otp_to_send = create_user(email, password)
        
        if success:
            if not otp_to_send: # Should not happen if success is true
                app.logger.error(f"OTP not generated for {email} despite successful user creation call.")
                flash("An internal error occurred during OTP generation. Please try again or contact support.", "error")
                return redirect(url_for("signup"))

            email_sent_successfully = False
            for attempt in range(MAX_EMAIL_RETRIES):
                app.logger.info(f"Attempt {attempt + 1} to send confirmation email to {email}")
                print(f"EMAIL_SEND_ATTEMPT: Attempt {attempt + 1} for {email}") # For dev console visibility

                email_sent_this_attempt = send_system_email(
                    to_email=email,
                    subject="Your IntelLaw Email Confirmation Code",
                    template_name_no_ext="send_otp", # <<< NEW TEMPLATE
                    otp_code=otp_to_send # Pass OTP to email template
                )
                if email_sent_this_attempt:
                    email_sent_successfully = True; break
                else:
                    app.logger.warning(f"Failed to send confirmation email to {email} on attempt {attempt + 1}. Retrying in {EMAIL_RETRY_DELAY_SECONDS}s...")
                    print(f"EMAIL_SEND_FAIL: Failed attempt {attempt + 1} for {email}. Retrying...")
                    if attempt < MAX_EMAIL_RETRIES - 1: # Don't sleep after the last attempt
                        time.sleep(EMAIL_RETRY_DELAY_SECONDS) # Wait before retrying
            
            if email_sent_successfully:
                session["otp_confirm_email"] = email # Store email for OTP verification page
                flash("Registration initiated! A confirmation code has been sent to your email. Please enter it below.", "info")
                return redirect(url_for("confirm_otp_page"))
            else:
                app.logger.error(f"Failed to send confirmation email to {email} after {MAX_EMAIL_RETRIES} attempts.")
                print(f"EMAIL_SEND_FINAL_FAIL: All {MAX_EMAIL_RETRIES} attempts failed for {email}.")
                flash("Registration was successful, but we failed to send a confirmation email even after multiple attempts. Please contact support. Your account is created but requires manual email verification assistance.", "error")
                # Consider what to do here:
                # 1. Leave user as pending_email_confirmation. Admin might need to intervene or user contacts support.
                # 2. Optionally, delete the user record if email confirmation is absolutely critical and cannot be sent.
                #    (This would require a backend.delete_user_if_pending_email_confirmation(email) function)
                #    If deleting, ensure the user is informed their registration was not fully completed.
                #
                # For now, we'll leave the user record as is and redirect.
                return redirect(url_for("login")) # Or a specific error page
        else:
            flash(message_from_backend, "error") # E.g., "Email already registered..." or "DB error"
            return render_template("signup.html", email=email), 400
            
    return render_template("signup.html")

@app.route("/update-email-during-otp", methods=["GET"]) # Changed endpoint name to be more descriptive
def update_email_address_page(): # This is the function name used by url_for
    email_for_confirmation = session.get("otp_confirm_email")
    if not email_for_confirmation:
        flash("Your session for email update has expired. Please try signing up again.", "warning")
        return redirect(url_for("signup"))
    
    flash("The feature to update your email address at this stage is not yet implemented. If you made an error, please try signing up again with the correct email address.", "info")
    return redirect(url_for("signup"))

@app.route("/confirm-otp", methods=["GET", "POST"]) # <<< NEW ROUTE (OTP entry page)
def confirm_otp_page():
    email_for_confirmation = session.get("otp_confirm_email")
    if not email_for_confirmation:
        flash("Your session for OTP confirmation has expired or is invalid. Please start the signup process again.", "warning")
        return redirect(url_for("signup"))
    
    if request.method == "POST":
        submitted_otp = request.form.get("otp_code", "").strip() # Assuming JS combines into this hidden field
        
        if not submitted_otp or len(submitted_otp) != 6 or not submitted_otp.isdigit(): # Basic validation
            flash("Please enter a valid 6-digit OTP.", "error")
            return render_template("confirm_otp_form.html", email=email_for_confirmation)

        success, message = verify_otp_and_activate_user(email_for_confirmation, submitted_otp)
        if success:
            session.pop("otp_confirm_email", None) # Clear session variable
            flash(message, "success") # "Email confirmed... account is now active!"
            # Log them in directly or redirect to login
            # For direct login:
            # user = get_user_by_email(email_for_confirmation)
            # if user and user.get("status") == STATUS_ACTIVE:
            #     session["user_email"] = user.get("email")
            #     session["user_role"] = user.get("role")
            #     session["user_full_name"] = user.get("full_name")
            #     return redirect(url_for("app_frame"))
            return redirect(url_for("login"))
        else:
            flash(message, "error") # "Invalid OTP", "OTP Expired"
            # Keep them on the OTP page to try again or resend
            return render_template("confirm_otp_form.html", email=email_for_confirmation)

    return render_template("confirm_otp_form.html", email=email_for_confirmation)

@app.route("/resend-otp", methods=["POST"]) # <<< NEW ROUTE
def resend_otp():
    email_to_resend = session.get("otp_confirm_email")
    if not email_to_resend:
        flash("Cannot resend OTP: Your session is invalid. Please try signing up again.", "error")
        return redirect(url_for("signup")) 

    new_otp, message_from_backend = regenerate_otp_for_user(email_to_resend)

    if new_otp:
        email_sent_successfully = False
        for attempt in range(MAX_EMAIL_RETRIES): # Retry sending the new OTP
            email_sent = send_system_email(
                to_email=email_to_resend,
                subject="Your New IntelLaw Confirmation Code",
                template_name_no_ext="send_otp",
                otp_code=new_otp
            )
            if email_sent: email_sent_successfully = True; break
            if attempt < MAX_EMAIL_RETRIES -1 : time.sleep(EMAIL_RETRY_DELAY_SECONDS)
        
        if email_sent_successfully:
            flash(f"A new confirmation code has been sent to {email_to_resend}.", "info")
        else:
            flash(f"Failed to resend OTP to {email_to_resend} after multiple attempts. Please try again later or contact support.", "error")
    else:
        flash(message_from_backend, "error") # "User not found or not awaiting confirmation" etc.

    return redirect(url_for("confirm_otp_page")) # Stay on OTP entry page

@app.route("/confirm_email/<token>")
def confirm_email_route(token):
    try:
        # Token expires in 1 day (86400 seconds) by default with itsdangerous
        email = ts.loads(token, salt='email-confirm-salt', max_age=86400)
    except SignatureExpired:
        flash("The confirmation link has expired. Please try signing up again, or if you have an account, try logging in to request a new link (feature to be added).", "danger")
        return redirect(url_for("signup")) # Or a dedicated page to resend confirmation
    except BadTimeSignature: # Could be BadSignature or other itsdangerous errors too
        flash("The confirmation link is invalid or has been tampered with. Please ensure you used the correct link.", "danger")
        return redirect(url_for("signup"))
    except Exception as e: # Catch-all for other itsdangerous errors or unexpected issues
        app.logger.error(f"Token deserialization error: {e}")
        flash("The confirmation link is invalid.", "danger")
        return redirect(url_for("signup"))

    # Attempt to confirm the email in the database
    success, message_from_backend = confirm_user_email_in_db(email) # Backend function updates status
    
    if success:
        flash(f"{message_from_backend} You can now log in.", "success")
    else:
        flash(f"Email confirmation failed: {message_from_backend}", "danger")
        # Example: If message says "already confirmed and active", guide to login.
        # If "user not found", guide to signup.
    return redirect(url_for("login"))


@app.route("/pending_activation")
def pending_activation():
    if "user_email" in session:
        user_in_session = get_user_by_email(session["user_email"])
        if user_in_session and user_in_session.get("status") == STATUS_ACTIVE:
            return redirect(url_for("app_frame"))
        if user_in_session and user_in_session.get("status") != STATUS_SUSPENDED:
            session.clear()
            flash("Your account status has changed. Please log in again.", "info")
            return redirect(url_for("login"))

    email_to_check = session.pop("pending_email_for_status_check", None)
    user_data = None
    page_message = "Your account is awaiting admin approval. Please check back later."

    if email_to_check: user_data = get_user_by_email(email_to_check)
    
    if user_data:
        if user_data.get("status") == STATUS_ACTIVE:
            flash("Your account is now active! Please log in.", "success")
            return redirect(url_for("login"))
        elif user_data.get("status") == STATUS_PENDING_EMAIL_CONFIRMATION:
            flash("Your email is not yet confirmed. Please check your inbox.", "warning")
            return redirect(url_for("login"))
        elif user_data.get("status") == STATUS_DEACTIVATED:
            flash("Your account is deactivated. Contact support.", "error")
            return redirect(url_for("login"))
        elif user_data.get("status") != STATUS_SUSPENDED:
            flash(f"Account status: '{user_data.get('status')}'. Contact support.", "warning")
            return redirect(url_for("login"))
    elif "user_email" not in session and not email_to_check:
        flash("Please log in to check your account status.", "info")
        return redirect(url_for("login"))

    display_email = email_to_check or (session.get("user_email") if "user_email" in session else None)
    return render_template("pending_activation.html", message=page_message, email=display_email)


@app.route("/settings")
def settings():

    if "user_email" not in session:
        flash("Please log in to access settings.", "info")
        return redirect(url_for("login"))
    current_user_data = get_user_by_email(session["user_email"])
    if not current_user_data or current_user_data.get("email") != APP_ADMIN_EMAIL:
        flash("Access denied. Restricted to primary system administrator.", "error")
        return redirect(url_for("app_frame") if current_user_data else url_for("login"))

    db_users = get_all_users_from_db()
    processed_users = []
    # The 'can_delete' logic in the settings route should now reflect that delete means 'deactivate'
    # You might want to show 'deactivated' users differently or allow 'reactivation'.
    # For now, `can_delete` will be true if the user is not self/super-admin AND not already 'deactivated'.
    for user_data_from_db in db_users:
        name_to_use = user_data_from_db.get("full_name", user_data_from_db.get("email", "N/A").split('@')[0])
        
        can_action = True # General flag for enabling actions
        is_deletable = True # Specifically for the delete/deactivate button
        
        if user_data_from_db.get("email") == APP_ADMIN_EMAIL:
            can_action = False
            is_deletable = False
        if user_data_from_db.get("status") == STATUS_PENDING_EMAIL_CONFIRMATION:
            can_action = False
            is_deletable = True
        
        if user_data_from_db.get("status") == STATUS_DEACTIVATED:
            is_deletable = True # Cannot "delete" an already deactivated account, maybe "activate" instead

        processed_users.append({
            # ... (other fields) ...
            "id": user_data_from_db.get("_id"),
            "role": user_data_from_db.get("role", "user"),
            "name": name_to_use,
            "email": user_data_from_db.get("email"),
            "status": user_data_from_db.get("status", "unknown"), # This will now show the precise status
            "last_active": format_datetime_for_display(user_data_from_db.get("last_login_at")),
            "created_at": format_datetime_for_display(user_data_from_db.get("created_at")),
            "name_initials": get_name_initials(name_to_use),
            "avatar_url": user_data_from_db.get("avatar_url"),
            "can_change_status": can_action, # For enabling/disabling the status select
            "is_deletable": is_deletable # For the delete/deactivate button
        })
    
    status_order = {
        STATUS_PENDING_EMAIL_CONFIRMATION: 0,
        STATUS_SUSPENDED: 1,
        STATUS_ACTIVE: 2,
        STATUS_DEACTIVATED: 3
    }
    # ... (your sorting logic, ensure it uses the renamed status constants if needed) ...
    def get_original_created_at(user_dict):
        original_user = next((u for u in db_users if u.get("email") == user_dict["email"]), None)
        if original_user and isinstance(original_user.get("created_at"), datetime): return original_user.get("created_at")
        return datetime.min.replace(tzinfo=timezone.utc)
    processed_users.sort(key=get_original_created_at, reverse=True)
    processed_users.sort(key=lambda u: status_order.get(u["status"], 99))

    return render_template("settings.html", users=processed_users,
                           # Pass status constants to template for dropdown options if needed,
                           # though hardcoding in template is also an option for fixed statuses.
                           STATUS_PENDING_EMAIL_CONFIRMATION=STATUS_PENDING_EMAIL_CONFIRMATION,
                           STATUS_ACTIVE=STATUS_ACTIVE,
                           STATUS_SUSPENDED=STATUS_SUSPENDED,
                           STATUS_DEACTIVATED=STATUS_DEACTIVATED
                           )


@app.route("/admin/update_user_status", methods=["POST"])
def admin_update_user_status():
    if "user_email" not in session:
        return {"success": False, "message": "Unauthorized - Not logged in"}, 403
    
    admin_user = get_user_by_email(session["user_email"])
    if not admin_user or admin_user.get("role") != "admin":
        return {"success": False, "message": "Unauthorized - Not an admin"}, 403

    data = request.get_json()
    target_user_email = data.get("email")
    new_status = data.get("new_status")

    if not target_user_email or not new_status:
        return {"success": False, "message": "Email and new status are required."}, 400

    # Prevent admin from deactivating/suspending their own account via this endpoint
    if target_user_email == admin_user.get("email") and new_status != STATUS_ACTIVE:
        # Admins can change their own status to active (e.g., if it was accidentally changed elsewhere)
        # But they cannot make themselves non-active through this specific UI flow.
        return {"success": False, "message": "Admins cannot set their own status to non-active using this form."}, 400

    success, message_from_backend = update_user_status_in_db(target_user_email, new_status)
    
    if success:
        return {"success": True, "message": message_from_backend, "email": target_user_email, "new_status": new_status}, 200
    else:
        return {"success": False, "message": message_from_backend}, 400

@app.route("/admin/hard_delete_user", methods=["POST"]) # <<< NEW ROUTE
def admin_hard_delete_user():
    if "user_email" not in session:
        flash("Unauthorized. Please log in.", "error")
        return redirect(url_for("login"))

    admin_user = get_user_by_email(session["user_email"])
    if not admin_user or admin_user.get("role") != "admin":
        flash("Access denied. Admins only.", "error")
        return redirect(url_for("app_frame"))

    target_user_email_to_delete = request.form.get("email_to_hard_delete")

    if not target_user_email_to_delete:
        flash("No user specified for deletion.", "error")
        return redirect(url_for("settings"))

    # Security Check: Prevent admin from hard deleting themselves or the primary admin
    if target_user_email_to_delete == admin_user.get("email"):
        flash("You cannot permanently delete your own account.", "error")
        return redirect(url_for("settings"))
    
    if target_user_email_to_delete == APP_ADMIN_EMAIL:
        flash(f"The primary admin account ({APP_ADMIN_EMAIL}) cannot be permanently deleted.", "error")
        return redirect(url_for("settings"))

    success, message = hard_delete_user_from_db(target_user_email_to_delete)

    if success:
        flash(message, "success")
    else:
        flash(f"Failed to delete user: {message}", "error")
    
    return redirect(url_for("settings"))

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/app")
def app_frame():
    if "user_email" not in session:
        return redirect(url_for("login"))
    
    user = get_user_by_email(session["user_email"]) # Fetch fresh data
    if not user: # User in session but not in DB (e.g. deleted during session)
        session.clear()
        flash("Session error or account not found. Please log in again.", "error")
        return redirect(url_for("login"))

    if user.get("status") != STATUS_ACTIVE:
        # Handle non-active users based on their specific status by redirecting appropriately
        if user.get("status") == STATUS_SUSPENDED:
            session["pending_email_for_status_check"] = user.get("email")
            flash("Your account is still pending admin approval.", "info")
            return redirect(url_for("pending_activation"))
        elif user.get("status") == STATUS_PENDING_EMAIL_CONFIRMATION:
            session.clear() # Log them out, they need to confirm email
            flash("Your email needs to be confirmed. Please check your inbox or log in again.", "warning")
            return redirect(url_for("login"))
        elif user.get("status") == STATUS_DEACTIVATED: # <<< ADDED CHECK
            session.clear()
            flash("Your account has been deactivated. Please contact support.", "error")
            return redirect(url_for("login"))
        
    # If we reach here, user status is ACTIVE
    user_role = session.get("user_role", "user") # Role was set at login for active user
    user_full_name = session.get("user_full_name", user.get("email").split('@')[0]) # Get full name for display

    # Pass user_full_name and user_role to the template if your iframe page uses them
    return render_template(
        "gradio_iframe.html",
        user_role=user_role,
        user_full_name=user_full_name, # Optional: if your template uses it
        gradio_app_url=GRADIO_APP_URL
    )

if __name__ == "__main__":
    print(f"Launching Flask server for authentication and iframe on 0.0.0.0:5000...")
    print(f"Gradio app is expected to be running at: {GRADIO_APP_URL}")
    app.run(host="0.0.0.0", port=5000, debug=True)