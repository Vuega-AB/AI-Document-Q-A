from functools import wraps
import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta # Ensure datetime, timezone, timedelta are imported
from flask_mail import Mail, Message # ADDED
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature # ADDED
import time # Add this if not present
import pyotp # ADDED
import qrcode # ADDED
import io # ADDED (for QR code image generation)
import base64

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
    get_full_user_for_auth,
    # MONGO_URI, # If used directly in Flask app, else remove (backend handles its own)
    get_all_users_from_db,
    set_user_totp_secret, enable_user_2fa, verify_totp_code,
    update_user_status_in_db,
    # confirm_user_email_in_db, # <<< ADDED
    STATUS_PENDING_EMAIL_CONFIRMATION,
    STATUS_ACTIVE,
    STATUS_SUSPENDED

)

load_dotenv()

APP_ADMIN_EMAIL = os.getenv("APP_ADMIN_EMAIL", "saragaballa2002@gmail.com")
APP_ADMIN_PASSWORD = os.getenv("APP_ADMIN_PASSWORD", "11112002")
GRADIO_APP_URL = os.getenv("GRADIO_APP_URL", "http://localhost:7860")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

if not FLASK_SECRET_KEY:
    FLASK_SECRET_KEY = "dev_default_unsafe_secret_key_CHANGE_ME_IMMEDIATELY"


app = Flask(__name__, template_folder="templates")
app.secret_key = FLASK_SECRET_KEY

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

# main_flask_app.py

def require_login(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        current_endpoint = request.endpoint
        
        if "force_2fa_setup_email" in session:
            allowed_endpoints_during_forced_setup = ['setup_2fa_page', 'logout', 'static']
            if current_endpoint in allowed_endpoints_during_forced_setup:
                print(f"DEBUG [@require_login]: FORCED 2FA. Allowing endpoint: {current_endpoint}")
                return f(*args, **kwargs) # <<< ALLOWS setup_2fa_page TO RUN
            else:
                flash("Please complete Two-Factor Authentication setup to continue.", "warning")
                print(f"DEBUG [@require_login]: FORCED 2FA. Denied endpoint: {current_endpoint}. Redirecting to setup.")
                return redirect(url_for('setup_2fa_page', new="true")) # Keep them on setup

        if "2fa_login_email" in session: # No need to check session.get("user_email") == session.get("2fa_login_email") here
            allowed_endpoints_during_2fa_challenge = ['login_2fa_page', 'resend_otp', 'logout', 'static']
            if current_endpoint in allowed_endpoints_during_2fa_challenge:
                print(f"DEBUG [@require_login]: 2FA CHALLENGE. Allowing endpoint: {current_endpoint}")
                return f(*args, **kwargs)
            else:
                flash("Please complete your 2FA login.", "info")
                print(f"DEBUG [@require_login]: 2FA CHALLENGE. Denied endpoint: {current_endpoint}. Redirecting to 2FA login.")
                return redirect(url_for('login_2fa_page'))

        if "user_email" not in session:
            flash("Please log in to access this page.", "info")
            print(f"DEBUG [@require_login]: NO SESSION (user_email). Denied endpoint: {current_endpoint}. Redirecting to login.")
            return redirect(url_for('login', next=request.url))
        user_for_check = get_full_user_for_auth(session["user_email"])
        if not user_for_check:
            session.clear(); flash("Session invalid (user not found).", "error"); return redirect(url_for('login'))
        
        if user_for_check.get("status") != STATUS_ACTIVE:
            session.clear(); flash(f"Account not active.", "warning"); return redirect(url_for('login'))
        
        if user_for_check.get("is_2fa_enabled") and not session.get("2fa_verified_this_session"):
            allowed_endpoints_before_2fa_verify = [
                'setup_2fa_page', 
                'login_2fa_page', 'resend_otp', 'logout', 'static' # Ensure all 2FA management and escape routes are here
            ]
            if current_endpoint not in allowed_endpoints_before_2fa_verify:
                session["2fa_login_email"] = session["user_email"] 
                flash("Two-Factor Authentication verification required for this page.", "info")
                print(f"DEBUG [@require_login]: SESSION OK, 2FA enabled but not verified. Endpoint: {current_endpoint}. Redirecting to 2FA login.")
                return redirect(url_for("login_2fa_page"))
            
        print(f"DEBUG [@require_login]: Access GRANTED to endpoint: {current_endpoint} for user {session['user_email']}")
        return f(*args, **kwargs)
    return decorated_function

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
def index(): # ... (Logic based on status and 2FA session state)
    if "user_email" in session:
        if "force_2fa_setup_email" in session:
            print("IN INDEX TO setup_2fa_page ")
            return redirect(url_for("setup_2fa_page"))
        
        if "2fa_login_email" in session and session["2fa_login_email"] == session.get("user_email"):
            return redirect(url_for("login_2fa_page")) # Mid-2FA login

        user = get_user_by_email(session["user_email"])
        # if not user: session.clear(); flash("Session invalid.", "error"); return redirect(url_for("login"))
        if user.get("status") == STATUS_ACTIVE:
            return redirect(url_for("app_frame"))
        else:
            session.clear() 
            flash("Account requires attention. Please log in.", "info")
            return redirect(url_for("login"))
    return redirect(url_for("login"))

# main_flask_app.py

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_email" in session and "2fa_login_email" not in session and "force_2fa_setup_email" not in session:
        print("IAM IN THE LOGIN FUNCTION, GOING TO INDEX")
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pwd = request.form.get("password", "")
        if not email or not pwd:
            flash("Email and password required.", "error"); return render_template("login.html", email=email)
        
        db_user_for_login = get_full_user_for_auth(email)

        if db_user_for_login and verify_password(pwd, db_user_for_login.get("password")):
            user_status = db_user_for_login.get("status")
            user_role = db_user_for_login.get("role", "user") # Get role early

            # First, handle non-active states for ALL users
            if user_status == STATUS_PENDING_EMAIL_CONFIRMATION:
                session["otp_confirm_email"] = email 
                flash("Email not confirmed. Enter OTP sent to your email.", "warning")
                return redirect(url_for("confirm_otp_page"))
            elif user_status == STATUS_SUSPENDED:
                return redirect(url_for("pending_activation")) 
            elif user_status != STATUS_ACTIVE: # Any other non-active status
                flash(f"Account status '{user_status}' prevents login. Contact support.", "error")
                return render_template("login.html", email=email)
            
            is_2fa_user_enabled = db_user_for_login.get("is_2fa_enabled") is True 
            is_app_admin_bypassing_2fa = (email == APP_ADMIN_EMAIL)
            has_user_completed_initial_login = db_user_for_login.get("has_completed_initial_login") is True

            if not is_app_admin_bypassing_2fa:
                session["2fa_login_email"] = email 
                print(f"DEBUG: User {email} (Role: {user_role}) is ACTIVE & 2FA enabled. Redirecting to 2FA page.")
                
                if not has_user_completed_initial_login:
                    print(f"DEBUG: User {email} has not completed initial login. Redirecting to 2FA setup page.")
                    session["force_2fa_setup_email"] = email
                    session["user_email_pre_2fa_force"] = email
                    return redirect(url_for("setup_2fa_page", new="true")) 
                
                return redirect(url_for("login_2fa_page"))
            else:
                if is_app_admin_bypassing_2fa:
                    print(f"DEBUG: APP_ADMIN_EMAIL ({email}) logged in, bypassing 2FA prompt (2FA status: {is_2fa_user_enabled}).")
                    session["2fa_verified_this_session"] = True

                    print(f"DEBUG: User {email} (Role: {user_role}) logging in. 2FA enabled: {is_2fa_user_enabled}, Bypassed: {is_app_admin_bypassing_2fa}, Initial Login Completed: {db_user_for_login.get('has_completed_initial_login')}")
                    session["user_email"] = email
                    session["user_role"] = user_role # Use role fetched earlier
                    session["user_full_name"] = db_user_for_login.get("full_name", email.split('@')[0])
                    if not is_2fa_user_enabled : # If no 2FA, ensure flag is not set or is false
                        session.pop("2fa_verified_this_session", None)
                    flash("Logged in successfully!", "success")
                    return redirect(url_for("app_frame"))
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
                return redirect(url_for("login")) # Or a specific error page
        else:
            flash(message_from_backend, "error")
            return render_template("signup.html", email=email), 400
            
    return render_template("signup.html")

@app.route("/login/2fa", methods=["GET", "POST"])
def login_2fa_page():
    if "user_email" in session and "2fa_login_email" not in session :
        return redirect(url_for("index"))
    
    email_for_2fa = session.get("2fa_login_email")
    if not email_for_2fa:
        flash("2FA session error. Please log in again.", "error"); return redirect(url_for("login"))

    # Use the new backend function
    db_user = get_full_user_for_auth(email_for_2fa) # <<< CHANGED
    
    if not db_user or not db_user.get("is_2fa_enabled") or not db_user.get("totp_secret"):
        session.pop("2fa_login_email", None)
        flash("2FA not configured or error. Log in again.", "error"); return redirect(url_for("login"))

    if request.method == "POST":
        code = request.form.get("totp_code", "").strip()
        recovery_mode = "use_recovery_code" in request.form # Check if the button was clicked
        login_success, message = False, "Invalid code."

        if recovery_mode:
            if code: 
                login_success, message = verify_recovery_code(email_for_2fa, code) # Uses backend
            else: 
                flash("Recovery code cannot be empty.", "error")
                message = "Recovery code cannot be empty." # ensure message is set
        else: # TOTP code mode
            if code:
                # verify_totp_code from backend takes the plaintext secret
                if verify_totp_code(db_user.get("totp_secret"), code):
                    login_success, message = True, "Logged in successfully with 2FA!"
                else:
                    message = "Invalid authentication code."
            else: 
                flash("Authentication code cannot be empty.", "error")
                message = "Authentication code cannot be empty."
        
        if login_success:
            session.pop("2fa_login_email", None)
            session["user_email"] = db_user["email"]
            session["user_role"] = db_user.get("role", "user")
            session["user_full_name"] = db_user.get("full_name", db_user["email"].split('@')[0])
            flash(message, "success"); return redirect(url_for("app_frame"))
        else:
            flash(message, "error")
            # Stay on 2FA page to allow another attempt
    return render_template("login_2fa.html", email=email_for_2fa)

# # main_flask_app.py
# @app.route("/profile")
# @require_login
# def profile_page():
#     user_for_profile = get_user_by_email(session["user_email"]) # This is fine as it gets 'is_2fa_enabled'
#     if not user_for_profile: return redirect(url_for("logout"))
#     return render_template("profile.html", user=user_for_profile)


@app.route("/2fa/setup", methods=["GET", "POST"])
@require_login 
def setup_2fa_page():
    print("IAM IN THE setup_2fa_page FUNCTION")
    is_forced_setup = "force_2fa_setup_email" in session
    current_user_email = session.get("force_2fa_setup_email") or session.get("user_email")

    if not current_user_email:
        flash("User session not found for 2FA setup. Please log in.", "error")
        return redirect(url_for("login"))

    db_user = get_full_user_for_auth(current_user_email)
    if not db_user:
        flash("User not found in database.", "error")
        session.clear()
        return redirect(url_for("login"))
    
    if db_user.get("is_2fa_enabled") and not is_forced_setup:
        flash("Two-Factor Authentication is already enabled for your account.", "info")
        return redirect(url_for("2fa_login_page"))

    if request.method == "POST":
        submitted_code = request.form.get("totp_code", "").strip()
        totp_secret_to_verify = session.get("2fa_setup_secret")

        if not totp_secret_to_verify:
            flash("2FA setup session expired. Please scan the QR code and try again.", "error")
            if is_forced_setup:
                session.pop("force_2fa_setup_email", None)
                session.pop("user_email_pre_2fa_force", None)
            session.pop("2fa_setup_secret", None)
            return redirect(url_for("setup_2fa_page", new="true")) 

        if not submitted_code:
             flash("Verification code cannot be empty.", "error")
        
        elif verify_totp_code(totp_secret_to_verify, submitted_code):
            set_ok, _ = set_user_totp_secret(current_user_email, totp_secret_to_verify)
            if not set_ok:
                flash("Failed to save 2FA configuration. Please try again.", "error")
                return redirect(url_for("setup_2fa_page", new="true"))
            
            enable_ok, enable_msg = enable_user_2fa(current_user_email) 
            
            if enable_ok:
                session.pop("2fa_setup_secret", None)
                session["2fa_verified_this_session"] = True
                
                if is_forced_setup:
                    original_email_for_login = session.pop("user_email_pre_2fa_force", None)
                    if original_email_for_login == current_user_email: 
                        session["user_email"] = current_user_email
                        session["user_role"] = db_user.get("role", "user")
                        session["user_full_name"] = db_user.get("full_name", current_user_email.split('@')[0])
                        session.pop("force_2fa_setup_email", None)
                        flash("2FA enabled successfully! You are now logged in.", "success")
                        return redirect(url_for("app_frame")) 
                    else:
                        app.logger.error("Email mismatch during forced 2FA completion.")
                        session.clear()
                        flash("A critical error occurred during login after 2FA setup. Please log in again.", "error")
                        return redirect(url_for("login"))
                else: # Voluntary setup from profile
                    flash("2FA enabled successfully!", "success")
                    session["2fa_login_email"] = current_user_email 
                    session.pop("2fa_verified_this_session", None) # Ensure challenge happens
                    flash("2FA enabled! Please verify with your authenticator app to complete login.", "info")
                    return redirect(url_for("login_2fa_page"))
            else: 
                flash(f"Failed to fully enable 2FA: {enable_msg}. Please try again.", "error")
                disable_user_2fa(current_user_email) # Attempt to roll back
                return redirect(url_for("setup_2fa_page", new="true"))
        else: # verify_totp_code returned False
            flash("Invalid authentication code. Please check your authenticator app and try again.", "error")

    if "2fa_setup_secret" not in session or request.args.get("new") == "true":
        session["2fa_setup_secret"] = pyotp.random_base32()
    
    totp_secret_for_qr = session["2fa_setup_secret"] # Use the one from session (either new or existing for this attempt)
    
    provisioning_uri = pyotp.totp.TOTP(totp_secret_for_qr).provisioning_uri(
        name=current_user_email, 
        issuer_name="IntelLaw" # IMPORTANT: Use your actual application name
    )
    
    img = qrcode.make(provisioning_uri)
    buf = io.BytesIO()
    img.save(buf)
    buf.seek(0)
    qr_code_data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('ascii')
    
    return render_template("setup_2fa.html", 
                           qr_code_data_url=qr_code_data_url, 
                           otp_secret_display=totp_secret_for_qr, # For manual entry
                           is_forced_setup=is_forced_setup,
                           user_email_for_display=current_user_email)

# @app.route("/2fa/recovery-codes")
# @require_login # Ensures user is at least partially authenticated (e.g., password verified)
# def show_recovery_codes_page():
#     was_forced_setup = session.pop("was_forced_2fa_setup_just_completed", False) 

#     # Recovery codes are stored in session temporarily after successful 2FA enable.
#     recovery_codes = session.pop("recovery_codes_to_display", None)

#     if not recovery_codes:
#         # This can happen if the user refreshes the page or navigates here directly later.
#         flash("Recovery codes are shown only once immediately after enabling 2FA. If you've lost them, you may need to disable and then re-enable 2FA to generate new codes.", "warning")
#         return redirect(url_for("profile_page")) # Or app_frame if no profile_page

#     # Determine where the "Done" button should lead.
#     if was_forced_setup:
#         done_redirect_url = url_for("app_frame")
#         if "user_email" not in session and session.get("user_email_pre_2fa_force"): # Double check if needed
#             email = session.pop("user_email_pre_2fa_force")
#             db_user = get_full_user_for_auth(email) # Get full details
#             if db_user:
#                 session["user_email"] = db_user.get("email")
#                 session["user_role"] = db_user.get("role", "user")
#                 session["user_full_name"] = db_user.get("full_name", db_user.get("email").split('@')[0])
#     else:
#         # If it was a voluntary 2FA setup from their profile, "Done" takes them back to profile.
#         done_redirect_url = url_for("profile_page")

#     return render_template("recovery_codes.html", 
#                            recovery_codes=recovery_codes, 
#                            done_redirect_url=done_redirect_url)

# @app.route("/profile/2fa/disable", methods=["POST"])
# @require_login
# def disable_2fa_route():
#     current_user_email = session["user_email"]
#     # Use the new backend function to get password for verification
#     db_user = get_full_user_for_auth(current_user_email) # <<< CHANGED
    
#     password_to_confirm = request.form.get("password_confirm_2fa_disable")

#     if not db_user or not password_to_confirm or not verify_password(password_to_confirm, db_user.get("password")):
#         flash("Incorrect password. 2FA not disabled.", "error")
#         return redirect(url_for("profile_page"))

#     success, message = disable_user_2fa(current_user_email) # Uses backend
#     flash(message, "success" if success else "error")
#     return redirect(url_for("profile_page"))


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
            flash(message, "success") # "Email confirmed... 
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

# @app.route("/confirm_email/<token>")
# def confirm_email_route(token):
#     try:
#         # Token expires in 1 day (86400 seconds) by default with itsdangerous
#         email = ts.loads(token, salt='email-confirm-salt', max_age=86400)
#     except SignatureExpired:
#         flash("The confirmation link has expired. Please try signing up again, or if you have an account, try logging in to request a new link (feature to be added).", "danger")
#         return redirect(url_for("signup")) # Or a dedicated page to resend confirmation
#     except BadTimeSignature: # Could be BadSignature or other itsdangerous errors too
#         flash("The confirmation link is invalid or has been tampered with. Please ensure you used the correct link.", "danger")
#         return redirect(url_for("signup"))
#     except Exception as e: # Catch-all for other itsdangerous errors or unexpected issues
#         app.logger.error(f"Token deserialization error: {e}")
#         flash("The confirmation link is invalid.", "danger")
#         return redirect(url_for("signup"))

#     # Attempt to confirm the email in the database
#     success, message_from_backend = confirm_user_email_in_db(email) # Backend function updates status
    
#     if success:
#         flash(f"{message_from_backend} You can now log in.", "success")
#     else:
#         flash(f"Email confirmation failed: {message_from_backend}", "danger")
#         # Example: If message says "already confirmed and active", guide to login.
#         # If "user not found", guide to signup.
#     return redirect(url_for("login"))


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
        STATUS_ACTIVE: 2
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
                           STATUS_SUSPENDED=STATUS_SUSPENDED
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
    if not user or user.get("status") != STATUS_ACTIVE:
        session.clear() # Defensive clear
        flash("Access issue. Please log in.", "error")
        return redirect(url_for("login"))
    
    # If 2FA was pending but they navigated here, redirect to 2FA page.
    if "2fa_login_email" in session and session["2fa_login_email"] == user:
        flash("Please complete 2FA login.", "info")
        return redirect(url_for("login_2fa_page"))

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