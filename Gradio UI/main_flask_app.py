import os
from flask import Flask, request, render_template, redirect, url_for, session, flash
from dotenv import load_dotenv

# your existing backend imports…
from backend import (
    get_user_by_email,
    verify_password,
)

load_dotenv()
app = Flask(__name__, template_folder="templates")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        pwd   = request.form["password"]
        user  = get_user_by_email(email)
        if user and verify_password(pwd, user["password"]):
            session["user_email"] = email
            return redirect(url_for("app_frame"))
        flash("Invalid credentials", "error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/app")
def app_frame():
    if "user_email" not in session:
        return redirect(url_for("login"))
    # Simply render an HTML page with an iframe to the Gradio server
    return render_template("gradio_iframe.html")

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000)
