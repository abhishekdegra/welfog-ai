import os

import click
from flask import abort, flash, redirect, render_template, request, url_for
from flask.cli import with_appcontext
from flask_login import login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from admin_models import AdminUser
from support_paths import KNOWLEDGE_DIR
from extensions import db
from services.kb_service import get_allowed_knowledge_filenames, refresh_knowledge_cache
from utils.validators import _safe_knowledge_filename


def register_admin_routes(app):
    @app.route("/welfog-admin-login", methods=["GET", "POST"])
    def admin_login():
        if request.method == "POST":
            u = (request.form.get("username") or "").strip()
            p = request.form.get("password") or ""
            user = AdminUser.query.filter_by(username=u).first()

            if user:
                valid = check_password_hash(user.password, p) if user.password.startswith("pbkdf2:") else (user.password == p)
                if valid:
                    if not user.password.startswith("pbkdf2:"):
                        user.password = generate_password_hash(p, method="pbkdf2:sha256")
                        db.session.commit()
                    login_user(user)
                    return redirect(url_for("admin_dashboard"))

            flash("Invalid username or password.", "error")
        return render_template("admin_login.html")

    @app.route("/welfog-admin")
    @login_required
    def admin_dashboard():
        return redirect(url_for("admin_knowledge"))

    @app.route("/welfog-admin/knowledge")
    @login_required
    def admin_knowledge():
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR)
        files = get_allowed_knowledge_filenames()
        return render_template("admin_knowledge.html", files=files, nav_active="knowledge")

    @app.route("/welfog-admin/settings")
    @login_required
    def admin_settings():
        return render_template(
            "admin_settings.html",
            nav_active="settings",
            knowledge_dir=os.path.abspath(KNOWLEDGE_DIR),
        )

    @app.route("/welfog-admin/new-file", methods=["GET", "POST"])
    @login_required
    def admin_add_file():
        if request.method == "POST":
            raw_name = (request.form.get("name") or "").strip()
            initial_content = request.form.get("content") or ""
            safe_name = _safe_knowledge_filename(raw_name)
            if not safe_name:
                flash("Invalid filename. Use letters, numbers, underscore or hyphen.", "error")
                return render_template("admin_new_file.html", nav_active="knowledge")

            filename = f"{safe_name}.txt"
            file_path = os.path.abspath(os.path.join(KNOWLEDGE_DIR, filename))
            if not file_path.startswith(os.path.abspath(KNOWLEDGE_DIR) + os.sep):
                abort(403)
            if os.path.exists(file_path):
                flash("File already exists. Choose a different name.", "error")
                return render_template("admin_new_file.html", nav_active="knowledge")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(initial_content.strip())
            refresh_knowledge_cache()
            flash(f"Created {filename} successfully.", "success")
            return redirect(url_for("admin_knowledge"))
        return render_template("admin_new_file.html", nav_active="knowledge")

    @app.route("/welfog-admin/edit/<filename>", methods=["GET", "POST"])
    @login_required
    def edit_kb_file(filename):
        allowed_files = set(get_allowed_knowledge_filenames())
        if filename not in allowed_files:
            abort(404)
        file_path = os.path.abspath(os.path.join(KNOWLEDGE_DIR, filename))
        if not file_path.startswith(os.path.abspath(KNOWLEDGE_DIR) + os.sep):
            abort(403)
        if request.method == "POST":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(request.form["content"])
            refresh_knowledge_cache()
            flash(f"Saved {filename} and refreshed index.", "success")
            return redirect(url_for("admin_knowledge"))

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return render_template("admin_edit.html", filename=filename, content=content, nav_active="knowledge")

    @app.route("/welfog-admin/delete/<filename>", methods=["POST"])
    @login_required
    def delete_kb_file(filename):
        allowed_files = set(get_allowed_knowledge_filenames())
        if filename not in allowed_files:
            abort(404)
        file_path = os.path.abspath(os.path.join(KNOWLEDGE_DIR, filename))
        if not file_path.startswith(os.path.abspath(KNOWLEDGE_DIR) + os.sep):
            abort(403)
        try:
            os.remove(file_path)
        except OSError:
            flash("Could not delete file.", "error")
            return redirect(url_for("admin_knowledge"))
        refresh_knowledge_cache()
        flash(f"Deleted {filename}.", "success")
        return redirect(url_for("admin_knowledge"))

    @app.route("/admin-logout")
    @login_required
    def admin_logout():
        logout_user()
        return redirect(url_for("admin_login"))

    @app.cli.command("create-admin")
    @click.argument("username")
    @click.argument("password")
    @with_appcontext
    def create_admin(username, password):
        db.create_all()
        username = (username or "").strip()
        if len(username) < 3:
            print("Username must be at least 3 chars.")
            return
        if len(password or "") < 8:
            print("Password must be at least 8 chars.")
            return
        existing = AdminUser.query.filter_by(username=username).first()
        if existing:
            print(f"Admin '{username}' already exists.")
            return
        new_admin = AdminUser(username=username, password=generate_password_hash(password, method="pbkdf2:sha256"))
        db.session.add(new_admin)
        db.session.commit()
        print(f"Admin '{username}' created successfully.")
