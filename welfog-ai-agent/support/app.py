"""
Welfog support application entrypoint.

- Public chat + MySQL history: routes.chat_routes
- Admin + SQLite (knowledge files): routes.admin_routes, admin_models, extensions
- AI / KB / APIs: services/* and utils/* (not duplicated here)
"""
import os
from secrets import token_hex

from dotenv import load_dotenv
from flask import Flask

from admin_models import AdminUser
from extensions import db, login_manager
from routes.admin_routes import register_admin_routes
from routes.chat_routes import register_chat_routes
from services.mysql_service import init_mysql_chat_schema
from support_paths import BASE_DIR

load_dotenv()


def create_app():
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY") or token_hex(32)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "welfog_v2.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["REMEMBER_COOKIE_HTTPONLY"] = True
    app.config["REMEMBER_COOKIE_SAMESITE"] = "Lax"

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "admin_login"

    @login_manager.user_loader
    def load_user(user_id):
        return AdminUser.query.get(int(user_id))

    init_mysql_chat_schema()
    register_chat_routes(app)
    register_admin_routes(app)
    return app


app = create_app()


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(port=5000, debug=True)
