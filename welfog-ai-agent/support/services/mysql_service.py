import json
import os
from secrets import token_hex

import pymysql


def get_mysql_connection():
    """Public chat history MySQL (XAMPP). Override via .env: MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE."""
    try:
        return pymysql.connect(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "welfog_ai"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
        )
    except Exception as e:
        # Avoid emoji in logs (Windows console encoding can crash on unicode)
        print(f"MySQL Connection Error: {e}")
        return None


def generate_chat_token():
    return token_hex(16)


def init_mysql_chat_schema():
    """chat_sessions = sidebar / chat id; chats = har message ka row (chat_id, chat_data JSON)."""
    conn = get_mysql_connection()
    if not conn:
        print("MySQL unreachable — chat save/load will not work until DB is running.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    chat_token VARCHAR(32) NOT NULL,
                    user_id VARCHAR(128) NOT NULL,
                    title VARCHAR(512) NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_user_created (user_id, created_at),
                    UNIQUE KEY uq_chat_token (chat_token)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    chat_token VARCHAR(32) NOT NULL,
                    chat_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_chat_token (chat_token),
                    INDEX idx_chat_messages (chat_token, id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )

            # Backfill random chat tokens for older tables
            cur.execute("SHOW COLUMNS FROM chat_sessions LIKE 'chat_token'")
            if not cur.fetchone():
                cur.execute("ALTER TABLE chat_sessions ADD COLUMN chat_token VARCHAR(32) NULL")
                cur.execute("SELECT id FROM chat_sessions")
                for row in cur.fetchall():
                    cur.execute(
                        "UPDATE chat_sessions SET chat_token = %s WHERE id = %s",
                        (token_hex(16), row["id"]),
                    )
                cur.execute("ALTER TABLE chat_sessions MODIFY chat_token VARCHAR(32) NOT NULL")
                cur.execute("SHOW INDEX FROM chat_sessions WHERE Key_name = 'uq_chat_token'")
                if not cur.fetchone():
                    cur.execute("ALTER TABLE chat_sessions ADD UNIQUE KEY uq_chat_token (chat_token)")
            else:
                cur.execute("SELECT id FROM chat_sessions WHERE chat_token IS NULL OR chat_token = ''")
                for row in cur.fetchall():
                    cur.execute(
                        "UPDATE chat_sessions SET chat_token = %s WHERE id = %s",
                        (token_hex(16), row["id"]),
                    )
                cur.execute("ALTER TABLE chat_sessions MODIFY chat_token VARCHAR(32) NOT NULL")
                cur.execute("SHOW INDEX FROM chat_sessions WHERE Key_name = 'uq_chat_token'")
                if not cur.fetchone():
                    cur.execute("ALTER TABLE chat_sessions ADD UNIQUE KEY uq_chat_token (chat_token)")

            cur.execute("SHOW COLUMNS FROM chats LIKE 'chat_token'")
            if not cur.fetchone():
                cur.execute("ALTER TABLE chats ADD COLUMN chat_token VARCHAR(32) NULL")
                cur.execute(
                    "UPDATE chats c JOIN chat_sessions s ON c.chat_id = s.id SET c.chat_token = s.chat_token"
                )
                cur.execute("UPDATE chats SET chat_id = chat_token WHERE chat_id IS NULL OR chat_id = ''")
                cur.execute("ALTER TABLE chats MODIFY chat_token VARCHAR(32) NOT NULL")
                cur.execute("SHOW INDEX FROM chats WHERE Key_name = 'uq_chat_token'")
                if not cur.fetchone():
                    cur.execute("ALTER TABLE chats ADD UNIQUE KEY uq_chat_token (chat_token)")
            else:
                cur.execute("UPDATE chats c JOIN chat_sessions s ON c.chat_id = s.id SET c.chat_token = s.chat_token WHERE c.chat_token IS NULL OR c.chat_token = ''")
                cur.execute("UPDATE chats SET chat_id = chat_token WHERE chat_id IS NULL OR chat_id = ''")
                cur.execute("ALTER TABLE chats MODIFY chat_token VARCHAR(32) NOT NULL")
                cur.execute("SHOW INDEX FROM chats WHERE Key_name = 'uq_chat_token'")
                if not cur.fetchone():
                    cur.execute("ALTER TABLE chats ADD UNIQUE KEY uq_chat_token (chat_token)")
        conn.commit()
    except Exception as e:
        print(f"MySQL schema init error: {e}")
    finally:
        conn.close()


def db_store_message(chat_id, sender, message):
    conn = get_mysql_connection()
    if not conn:
        return

    try:
        sid = str(chat_id)
        numeric_chat_id = int(sid) if sid.isdigit() and len(sid) < 20 else None
        with conn.cursor() as cursor:
            if numeric_chat_id is not None:
                cursor.execute(
                    "SELECT chat_data FROM chats WHERE chat_token = %s OR chat_id = %s LIMIT 1",
                    (chat_id, numeric_chat_id),
                )
            else:
                cursor.execute(
                    "SELECT chat_data FROM chats WHERE chat_token = %s LIMIT 1",
                    (chat_id,),
                )
            row = cursor.fetchone()

            new_message = {"sender": sender, "text": message}
            if row and row.get("chat_data"):
                try:
                    chat_data = json.loads(row["chat_data"])
                except (TypeError, json.JSONDecodeError):
                    chat_data = []

                if isinstance(chat_data, dict):
                    chat_data = [chat_data]
                elif not isinstance(chat_data, list):
                    chat_data = []

                chat_data.append(new_message)
                chat_data_json = json.dumps(chat_data, ensure_ascii=False)
                cursor.execute(
                    "UPDATE chats SET chat_data = %s, chat_id = %s, updated_at = NOW() WHERE chat_token = %s OR chat_id = %s",
                    (chat_data_json, chat_id, chat_id, numeric_chat_id if numeric_chat_id is not None else -1),
                )
            else:
                chat_data_json = json.dumps([new_message], ensure_ascii=False)
                cursor.execute(
                    "INSERT INTO chats (chat_token, chat_id, chat_data, created_at, updated_at) VALUES (%s, %s, %s, NOW(), NOW())",
                    (chat_id, chat_id, chat_data_json),
                )

        conn.commit()
    except Exception as e:
        print(f"MySQL Insert/Update Error: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


def db_get_recent_messages(chat_id, limit=12):
    """Last N messages for this chat (includes current user message once stored)."""
    if not chat_id or limit <= 0:
        return []
    conn = get_mysql_connection()
    if not conn:
        return []
    try:
        sid = str(chat_id)
        numeric_chat_id = int(sid) if sid.isdigit() and len(sid) < 20 else None
        with conn.cursor() as cur:
            if numeric_chat_id is not None:
                cur.execute(
                    "SELECT chat_data FROM chats WHERE chat_token = %s OR chat_id = %s LIMIT 1",
                    (chat_id, numeric_chat_id),
                )
            else:
                cur.execute(
                    "SELECT chat_data FROM chats WHERE chat_token = %s LIMIT 1",
                    (chat_id,),
                )
            row = cur.fetchone()

        if not row or not row.get("chat_data"):
            return []

        try:
            chat_data = json.loads(row["chat_data"])
        except (TypeError, json.JSONDecodeError):
            return []

        if isinstance(chat_data, dict):
            chat_data = [chat_data]
        elif not isinstance(chat_data, list):
            return []

        recent = chat_data[-limit:]
        out = []
        for item in recent:
            if not isinstance(item, dict):
                continue
            out.append({
                "sender": item.get("sender"),
                "message": item.get("text") if item.get("text") is not None else item.get("message"),
            })
        return out
    except Exception as e:
        print(f"db_get_recent_messages: {e}")
        return []
    finally:
        conn.close()
