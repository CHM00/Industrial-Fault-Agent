"""Configuration-backed user sessions and RBAC for the pilot application."""

from __future__ import annotations

import contextvars
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass


ROLE_LEVEL = {"viewer": 10, "operator": 20, "expert": 30, "admin": 40}


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str = "viewer"
    tenant_id: str = "default"
    authenticated: bool = True
    display_name: str = ""

    def can(self, required_role: str) -> bool:
        return ROLE_LEVEL.get(self.role, 0) >= ROLE_LEVEL.get(required_role, 999)


DEVELOPMENT_PRINCIPAL = Principal("local-development", "admin", "default", False)
current_principal: contextvars.ContextVar[Principal] = contextvars.ContextVar(
    "current_principal", default=DEVELOPMENT_PRINCIPAL
)


class AuthConfigurationError(RuntimeError):
    pass


class ApiKeyAuth:
    def __init__(
        self,
        enabled: bool | None = None,
        raw_config: str | None = None,
        raw_users: str | None = None,
        session_secret: str | None = None,
    ):
        app_env = os.environ.get("APP_ENV", "development").lower()
        self.enabled = enabled if enabled is not None else os.environ.get(
            "AUTH_ENABLED", "true" if app_env == "production" else "false"
        ).lower() == "true"
        self._entries: list[tuple[str, Principal]] = []
        self._users: dict[str, tuple[Principal, str]] = {}
        self.session_cookie = os.environ.get("AUTH_SESSION_COOKIE", "fault_agent_session")
        self.session_ttl_seconds = max(300, int(os.environ.get("AUTH_SESSION_TTL_SECONDS", "28800")))
        self.session_secure = os.environ.get("AUTH_SESSION_SECURE", "false").lower() == "true"
        self._session_secret = (
            session_secret if session_secret is not None else os.environ.get("AUTH_SESSION_SECRET", "")
        ).encode("utf-8")
        config = raw_config if raw_config is not None else os.environ.get("API_KEYS_JSON", "")
        if config:
            try:
                parsed = json.loads(config)
                for api_key, value in parsed.items():
                    if not isinstance(value, dict):
                        raise ValueError("每个 key 必须映射到对象")
                    role = value.get("role", "viewer")
                    if role not in ROLE_LEVEL:
                        raise ValueError(f"未知角色: {role}")
                    principal = Principal(
                        subject=str(value.get("subject") or "pilot-user"),
                        role=role,
                        tenant_id=str(value.get("tenant_id") or "default"),
                        display_name=str(value.get("display_name") or value.get("subject") or "pilot-user"),
                    )
                    self._entries.append((self._digest(api_key), principal))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise AuthConfigurationError(f"API_KEYS_JSON 配置无效: {exc}") from exc
        bootstrap = os.environ.get("PILOT_API_KEY", "")
        if bootstrap:
            self._entries.append(
                (self._digest(bootstrap), Principal("pilot-admin", "admin", "default"))
            )
        users_config = raw_users if raw_users is not None else os.environ.get("AUTH_USERS_JSON", "")
        users_file = "" if raw_users is not None else os.environ.get("AUTH_USERS_FILE", "").strip()
        if users_file:
            try:
                with open(users_file, "r", encoding="utf-8") as handle:
                    users_config = handle.read()
            except OSError as exc:
                raise AuthConfigurationError(f"无法读取 AUTH_USERS_FILE: {exc}") from exc
        if users_config:
            try:
                users = json.loads(users_config)
                for username, value in users.items():
                    if not isinstance(value, dict):
                        raise ValueError("每个用户必须映射到对象")
                    role = str(value.get("role", "viewer"))
                    if role not in ROLE_LEVEL:
                        raise ValueError(f"未知角色: {role}")
                    credential = str(value.get("password_hash") or value.get("password") or "")
                    if not credential:
                        raise ValueError(f"用户 {username} 缺少 password_hash 或 password")
                    principal = Principal(
                        subject=str(value.get("subject") or username),
                        role=role,
                        tenant_id=str(value.get("tenant_id") or "default"),
                        display_name=str(value.get("display_name") or username),
                    )
                    self._users[str(username)] = (principal, credential)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise AuthConfigurationError(f"用户认证配置无效: {exc}") from exc
            if not self._session_secret:
                raise AuthConfigurationError("配置 AUTH_USERS_JSON/AUTH_USERS_FILE 时必须设置 AUTH_SESSION_SECRET")

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @property
    def configured(self) -> bool:
        return bool(self._entries or self._users) or not self.enabled

    @property
    def login_enabled(self) -> bool:
        return bool(self._users)

    @staticmethod
    def hash_password(password: str, iterations: int = 260_000) -> str:
        salt = secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return "pbkdf2_sha256${}${}${}".format(
            iterations,
            base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
            base64.urlsafe_b64encode(derived).decode("ascii").rstrip("="),
        )

    @staticmethod
    def _verify_password(password: str, credential: str) -> bool:
        if not credential.startswith("pbkdf2_sha256$"):
            return hmac.compare_digest(password, credential)
        try:
            _, raw_iterations, raw_salt, raw_expected = credential.split("$", 3)
            padding = lambda value: value + "=" * (-len(value) % 4)
            salt = base64.urlsafe_b64decode(padding(raw_salt))
            expected = base64.urlsafe_b64decode(padding(raw_expected))
            actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(raw_iterations))
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False

    def login(self, username: str, password: str) -> Principal | None:
        entry = self._users.get(username)
        if not entry or not self._verify_password(password, entry[1]):
            return None
        return entry[0]

    def issue_session(self, principal: Principal) -> str:
        payload = {
            "sub": principal.subject,
            "role": principal.role,
            "tenant": principal.tenant_id,
            "exp": int(time.time()) + self.session_ttl_seconds,
        }
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).decode("ascii").rstrip("=")
        signature = hmac.new(self._session_secret, encoded.encode("ascii"), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"

    def authenticate_session(self, token: str | None) -> Principal | None:
        if not token or not self._session_secret:
            return None
        try:
            encoded, signature = token.split(".", 1)
            expected = hmac.new(self._session_secret, encoded.encode("ascii"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                return None
            raw = encoded + "=" * (-len(encoded) % 4)
            payload = json.loads(base64.urlsafe_b64decode(raw).decode("utf-8"))
            if int(payload.get("exp", 0)) < int(time.time()):
                return None
            for principal, _ in self._users.values():
                if principal.subject == payload.get("sub") and principal.tenant_id == payload.get("tenant"):
                    return principal
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
        return None

    def authenticate(self, authorization: str | None, x_api_key: str | None, session_token: str | None = None) -> Principal | None:
        if not self.enabled:
            return DEVELOPMENT_PRINCIPAL
        principal = self.authenticate_session(session_token)
        if principal is not None:
            return principal
        token = (x_api_key or "").strip()
        if not token and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not token:
            return None
        digest = self._digest(token)
        for expected, principal in self._entries:
            if hmac.compare_digest(digest, expected):
                return principal
        return None


def get_principal() -> Principal:
    return current_principal.get()


if __name__ == "__main__":
    import getpass
    password = getpass.getpass("请输入要生成哈希的密码: ")
    confirmation = getpass.getpass("请再次输入: ")
    if password != confirmation:
        raise SystemExit("两次密码不一致")
    print(ApiKeyAuth.hash_password(password))
