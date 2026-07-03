"""Auth credential models and data-access layer."""

from __future__ import annotations
from open_webui.env import AWS_REGION, COGNITO_CLIENT_ID
import logging
import uuid
from typing import Optional
import requests
from aws_requests_auth.aws_auth import AWSRequestsAuth
import boto3

import bcrypt
from open_webui.internal.db import Base, JSONField, get_async_db_context
from open_webui.models.users import User, UserModel, UserProfileImageResponse, Users
from open_webui.utils.validate import validate_profile_image_url
from pydantic import BaseModel, field_validator
from sqlalchemy import Boolean, Column, String, Text, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

# Pre-computed hash verified on signin paths that lack a real credential
# (unknown user, inactive account) so response timing cannot reveal
# whether an account exists (CWE-208).
PLACEHOLDER_HASH = bcrypt.hashpw(b'placeholder', bcrypt.gensalt()).decode('utf-8')


class Auth(Base):  # credential ↔ user linkage
    """Maps a user ID to an email/password pair with an active flag."""

    __tablename__ = 'auth'

    id = Column(String, primary_key=True, unique=True)  # mirrors User.id
    email = Column(String)  # login address, kept in sync with User.email
    password = Column(Text)  # argon2 / bcrypt hash
    active = Column(Boolean)  # account soft-disable toggle


class AuthModel(BaseModel):
    """Pydantic mirror of the ``auth`` table row."""

    id: str
    email: str
    password: str
    active: bool = True


class Token(BaseModel):
    """JWT bearer-token response wrapper."""

    token: str
    token_type: str


class ApiKey(BaseModel):
    api_key: str | None = None


class SigninResponse(Token, UserProfileImageResponse):
    pass


class SigninForm(BaseModel):
    email: str
    password: str


class LdapForm(BaseModel):
    user: str
    password: str


class ProfileImageUrlForm(BaseModel):
    profile_image_url: str


class UpdatePasswordForm(BaseModel):
    password: str
    new_password: str


class SignupForm(BaseModel):
    name: str
    email: str
    password: str
    profile_image_url: str | None = '/user.png'

    @field_validator('profile_image_url')
    @classmethod
    def check_profile_image_url(cls, v: str | None) -> str | None:
        if v is not None:
            return validate_profile_image_url(v)
        return v


class AddUserForm(SignupForm):
    role: str | None = 'pending'


# --- data-access layer ---


class AuthsTable:
    """Provides CRUD operations for the Auth ↔ User lifecycle."""

    async def insert_new_auth(
        self,
        email: str,
        password: str,
        name: str,
        profile_image_url: str = '/user.png',
        role: str = 'pending',
        oauth: dict | None = None,
        db: AsyncSession | None = None,
    ) -> UserModel | None:
        """Create an Auth + User pair inside a single transaction."""
        async with get_async_db_context(db) as session:
            log.info('insert_new_auth')

            new_id = str(uuid.uuid4())

            credential = Auth(
                id=new_id,
                email=email,
                password=password,
                active=True,
            )
            session.add(credential)

            created_user = await Users.insert_new_user(
                new_id,
                name,
                email,
                profile_image_url,
                role,
                oauth=oauth,
                db=session,
            )
            # persist both records and reload generated defaults
            await session.commit()
            await session.refresh(credential)
            return created_user if credential and created_user else None

    async def authenticate_user(
        self,
        email: str,
        verify_password: callable,
        db: AsyncSession | None = None,
    ) -> UserModel | None:
        """Verify email + password credentials via AWS Cognito and return the matching user."""
        log.info('authenticate_user via Cognito: %s', email)

        import asyncio
        import boto3
        import inspect

        password_str = ""
        cognito_id = None

        # --- NUEVA ESTRATEGIA DE EXTRACCIÓN ADAPTADA A OPEN WEBUI ---

        # 1. Si es un string directo
        if isinstance(verify_password, str):
            password_str = verify_password

        # 2. Si es una función (como la lambda que muestra el error) o un callable
        elif callable(verify_password):
            try:
                # Las lambdas de validación de contraseñas suelen requerir la contraseña original
                # para verificarla, pero a veces son wrappers que devuelven el string o resuelven un valor.
                # Intentamos ejecutarla directamente sin argumentos:
                password_str = verify_password()
            except TypeError:
                # Si la lambda/función pide parámetros (ej. verify_password(plain_password)),
                # nos vemos obligados a usar la inspección de clausura (Closure) para robar el valor:
                if hasattr(verify_password, "__closure__") and verify_password.__closure__:
                    for cell in verify_password.__closure__:
                        if isinstance(cell.cell_contents, str) and len(cell.cell_contents) > 0:
                            password_str = cell.cell_contents
                            break

        # 3. Inspeccionar el entorno local (Frame) por si la lambda está vacía pero la variable vive arriba
        if not password_str:
            try:
                frame = inspect.currentframe().f_back
                if "password" in frame.f_locals:
                    password_str = frame.f_locals["password"]
                elif "form_data" in frame.f_locals:
                    password_str = frame.f_locals["form_data"].password
                elif "password_hash" in frame.f_locals:  # A veces Open WebUI la llama así en el frame
                    password_str = frame.f_locals["password_hash"]
            except Exception:
                pass

        # Fallback de emergencia drástico (Si sigue siendo un objeto, fallará en Cognito,
        # por lo que preferimos registrar el error antes de enviar basura)
        if not password_str or callable(password_str):
            log.error("No se pudo extraer el string de la contraseña del objeto callable: %s", type(verify_password))
            return None

        # --- LLAMADA ASÍNCRONA SEGURA A AWS ---
        def _execute_cognito_call(p_str):
            client = boto3.client('cognito-idp', region_name=AWS_REGION)
            try:
                return client.initiate_auth(
                    ClientId=COGNITO_CLIENT_ID,
                    AuthFlow='USER_PASSWORD_AUTH',
                    AuthParameters={'USERNAME': email, 'PASSWORD': p_str},
                )
            except Exception as aws_err:
                log.error("Error directo de AWS Cognito: %s", aws_err)
                return None

        cognito_response = await asyncio.to_thread(_execute_cognito_call, password_str)

        if not cognito_response or 'AuthenticationResult' not in cognito_response:
            log.warning("Cognito authentication failed for user: %s", email)
            return None

        # --- EXTRACCIÓN DE ROL DESDE COGNITO ---
        assigned_role = 'user'
        try:
            id_token = cognito_response['AuthenticationResult']['IdToken']
            import base64
            import json

            payload_b64 = id_token.split('.')[1]
            payload_json = base64.urlsafe_b64decode(payload_b64 + '=' * (4 - len(payload_b64) % 4))
            token_data = json.loads(payload_json)

            cognito_id = token_data.get('sub')
            cognito_groups = token_data.get('cognito:groups', [])
            if cognito_groups:
                if 'admin' in cognito_groups:
                    assigned_role = 'admin'
                log.info("Rol extraído de Cognito Groups: %s", assigned_role)
        except Exception as token_err:
            log.error("Error al extraer grupos de Cognito: %s", token_err)

        # --- RESOLVER SESIÓN LOCAL ---
        resolved = await Users.get_user_by_email(email, db=db)

        if not resolved:
            log.info("User verified via Cognito but missing locally. Auto-provisioning user...")
            new_id = str(uuid.uuid4())
            resolved = await Users.insert_new_user(
                id=new_id,
                name=email.split('@')[0],
                email=email,
                profile_image_url='/user.png',
                role=assigned_role,
                oauth={"provider": "cognito"},
                db=db,
            )
        else:
            if resolved.role != assigned_role:
                log.info("Sincronizando rol de usuario local con Cognito: %s", assigned_role)
                resolved.role = assigned_role

        api_key = await self.get_api_key(cognito_id)
        if api_key is not None:
            success = await Users.update_user_api_key_by_id(resolved.id, api_key, db=db)
            if success:
                log.info("Assigned api key")

        return resolved

    async def authenticate_user_by_api_key(
        self,
        api_key: str,
        db: AsyncSession | None = None,
    ) -> UserModel | None:
        """Look up the user that owns the given API key."""
        log.info('authenticate_user_by_api_key')
        if not api_key:
            return
        # delegate to the Users model for the actual lookup
        return await Users.get_user_by_api_key(api_key, db=db)

    async def authenticate_user_by_email(
        self,
        email: str,
        db: AsyncSession | None = None,
    ) -> UserModel | None:
        """Single-query auth via JOIN on Auth ↔ User, filtered by active flag."""
        log.info('authenticate_user_by_email: %s', email)
        # single JOIN avoids N+1 — returns (Auth, User) tuple or None
        async with get_async_db_context(db) as session:
            joined_query = (
                select(Auth, User).join(User, Auth.id == User.id).where(Auth.email == email, Auth.active.is_(True))
            )
            match = (await session.execute(joined_query)).first()
            if not match:
                return
            _, found_user = match
            return UserModel.model_validate(found_user)

    async def update_email_by_id(
        self,
        user_id: str,
        email: str,
        db: AsyncSession | None = None,
    ) -> bool:
        """Set a new email on the auth record and propagate to the user row."""
        async with get_async_db_context(db) as session:
            auth_row = await session.get(Auth, user_id)
            if auth_row is None:
                return False
            auth_row.email = email
            await session.commit()
            await Users.update_user_by_id(user_id, {'email': email}, db=session)
            return True
        # --- password modification ---

    async def update_user_password_by_id(
        self,
        user_id: str,
        new_password: str,
        db: AsyncSession | None = None,
    ) -> bool:
        """Set a new password hash for an existing user."""
        async with get_async_db_context(db) as session:
            auth_row = await session.get(Auth, user_id)
            if auth_row is None:
                return False
            auth_row.password = new_password
            await session.commit()
            return True

    async def delete_auth_by_id(
        self,
        id: str,
        db: AsyncSession | None = None,
    ) -> bool:
        """Remove a user and their auth credential in one transaction."""
        async with get_async_db_context(db) as session:
            if not await Users.delete_user_by_id(id, db=session):
                return False
            await session.execute(delete(Auth).where(Auth.id == id))
            await session.commit()
            return True

    async def get_api_key(self, userId: str):
        URL_BASE_API = "https://svc.pyrun.cloud"
        api_url = f"{URL_BASE_API}/agentinfoec2?studentId={userId}"
        host_limpio = str(URL_BASE_API).replace("https://", "").replace("http://", "").split("/")[0]

        session = boto3.Session()
        credentials = session.get_credentials().get_frozen_credentials()

        auth = AWSRequestsAuth(
            aws_access_key=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_token=credentials.token,
            aws_host=host_limpio,
            aws_region="us-east-1",
            aws_service="execute-api",
        )

        log.info("Enviando petición GET firmada con AWSRequestsAuth a /agentinfoec2...")
        try:
            response = requests.get(api_url, auth=auth)
            log.info(f"\n[Resultado del Servidor]")
            log.info(f"Status Code: {response.status_code}")
            log.info(f"Body: {response.text}")
            data = response.json()
            return data.get("llmApiKey")
        except Exception as e:
            log.error(f"❌ Error al conectar con la API: {e}")


Auths = AuthsTable()  # singleton — module-level instance
