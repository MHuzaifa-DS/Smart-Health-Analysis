"""
tests/test_auth.py — Tests for authentication endpoints.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.utils.jwt import create_access_token, verify_access_token, verify_refresh_token


# ── JWT utility tests ──────────────────────────────────────────────────────────

class TestJWTUtils:
    def test_create_and_verify_access_token(self):
        token = create_access_token("user-123", "test@example.com")
        assert token
        user_id = verify_access_token(token)
        assert user_id == "user-123"

    def test_invalid_token_returns_none(self):
        result = verify_access_token("not.a.valid.token")
        assert result is None

    def test_refresh_token_is_separate_from_access_token(self):
        from app.utils.jwt import create_refresh_token
        refresh = create_refresh_token("user-456")
        # Refresh token should not pass as access token
        assert verify_access_token(refresh) is None
        assert verify_refresh_token(refresh) == "user-456"

    def test_tampered_token_rejected(self):
        token = create_access_token("user-789", "x@x.com")
        tampered = token[:-5] + "XXXXX"
        assert verify_access_token(tampered) is None


# ── /auth/me endpoint ─────────────────────────────────────────────────────────

class TestGetMe:
    def test_get_me_authenticated(self, client, auth_headers, mock_supabase):
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "full_name": "Test User",
            "age": 35,
            "gender": "male",
            "blood_type": "O+",
            "created_at": "2024-01-01T00:00:00+00:00",
        }

        admin_mock = MagicMock()
        admin_mock.auth.admin.get_user_by_id.return_value.user.email = "test@example.com"

        with patch("app.database.get_supabase_admin", return_value=admin_mock):
            response = client.get("/auth/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "00000000-0000-0000-0000-000000000001"
        assert data["email"] == "test@example.com"
        assert data["age"] == 35

    def test_get_me_unauthenticated(self, client):
        response = client.get("/auth/me")
        assert response.status_code == 401

    def test_get_me_invalid_token(self, client):
        response = client.get("/auth/me", headers={"Authorization": "Bearer invalid"})
        assert response.status_code == 401


# ── /auth/login endpoint ──────────────────────────────────────────────────────

class TestLogin:
    def test_login_success(self, client, mock_supabase):
        # Mock Supabase auth sign_in_with_password
        mock_user = MagicMock()
        mock_user.id = "00000000-0000-0000-0000-000000000001"
        mock_user.email = "test@example.com"
        mock_supabase.auth.sign_in_with_password.return_value.user = mock_user

        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"id": "00000000-0000-0000-0000-000000000001", "full_name": "Test", "age": 35, "gender": "male"}
        ]

        response = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "password123",
        })

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self, client, mock_supabase):
        mock_supabase.auth.sign_in_with_password.side_effect = Exception("Invalid login credentials")
        response = client.post("/auth/login", json={
            "email": "bad@example.com",
            "password": "wrongpassword",
        })
        assert response.status_code == 401

    def test_login_missing_fields(self, client):
        response = client.post("/auth/login", json={"email": "test@example.com"})
        assert response.status_code == 422


# ── /auth/profile update ──────────────────────────────────────────────────────

class TestProfileUpdate:
    def test_update_profile(self, client, auth_headers, mock_supabase):
        admin_mock = MagicMock()
        admin_mock.auth.admin.get_user_by_id.return_value.user.email = "test@example.com"
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = {
            "id": "00000000-0000-0000-0000-000000000001",
            "full_name": "Old Name", "age": 35, "gender": "male", "created_at": "2024-01-01T00:00:00+00:00"
        }
        mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
            {"id": "00000000-0000-0000-0000-000000000001", "full_name": "New Name", "age": 36, "gender": "male"}
        ]

        with patch("app.database.get_supabase_admin", return_value=admin_mock):
            response = client.put("/auth/profile", json={"full_name": "New Name", "age": 36}, headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["full_name"] == "New Name"
