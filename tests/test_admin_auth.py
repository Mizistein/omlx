# SPDX-License-Identifier: Apache-2.0
"""Tests for admin authentication and chat page API key injection."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.server  # noqa: F401 — ensure server module is imported first
import omlx.admin.routes as admin_routes


def _mock_global_settings(api_key=None):
    """Create a mock GlobalSettings with the given API key."""
    mock = MagicMock()
    mock.auth.api_key = api_key
    return mock


def _patch_getter(mock_settings):
    """Replace the module-level _get_global_settings with a lambda returning mock."""
    original = admin_routes._get_global_settings
    admin_routes._get_global_settings = lambda: mock_settings
    return original


def _restore_getter(original):
    """Restore the original _get_global_settings."""
    admin_routes._get_global_settings = original


class TestAutoLogin:
    """Tests for GET /admin/auto-login endpoint."""

    def test_auto_login_success_redirects_to_dashboard(self):
        """Valid API key should redirect to the specified path with session cookie."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="test-key", redirect="/admin/dashboard")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin/dashboard"
            # Check that session cookie is set
            cookie_header = result.headers.get("set-cookie", "")
            assert "omlx_admin_session" in cookie_header
        finally:
            _restore_getter(original)

    def test_auto_login_success_redirects_to_chat(self):
        """Valid API key should redirect to chat page."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="test-key", redirect="/admin/chat")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin/chat"
        finally:
            _restore_getter(original)

    def test_auto_login_invalid_key_redirects_to_login(self):
        """Invalid API key should redirect to login page without session cookie."""
        mock_settings = _mock_global_settings(api_key="correct-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="wrong-key", redirect="/admin/dashboard")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
            cookie_header = result.headers.get("set-cookie", "")
            assert "omlx_admin_session" not in cookie_header
        finally:
            _restore_getter(original)

    def test_auto_login_empty_key_redirects_to_login(self):
        """Empty API key should redirect to login page."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="", redirect="/admin/dashboard")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter(original)

    def test_auto_login_no_server_key_redirects_to_login(self):
        """No server API key configured should redirect to login page."""
        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="any-key", redirect="/admin/dashboard")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter(original)

    def test_auto_login_invalid_redirect_returns_400(self):
        """Redirect path not starting with /admin should return 400."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    admin_routes.auto_login(
                        key="test-key", redirect="https://evil.com"
                    )
                )
            assert exc_info.value.status_code == 400
            assert "Invalid redirect path" in exc_info.value.detail
        finally:
            _restore_getter(original)

    def test_auto_login_redirect_to_admin_root(self):
        """Redirect to /admin (exact match) should be allowed."""
        mock_settings = _mock_global_settings(api_key="test-key")
        original = _patch_getter(mock_settings)
        try:
            result = asyncio.run(
                admin_routes.auto_login(key="test-key", redirect="/admin")
            )
            assert result.status_code == 302
            assert result.headers["location"] == "/admin"
        finally:
            _restore_getter(original)


class TestChatPageApiKeyInjection:
    """Tests for GET /admin/chat API key template injection."""

    def test_chat_page_passes_api_key_in_context(self):
        """Chat page should include API key in template context."""
        mock_settings = _mock_global_settings(api_key="test-chat-key")
        original = _patch_getter(mock_settings)
        try:
            mock_request = MagicMock()
            with patch.object(admin_routes, "templates") as mock_templates:
                mock_templates.TemplateResponse.return_value = MagicMock()
                asyncio.run(
                    admin_routes.chat_page(request=mock_request, is_admin=True)
                )
                mock_templates.TemplateResponse.assert_called_once_with(
                    "chat.html",
                    {"request": mock_request, "api_key": "test-chat-key"},
                )
        finally:
            _restore_getter(original)

    def test_chat_page_passes_empty_when_no_key(self):
        """Chat page should pass empty string when no API key is configured."""
        mock_settings = _mock_global_settings(api_key=None)
        original = _patch_getter(mock_settings)
        try:
            mock_request = MagicMock()
            with patch.object(admin_routes, "templates") as mock_templates:
                mock_templates.TemplateResponse.return_value = MagicMock()
                asyncio.run(
                    admin_routes.chat_page(request=mock_request, is_admin=True)
                )
                call_args = mock_templates.TemplateResponse.call_args
                context = call_args[0][1]
                assert context["api_key"] == ""
        finally:
            _restore_getter(original)

    def test_chat_page_passes_empty_when_no_settings(self):
        """Chat page should pass empty string when global settings is None."""
        original = admin_routes._get_global_settings
        admin_routes._get_global_settings = lambda: None
        try:
            mock_request = MagicMock()
            with patch.object(admin_routes, "templates") as mock_templates:
                mock_templates.TemplateResponse.return_value = MagicMock()
                asyncio.run(
                    admin_routes.chat_page(request=mock_request, is_admin=True)
                )
                call_args = mock_templates.TemplateResponse.call_args
                context = call_args[0][1]
                assert context["api_key"] == ""
        finally:
            admin_routes._get_global_settings = original
