"""Unit tests for configuration redaction and safe dictionary output."""

import unittest

try:
    from core.config import get_settings
    HAS_PYDANTIC_SETTINGS = True
except ImportError:
    HAS_PYDANTIC_SETTINGS = False


@unittest.skipUnless(HAS_PYDANTIC_SETTINGS, "pydantic_settings not installed in current Python env")
class TestConfigSafety(unittest.TestCase):
    def test_as_safe_dict_redacts_sensitive_keys(self):
        settings = get_settings()
        safe_dict = settings.as_safe_dict()

        def _assert_no_raw_secret(d: dict, path=""):
            for k, v in d.items():
                current_path = f"{path}.{k}" if path else k
                if isinstance(v, dict):
                    _assert_no_raw_secret(v, current_path)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            _assert_no_raw_secret(item, current_path)
                else:
                    if any(term in k.lower() for term in ("key", "token", "secret", "password")) and v:
                        self.assertEqual(
                            v,
                            "***",
                            f"Field '{current_path}' was not masked with '***': {v}",
                        )

        _assert_no_raw_secret(safe_dict)


if __name__ == "__main__":
    unittest.main()
