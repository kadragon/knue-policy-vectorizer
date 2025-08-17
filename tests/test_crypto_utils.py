"""Tests for crypto_utils module."""

import pytest

from src.crypto_utils import CryptoUtils


def test_data_integrity_hash():
    """Test data integrity hashing function."""
    # Test basic functionality
    data = "test configuration data"
    hash1 = CryptoUtils.calculate_data_integrity_hash(data)
    hash2 = CryptoUtils.calculate_data_integrity_hash(data)

    # Same input should produce same hash
    assert hash1 == hash2

    # Hash should be 64 characters (SHA-256 hex)
    assert len(hash1) == 64

    # Hash should be hex string
    assert all(c in "0123456789abcdef" for c in hash1)

    # Different input should produce different hash
    different_data = "different data"
    hash3 = CryptoUtils.calculate_data_integrity_hash(different_data)
    assert hash1 != hash3


def test_data_integrity_hash_consistency():
    """Test that data integrity hash is consistent across calls."""
    test_cases = [
        "",
        "simple",
        "한글 텍스트",
        '{"config": {"value": 123}}',
        "very long text " * 100,
    ]

    for data in test_cases:
        hash1 = CryptoUtils.calculate_data_integrity_hash(data)
        hash2 = CryptoUtils.calculate_data_integrity_hash(data)
        assert hash1 == hash2, f"Hash inconsistent for: {data[:50]}"


def test_password_hashing_basic():
    """Test basic password hashing functionality."""
    password = "test_password_123"

    # Test with auto-generated salt
    hash1, salt1 = CryptoUtils.hash_password_securely(password)
    hash2, salt2 = CryptoUtils.hash_password_securely(password)

    # Different salts should produce different hashes
    assert hash1 != hash2
    assert salt1 != salt2

    # Both hashes should be hex strings
    assert all(c in "0123456789abcdef" for c in hash1)
    assert all(c in "0123456789abcdef" for c in hash2)

    # Salt should be 32 bytes
    assert len(salt1) == 32
    assert len(salt2) == 32


def test_password_hashing_with_salt():
    """Test password hashing with provided salt."""
    password = "test_password_123"
    salt = b"a" * 32  # Fixed salt

    # Same password and salt should produce same hash
    hash1, returned_salt1 = CryptoUtils.hash_password_securely(password, salt)
    hash2, returned_salt2 = CryptoUtils.hash_password_securely(password, salt)

    assert hash1 == hash2
    assert returned_salt1 == returned_salt2 == salt


def test_password_verification():
    """Test password verification function."""
    password = "correct_password"
    wrong_password = "wrong_password"

    # Generate hash
    stored_hash, salt = CryptoUtils.hash_password_securely(password)

    # Correct password should verify
    assert CryptoUtils.verify_password(password, stored_hash, salt)

    # Wrong password should not verify
    assert not CryptoUtils.verify_password(wrong_password, stored_hash, salt)


def test_password_security_properties():
    """Test security properties of password hashing."""
    password = "secure_password_123"

    # Generate multiple hashes with different salts
    hashes_and_salts = [CryptoUtils.hash_password_securely(password) for _ in range(10)]

    # All hashes should be different (due to random salts)
    hashes = [h for h, s in hashes_and_salts]
    assert len(set(hashes)) == len(hashes), "Hashes should be unique"

    # All salts should be different
    salts = [s for h, s in hashes_and_salts]
    assert len(set(salts)) == len(salts), "Salts should be unique"

    # All should verify correctly
    for hash_val, salt in hashes_and_salts:
        assert CryptoUtils.verify_password(password, hash_val, salt)


def test_unicode_handling():
    """Test handling of Unicode characters."""
    # Test data integrity hash with Unicode
    unicode_data = "한국어 정책 문서 테스트 🔒"
    hash1 = CryptoUtils.calculate_data_integrity_hash(unicode_data)
    hash2 = CryptoUtils.calculate_data_integrity_hash(unicode_data)
    assert hash1 == hash2

    # Test password hashing with Unicode
    unicode_password = "비밀번호_테스트_🔐"
    hash_val, salt = CryptoUtils.hash_password_securely(unicode_password)
    assert CryptoUtils.verify_password(unicode_password, hash_val, salt)


def test_edge_cases():
    """Test edge cases and error conditions."""
    # Empty string
    empty_hash = CryptoUtils.calculate_data_integrity_hash("")
    assert len(empty_hash) == 64

    # Empty password
    empty_pwd_hash, salt = CryptoUtils.hash_password_securely("")
    assert CryptoUtils.verify_password("", empty_pwd_hash, salt)

    # Very long data
    long_data = "x" * 100000
    long_hash = CryptoUtils.calculate_data_integrity_hash(long_data)
    assert len(long_hash) == 64
