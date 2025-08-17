"""
Cryptographic utilities with security best practices.

This module provides secure cryptographic functions following industry standards:
- Data integrity hashing (SHA-256 for non-sensitive data)
- Secure password hashing (PBKDF2 for credentials)

SECURITY GUIDELINES:
- Use calculate_data_integrity_hash() for file hashing, checksums, and data verification
- Use hash_password_securely() for any password or credential hashing
- Never use fast hash functions (MD5, SHA-1, raw SHA-256) for password hashing
"""

import hashlib
import hmac
import os
from typing import Optional


class CryptoUtils:
    """Cryptographic utilities with clear separation of use cases"""

    @staticmethod
    def calculate_data_integrity_hash(data: str) -> str:
        """
        Calculate SHA-256 hash for data integrity verification.

        SECURITY NOTE: This function is intended ONLY for data integrity checking,
        NOT for password hashing or other security-sensitive operations.

        Use cases:
        - File content verification
        - Configuration checksums
        - Document ID generation
        - Data deduplication

        Args:
            data: Data to hash (configuration, file contents, etc.)

        Returns:
            SHA-256 hex digest for integrity verification
        """
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_password_securely(
        password: str, salt: Optional[bytes] = None
    ) -> tuple[str, bytes]:
        """
        Hash password using secure PBKDF2 algorithm.

        SECURITY NOTE: This function uses PBKDF2 with SHA-256, which is
        computationally expensive and resistant to brute-force attacks.
        Use this for any password or credential hashing needs.

        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)

        Returns:
            Tuple of (hex_digest, salt_used)
        """
        if salt is None:
            salt = os.urandom(32)  # 256-bit salt

        # PBKDF2 with 100,000 iterations (recommended minimum)
        pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return pwd_hash.hex(), salt

    @staticmethod
    def verify_password(password: str, stored_hash: str, salt: bytes) -> bool:
        """
        Verify password against stored hash using secure comparison.

        Args:
            password: Password to verify
            stored_hash: Stored password hash (hex)
            salt: Salt used for original hashing

        Returns:
            True if password matches, False otherwise
        """
        computed_hash, _ = CryptoUtils.hash_password_securely(password, salt)
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(stored_hash, computed_hash)
