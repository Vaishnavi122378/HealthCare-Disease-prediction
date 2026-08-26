import hashlib
import secrets
import getpass
import re

from src.database import (
    create_table,
    get_user,
    create_user
)


def hash_password(password):

    iterations = 310000

    salt = secrets.token_bytes(16)

    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations
    )

    return (
        f"pbkdf2_sha256$"
        f"{iterations}$"
        f"{salt.hex()}$"
        f"{key.hex()}"
    )


create_table()

print()
print("=" * 55)
print("        HEARTCARE AI - ADMIN SETUP")
print("=" * 55)
print()


while True:

    username = input(
        "Create username: "
    ).strip().lower()

    if not username:

        print("Username cannot be empty.")
        continue

    if not re.fullmatch(
        r"[a-z0-9_.-]{3,30}",
        username
    ):

        print(
            "Username must contain only letters, "
            "numbers, '.', '_' or '-'."
        )

        continue

    if get_user(username):

        print(
            "That username already exists."
        )

        continue

    break


while True:

    password = getpass.getpass(
        "Create password: "
    )

    if len(password) < 10:

        print(
            "Password must contain at least "
            "10 characters."
        )

        continue

    confirm_password = getpass.getpass(
        "Confirm password: "
    )

    if password != confirm_password:

        print(
            "Passwords do not match."
        )

        continue

    break


password_hash = hash_password(password)


create_user(
    username,
    password_hash
)


print()
print("=" * 55)
print("ADMIN USER CREATED SUCCESSFULLY")
print("=" * 55)
print()

print(f"Username: {username}")

print()
print(
    "You can now start the Streamlit application."
)
print()