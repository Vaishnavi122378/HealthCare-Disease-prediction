from database import clear_failed_login_attempts

username = input(
    "Enter your login username: "
).strip().lower()

clear_failed_login_attempts(username)

print(
    f"Failed login attempts cleared for: {username}"
)