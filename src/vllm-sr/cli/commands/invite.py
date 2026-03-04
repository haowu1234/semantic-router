"""Invite code generation command for vLLM Semantic Router."""

import base64
import click
import hashlib
import hmac
import json
import os
import sys
import time
from cli.utils import getLogger

log = getLogger(__name__)


def parse_duration(duration_str: str) -> int:
    """
    Parse duration string to seconds.
    
    Supported formats:
        - "30m" -> 30 minutes
        - "24h" -> 24 hours
        - "7d" -> 7 days
        - "0" or "never" -> 0 (never expires)
        - Unix timestamp (integer)
    
    Args:
        duration_str: Duration string or timestamp
        
    Returns:
        int: Duration in seconds, or 0 for never expires
    """
    duration_str = duration_str.strip().lower()
    
    if duration_str in ("0", "never", "none"):
        return 0
    
    # Try parsing as integer (unix timestamp or seconds)
    try:
        val = int(duration_str)
        # If it looks like a timestamp (> 1 billion), use as-is
        if val > 1_000_000_000:
            return val - int(time.time())  # Convert to duration from now
        return val
    except ValueError:
        pass
    
    # Parse duration with suffix
    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }
    
    for suffix, mult in multipliers.items():
        if duration_str.endswith(suffix):
            try:
                num = int(duration_str[:-1])
                return num * mult
            except ValueError:
                pass
    
    raise ValueError(f"Invalid duration format: {duration_str}. Use formats like '30m', '24h', '7d', or '0' for never expires.")


def generate_invite_code(secret: str, exp_seconds: int, scope: str, note: str = "") -> str:
    """
    Generate a signed invite code.
    
    Format: invite-{base64_payload}.{hex_signature}
    
    Args:
        secret: HMAC secret key
        exp_seconds: Expiration time in seconds from now (0 = never expires)
        scope: Permission scope ("write" or "admin")
        note: Optional note for auditing
        
    Returns:
        str: The generated invite code
    """
    # Build payload
    payload = {
        "exp": int(time.time()) + exp_seconds if exp_seconds > 0 else 0,
        "scope": scope,
    }
    if note:
        payload["note"] = note
    
    # Encode payload as URL-safe base64 (no padding)
    payload_json = json.dumps(payload, separators=(",", ":"))
    payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).rstrip(b"=").decode()
    
    # Generate HMAC-SHA256 signature
    mac = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256)
    signature = mac.hexdigest()
    
    return f"invite-{payload_b64}.{signature}"


def verify_invite_code(code: str, secret: str) -> dict | None:
    """
    Verify an invite code and return payload if valid.
    
    Args:
        code: The invite code to verify
        secret: HMAC secret key
        
    Returns:
        dict: The payload if valid, None otherwise
    """
    # Remove prefix if present
    code = code.removeprefix("invite-")
    
    parts = code.split(".", 1)
    if len(parts) != 2:
        return None
    
    payload_b64, sig_hex = parts
    
    # Verify signature
    mac = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256)
    expected_sig = mac.hexdigest()
    
    if not hmac.compare_digest(sig_hex, expected_sig):
        return None
    
    # Decode payload (add padding back for base64)
    try:
        # Add padding
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_json = base64.urlsafe_b64decode(payload_b64).decode()
        payload = json.loads(payload_json)
    except Exception:
        return None
    
    # Check expiration
    exp = payload.get("exp", 0)
    if exp > 0 and time.time() > exp:
        return None
    
    return payload


@click.group()
def invite():
    """
    Manage invite codes for dashboard beta access.
    
    Invite codes allow users to unlock write permissions in readonly mode.
    
    Examples:
        vllm-sr invite generate --exp 7d --scope write --note "beta-user-001"
        vllm-sr invite verify "invite-xxx.yyy"
    """
    pass


@invite.command("generate")
@click.option(
    "--secret",
    envvar="INVITE_SECRET",
    required=True,
    help="HMAC secret key (or set INVITE_SECRET env var)",
)
@click.option(
    "--exp",
    default="7d",
    help="Expiration time: '30m', '24h', '7d', or '0' for never expires (default: 7d)",
)
@click.option(
    "--scope",
    type=click.Choice(["write", "admin"], case_sensitive=False),
    default="write",
    help="Permission scope (default: write)",
)
@click.option(
    "--note",
    default="",
    help="Optional note for auditing (e.g., 'beta-user-001')",
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Output in JSON format",
)
def generate(secret, exp, scope, note, json_output):
    """
    Generate a new invite code.
    
    The invite code is a HMAC-SHA256 signed token that grants temporary
    write permissions in readonly mode.
    
    Examples:
    
    \b
        # Generate with 7-day expiration (default)
        vllm-sr invite generate --secret "my-secret-key"
        
        # Generate with custom expiration
        vllm-sr invite generate --secret "my-secret" --exp 24h
        
        # Generate with note for auditing
        vllm-sr invite generate --secret "my-secret" --note "beta-user-alice"
        
        # Generate never-expiring code (for admin)
        vllm-sr invite generate --secret "my-secret" --exp 0 --scope admin
        
        # Use environment variable for secret
        export INVITE_SECRET="my-secret-key"
        vllm-sr invite generate --exp 7d --note "beta-user-001"
    """
    try:
        # Parse expiration
        exp_seconds = parse_duration(exp)
        
        # Generate code
        code = generate_invite_code(secret, exp_seconds, scope.lower(), note)
        
        if json_output:
            # Calculate actual expiration timestamp
            exp_ts = int(time.time()) + exp_seconds if exp_seconds > 0 else 0
            output = {
                "code": code,
                "scope": scope.lower(),
                "exp": exp_ts,
                "exp_human": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(exp_ts)) if exp_ts > 0 else "never",
                "note": note,
            }
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo("")
            click.echo("✓ Invite code generated successfully!")
            click.echo("")
            click.echo(f"  Code:   {code}")
            click.echo(f"  Scope:  {scope.lower()}")
            if exp_seconds > 0:
                exp_ts = int(time.time()) + exp_seconds
                exp_human = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(exp_ts))
                click.echo(f"  Expiry: {exp_human} ({exp})")
            else:
                click.echo("  Expiry: Never")
            if note:
                click.echo(f"  Note:   {note}")
            click.echo("")
            click.echo("Share this code with the user. They can enter it in the dashboard")
            click.echo("to unlock write permissions.")
            click.echo("")
            
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)
    except Exception as e:
        log.error(f"Error generating invite code: {e}")
        sys.exit(1)


@invite.command("verify")
@click.argument("code")
@click.option(
    "--secret",
    envvar="INVITE_SECRET",
    required=True,
    help="HMAC secret key (or set INVITE_SECRET env var)",
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Output in JSON format",
)
def verify(code, secret, json_output):
    """
    Verify an invite code.
    
    Checks if the code is valid and not expired.
    
    Examples:
    
    \b
        # Verify a code
        vllm-sr invite verify "invite-xxx.yyy" --secret "my-secret"
        
        # Use environment variable
        export INVITE_SECRET="my-secret-key"
        vllm-sr invite verify "invite-xxx.yyy"
    """
    try:
        payload = verify_invite_code(code, secret)
        
        if payload is None:
            if json_output:
                click.echo(json.dumps({"valid": False, "error": "Invalid or expired invite code"}))
            else:
                click.echo("")
                click.echo("✗ Invalid or expired invite code")
                click.echo("")
            sys.exit(1)
        
        if json_output:
            output = {
                "valid": True,
                "scope": payload.get("scope", "write"),
                "exp": payload.get("exp", 0),
                "exp_human": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(payload["exp"])) if payload.get("exp", 0) > 0 else "never",
                "note": payload.get("note", ""),
            }
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo("")
            click.echo("✓ Valid invite code!")
            click.echo("")
            click.echo(f"  Scope:  {payload.get('scope', 'write')}")
            exp = payload.get("exp", 0)
            if exp > 0:
                exp_human = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(exp))
                remaining = exp - int(time.time())
                if remaining > 86400:
                    remaining_human = f"{remaining // 86400} days"
                elif remaining > 3600:
                    remaining_human = f"{remaining // 3600} hours"
                else:
                    remaining_human = f"{remaining // 60} minutes"
                click.echo(f"  Expiry: {exp_human} ({remaining_human} remaining)")
            else:
                click.echo("  Expiry: Never")
            if payload.get("note"):
                click.echo(f"  Note:   {payload['note']}")
            click.echo("")
            
    except Exception as e:
        log.error(f"Error verifying invite code: {e}")
        sys.exit(1)
