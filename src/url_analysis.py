"""Transparent, local-only URL risk signals for the dashboard.

This is deliberately separate from the trained classifier.  The classifier
expects webpage-derived features (for example external links and form data)
that cannot be truthfully inferred from a URL string alone.
"""
import ipaddress
import re
from urllib.parse import urlparse


SENSITIVE_WORDS = {
    "account", "bank", "confirm", "login", "password", "secure", "signin",
    "update", "verify", "wallet",
}


def analyze_url(raw_url):
    """Return explainable risk signals without making any network request."""
    value = raw_url.strip()
    if not value:
        raise ValueError("Enter a URL to analyse.")
    candidate = value if "://" in value else f"https://{value}"
    parsed = urlparse(candidate)
    if not parsed.hostname:
        raise ValueError("Enter a valid URL, for example https://example.com.")

    host = parsed.hostname.lower()
    path_query = f"{parsed.path}?{parsed.query}".lower()
    signals, score = [], 0.0

    def add(condition, weight, title, explanation):
        nonlocal score
        if condition:
            score += weight
            signals.append({"signal": title, "explanation": explanation, "weight": weight})

    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False

    words_found = sorted(word for word in SENSITIVE_WORDS if word in f"{host}{path_query}")
    subdomain_count = max(0, len(host.split(".")) - 2)
    add(parsed.scheme != "https", 0.18, "No HTTPS", "The URL does not use HTTPS.")
    add(is_ip, 0.25, "IP-address host", "The host is an IP address rather than a domain name.")
    add("@" in value, 0.20, "At-sign in URL", "An at-sign can obscure the real destination.")
    add(len(value) >= 90, 0.10, "Very long URL", "Long URLs can hide misleading paths or parameters.")
    add(not is_ip and subdomain_count >= 2, 0.10, "Many subdomains", "Multiple subdomains can make a destination harder to read.")
    add(value.count("-") >= 3, 0.06, "Multiple dashes", "Several dashes are a common obfuscation pattern.")
    add(sum(ch.isdigit() for ch in value) >= 5, 0.06, "Many numeric characters", "An unusually numeric URL can be a suspicious signal.")
    add("%" in value, 0.05, "Encoded characters", "URL encoding can be used to make a link harder to inspect.")
    add(len(parsed.query.split("&")) >= 3 and bool(parsed.query), 0.05, "Complex query string", "Several URL parameters add complexity to the destination.")
    add(bool(words_found), min(0.18, 0.06 * len(words_found)), "Sensitive wording", f"Found: {', '.join(words_found)}.")
    add("//" in parsed.path, 0.05, "Double slash in path", "An extra path separator may be used to obscure a URL.")

    score = min(score, 1.0)
    risk = "High" if score >= 0.45 else "Medium" if score >= 0.20 else "Low"
    verdict = "Suspicious URL patterns detected" if risk != "Low" else "No strong suspicious URL patterns detected"
    return {
        "input_url": value,
        "host": host,
        "scheme": parsed.scheme,
        "risk": risk,
        "risk_score": score,
        "verdict": verdict,
        "signals": signals,
    }
