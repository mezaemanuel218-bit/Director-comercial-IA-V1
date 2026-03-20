from .zoho_client import get_records


def fetch_leads():
    return get_records("Leads")


def fetch_contacts():
    return get_records("Contacts")


def fetch_accounts():
    return get_records("Accounts")


def fetch_notes():
    return get_records("Notes")


def fetch_calls():
    return get_records("Calls")


def fetch_events():
    return get_records("Events")


def fetch_tasks():
    return get_records("Tasks")


def fetch_deals():
    return get_records("Deals")


def fetch_users():
    return get_records("Users")