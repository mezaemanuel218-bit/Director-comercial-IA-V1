import requests
from crm.zoho_auth import get_access_token


def get_records(module):

    token = get_access_token()

    headers = {
        "Authorization": f"Zoho-oauthtoken {token}"
    }

    url = f"https://www.zohoapis.com/crm/v2/{module}"

    page = 1
    records = []

    while True:

        params = {
            "page": page,
            "per_page": 200
        }

        r = requests.get(url, headers=headers, params=params, timeout=60)
        r.raise_for_status()

        data = r.json()

        if "data" not in data:
            if data:
                print(f"{module}: respuesta sin bloque data -> {data}")
            break

        records.extend(data["data"])

        info = data.get("info", {})

        if not info.get("more_records"):
            break

        page += 1

    print(f"{module}: {len(records)} registros")

    return records
