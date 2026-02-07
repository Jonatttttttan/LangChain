import os
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from proto.marshal.rules import dates

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

def get_creds():
    creds = None

    # 1) Reaproveita token salvo
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json", SCOPES)
    # 2) Se não tiver token ou estiver inválido, renova ou pede login uma vez
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request()) # renova sem abrir navegador
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        # 3) Salva para próximas execuções
        with open("token.json", "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    return creds

def criar_evento(summary = "Teste", date = "2026-02-19", time="14:00:00"):
    timeend = time.replace(time.split(":")[0], str(int(time.split(":")[0]) + 1))
    print(timeend)
    creds = get_creds()

    service = build("calendar", "v3", credentials=creds)
    evento = {
        "summary": summary,
        "description": "Evento criado via API",
        "start":{
            "dateTime": date + "T" + time + "-03:00",
            "timeZone": "America/Sao_Paulo",
        },
        "end": {
            "dateTime": date + "T" + timeend + "-03:00",
            "timeZone": "America/Sao_Paulo",
        },
    }

    criado = service.events().insert(calendarId="primary", body=evento).execute()

    print("Evento criado")
    print("Link", criado["htmlLink"])

#criar_evento()

