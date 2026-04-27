import os,json, base64, asyncio, websockets


from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv


load_dotenv()

VOICE = "alloy" # voz do GPT-4o
PCM_SR = 16000 # taxa de amostragem que usaremos no lado do cliente
PORT = 5050

app = FastAPI()

@app.websocket("/voice")
async def voice_bridge(ws: WebSocket) -> None:
    """
    1. O navegador abre a conexão ws://host:5050/voice
    2. O navegador transmite chunks PCM mono de 16 bits
        codificados em base64: {"audio" : "b64"}
    3. Encaminhamos os chunks para o OpenAI Realtime ('input_audio_buffer.append')
    4. Reenviamos os deltas de áudio do assistente de volta para
        o navegador da mesma forma
    5. Monitoramos eventos 'speech_started' e enviamos um truncate
        caso o usuário interrompa
    """
    await ws.accept()

    '''openai_ws = await websockets.connect(
        "wss://api.openai.com/v1/realtime?" +
        "model=gpt-4o-realtime-preview-2024-10-01",
        extra_headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta" : "realtime=v1"
        },
        max_size=None, max_queue=None # ilimitando para simplificar a demonstração

    )'''
    openai_ws = await websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-realtime",
        extra_headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta": "realtime=v1"
        }
    )

    # Inicializar a sessão em tempo real
    await openai_ws.send(json.dumps({
        "type" : "session.update",
        "session" : {
            "turn_detection" : {"type" : "server_vad"},
            "input_audio_format" : "pcm16",
            "output_audio_format" : "pcm16",
            "voice" : VOICE,
            "modalities" : ["audio"],
            "instructions" : "Você é um assistente de IA conciso e prestativo. Responda sempre em português."

        }
    }

    ))

    last_assistant_item = None # acompanhar a resposta atual do assistente


    latest_pcm_ts = 0 # timestamp em ms do cliente
    pending_marks = [] # Marcas de fala pendentes

    ''' async def from_client() -> None:
        """Transmitir chunks de áudio PCM do microfone do navegador --> OpneAI."""
        nonlocal latest_pcm_ts
        async for msg in ws.iter_text():
            data = json.loads(msg)
            pcm = base64.b64decode(data["audio"])
            latest_pcm_ts += int(len(pcm) / (PCM_SR * 2) * 1000)
            await openai_ws.send(json.dumps({
                "type" : "input_audio_buffer.append",
                "audio" : base64.b64encode(pcm).decode('ascii')
            }))
            await openai_ws.send(json.dumps({
                "type": "input_audio_buffer.commit"
            }))
            await openai_ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["audio"]
                }
            }))'''

    async def from_client():
        async for msg in ws.iter_text():
            data = json.loads(msg)

            # 🔥 usuário terminou de falar
            if "stop" in data:
                print("🛑 STOP recebido")

                await openai_ws.send(json.dumps({
                    "type": "input_audio_buffer.commit"
                }))

                await openai_ws.send(json.dumps({
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio"]
                    }
                }))
                continue

            # 🎤 áudio normal
            pcm = base64.b64decode(data["audio"])

            print("🎤 recebendo áudio")

            await openai_ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii")
            }))
    async def to_client() -> None:
        """Transmitir áudio do assistente + gerenciar interrupções."""
        nonlocal last_assistant_item, pending_marks
        async for raw in openai_ws:
            msg = json.loads(raw)

            # Assistente fala
            if msg.get("type") == "response.audio.delta":
                pcm = base64.b64decode(msg["delta"])
                await ws.send_json({ "audio" :
                                     base64.b64encode(pcm).decode("ascii")})
                last_assistant_item = msg.get("item_id")
            # Usuário começou a falar --> cancelar fala do assistente
            started = "input_audio_buffer.speech_started"
            if msg.get("type") == started and last_assistant_item:
                await openai_ws.send(json.dumps({
                    "type" : "conversation.item.truncate",
                    "item_id" : last_assistant_item,
                    "content_index" : 0,
                    "audio_end_ms" : 0 # parar imediatamente
                }))
                last_assistant_item = None
                pending_marks.clear()
            print("EVENTO:", msg)

    '''try:
        await asyncio.gather(from_client(), to_client())
    finally:
        await openai_ws.close()
        #await ws.close()
        if ws.client_state.name != "DISCONNECTED":
            await ws.close()'''
    try:
        await asyncio.gather(
            from_client(),
            to_client(),
        )
    except Exception as e:
        print("Erro:", e)
    finally:
        try:
            await openai_ws.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("realtime_voice_minimal:app", host="0.0.0.0", port=PORT)



