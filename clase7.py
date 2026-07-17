import asyncio
import websockets

IP_ESP32 = "192.168.1.5"  # IP que imprime el ESP32 en el Monitor Serial
ESP32_WS_URL = f"ws://{IP_ESP32}:81"

async def saludo():
    while True:
        try:
            async with websockets.connect(ESP32_WS_URL) as ws:
                print("[WebSocket] Conectado a ESP32")
                while True:
                    await ws.send("hola")
                    print("Python envio: hola")

                    respuesta = await ws.recv()
                    print(f"ESP32 respondio: {respuesta}")

                    await asyncio.sleep(1)
        except Exception as e:
            print(f"[WebSocket] Reconectando... ({e})")
            await asyncio.sleep(1)



if __name__ == "__main__":
    asyncio.run(saludo())
