#include <WiFi.h>
#include <WebSocketsServer.h>

const char* SSID = "Flia. Barradas";
const char* PASSWORD = "D1B9M21C";

const int PIN_R[3] = {32, 33, 25};
const int PIN_G[3] = {26, 27, 14};
const int PIN_B[3] = {12, 13, 15};

WebSocketsServer ws(81);

void apagarTodos() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_R[i], LOW);
    digitalWrite(PIN_G[i], LOW);
    digitalWrite(PIN_B[i], LOW);
  }
}

void encenderColor(char color, int cantidad) {
  apagarTodos();

  const int* pines = nullptr;
  if (color == 'R') pines = PIN_R;
  else if (color == 'G') pines = PIN_G;
  else if (color == 'B') pines = PIN_B;
  else return;

  for (int i = 0; i < cantidad && i < 3; i++) {
    digitalWrite(pines[i], HIGH);
  }
}

void onWsEvent(uint8_t client, WStype_t type, uint8_t* payload, size_t len) {
  if (type == WStype_CONNECTED) {
    Serial.println("Cliente conectado!");
  }
  if (type == WStype_DISCONNECTED) {
    Serial.println("Cliente desconectado.");
  }
  if (type == WStype_TEXT) {
    String cmd = String((char*)payload);
    cmd.trim();
    Serial.println("CMD: " + cmd);

    if (cmd.length() >= 2) {
      char color = cmd.charAt(0);
      int cantidad = cmd.charAt(1) - '0';
      encenderColor(color, cantidad);
    }
  }
}

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 3; i++) {
    pinMode(PIN_R[i], OUTPUT);
    pinMode(PIN_G[i], OUTPUT);
    pinMode(PIN_B[i], OUTPUT);
  }
  apagarTodos();

  WiFi.begin(SSID, PASSWORD);
  Serial.print("Conectando a WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nIP: " + WiFi.localIP().toString());

  ws.begin();
  ws.onEvent(onWsEvent);
  Serial.println("WebSocket server iniciado en puerto 81");
}

void loop() {
  ws.loop();
}
