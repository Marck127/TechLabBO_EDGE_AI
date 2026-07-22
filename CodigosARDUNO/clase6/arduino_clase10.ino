const int PIN_R[3] = {2, 3, 4};
const int PIN_G[3] = {5, 6, 7};
const int PIN_B[3] = {8, 9, 10};

void apagarTodos() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_R[i], LOW);
    digitalWrite(PIN_G[i], LOW);
    digitalWrite(PIN_B[i], LOW);
  }
}

void encenderColor(char color, int cantidad) {
  apagarTodos();

  int* pines = nullptr;
  if (color == 'R') pines = (int*)PIN_R;
  else if (color == 'G') pines = (int*)PIN_G;
  else if (color == 'B') pines = (int*)PIN_B;
  else return;

  for (int i = 0; i < cantidad && i < 3; i++) {
    digitalWrite(pines[i], HIGH);
  }
}

void setup() {
  Serial.begin(9600);

  for (int i = 0; i < 3; i++) {
    pinMode(PIN_R[i], OUTPUT);
    pinMode(PIN_G[i], OUTPUT);
    pinMode(PIN_B[i], OUTPUT);
  }
  apagarTodos();
}

void loop() {
  if (Serial.available() > 0) {
    String comando = Serial.readStringUntil('\n');
    comando.trim();

    if (comando.length() >= 2) {
      char color = comando.charAt(0);
      int cantidad = comando.charAt(1) - '0';
      encenderColor(color, cantidad);
    }
  }
}
