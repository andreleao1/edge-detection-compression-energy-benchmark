/*
 * Monitor de Energia INA219 - Saída Serial
 * 
 * Este código lê dados do sensor INA219 e envia via Serial
 * no formato que pode ser facilmente parseado por um script Python
 * 
 * Formato de saída:
 * SENSOR,timestamp,tensao,corrente,potencia,status,temperatura_c
 */

#include <Wire.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ==================== CONFIGURAÇÕES ====================

// Intervalo entre leituras (em milissegundos)
const unsigned long INTERVALO_LEITURA = 5000; // 5 segundos

// Configurações do Sensor
const float SHUNT_RESISTANCE_OHM = 0.1;
const uint8_t DS18B20_PIN = 4;

// ==================== OBJETOS GLOBAIS ====================

Adafruit_INA219 ina219;
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);

// Estrutura para armazenar leituras do sensor
struct SensorData {
  float voltage;      // Tensão em Volts
  float current;      // Corrente em Amperes
  float power;        // Potência em Watts
  bool valid;         // Indica se a leitura é válida
  unsigned long timestamp; // Timestamp em milissegundos
  float temperatureC; // Temperatura ambiente em graus Celsius
};

// ==================== FUNÇÕES DE INICIALIZAÇÃO ====================

/**
 * Inicializa o sensor INA219
 * @return true se inicialização bem-sucedida, false caso contrário
 */
bool initSensor() {
  Serial.println("# Inicializando INA219...");
  
  if (!ina219.begin()) {
    Serial.println("# ERRO: Falha ao inicializar INA219");
    Serial.println("# Verifique: alimentação, GND comum e conexões SDA/SCL");
    return false;
  }
  
  Serial.println("# INA219 inicializado com sucesso");
  
  // Calibração padrão
  // ina219.setCalibration_32V_2A();
  // ina219.setCalibration_32V_1A();
  // ina219.setCalibration_16V_400mA();
  
  delay(100);
  return true;
}

/**
 * Escaneia o barramento I2C para detectar dispositivos
 */
void scanI2C() {
  Serial.println("# === Scanner I2C ===");
  
  byte error, address;
  int deviceCount = 0;
  
  for (address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    
    if (error == 0) {
      Serial.print("# Dispositivo encontrado em 0x");
      if (address < 16) Serial.print("0");
      Serial.print(address, HEX);
      Serial.println();
      deviceCount++;
    }
  }
  
  if (deviceCount == 0) {
    Serial.println("# Nenhum dispositivo I2C encontrado");
  } else {
    Serial.print("# Total de dispositivos: ");
    Serial.println(deviceCount);
  }
}

// ==================== FUNÇÕES DE LEITURA ====================

/**
 * Lê dados do sensor INA219
 * @return Estrutura SensorData com as leituras
 */
SensorData readSensorData() {
  SensorData data;

  // A conversão padrão do DS18B20 pode levar até 750 ms. Faça-a antes de
  // registrar o timestamp e ler o INA219 para manter os valores sincronizados.
  ds18b20.requestTemperatures();
  data.temperatureC = ds18b20.getTempCByIndex(0);
  data.timestamp = millis();
  
  // Lê valores do sensor
  float shuntVoltage_mV = ina219.getShuntVoltage_mV();
  float busVoltage_V = ina219.getBusVoltage_V();
  float current_mA = ina219.getCurrent_mA();
  float power_mW = ina219.getPower_mW();
  
  // Verifica se as leituras são válidas
  if (isnan(busVoltage_V) || isnan(current_mA) || isnan(power_mW)) {
    data.valid = false;
    
    // Tenta calcular corrente a partir do shunt se disponível
    if (!isnan(shuntVoltage_mV)) {
      current_mA = shuntVoltage_mV / SHUNT_RESISTANCE_OHM;
    }
  } else {
    data.valid = true;
  }
  
  // Converte para unidades padrão
  data.voltage = busVoltage_V;
  data.current = current_mA / 1000.0;  // mA para A
  data.power = power_mW / 1000.0;      // mW para W
  
  // Garante valores positivos
  if (data.current < 0) {
    data.current = -data.current;
  }
  
  return data;
}

// ==================== FUNÇÕES DE SAÍDA ====================

/**
 * Envia dados do sensor no formato CSV via Serial
 * Formato: SENSOR,timestamp,tensao,corrente,potencia,status,temperatura_c
 * @param data Dados do sensor
 */
void sendSerialData(const SensorData& data) {
  Serial.print("SENSOR,");
  Serial.print(data.timestamp);
  Serial.print(",");
  Serial.print(data.voltage, 3);
  Serial.print(",");
  Serial.print(data.current, 3);
  Serial.print(",");
  Serial.print(data.power, 3);
  Serial.print(",");
  Serial.print(data.valid ? 1 : 0);
  Serial.print(",");
  // DallasTemperature retorna DEVICE_DISCONNECTED_C (-127) se não detectar
  // nenhum sensor; mantenha o sentinela explícito para o parser identificar.
  Serial.println(data.temperatureC, 2);
}

/**
 * Envia dados no formato JSON via Serial (alternativo)
 * @param data Dados do sensor
 */
void sendSerialDataJSON(const SensorData& data) {
  Serial.print("{\"timestamp\":");
  Serial.print(data.timestamp);
  Serial.print(",\"voltage\":");
  Serial.print(data.voltage, 3);
  Serial.print(",\"current\":");
  Serial.print(data.current, 3);
  Serial.print(",\"power\":");
  Serial.print(data.power, 3);
  Serial.print(",\"valid\":");
  Serial.print(data.valid ? 1 : 0);
  Serial.print(",\"temperature_c\":");
  Serial.print(data.temperatureC, 2);
  Serial.println("}");
}

// ==================== SETUP E LOOP ====================

void setup() {
  // Inicializa Serial
  Serial.begin(9600);
  while (!Serial) {
    ; // Aguarda porta serial
  }
  
  Serial.println("# =================================");
  Serial.println("# Monitor de Energia INA219");
  Serial.println("# Saída via Serial");
  Serial.println("# =================================");
  Serial.println("# ");
  
  // Inicializa I2C
  Wire.begin();
  delay(100);
  
  // Escaneia barramento I2C
  scanI2C();
  Serial.println("# ");
  
  // Inicializa sensor
  if (!initSensor()) {
  Serial.println("# AVISO: Sistema continuará sem sensor");
  }

  ds18b20.begin();
  Serial.println("# DS18B20 inicializado no pino 4");
  
  Serial.println("# ");
  Serial.println("# === Sistema Pronto ===");
  Serial.println("# Formato: SENSOR,timestamp,tensao,corrente,potencia,status,temperatura_c");
  Serial.println("# ");
  
  delay(1000);
}

void loop() {
  static unsigned long lastReading = 0;
  
  // Faz leitura no intervalo definido
  if (millis() - lastReading >= INTERVALO_LEITURA) {
    SensorData data = readSensorData();
    
    // Envia dados via Serial em formato CSV
    sendSerialData(data);
    
    // OU use JSON (comente a linha acima e descomente esta):
    // sendSerialDataJSON(data);
    
    lastReading = millis();
  }
}
