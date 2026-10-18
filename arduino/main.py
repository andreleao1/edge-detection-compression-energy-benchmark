#!/usr/bin/env python3
"""
Serial to Prometheus Exporter
Lê dados do Arduino via Serial e expõe métricas para o Prometheus

Requisitos:
    pip install pyserial prometheus-client

Uso:
    python serial_to_prometheus.py
"""

import serial
import time
import sys
from collections import deque
from datetime import datetime, date
from prometheus_client import start_http_server, Gauge, Counter
import argparse
import threading
import logging
from pathlib import Path
import yaml

# ==================== CONFIGURAÇÕES ====================

# Configurações padrão, sobrescritas por settings.yaml
DEFAULT_SERIAL_PORT = 'COM5'
DEFAULT_BAUD_RATE = 9600
DEFAULT_PROMETHEUS_PORT = 8080

# Numero de leituras identicas seguidas (tensao+corrente+potencia) que
# caracterizam um sensor/Arduino travado (INA219 preso retornando sempre o
# mesmo valor). Com o intervalo de 5s do sketch, o default de 4 leituras
# equivale a ~20s de dado congelado antes de disparar o reset.
DEFAULT_STUCK_THRESHOLD = 4

# Valores aceitos para --period/--mode. 'period' e sempre passado explicito
# (nunca inferido do relogio) porque uma sessao pode cruzar o limite entre
# periodos, o que tornaria a inferencia automatica ambigua.
VALID_PERIODS = {'manha', 'tarde', 'noite'}
VALID_MODES = {'baseline', 'workload'}


def load_settings():
    """Carrega as configurações locais do arquivo settings.yaml ao lado do script."""
    settings_path = Path(__file__).resolve().parent / 'settings.yaml'
    try:
        with settings_path.open(encoding='utf-8') as settings_file:
            settings = yaml.safe_load(settings_file)
    except FileNotFoundError as exc:
        raise RuntimeError(f'Arquivo de configuração não encontrado: {settings_path}') from exc

    if not isinstance(settings, dict):
        raise ValueError(f'O arquivo {settings_path} precisa conter um mapa YAML.')

    resolved = {
        'serial_port': settings.get('serial_port', DEFAULT_SERIAL_PORT),
        'baud_rate': int(settings.get('baud_rate', DEFAULT_BAUD_RATE)),
        'prometheus_port': int(settings.get('prometheus_port', DEFAULT_PROMETHEUS_PORT)),
        'stuck_threshold': int(settings.get('stuck_threshold', DEFAULT_STUCK_THRESHOLD)),
    }
    if not isinstance(resolved['serial_port'], str) or not resolved['serial_port'].strip():
        raise ValueError('serial_port em settings.yaml precisa ser um texto não vazio.')
    if resolved['baud_rate'] <= 0 or resolved['prometheus_port'] <= 0 or resolved['stuck_threshold'] <= 0:
        raise ValueError('baud_rate, prometheus_port e stuck_threshold precisam ser positivos.')
    return resolved

# ==================== MÉTRICAS PROMETHEUS ====================

# Labels de sessao (session_id, period, day, mode) sao fixos para a vida
# inteira do processo — uma nova sessao de coleta significa reiniciar este
# script com novos valores de flag, nao trocar o label em tempo de execucao.
SESSION_LABELS = ['session_id', 'period', 'day', 'mode']

# Define as métricas Prometheus
energia_tensao = Gauge('energia_tensao', 'Tensao em volts', SESSION_LABELS)
energia_corrente = Gauge('energia_corrente', 'Corrente em amperes', SESSION_LABELS)
energia_potencia = Gauge('energia_potencia', 'Potencia em watts', SESSION_LABELS)
energia_temperatura_ambiente = Gauge(
    'energia_temperatura_ambiente_celsius',
    'Temperatura ambiente em graus Celsius (DS18B20)',
    SESSION_LABELS,
)
energia_status = Gauge('energia_status', 'Status da leitura (1=valido, 0=invalido)', SESSION_LABELS)
leituras_total = Counter('energia_leituras_total', 'Total de leituras realizadas')
erros_total = Counter('energia_erros_total', 'Total de erros de leitura')
resets_travamento_total = Counter(
    'energia_resets_travamento_total',
    'Total de resets do Arduino por leitura travada (valor fixo repetido)'
)

# ==================== CONFIGURAÇÃO DE LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CLASSE PRINCIPAL ====================

class ArduinoSerialReader:
    """Lê dados do Arduino via Serial e atualiza métricas Prometheus"""
    
    def __init__(self, port, baud_rate, session_labels, stuck_threshold=DEFAULT_STUCK_THRESHOLD):
        self.port = port
        self.baud_rate = baud_rate
        self.session_labels = session_labels
        self.serial = None
        self.running = False
        self.stuck_threshold = stuck_threshold
        self._recent_readings = deque(maxlen=stuck_threshold)

    def connect(self):
        """Conecta à porta serial"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1
            )
            logger.info(f"✅ Conectado à porta {self.port} ({self.baud_rate} baud)")
            time.sleep(2)  # Aguarda Arduino resetar
            return True
        except serial.SerialException as e:
            logger.error(f"❌ Erro ao conectar na porta {self.port}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro inesperado: {e}")
            return False
    
    def disconnect(self):
        """Desconecta da porta serial"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Porta serial fechada")
    
    def parse_csv_line(self, line):
        """
        Parse de linha CSV do Arduino
        Formatos aceitos:
        SENSOR,timestamp,tensao,corrente,potencia,status
        SENSOR,timestamp,tensao,corrente,potencia,status,temperatura_c
        """
        try:
            parts = line.strip().split(',')
            
            if len(parts) not in (6, 7):
                return None
            
            if parts[0] != 'SENSOR':
                return None
            
            data = {
                'timestamp': int(parts[1]),
                'voltage': float(parts[2]),
                'current': float(parts[3]),
                'power': float(parts[4]),
                'valid': int(parts[5])
            }

            if len(parts) == 7:
                temperature_c = float(parts[6])
                # -127 °C é o sentinela DEVICE_DISCONNECTED_C da biblioteca
                # DallasTemperature, portanto não deve virar uma métrica real.
                if temperature_c > -100:
                    data['temperature_c'] = temperature_c
            
            return data
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Erro ao fazer parse da linha: {line.strip()} - {e}")
            return None
    
    def update_metrics(self, data):
        """Atualiza as métricas do Prometheus"""
        try:
            energia_tensao.labels(**self.session_labels).set(data['voltage'])
            energia_corrente.labels(**self.session_labels).set(data['current'])
            energia_potencia.labels(**self.session_labels).set(data['power'])
            energia_status.labels(**self.session_labels).set(data['valid'])
            if 'temperature_c' in data:
                energia_temperatura_ambiente.labels(**self.session_labels).set(
                    data['temperature_c']
                )
            leituras_total.inc()

            temperature_info = (
                f"T_amb:{data['temperature_c']:.2f}°C | "
                if 'temperature_c' in data else ""
            )

            logger.info(
                f"📊 V:{data['voltage']:.3f}V | "
                f"I:{data['current']:.3f}A | "
                f"P:{data['power']:.3f}W | "
                f"{temperature_info}"
                f"Status:{'OK' if data['valid'] else 'INVÁLIDO'}"
            )
            
        except Exception as e:
            logger.error(f"Erro ao atualizar métricas: {e}")
            erros_total.inc()

    def _is_stuck(self, data):
        """Detecta sensor/Arduino travado: mesma tripla (tensao, corrente,
        potencia) se repetindo por 'stuck_threshold' leituras seguidas.
        Comparacao exata (sem tolerancia) — o ruido do ADC do INA219 torna
        praticamente impossivel repetir o mesmo valor por acaso em varias
        leituras reais consecutivas."""
        self._recent_readings.append((data['voltage'], data['current'], data['power']))

        if len(self._recent_readings) < self.stuck_threshold:
            return False

        return len(set(self._recent_readings)) == 1

    def _handle_stuck_sensor(self):
        """Reseta o Arduino inteiro reabrindo a porta serial. Placas com
        bootloader padrao (Uno/Nano/Mega) resetam sozinhas quando a linha DTR
        e alternada ao abrir a conexao — o mesmo mecanismo usado pela IDE
        para fazer upload — entao isso reinicializa tambem o barramento I2C
        do microcontrolador, nao so o sensor."""
        logger.warning(
            f"⚠️ Leitura travada detectada ({self.stuck_threshold} valores "
            f"identicos seguidos) — reiniciando Arduino..."
        )
        resets_travamento_total.inc()
        self._recent_readings.clear()

        self.disconnect()
        if not self.connect():
            logger.error("❌ Falha ao reconectar apos reset por travamento.")

    def read_loop(self):
        """Loop principal de leitura"""
        logger.info("🔄 Iniciando loop de leitura...")
        
        while self.running:
            try:
                if self.serial and self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Ignora linhas de comentário (começam com #)
                    if line.startswith('#'):
                        logger.debug(f"Comentário: {line}")
                        continue
                    
                    # Ignora linhas vazias
                    if not line:
                        continue
                    
                    # Parse e atualiza métricas
                    data = self.parse_csv_line(line)
                    if data:
                        self.update_metrics(data)
                        if self._is_stuck(data):
                            self._handle_stuck_sensor()
                    else:
                        logger.debug(f"Linha ignorada: {line}")
                
                time.sleep(0.1)  # Pequeno delay para não sobrecarregar CPU
                
            except serial.SerialException as e:
                logger.error(f"❌ Erro de comunicação serial: {e}")
                erros_total.inc()
                time.sleep(1)
                
                # Tenta reconectar
                logger.info("🔄 Tentando reconectar...")
                self.disconnect()
                if not self.connect():
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                logger.info("Interrompido pelo usuário")
                self.running = False
                break
                
            except Exception as e:
                logger.error(f"❌ Erro inesperado: {e}")
                erros_total.inc()
                time.sleep(1)
    
    def start(self):
        """Inicia a leitura em thread separada"""
        self.running = True
        thread = threading.Thread(target=self.read_loop, daemon=True)
        thread.start()
        return thread

# ==================== FUNÇÃO PRINCIPAL ====================

def _prompt_choice(title: str, choices: list) -> str:
    """Menu numerado no terminal. 'choices' e uma lista de (valor, rotulo) —
    retorna o valor correspondente a opcao escolhida. Usado para --period e
    --mode quando omitidos na linha de comando, evitando erro de digitacao
    nesses dois campos que precisam bater entre rasp_monitor (Pi) e este
    script (desktop)."""
    print(title)
    for i, (_, label) in enumerate(choices, start=1):
        print(f"  {i} - {label}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1][0]
        print(f"Opcao invalida: {raw!r} — digite o numero correspondente.\n")


def main():
    """Função principal"""

    try:
        settings = load_settings()
    except (OSError, RuntimeError, ValueError, yaml.YAMLError) as e:
        logger.error(f'❌ Erro ao carregar settings.yaml: {e}')
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Serial to Prometheus Exporter - Lê dados do Arduino e expõe para Prometheus'
    )
    parser.add_argument(
        '-p', '--port',
        default=settings['serial_port'],
        help=f'Porta serial (padrão de settings.yaml: {settings["serial_port"]})'
    )
    parser.add_argument(
        '-b', '--baud',
        type=int,
        default=settings['baud_rate'],
        help=f'Baud rate (padrão de settings.yaml: {settings["baud_rate"]})'
    )
    parser.add_argument(
        '--prometheus-port',
        type=int,
        default=settings['prometheus_port'],
        help=f'Porta do servidor Prometheus (padrão de settings.yaml: {settings["prometheus_port"]})'
    )
    parser.add_argument(
        '--stuck-threshold',
        type=int,
        default=settings['stuck_threshold'],
        help=(
            'Numero de leituras identicas seguidas para considerar o sensor '
            f'travado e reiniciar o Arduino (padrão de settings.yaml: {settings["stuck_threshold"]})'
        )
    )
    parser.add_argument(
        '--session-id',
        default=None,
        help=(
            'Identificador unico da sessao de coleta. Se omitido, usa '
            '"<day>-<period>" — assim bate automaticamente com o rasp_monitor '
            'rodando em outra maquina, sem precisar copiar um valor entre elas. '
            'Passe explicito so para desambiguar um rerun do mesmo day+period.'
        )
    )
    parser.add_argument(
        '--period',
        default=None,
        choices=sorted(VALID_PERIODS),
        help=(
            'Periodo da sessao — sempre explicito, nunca inferido do relogio. '
            'Se omitido, pergunta interativamente.'
        )
    )
    parser.add_argument(
        '--day',
        default=date.today().isoformat(),
        help='Data da sessao (YYYY-MM-DD). Padrao: hoje.'
    )
    parser.add_argument(
        '--mode',
        default=None,
        choices=sorted(VALID_MODES),
        help=(
            'baseline (idle, sem inferencia rodando) ou workload (modelos '
            'rodando). Se omitido, pergunta interativamente.'
        )
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Modo verboso (debug)'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.mode is None:
        args.mode = _prompt_choice(
            "Selecione o modo:",
            [('baseline', 'baseline'), ('workload', 'workload')],
        )
    if args.period is None:
        args.period = _prompt_choice(
            "Selecione o periodo:",
            [('manha', 'Manha'), ('tarde', 'Tarde'), ('noite', 'Noite')],
        )

    session_id = args.session_id or f'{args.day}-{args.period}'
    session_labels = {
        'session_id': session_id,
        'period': args.period,
        'day': args.day,
        'mode': args.mode,
    }
    
    # Banner
    print("=" * 60)
    print("  Serial to Prometheus Exporter")
    print("  Monitor de Energia Arduino INA219")
    print("=" * 60)
    print(f"  Porta Serial: {args.port} ({args.baud} baud)")
    print(f"  Servidor Prometheus: http://localhost:{args.prometheus_port}")
    print(f"  Limiar de travamento: {args.stuck_threshold} leituras identicas")
    print(f"  Sessao: session_id={session_id} period={args.period} day={args.day} mode={args.mode}")
    print("=" * 60)
    print()

    # Inicia servidor HTTP do Prometheus
    try:
        start_http_server(args.prometheus_port)
        logger.info(f"🌐 Servidor Prometheus iniciado na porta {args.prometheus_port}")
        logger.info(f"   Acesse: http://localhost:{args.prometheus_port}/metrics")
    except OSError as e:
        logger.error(f"❌ Erro ao iniciar servidor HTTP: {e}")
        logger.error(f"   A porta {args.prometheus_port} pode estar em uso")
        sys.exit(1)

    # Cria leitor serial
    reader = ArduinoSerialReader(
        args.port, args.baud, session_labels, stuck_threshold=args.stuck_threshold
    )
    
    # Conecta à porta serial
    if not reader.connect():
        logger.error("❌ Falha ao conectar. Verifique a porta serial e tente novamente.")
        logger.info("\nDica: No Linux, use 'ls /dev/tty*' para listar portas")
        logger.info("      No Windows, verifique o Gerenciador de Dispositivos")
        sys.exit(1)
    
    # Inicia leitura
    thread = reader.start()
    
    logger.info("✅ Sistema pronto! Pressione Ctrl+C para parar")
    logger.info("")
    
    try:
        # Mantém o programa rodando
        while thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Encerrando...")
        reader.running = False
        thread.join(timeout=2)
    finally:
        reader.disconnect()
        logger.info("👋 Programa encerrado")

# ==================== PONTO DE ENTRADA ====================

if __name__ == "__main__":
    main()
