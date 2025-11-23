import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import threading
from collections import deque
from escpos.printer import Usb

# -------------------- CONFIG --------------------

BAUD = 115200
screen_width = 1280
screen_height = 720

# -------------------- INIZIALIZZAZIONE VIDEO --------------------

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cv2.namedWindow("Videowall", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Videowall", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# -------------------- SERIAL MANAGER (interno) --------------------
# -------------------- SERIAL MANAGER SEMPLIFICATO --------------------
class SerialManager:
    def __init__(self, baud=BAUD):
        self.baud = baud
        self.ser = None
        self.event = None
        self.lock = threading.Lock()
        self.stop_flag = False
        self.thread = None
        self._open_port()

    def _open_port(self):
        ports = [p.device for p in serial.tools.list_ports.comports()
                 if ('ACM' in p.device or 'USB' in p.device)]

        if not ports:
            print("Nessuna porta seriale trovata")
            return
        
        try:
            self.ser = serial.Serial(ports[0], self.baud, timeout=0.1)
            self.ser.reset_input_buffer()
            print("Seriale aperta:", ports[0])
        except Exception as e:
            print("Errore apertura seriale:", e)

    def start(self):
        if not self.ser:
            return
        self.stop_flag = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=1)
        if self.ser:
            try:
                self.ser.close()
            except:
                pass

    def _read_loop(self):
        while not self.stop_flag:
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip().upper()

                    if line.startswith("PRES"):
                        with self.lock:
                            self.event = line  # ultimo evento ricevuto
                        print("Evento seriale:", line)
            except Exception as e:
                print("Errore seriale:", e)
            time.sleep(0.005)

    def get_last_event(self):
        with self.lock:
            ev = self.event
            self.event = None
            return ev
        
# -------------------- ISTANZIA E START --------------------
sm = SerialManager()
sm.start()


def transition(screen_width, screen_height):

    # ============================
    # STEP 1 — Quadrante 4 blu
    # ============================

    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.full((screen_height//2, screen_width//2, 3), (50, 200, 128), dtype=np.uint8)
    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)  
    
    time.sleep(1)

    # ============================
    # STEP 2 — Quadrante 1 blu
    # ============================

    qf1 = np.full((screen_height//2, screen_width//2, 3), (255, 0, 128), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    
    time.sleep(1)

    # ============================
    # STEP 3 — Quadrante 3 blu
    # ============================

    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.full((screen_height//2, screen_width//2, 3), (128, 128, 0), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall) 
    
    time.sleep(1)

    # ============================
    # STEP 4 — Quadrante 2 blu
    # ============================

    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.full((screen_height//2, screen_width//2, 3), (255, 255, 255), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall) 
    
    time.sleep(1)



def process_printer_test():
    
    """
    Funzione di test per stampante ESC/POS via USB.
    Stampa "hello world" e ritorna subito.
    """
    try:
        
        # Vendor ID e Product ID della tua stampante
        # Puoi scoprirli con 'lsusb' su Linux
        VENDOR_ID = 0x0483  # sostituisci con quello corretto
        PRODUCT_ID = 0x5840  # sostituisci con quello corretto
        INTERFACE = 0
        OUT_EP = 0x04
        IN_EP = 0x82

        p = Usb(VENDOR_ID, PRODUCT_ID, interface=INTERFACE, in_ep=IN_EP, out_ep=OUT_EP)
        p.text("hello world\n")
        p.cut()
        print("Stampato su stampante ESC/POS")
        time.sleep(0.5)  # piccola pausa
    except Exception as e:
        print("Errore stampante:", e)

def serial_event_triggered(n):
    ev = sm.get_last_event()
    return ev == f"PRES{n}"

def build_videowall(qf1, qf2, qf3, qf4):
    wall = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    wall[0:screen_height//2, 0:screen_width//2] = qf1
    wall[0:screen_height//2, screen_width//2:screen_width] = qf2
    wall[screen_height//2:screen_height, 0:screen_width//2] = qf3
    wall[screen_height//2:screen_height, screen_width//2:screen_width] = qf4
    return wall

def process_frame1():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        qf1 = cv2.resize(frame, (screen_width//2, screen_height//2))
        qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        
        
        # Face detection e grayscale per il quadrante 1

        gray = cv2.cvtColor(qf1, cv2.COLOR_BGR2GRAY)
        display_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))
        for (x, y, w, h) in faces:
            cv2.rectangle(display_gray, (x, y), (x+w, y+h), (255,255,255), 2)
        qf1 = display_gray
        
        # Costruisci videowall
        videowall = build_videowall(qf1,qf2,qf3,qf4)
        cv2.imshow("Videowall", videowall)

        # ESC per uscire
        if cv2.waitKey(1) == 27:
            exit(0)

        if serial_event_triggered(1):
            # piccolo pausa per sicurezza (debounce software lato Pi, opz.)
            time.sleep(0.05)
            break

def process_frame2():
    while True:
        qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf2 = np.full((screen_height//2, screen_width//2, 3), (0,0,255), dtype=np.uint8) # rosso
        qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)

        videowall = build_videowall(qf1, qf2, qf3, qf4)
        cv2.imshow("Videowall", videowall)

        if cv2.waitKey(1) == 27:
            exit(0)
        if serial_event_triggered(2):
            # piccolo pausa per sicurezza (debounce software lato Pi, opz.)
            time.sleep(0.05)
            break


def process_frame3():
    while True:
        qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf3 = np.full((screen_height//2, screen_width//2, 3), (0,255,0), dtype=np.uint8) # verde
        qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)

        videowall = build_videowall(qf1, qf2, qf3, qf4)
        cv2.imshow("Videowall", videowall)

        if cv2.waitKey(1) == 27:
            exit(0)
        if serial_event_triggered(3):
            # piccolo pausa per sicurezza (debounce software lato Pi, opz.)
            time.sleep(0.05)
            break


def process_frame4():
    while True:
        qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf4 = np.full((screen_height//2, screen_width//2, 3), (255,0,0), dtype=np.uint8) # blu

        videowall = build_videowall(qf1, qf2, qf3, qf4)
        cv2.imshow("Videowall", videowall)

        if cv2.waitKey(1) == 27:
            exit(0)

        if serial_event_triggered(4):
            # piccolo pausa per sicurezza (debounce software lato Pi, opz.)
            time.sleep(0.05)
            break


while True:
    process_frame1()
    transition(screen_width, screen_height)
    process_frame2()
    process_frame3()
    process_frame4()
    process_printer_test()
                       