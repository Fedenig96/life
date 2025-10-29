import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import threading
from collections import deque


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
class SerialManager:
    def __init__(self, baud=BAUD):
        self.baud = baud
        self.serials = []
        self.lock = threading.Lock()
        self.events = deque()
        self.stop_flag = False
        self.thread = None
        self._open_all_ports()

    def _open_all_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports() if ('ACM' in p.device or 'USB' in p.device)]
        for p in ports:
            try:
                s = serial.Serial(p, self.baud, timeout=0.1)
                s.reset_input_buffer()
                self.serials.append(s)
                print("Opened serial port:", p)
            except Exception as e:
                print("Failed to open", p, e)


    def start(self):
        self.stop_flag = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=1)
        for s in self.serials:
            try:
                s.close()
            except:
                pass

    def _read_loop(self):
        while not self.stop_flag:
            for s in list(self.serials):
                try:
                    if s.in_waiting:
                        line = s.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            parts = line.split(',')
                            # expected format: ID,<n>,PRESS
                            if len(parts) >= 3 and parts[0] == 'ID':
                                try:
                                    dev_id = int(parts[1])
                                except:
                                    dev_id = None
                                event = parts[2].strip().upper()
                                with self.lock:
                                    self.events.append((dev_id, event))
                                print("Serial event:", dev_id, event)
                            else:
                                print("Serial raw:", line)
                except Exception as e:
                    print("Serial read error:", e)
            time.sleep(0.01)

    def get_event(self):
        with self.lock:
            return self.events.popleft() if self.events else None


    def pop_matching(self, match_dev_id, match_event='PRESS'):
        """
        Cerca il primo evento (dev_id, event) nella coda self.events che corrisponde a
        (match_dev_id, match_event). Se lo trova lo rimuove e ritorna True, altrimenti False.
        """
        with self.lock:
            for ev in list(self.events):          # iteriamo su copia per evitare problemi
                # ev � (dev_id, event)
                if len(ev) >= 2:
                    dev_id, event = ev[0], ev[1]
                else:
                    continue
                if dev_id == match_dev_id and event == match_event:
                    try:
                        self.events.remove(ev)
                    except ValueError:
                        pass
                    # debug utile
                    print("SerialManager.pop_matching: consumed", ev)
                    return True
        return False
        
# -------------------- ISTANZIA E START --------------------
sm = SerialManager()
sm.start()


def serial_event_triggered(expected_id):
    """
    Restituisce True SOLO se � presente e viene consumato un evento
    (expected_id, 'PRESS') nella coda.
    """
    # debug (opzionale) per vedere lo stato della coda
    # with sm.lock:
    #     print("Queue now:", list(sm.events))

    return sm.pop_matching(expected_id, 'PRESS')

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
    process_frame2()
    process_frame3()
    process_frame4()
    
                       