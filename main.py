import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import threading
from collections import deque
from escpos.printer import Usb
from PIL import Image

# -------------------- CONFIG --------------------
saved_frame = None   # conterrà l'immagine grayscale salvata
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


def transition1(screen_width, screen_height):

    # ============================
    # STEP 1 — Quadrante 4 blu
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.full((screen_height//2, screen_width//2, 3), (255,0,0), dtype=np.uint8)
    cv2.putText(qf2, "SCEGLI BENE", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "NON TORNI INDIETRO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 2 — Quadrante 1 rosa
    # ============================
    qf1 = np.full((screen_height//2, screen_width//2, 3), (255, 0, 128), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    cv2.putText(qf1, "DOVE SONO?", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf4, "ASPETTA", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 3 — Quadrante 3 giallo
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.full((screen_height//2, screen_width//2, 3), (128, 128, 0), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)

    cv2.putText(qf2, "ARRIVO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "LOADING...", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 4 — Quadrante 2 bianco
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.full((screen_height//2, screen_width//2, 3), (255, 255, 255), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    cv2.putText(qf2, "ARRIVO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "LOADING...", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)


def transition2(screen_width, screen_height):

    # ============================
    # STEP 1 — Quadrante 4 blu
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.full((screen_height//2, screen_width//2, 3), (255,0,0), dtype=np.uint8)
    cv2.putText(qf2, "SCEGLI BENE", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "NON TORNI INDIETRO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 2 — Quadrante 1 rosa
    # ============================
    qf1 = np.full((screen_height//2, screen_width//2, 3), (255, 0, 128), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    cv2.putText(qf1, "DOVE SONO?", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf4, "ASPETTA", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 3 — Quadrante 3 giallo
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.full((screen_height//2, screen_width//2, 3), (128, 128, 0), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)

    cv2.putText(qf2, "ARRIVO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "LOADING...", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 4 — Quadrante 2 bianco
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.full((screen_height//2, screen_width//2, 3), (255, 255, 255), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    cv2.putText(qf2, "ARRIVO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "LOADING...", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)


def transition3(screen_width, screen_height):

    # ============================
    # STEP 1 — Quadrante 4 blu
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.full((screen_height//2, screen_width//2, 3), (255,0,0), dtype=np.uint8)
    cv2.putText(qf2, "SCEGLI BENE", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "NON TORNI INDIETRO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 2 — Quadrante 1 rosa
    # ============================
    qf1 = np.full((screen_height//2, screen_width//2, 3), (255, 0, 128), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    cv2.putText(qf1, "DOVE SONO?", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf4, "ASPETTA", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 3 — Quadrante 3 giallo
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.full((screen_height//2, screen_width//2, 3), (128, 128, 0), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)

    cv2.putText(qf2, "ARRIVO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "LOADING...", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 4 — Quadrante 2 bianco
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.full((screen_height//2, screen_width//2, 3), (255, 255, 255), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    cv2.putText(qf2, "ARRIVO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "LOADING...", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)


def transition4(screen_width, screen_height):

    # ============================
    # STEP 1 — Quadrante 4 blu
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.full((screen_height//2, screen_width//2, 3), (255,0,0), dtype=np.uint8)
    cv2.putText(qf2, "SCEGLI BENE", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "NON TORNI INDIETRO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 2 — Quadrante 1 rosa
    # ============================
    qf1 = np.full((screen_height//2, screen_width//2, 3), (255, 0, 128), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    cv2.putText(qf1, "DOVE SONO?", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf4, "ASPETTA", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 3 — Quadrante 3 giallo
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf3 = np.full((screen_height//2, screen_width//2, 3), (128, 128, 0), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)

    cv2.putText(qf2, "DISTRUGGE", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "E LA FINE", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "RIORDINA", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "FRAMMENTI", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)

    # ============================
    # STEP 4 — Quadrante 2 bianco
    # ============================
    qf1 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf2 = np.full((screen_height//2, screen_width//2, 3), (0, 255, 0), dtype=np.uint8)
    qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
    cv2.putText(qf1, "CI SIAMO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf2, "SARA' TUO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf4, "PRENDILO", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
    cv2.putText(qf3, "PRINT PROCESS/EXEC...", (120, 80), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)

    videowall = build_videowall(qf1, qf2, qf3, qf4)
    cv2.imshow("Videowall", videowall)
    cv2.waitKey(1)
    time.sleep(1)




def process_printer_test():
    
    global saved_frame

    if saved_frame is None:
        print("Nessun frame salvato da stampare!")
        return
    
    try:
        
        # Vendor ID e Product ID della tua stampante
        # Puoi scoprirli con 'lsusb' su Linux
        VENDOR_ID = 0x0483  # sostituisci con quello corretto
        PRODUCT_ID = 0x5840  # sostituisci con quello corretto
        INTERFACE = 0
        OUT_EP = 0x04
        IN_EP = 0x82

        # p = Usb(VENDOR_ID, PRODUCT_ID, in_ep=IN_EP, out_ep=OUT_EP)

        img = saved_frame.copy() 
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Ridimensiona alla larghezza massima della stampante
        max_width = 384  # tipico per molte termiche ESC/POS
        scale = max_width / img.shape[1]
        new_height = int(img.shape[0] * scale)
        img_resized = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_LINEAR)
        img_bright = cv2.add(img_resized,30)
        # Converti in PIL Image 1-bit
        pil_img = Image.fromarray(img_bright)
        pil_img = pil_img.convert('1')  # bianco/nero 1-bit

        # Invia alla stampante
        p = Usb(VENDOR_ID, PRODUCT_ID, in_ep=IN_EP, out_ep=OUT_EP)
        p.image(pil_img)
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
    global saved_frame
    saved_frame = None  # reset
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        qf1 = cv2.resize(frame, (screen_width//2, screen_height//2))
        qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf4 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        
        cv2.putText(
            qf1,
            "GUARDA QUI",
            (120, 80),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        # Face detection e grayscale per il quadrante 1

        gray = cv2.cvtColor(qf1, cv2.COLOR_BGR2GRAY)
        display_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))
        for (x, y, w, h) in faces:
            cv2.rectangle(display_gray, (x, y), (x+w, y+h), (255,255,255), 2)
            
        qf1 = display_gray

        saved_frame = gray.copy()
        
        # Costruisci videowall
        videowall = build_videowall(qf1,qf2,qf3,qf4)
        cv2.imshow("Videowall", videowall)

        # ESC per uscire
        if cv2.waitKey(1) == 27:
            exit(0)

        if serial_event_triggered(1):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                saved_frame = gray.copy()
                print("Frame salvato:", saved_frame.shape)
            # piccolo pausa per sicurezza (debounce software lato Pi, opz.)
            time.sleep(0.05)
            break

def process_frame2():
    global saved_frame

    if saved_frame is not None:
        preview = cv2.resize(saved_frame, (screen_width//2, screen_height//2))
    

    while True:


        descaling_factor = 4  # puoi cambiare questo valore a piacere

        small = cv2.resize(preview, 
                           (preview.shape[1] // descaling_factor, preview.shape[0] // descaling_factor), 
                           interpolation=cv2.INTER_AREA)
        # riporta alla dimensione originale del quadrante
        modified_preview = cv2.resize(small, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_NEAREST)

        saved_frame = modified_preview.copy() 
        

        qf1 = np.full((screen_height//2, screen_width//2, 3), (0,0,255), dtype=np.uint8) # rosso
        qf2 = cv2.cvtColor(modified_preview, cv2.COLOR_GRAY2BGR)
        qf3 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf4 = np.full((screen_height//2, screen_width//2, 3), (255,0,0), dtype=np.uint8) # rosso

        videowall = build_videowall(qf1, qf2, qf3, qf4)
        cv2.imshow("Videowall", videowall)

        if cv2.waitKey(1) == 27:
            exit(0)
        if serial_event_triggered(2):
            # piccolo pausa per sicurezza (debounce software lato Pi, opz.)
            time.sleep(0.05)
            break


def process_frame3():
    global saved_frame
    if saved_frame is not None:
        preview = cv2.resize(saved_frame, (screen_width//2, screen_height//2))

    
    # Coordinate del rettangolo
    x, y = 100, 50      # angolo in alto a sinistra
    w, h = 80, 80       # larghezza e altezza del rettangolo
    noise_intensity = 30  # regolabile, quantità di rumore

    while True:
        # Creo la maschera circolare
        mask = np.zeros_like(preview, dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)  # 255 dentro il cerchio

        # Creo il rumore
        noise = np.random.randint(-noise_intensity, noise_intensity+1, preview.shape, dtype=np.int16)
        noisy_img = preview.astype(np.int16) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        # Applico il noise solo fuori dal cerchio
        modified = noisy_img.copy()
        modified[mask == 255] = preview[mask == 255]   # l'area dentro il cerchio resta originale

        # Aggiorno saved_frame per il prossimo step
        saved_frame = modified.copy()

        qf1 = np.full((screen_height//2, screen_width//2, 3), (255,0,255), dtype=np.uint8) # rosso
        qf2 = np.zeros((screen_height//2, screen_width//2, 3), dtype=np.uint8)
        qf3 = cv2.cvtColor(modified, cv2.COLOR_GRAY2BGR)
        qf4 = np.full((screen_height//2, screen_width//2, 3), (255,255,0), dtype=np.uint8) # rosso

        videowall = build_videowall(qf1, qf2, qf3, qf4)
        cv2.imshow("Videowall", videowall)

        if cv2.waitKey(1) == 27:
            exit(0)
        if serial_event_triggered(3):
            saved_frame = modified.copy()  # aggiorna saved_frame per il prossimo step
           
            # piccolo pausa per sicurezza (debounce software lato Pi, opz.)
            time.sleep(0.05)
            break


def process_frame4():
    global saved_frame
    if saved_frame is not None:
        preview = cv2.resize(saved_frame, (screen_width//2, screen_height//2))
    while True:
        x_offset = 50  # quanto traslare in pixel
        quad_width = 100  # larghezza di ciascun "quadrante" da spostare

        modified = preview.copy()
        h, w = modified.shape[:2]

        # Quadrante 1 (alto-sinistra)
        start_x = 0
        end_x = start_x + quad_width
        quad = modified[:, start_x:end_x].copy()
        blank = np.zeros_like(quad)
        blank[:, x_offset:] = quad[:, :quad.shape[1]-x_offset]
        modified[:, start_x:end_x] = blank

        # Quadrante 2 (alto-destra)
        start_x = quad_width
        end_x = start_x + quad_width
        quad = modified[:, start_x:end_x].copy()
        blank = np.zeros_like(quad)
        blank[:, x_offset:] = quad[:, :quad.shape[1]-x_offset]
        modified[:, start_x:end_x] = blank


        saved_frame = modified.copy()  # pronto per il print



        qf1 = np.full((screen_height//2, screen_width//2, 3), (0,0,255), dtype=np.uint8) # rosso
        qf2 = np.full((screen_height//2, screen_width//2, 3), (128,128,0), dtype=np.uint8) # rosso
        qf3 = np.full((screen_height//2, screen_width//2, 3), (0,128,128), dtype=np.uint8) # rosso
        qf4 = cv2.cvtColor(modified, cv2.COLOR_GRAY2BGR)

        
        videowall = build_videowall(qf1, qf2, qf3, qf4)
        cv2.imshow("Videowall", videowall)

        if cv2.waitKey(1) == 27:
            exit(0)

        if serial_event_triggered(4):
            saved_frame = modified.copy() 
            # piccolo pausa per sicurezza (debounce software lato Pi, opz.)
            time.sleep(0.05)
            break


while True:
    process_frame1()
    transition1(screen_width, screen_height)
    process_frame2()
    transition2(screen_width, screen_height)
    process_frame3()
    transition3(screen_width, screen_height)
    process_frame4()
    transition4(screen_width, screen_height)
    process_printer_test()
                       