import cv2
import numpy as np
import mediapipe as mp
import keyboard

def main():
  mp_drawing = mp.solutions.drawing_utils # Variabel untuk menambahkan landmark ke dalam telapak tangan
  mp_hands = mp.solutions.hands # Variabel untuk memanggil class untuk deteksi telapak tangan
  cap = cv2.VideoCapture(0) # Variabel untuk mengambil gambar dari webcam
  bts_bawah = 40 # Variabel nilai batas bawah
  jarak_minimal = 50 # Variabel nilai minimal dalam bentuk persen untuk trigger kontrol permainan

  with mp_hands.Hands(max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
      # Mengambil gambar dari webcam
      success, image = cap.read()
      image_height, image_width, _ = image.shape
      if not success:
        print('[+] Camera kosong!!!')
        continue

      # Rubah posisi tampilan dari vertica
      # Merubah warna dari BGR ke RGB
      image = cv2.cvtColor(cv2.flip(image, 1),
                           cv2.COLOR_BGR2RGB)

      # Buat image menjadi tidak bisa dimodifikasi agar mempercepat proses perhitungan
      image.flags.writeable = False
      results = hands.process(image)
      image.flags.writeable = True

      # Rubah format gambar dari RGB ke BGR
      image = cv2.cvtColor(image,
                           cv2.COLOR_RGB2BGR)

      # Cek apakah tangan ditemukan pada webcam
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          # Mengambil nilai x dari jari telunjuk
          jarakXIFT = genXLandmark(HLandmark=hand_landmarks,
                                   hands=mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                   length=image_width)
          # Mengambil nilai y dari jari telunjuk
          jarakYIFT = genYLandmark(HLandmark=hand_landmarks,
                                   hands=mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                   length=image_height)

          # Mengambil nilai x dari ibu jari
          jarakXTT = genXLandmark(HLandmark=hand_landmarks,
                                  hands=mp_hands.HandLandmark.THUMB_TIP,
                                  length=image_width)
          # Mengambil nilai y dari ibu jari
          jarakYTT = genYLandmark(HLandmark=hand_landmarks,
                                  hands=mp_hands.HandLandmark.THUMB_TIP,
                                  length=image_height)

          # Menambahkan landmark ke dalam gambar
          mp_drawing.draw_landmarks(image,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS)

          # Membuat garis antar ibu jari dan jari telunjuk
          image = cv2.line(image,
                           (jarakXIFT, jarakYIFT),
                           (jarakXTT, jarakYTT),
                           (255,0,0),
                           2)

          # Rumus pythagoras
          jarakX = abs(jarakXIFT-jarakXTT) ** 2
          jarakY = abs(jarakYIFT-jarakYTT) ** 2
          checkJarak = np.sqrt(jarakX+jarakY)

          # Check jarak untuk memilih lompat atau tidak
          if checkJarak < bts_bawah:
            print('down', checkJarak)
            keyboard.send("space")
          else:
            print(f'up: {checkJarak}')
            # Hitung batas nilai bawah
            bts_bawah = round(jarak_minimal/100*checkJarak)

      # Show window
      cv2.imshow('T-rex Palm Detection', image)

      # Close window
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()

def genXLandmark(HLandmark,
                 hands,
                 length):
  return round(HLandmark.landmark[hands].x * length)

def genYLandmark(HLandmark,
                 hands,
                 length):
  return round(HLandmark.landmark[hands].y * length)

if __name__ == '__main__':
  main()