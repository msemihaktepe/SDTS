# Gerekli kütüphaneler
import cv2
from flask import Flask, render_template, Response
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import mediapipe as mp

app = Flask(__name__)

# Ses dosyasını yükle
sound = AudioSegment.from_wav("beep.wav")

# Mediapipe kütüphanesinden yüz ve göz takibi için yardımcı fonksiyonları içeri aktar
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates


# Göz aspect oranını hesaplayan fonksiyon
def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        # Göz referans noktalarına göre koordinatları al
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y,
                                            frame_width, frame_height)
            coords_points.append(coord)

        # Göz noktaları arasındaki mesafeleri hesapla
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Göz kapalılık oranını hesapla
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

        print(ear)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


# Her iki göz için ortalama kapalılık oranını hesaplayan fonksiyon
def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    # Sol ve sağ gözlerin kapalılık oranlarını hesapla
    left_ear, left_lm_coordinates = get_ear(
        landmarks,
        left_eye_idxs,
        image_w,
        image_h
    )

    right_ear, right_lm_coordinates = get_ear(
        landmarks,
        right_eye_idxs,
        image_w,
        image_h
    )

    # Her iki gözün ortalamasını al
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


# İki nokta arasındaki öklidyen mesafeyi hesaplayan fonksiyon
def distance(point_1, point_2):
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist



# Göz takip sistemi çalışma fonksiyonu
def track_alertness():
    # Kamera başlat
    video_capture = cv2.VideoCapture(0)

    # Kamera çözünürlüğünü al
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))

    EYE_AR_THRESH = 0.2  # Kapalı gözler için göz aspect oranı eşik değeri
    EYE_AR_CONSEC_FRAMES = 20  # Alarmı tetiklemek için gözün eşik altında olması gereken ardışık kare sayısı
    YAWN_AR_THRESH = 0.5  # Esneme için üst ve alt dudak arasındaki mesafe eşik değeri


    # Mediapipe yüz takibi için sınıfı oluştur
    mp_facemesh = mp.solutions.face_mesh
    face_mesh = mp_facemesh.FaceMesh()

    # Mediapipe yüz takibi için sınıfı oluştur
    chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
    chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
    chosen_mouth_idxs = [308, 38, 268, 62, 86, 316]

    COUNTER = 0
    ALARM_ON = False
    sound_playback = None

    while True:
        # Kameradan bir kare al
        ret, frame = video_capture.read()
        height, width, _ = frame.shape

        # Mediapipe ile yüz ve göz takibi yap
        result = face_mesh.process(frame)
        facial_landmarks_result = result.multi_face_landmarks

        if facial_landmarks_result:

            for facial_landmarks in facial_landmarks_result:
                # Göz ve ağız kapalılık oranlarını ve koordinatları hesapla
                ear, coordinates = calculate_avg_ear(facial_landmarks.landmark,
                                                     chosen_left_eye_idxs,
                                                     chosen_right_eye_idxs,
                                                     frame_width,
                                                     frame_height
                                                     )

                mouth_ear, _ = get_ear(facial_landmarks.landmark, chosen_mouth_idxs, frame_width, frame_height)

                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                    # Gözler belirli bir sayıda ardışık kare boyunca kapalıysa alarmı çal
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(
                            frame,
                            "UYUMA !!!!",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                        if sound_playback:
                            if not sound_playback.is_playing():
                                sound_playback=_play_with_simpleaudio(sound)
                        else:
                            sound_playback = _play_with_simpleaudio(sound)

                else:
                    COUNTER = 0

                if mouth_ear > YAWN_AR_THRESH:
                    # Ağız açıklık oranına göre uyuma uyarısı
                    cv2.putText(
                        frame,
                        "YORGUNSUN ESNIYORSUN !!!",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    if sound_playback:
                        if not sound_playback.is_playing():
                            sound_playback = _play_with_simpleaudio(sound)
                    else:
                        sound_playback = _play_with_simpleaudio(sound)

        # Kareyi görüntüle
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    # Kamera serbest bırak
    video_capture.release()
    cv2.destroyAllWindows()

# Flask uygulaması için route'lar
@app.route("/sdts")
def sdts():
    return render_template("index.html")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        track_alertness(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True)
