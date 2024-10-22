import cv2
import face_recognition

def load_and_encode_image(image_path):
    """画像を読み込み、顔の特徴をエンコードします。"""
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]  # 最初の顔の特徴を取得
    return encoding

def compare_faces(known_face_encoding, unknown_face_encoding):
    """2つの顔の特徴を比較し、同一人物かどうかを判断します。"""
    results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)
    return results[0]

# 知っている人物の画像を読み込み、特徴をエンコード
known_image_path = r'C:\Users\hyeji\Documents\대학\수업\3-1\メディア係演習１\感性情報処理\practice\practice\SHIMLoriHyejin.jpg'
known_face_encoding = load_and_encode_image(known_image_path)

# カメラを起動
video_capture = cv2.VideoCapture(0)

while True:
    # カメラからのフレームを取得
    ret, frame = video_capture.read()
    if not ret:
        break

    # フレームから顔の特徴をエンコード
    unknown_face_encodings = face_recognition.face_encodings(frame)

    # すべての顔を比較
    for unknown_face_encoding in unknown_face_encodings:
        is_same_person = compare_faces(known_face_encoding, unknown_face_encoding)
        
        # 結果を表示
        result_text = "True" if is_same_person else "False"
        cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # フレームを表示
    cv2.imshow('Video', frame)

    # 'q'キーを押すとループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
video_capture.release()
cv2.destroyAllWindows()
