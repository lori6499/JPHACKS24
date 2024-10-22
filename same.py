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

def resize_image(image, width, height):
    """画像を指定したサイズにリサイズします。"""
    return cv2.resize(image, (width, height))

# 知っている人物の画像パス
#　画像のパスは変えること
known_image_path = r'C:\Users\hyeji\Desktop\JPHACKS24\ElonMusk.jpg'
# 比較する顔の画像パス
unknown_image_path = r'C:\Users\hyeji\Desktop\JPHACKS24\ElonMusk２.jpg'

# 知っている人物の顔をエンコード
known_face_encoding = load_and_encode_image(known_image_path)

# 比較する顔をエンコード
unknown_face_encoding = load_and_encode_image(unknown_image_path)

# 同一人物かどうかを比較
is_same_person = compare_faces(known_face_encoding, unknown_face_encoding)

# 画像を表示
known_image = cv2.imread(known_image_path)
unknown_image = cv2.imread(unknown_image_path)

# 同一人物かどうかの結果を表示
if is_same_person:
    result_text = "True"
else:
    result_text = "False"


known_image_resized = resize_image(known_image, 300, 400)
unknown_image_resized = resize_image(unknown_image, 300, 400)


# 画像にテキストを追加
cv2.putText(known_image_resized, "Known: " + result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
cv2.putText(unknown_image_resized, "Unknown: " + result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# 画像を表示
cv2.imshow("Known Person", known_image_resized)
cv2.imshow("Unknown Person", unknown_image_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
