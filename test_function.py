import mediapipe as mp
import cv2


# Init detector
detector = mp.solutions.hands.Hands()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    annotated_image = frame.copy()
    result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   
    if result.multi_hand_landmarks is not None:
        for re in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(annotated_image, re,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())
        
    if not ret:
        break

    cv2.imshow("", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
