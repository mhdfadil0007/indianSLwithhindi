# import cv2
# import mediapipe as mp

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )

# cap = cv2.VideoCapture(0)

# print("STEP 1: Hand detection started")
# print("Press ESC to exit")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to access camera")
#         break

#     frame = cv2.flip(frame, 1)

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(rgb_frame)

#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS
#             )

#     cv2.imshow("Live Sign – Hand Detection", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
