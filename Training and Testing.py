import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# normalize utility
def normalize_lm(flat):
    lm = flat.reshape(21,3)
    wrist = lm[0].copy()
    lm -= wrist
    norm = np.linalg.norm(lm)
    return (lm/norm if norm>0 else lm).flatten()

# train helper
def train(csv, model_name):
    df = pd.read_csv(csv)
    X  = np.stack([normalize_lm(r) for r in df.drop('label',axis=1).values])
    y  = df['label'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y,
                                          test_size=0.2,
                                          stratify=y,
                                          random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(128,64),
                        max_iter=500,
                        random_state=42)
    mlp.fit(Xtr, ytr)
    print(f"{model_name} Accuracy: {accuracy_score(yte, mlp.predict(Xte))*100:.2f}%")
    print(classification_report(yte, mlp.predict(Xte), zero_division=0))
    joblib.dump(mlp, model_name + '.pkl')
    return mlp

# train left & right
print("Training LEFT-hand model (digits + signs)...")
left_clf  = train('left_hand_dataset.csv',  'left_hand_model')
print("Training RIGHT-hand model (A–Z)...")
right_clf = train('right_hand_dataset.csv', 'right_hand_model')

# real-time inference
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# reload for demo
left_clf  = joblib.load('left_hand_model.pkl')
right_clf = joblib.load('right_hand_model.pkl')

cap = cv2.VideoCapture(0)
print("Starting real-time dual-hand recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = hands.process(rgb)

    if res.multi_hand_landmarks and res.multi_handedness:
        for lm, hand_h in zip(res.multi_hand_landmarks, res.multi_handedness):
            side  = hand_h.classification[0].label  # 'Left' or 'Right'
            flat  = np.array([[p.x,p.y,p.z] for p in lm.landmark]).flatten()
            norm  = normalize_lm(flat)
            if side == 'Left':
                pred = left_clf.predict([norm])[0]
                text = f"Left: '{pred}'"
                color = (0,255,0)
            else:
                pred = right_clf.predict([norm])[0]
                text = f"Right: '{pred}'"
                color = (0,200,255)
            # draw
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, text, (10, 30 if side=='Right' else 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Dual-Hand Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
