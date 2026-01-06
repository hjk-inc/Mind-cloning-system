import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from .model import SovereignTitanNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

class SovereignTitanKernel:
    """
    Online self-supervised motion predictor
    Features:
    - Multi-step imagination (T+1, T+2)
    - Surprise detection
    - Progressive depth growth
    """
    def __init__(self):
        os.makedirs("models", exist_ok=True)
        self.model = SovereignTitanNet().to(DEVICE)
        self.lock = threading.Lock()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scaler = GradScaler() if USE_AMP else None
        self.criterion = nn.MSELoss().to(DEVICE)
        self.training_buffer, self.target_buffer = [], []
        self.pose_history = []
        self.train_event = threading.Event()
        self.last_train_size, self.step_count = 0, 0
        self.best_loss = float("inf")
        self.surprise_factor = 0.0

    def normalize_pose(self, raw):
        pose = raw.reshape(33, 3)
        center = (pose[23] + pose[24]) / 2.0
        scale = np.linalg.norm(pose[11] - pose[12]) + 1e-6
        return ((pose - center) / scale).flatten()

    def train_cycle(self, duration=300):
        target_depth = min(50 + (len(self.training_buffer)//1000)*25, 450)
        if os.path.exists("models/titan_best.pth"):
            with self.lock:
                ckpt = torch.load("models/titan_best.pth", map_location=DEVICE)
                self.model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.step_count = ckpt.get("step", 0)
                self.best_loss = ckpt.get("loss", self.best_loss)
                self.model.active_blocks = max(target_depth, ckpt.get("active_blocks",50))

        for i in range(max(0, self.model.active_blocks-25)):
            for p in self.model.backbone[i].parameters(): p.requires_grad=False
        self.optimizer = optim.AdamW(filter(lambda p:p.requires_grad, self.model.parameters()), lr=1e-4)

        with self.lock:
            x_data = torch.FloatTensor(np.array(self.training_buffer[-10000:], copy=True)).to(DEVICE)
            y_data = torch.FloatTensor(np.array(self.target_buffer[-10000:], copy=True)).to(DEVICE)
        if x_data.size(0) < 128:
            self.train_event.clear()
            return

        self.model.train()
        start = time.time()
        while time.time() - start < duration:
            idx = torch.randint(0, x_data.size(0), (128,), device=DEVICE)
            self.optimizer.zero_grad()
            with autocast() if USE_AMP else torch.enable_grad():
                loss = self.criterion(self.model(x_data[idx]), y_data[idx])
            if USE_AMP:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.step_count += 1

            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                with self.lock:
                    torch.save({
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "step": self.step_count,
                        "loss": self.best_loss,
                        "active_blocks": self.model.active_blocks
                    }, "models/titan_best.pth")
        self.train_event.clear()

    def run(self):
        cap = cv2.VideoCapture(0)
        with mp.solutions.holistic.Holistic(model_complexity=1) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                if results.pose_landmarks:
                    norm = self.normalize_pose(np.array([[l.x, l.y, l.z] for l in results.pose_landmarks.landmark]).flatten())
                    self.pose_history.append(norm)
                    if len(self.pose_history) > 4: self.pose_history.pop(0)

                    if len(self.pose_history) == 4:
                        window = np.concatenate(self.pose_history[:3])
                        actual_next = self.pose_history[3]

                        self.training_buffer.append(window)
                        self.target_buffer.append(actual_next)

                        self.model.eval()
                        with torch.no_grad(), self.lock:
                            inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                            pred_t1 = self.model(inp)
                            next_window = torch.cat([inp[:, 99:], pred_t1], dim=1)
                            pred_t2 = self.model(next_window)

                            err = nn.functional.mse_loss(pred_t1, torch.tensor(actual_next, dtype=torch.float32).unsqueeze(0).to(DEVICE))
                            self.surprise_factor = err.item() * 1000

                        for i, (pred, color) in enumerate([(pred_t1,(255,255,0)),(pred_t2,(255,0,255))]):
                            pts = pred.cpu().numpy().reshape(33,3)
                            for p in pts:
                                x = int((p[0]*0.2+0.5)*frame.shape[1])
                                y = int((p[1]*0.2+0.5)*frame.shape[0])
                                cv2.circle(frame,(x,y), 2 if i==0 else 1, color,-1)

                cv2.putText(frame,f"Surprise: {self.surprise_factor:.2f}", (10,50), 2,0.7,(0,255,0),2)
                cv2.imshow("ChronosEngine V17", frame)

                if len(self.training_buffer) - self.last_train_size >= 5000 and not self.train_event.is_set():
                    self.last_train_size = len(self.training_buffer)
                    self.training_buffer, self.target_buffer = self.training_buffer[-20000:], self.target_buffer[-20000:]
                    self.train_event.set()
                    threading.Thread(target=self.train_cycle, daemon=True).start()
                if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release()
        cv2.destroyAllWindows()
