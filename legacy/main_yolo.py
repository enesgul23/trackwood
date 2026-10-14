# -*- coding: utf-8 -*-
# wood_tracker_presentation_v3_fixed_final.py – Sayaç ve Sonuç Yazdırma Düzeltmeleri
# =====================================================================================
# Özellikler:
#  ▸ Bağımsız Sayaçlar: Her çizgi kendi geçişlerini bağımsız olarak sayar.
#  ▸ Düzeltilmiş Sayaç: Çizgiyi geçen *tüm* izler (hem update hem predict ile) sayılır.
#  ▸ Detaylı Yörünge Verisi Kaydı (CSV): Konum, BBox, Alan, Hız, Kalman durumu.
#  ▸ Ayrı Sayaç Geçiş Verisi Kaydı (CSV): Geçiş anı, ID, konum (Y), alan.
#  ▸ İstenilen sayıda dikey sanal sayım çizgisi ve ayrı sayaçlar.
#  ▸ Dinamik görselleştirme ölçeklemesi.
#  ▸ Basitleştirilmiş HUD (Sadece Kare Sayısı).
#  ▸ Yüksek çözünürlüklü video kaydı ve ölçekli canlı önizleme.
# -------------------------------------------------------------------------------------
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any, Optional

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# ───────────────────────────────  AYARLAR VE SABİTLER  ──────────────────────────────── #

# --- Temel Yapılandırma ---
CFG: Dict[str, Any] = {
    "video": "videolar/20060.MP4", # Video yolunu kontrol edin
    "model": "best400.pt",
    "out_dir": "results_v3_fixed_final",
    "conf": 0.25,
    "nms_iou": 0.6,
    # --- Sanal Çizgiler ---
    "lines": "v:20%,v:80%",
    # --- Görselleştirme Ayarları ---
    "vis": True,
    "preview_max_height": 960,
    "show_trails": True,
    "trail_length": 50,
    "line_counter_start_y": 150,
    # --- Takipçi Ayarları ---
    "max_age_f": 2,
    "min_hits": 3,
    "trk_iou": 0.3,
    # --- Veri Kaydı ---
    "save_trajectory_csv": True,
    "save_crossings_csv": True,
}

# --- Tip Tanımları ---
BBox = Tuple[int, int, int, int]
Point = Tuple[float, float]
CrossingInfo = Dict[str, Any]

# --- Görselleştirme Renkleri ve Stilleri (BGR formatında) ---
COL = {
    "bbox_active": (0, 255, 0),
    "trail": (255, 150, 50),
    "label_text": (255, 255, 255),
    "line": (0, 255, 255),
    "line_text": (255, 255, 255),
    "line_text_bg": (0, 0, 0),
    "hud_text": (255, 255, 255),
    "hud_bg": (0, 0, 0),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ──────────────────────────────  YARDIMCI FONKSİYONLAR ────────────────────────────── #
def iou(box_list1: np.ndarray, box_list2: np.ndarray) -> np.ndarray:
    if len(box_list1) == 0 or len(box_list2) == 0: return np.empty((len(box_list1), len(box_list2)))
    area1 = (box_list1[:, 2] - box_list1[:, 0]) * (box_list1[:, 3] - box_list1[:, 1])
    area2 = (box_list2[:, 2] - box_list2[:, 0]) * (box_list2[:, 3] - box_list2[:, 1])
    inter_x1 = np.maximum(box_list1[:, 0, np.newaxis], box_list2[:, 0])
    inter_y1 = np.maximum(box_list1[:, 1, np.newaxis], box_list2[:, 1])
    inter_x2 = np.minimum(box_list1[:, 2, np.newaxis], box_list2[:, 2])
    inter_y2 = np.minimum(box_list1[:, 3, np.newaxis], box_list2[:, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1); inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h; union = area1[:, np.newaxis] + area2 - intersection + 1e-6
    return intersection / union

def to_kalman_state(bbox: BBox) -> np.ndarray:
    x1, y1, x2, y2 = bbox; width = x2 - x1; height = y2 - y1
    center_x = x1 + width / 2; center_y = y1 + height / 2
    area = width * height; aspect_ratio = width / (height + 1e-6)
    return np.array([center_x, center_y, area, aspect_ratio], dtype=float)

def from_kalman_state(state: np.ndarray) -> Tuple[BBox, float]:
    cx, cy, s, r = state[:4]; s = max(s, 1e-6); r = max(r, 1e-6)
    width = math.sqrt(s * r); height = s / width
    x1 = int(cx - width / 2); y1 = int(cy - height / 2); x2 = int(cx + width / 2); y2 = int(cy + height / 2)
    return (x1, y1, x2, y2), float(s)

def draw_semi_transparent_rect(img: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Tuple[int, int, int], alpha: float):
    x1, y1 = top_left; x2, y2 = bottom_right; x1, x2 = min(x1, x2), max(x1, x2); y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
    if x1 >= x2 or y1 >= y2: return
    try:
        sub_img = img[y1:y2, x1:x2]; rect = np.full(sub_img.shape, color, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 1.0 - alpha, rect, alpha, 1.0); img[y1:y2, x1:x2] = res
    except Exception as e: print(f"Hata (Yarı Saydam Dikdörtgen): {e}")

# ─────────────────────────  KALMAN FİLTRESİ İLE TAKİP SINIFLARI  ────────────────────────────── #

class Track:
    _next_id: int = 1
    def __init__(self, initial_bbox: BBox, frame_index: int):
        self.id: int = Track._next_id; Track._next_id += 1
        self.kf = KalmanFilter(dim_x=7, dim_z=4); dt = 1.0
        self.kf.F = np.array([[1,0,0,0,dt,0,0],[0,1,0,0,0,dt,0],[0,0,1,0,0,0,dt],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], float)
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], float)
        self.kf.R *= 10.; self.kf.Q[4:, 4:] *= 0.01; self.kf.Q[:4, :4] *= 0.01; self.kf.P[4:, 4:] *= 1000.; self.kf.P *= 10.
        self.kf.x[:4] = to_kalman_state(initial_bbox).reshape(4, 1); self.kf.x[4:] = 0
        self.hits: int = 1; self.misses: int = 0; self.start_frame: int = frame_index; self.last_updated_frame: int = frame_index
        self.center_history: List[Tuple[int, float, float]] = [(frame_index, float(self.kf.x[0, 0]), float(self.kf.x[1, 0]))]
    def predict(self) -> BBox:
        self.kf.predict(); self.misses += 1; bbox, _ = from_kalman_state(self.kf.x); return bbox
    def update(self, detection_bbox: BBox, frame_index: int) -> None:
        measurement = to_kalman_state(detection_bbox).reshape(4, 1); self.kf.update(measurement)
        self.hits += 1; self.misses = 0; self.last_updated_frame = frame_index
        self.center_history.append((frame_index, float(self.kf.x[0, 0]), float(self.kf.x[1, 0])))
        max_hist_len = CFG.get("trail_length", 50) * 3
        if len(self.center_history) > max_hist_len: self.center_history.pop(0)
    def get_state_for_logging(self) -> Dict[str, Any]:
        bbox, area_k = from_kalman_state(self.kf.x); cx, cy = self.get_center(); vx, vy, vs = self.get_kalman_velocities()
        speed = self.calculate_speed(); area_bbox = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return {"cx": cx, "cy": cy, "bbox_x1": bbox[0], "bbox_y1": bbox[1], "bbox_x2": bbox[2], "bbox_y2": bbox[3],
                "bbox_area": area_bbox, "estimated_speed": speed, "kalman_area": area_k, "kalman_vx": vx, "kalman_vy": vy, "kalman_vs": vs}
    def get_bbox_and_area(self) -> Tuple[BBox, float]: return from_kalman_state(self.kf.x)
    def get_center(self) -> Point: return (float(self.kf.x[0, 0]), float(self.kf.x[1, 0]))
    def get_kalman_velocities(self) -> Tuple[float, float, float]: return (float(self.kf.x[4, 0]), float(self.kf.x[5, 0]), float(self.kf.x[6, 0]))
    def calculate_speed(self) -> float:
        if len(self.center_history) < 2: return 0.0
        f1, x1, y1 = self.center_history[-2]; f2, x2, y2 = self.center_history[-1]; delta_frames = f2 - f1
        if delta_frames <= 0: return 0.0
        distance = math.hypot(x2 - x1, y2 - y1); return distance / delta_frames

class Tracker:
    def __init__(self, iou_threshold: float, max_age: int, min_hits_to_activate: int):
        self.iou_threshold: float = iou_threshold; self.max_age: int = max_age; self.min_hits: int = min_hits_to_activate
        self.tracks: List[Track] = []; self._frame_count: int = 0
    def update(self, detections: List[BBox]) -> List[Track]:
        self._frame_count += 1; predicted_bboxes = []; active_track_indices = []
        for i, track in enumerate(self.tracks): predicted_bboxes.append(track.predict()); active_track_indices.append(i)
        predicted_bboxes_np = np.array(predicted_bboxes) if predicted_bboxes else np.empty((0, 4))
        detections_np = np.array(detections) if detections else np.empty((0, 4))
        matched_indices = []; unmatched_track_indices = list(range(len(predicted_bboxes_np))); unmatched_detection_indices = list(range(len(detections_np)))
        if predicted_bboxes_np.shape[0] > 0 and detections_np.shape[0] > 0:
            iou_matrix = iou(predicted_bboxes_np, detections_np); cost_matrix = 1.0 - iou_matrix
            try: row_ind, col_ind = linear_sum_assignment(cost_matrix)
            except ValueError: row_ind, col_ind = [], []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1.0 - self.iou_threshold):
                    original_track_idx = active_track_indices[r]; matched_indices.append((original_track_idx, c))
                    if r in unmatched_track_indices: unmatched_track_indices.remove(r)
                    if c in unmatched_detection_indices: unmatched_detection_indices.remove(c)
        for track_list_idx, detection_idx in matched_indices:
             if 0 <= track_list_idx < len(self.tracks) and 0 <= detection_idx < len(detections):
                self.tracks[track_list_idx].update(detections[detection_idx], self._frame_count)
        for detection_idx in unmatched_detection_indices:
            if 0 <= detection_idx < len(detections): self.tracks.append(Track(detections[detection_idx], self._frame_count))
        survivors = []; active_tracks_output = []
        for track in self.tracks:
            if track.misses <= self.max_age:
                survivors.append(track)
                if track.hits >= self.min_hits and track.misses == 0: active_tracks_output.append(track)
        self.tracks = survivors; return active_tracks_output

# ───────────────────────────  SANAL ÇİZGİ VE SAYMA  ─────────────────────────────── #

class VerticalLine:
    def __init__(self, line_id: str, x_coord: int, frame_height: int, color: Tuple[int, int, int], thickness: int = 2):
        self.line_id: str = line_id; self.x: int = x_coord; self.p1: Point = (x_coord, 0); self.p2: Point = (x_coord, frame_height - 1)
        self.color: Tuple[int, int, int] = color; self.thickness: int = thickness; self.crossed_ids: Set[int] = set()
    def _get_side(self, point: Point) -> float:
        val = point[0] - self.x; return 0 if abs(val) < 1 else math.copysign(1.0, val)
    def check_crossing(self, track: Track) -> Optional[CrossingInfo]:
        if len(track.center_history) < 2 or track.id in self.crossed_ids: return None
        f_prev, cx_prev, cy_prev = track.center_history[-2]; f_curr, cx_curr, cy_curr = track.center_history[-1]
        side_last = self._get_side((cx_curr, cy_curr)); side_prev = self._get_side((cx_prev, cy_prev))
        if side_last * side_prev < 0:
            self.crossed_ids.add(track.id); bbox, _ = track.get_bbox_and_area()
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            return {"line_id": self.line_id, "frame": f_curr, "track_id": track.id, "crossing_cy": cy_curr, "bbox_area_at_crossing": bbox_area}
        return None
    def draw(self, frame: np.ndarray) -> None:
        cv2.line(frame, (int(self.p1[0]), int(self.p1[1])), (int(self.p2[0]), int(self.p2[1])), self.color, self.thickness)

def parse_vertical_lines(spec: str, frame_height: int, frame_width: int) -> List[VerticalLine]:
    lines = []; tokens = [t.strip() for t in spec.split(',') if t.strip()]
    for i, token in enumerate(tokens):
        line_id = f'L{i+1}'
        try:
            parts = token.split(':'); assert len(parts) == 2 and parts[0].lower() == 'v'
            _, value_str = parts; is_percent = '%' in value_str; value = float(value_str.replace('%', ''))
            if is_percent: x = int(frame_width * value / 100.0)
            else: x = int(value)
            x = max(0, min(frame_width - 1, x)); color = COL.get("line", (0, 255, 255))
            lines.append(VerticalLine(line_id, x, frame_height, color))
        except Exception as e: print(f"Uyarı: Çizgi '{token}' ayrıştırılamadı: {e}")
    return lines

# ────────────────────────────  ANA İŞLEM DÖNGÜSÜ  ───────────────────────────────── #

def main():
    video_path = CFG["video"]; model_path = CFG["model"]; output_dir = Path(CFG["out_dir"])
    output_dir.mkdir(parents=True, exist_ok=True); print(f"Çıktı Dizini: {output_dir.resolve()}")
    cap = cv2.VideoCapture(video_path); assert cap.isOpened(), f"Hata: Video açılamadı: {video_path}"
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30; print(f"Video Açıldı: {video_path} ({frame_width}x{frame_height} @ {fps:.2f} FPS)")
    output_width = frame_width + (frame_width % 2); output_height = frame_height + (frame_height % 2)
    output_video_path = str(output_dir / f"{Path(video_path).stem}_annotated.mp4")
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)); print(f"Çıktı Videosu: {output_video_path}")
    detector = YOLO(model_path); print(f"YOLO Modeli Yüklendi: {model_path}")
    tracker = Tracker(iou_threshold=CFG["trk_iou"], max_age=int(fps * CFG["max_age_f"]), min_hits_to_activate=CFG["min_hits"])
    lines = parse_vertical_lines(CFG["lines"], frame_height, frame_width)
    if lines: print(f"Sanal Dikey Çizgiler ({len(lines)} adet) Ayrıştırıldı.")
    traj_csv_writer, traj_csv_file, traj_path = None, None, None
    cross_csv_writer, cross_csv_file, cross_path = None, None, None
    if CFG["save_trajectory_csv"]:
        traj_path = output_dir / f"{Path(video_path).stem}_trajectory_detailed.csv";
        try:
            traj_csv_file = open(traj_path, "w", newline="", encoding="utf-8-sig"); traj_csv_writer = csv.writer(traj_csv_file)
            traj_csv_writer.writerow(["frame", "track_id", "start_frame", "cx", "cy", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_area", "estimated_speed", "kalman_area", "kalman_vx", "kalman_vy", "kalman_vs"]); print(f"Yörünge CSV: {traj_path}")
        except IOError as e: print(f"Hata: Yörünge CSV açılamadı ({traj_path}): {e}")
    if CFG["save_crossings_csv"] and lines:
        cross_path = output_dir / f"{Path(video_path).stem}_line_crossings.csv";
        try:
            cross_csv_file = open(cross_path, "w", newline="", encoding="utf-8-sig"); cross_csv_writer = csv.writer(cross_csv_file)
            cross_csv_writer.writerow(["line_id", "frame", "track_id", "crossing_cy", "bbox_area_at_crossing"]); print(f"Geçiş CSV: {cross_path}")
        except IOError as e: print(f"Hata: Geçiş CSV açılamadı ({cross_path}): {e}")
    if CFG["vis"]:
        cv2.namedWindow("Takip Önizleme", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO); preview_h = CFG.get("preview_max_height", 960); preview_w = int(frame_width * min(1.0, preview_h / frame_height)); cv2.resizeWindow("Takip Önizleme", preview_w, preview_h)
    frame_index = 0; start_time = time.time(); print("\nVideo İşleme Başlatılıyor...")
    while True:
        ret, frame = cap.read();
        if not ret: print("\nVideo bitti."); break
        predictions = detector.predict(frame, conf=CFG["conf"], iou=CFG["nms_iou"], verbose=False)[0]
        detected_bboxes = [tuple(map(int, box)) for box in predictions.boxes.xyxy.cpu().numpy()]
        active_tracks = tracker.update(detected_bboxes)
        vis_frame = frame.copy(); sf = frame_height / 1080.0;
        def gs(s): return max(s * sf, 0.4);
        def gt(t): return max(1, int(t * sf + 0.5))
        for track in active_tracks:
            state_data = track.get_state_for_logging(); x1, y1, x2, y2 = state_data["bbox_x1"], state_data["bbox_y1"], state_data["bbox_x2"], state_data["bbox_y2"]
            if traj_csv_writer:
                try: traj_csv_writer.writerow([frame_index, track.id, track.start_frame, int(state_data["cx"]), int(state_data["cy"]), x1, y1, x2, y2, int(state_data["bbox_area"]), f"{state_data['estimated_speed']:.2f}", f"{state_data['kalman_area']:.1f}", f"{state_data['kalman_vx']:.2f}", f"{state_data['kalman_vy']:.2f}", f"{state_data['kalman_vs']:.2f}"])
                except Exception as e: print(f"Uyarı: Yörünge CSV yazma hatası: {e}")
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), COL["bbox_active"], gt(2)); label_text = f'{track.id}'
            (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, FONT, gs(0.5), gt(1)); lbl_x, lbl_y = max(0, x1), max(lbl_h + 5, y1 - 5)
            if lbl_x + lbl_w > frame_width: lbl_x = frame_width - lbl_w
            cv2.putText(vis_frame, label_text, (lbl_x, lbl_y), FONT, gs(0.5), COL["label_text"], gt(1), cv2.LINE_AA)
            if CFG["show_trails"] and len(track.center_history) > 1:
                points = [(int(p[1]), int(p[2])) for p in track.center_history[-CFG["trail_length"]:]]
                if len(points) > 1: pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2)); cv2.polylines(vis_frame, [pts], isClosed=False, color=COL["trail"], thickness=gt(2))
        line_crossing_counts = {line.line_id: len(line.crossed_ids) for line in lines}
        if lines and cross_csv_writer:
            for track in tracker.tracks: # TÜM izleri kontrol et
                for line in lines:
                    crossing_info = line.check_crossing(track) # Koşulsuz kontrol
                    if crossing_info:
                        try: cross_csv_writer.writerow([crossing_info["line_id"], crossing_info["frame"], crossing_info["track_id"], int(crossing_info["crossing_cy"]), int(crossing_info["bbox_area_at_crossing"])]); line_crossing_counts[line.line_id] = len(line.crossed_ids)
                        except Exception as e: print(f"Uyarı: Geçiş CSV yazma hatası: {e}")
        line_counter_start_y = CFG.get("line_counter_start_y", 150)
        for i, line in enumerate(lines):
            line.draw(vis_frame); count = line_crossing_counts.get(line.line_id, 0); count_text = f'{line.line_id}: {count}'
            text_y = line_counter_start_y + i * int(35 * sf); (lw, lh), l_base = cv2.getTextSize(count_text, FONT, gs(0.7), gt(2))
            bg_x1, bg_y1 = 10, text_y - lh - l_base - 3; bg_x2, bg_y2 = 10 + lw + 6, text_y + l_base + 3
            draw_semi_transparent_rect(vis_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), COL["line_text_bg"], alpha=0.6)
            cv2.putText(vis_frame, count_text, (15, text_y), FONT, gs(0.7), COL["line_text"], gt(2), cv2.LINE_AA)
        hud_text = f'Kare: {frame_index}'; (hud_w, hud_h), hud_base = cv2.getTextSize(hud_text, FONT, gs(0.8), gt(2))
        hud_x, hud_y = 10, frame_height - 15; bg_hx1, bg_hy1 = hud_x - 5, hud_y - hud_h - hud_base - 3; bg_hx2, bg_hy2 = hud_x + hud_w + 5, hud_y + hud_base + 3
        draw_semi_transparent_rect(vis_frame, (bg_hx1, bg_hy1), (bg_hx2, bg_hy2), COL["hud_bg"], alpha=0.6)
        cv2.putText(vis_frame, hud_text, (hud_x, hud_y), FONT, gs(0.8), COL["hud_text"], gt(2), cv2.LINE_AA)
        output_frame = cv2.resize(vis_frame, (output_width, output_height)); video_writer.write(output_frame)
        if CFG["vis"]:
            preview_frame = cv2.resize(vis_frame, (preview_w, preview_h)); cv2.imshow("Takip Önizleme", preview_frame)
            key = cv2.waitKey(1) & 0xFF;
            if key == ord('q') or key == 27: print("\nÇıkış istendi."); break
        frame_index += 1
        if frame_index % 100 == 0: elapsed = time.time() - start_time; fps_avg = frame_index / elapsed if elapsed > 0 else 0; print(f"  İşlenen Kare: {frame_index}, Geçen Süre: {elapsed:.1f}s, Ort. FPS: {fps_avg:.1f}")
    end_time = time.time(); total_time = end_time - start_time
    print("\n--- İşlem Tamamlandı ---"); print(f"Toplam İşlenen Kare: {frame_index}"); print(f"Toplam Süre: {total_time:.2f} saniye")
    if frame_index > 0 and total_time > 0: print(f"Genel Ortalama FPS: {frame_index / total_time:.2f}")
    print(f"Kaydedilen Video: {output_video_path}")
    if traj_csv_file and traj_path: print(f"Kaydedilen Yörünge CSV: {traj_path}")
    if cross_csv_file and cross_path: print(f"Kaydedilen Geçiş CSV: {cross_path}")
    for i, line in enumerate(lines): print(f"Çizgi {line.line_id} (x={line.x}) Toplam Geçiş: {len(line.crossed_ids)}")
    cap.release(); video_writer.release()
    if traj_csv_file: traj_csv_file.close()
    if cross_csv_file: cross_csv_file.close()
    if CFG["vis"]: cv2.destroyAllWindows(); print("Kaynaklar serbest bırakıldı.")

if __name__ == "__main__":
    try: Track._next_id = 1; main()
    except ImportError as e: print(f"Hata: Kütüphane bulunamadı: {e.name}. 'pip install ultralytics filterpy scipy numpy opencv-python' deneyin.")
    except Exception as e: print(f"\n!!! Beklenmeyen Hata: {e}"); import traceback; traceback.print_exc()