# -*- coding: utf-8 -*-

import time
import json
import csv
import base64
from datetime import datetime
from pathlib import Path

import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# ThingsBoard 配置
THINGSBOARD_SERVER = "localhost"
DEVICE_ACCESS_TOKEN = "tcWPElovLWhkCXY7ezVf"  # 设备访问令牌
PORT = 1883  # MQTT端口
# 视频流配置
VIDEO_STREAM_ENABLED = True
STREAM_FRAME_RATE = 5  # 上传帧率
#统计配置
CSV_FILE = "traffic_report1.csv"
last_report_time = time.time()
period_counter = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
total_counter = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
#MQTT客户端初始化
mqtt_client = mqtt.Client()
mqtt_connected = False
def on_connect(client, userdata, flags, rc):
    global mqtt_connected
    if rc == 0:
        print("成功连接到ThingsBoard!")
        mqtt_connected = True
    else:
        print(f"连接失败，错误码: {rc}")
        mqtt_connected = False
def on_disconnect(client, userdata, rc):
    global mqtt_connected
    mqtt_connected = False
    print("与ThingsBoard断开连接")
def connect_to_thingsboard():
    try:
        mqtt_client.username_pw_set(DEVICE_ACCESS_TOKEN)
        mqtt_client.on_connect = on_connect
        mqtt_client.on_disconnect = on_disconnect
        mqtt_client.connect(THINGSBOARD_SERVER, PORT, 60)
        mqtt_client.loop_start()
        print("正在连接到ThingsBoard...")
        time.sleep(2)  # 等待连接建立
        return mqtt_connected
    except Exception as e:
        print(f"连接ThingsBoard时出错: {e}")
        return False
def send_telemetry(data):
    """发送遥测数据到ThingsBoard"""
    if not mqtt_connected:
        print("MQTT未连接，无法发送数据")
        return False
    try:
        payload = json.dumps(data)
        mqtt_client.publish("v1/devices/me/telemetry", payload)
        print(f"数据已发送: {data}")
        return True
    except Exception as e:
        print(f"发送数据时出错: {e}")
        return False
#初始化模型
def initialize_model():
        model = YOLO("yolov12n.engine")
        return model
#流量统计类
class TrafficAnalyzer:
    def __init__(self):
        self.counting_line = None
        self.crossed_ids = set()
        self.class_counter = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        self.prev_positions = {}
        self.frame_count = 0
    def setup_counting_line(self, frame_width, frame_height):
        self.counting_line = [(0, frame_height // 2 + 60), (frame_width, frame_height // 2 + 60)]
    def process_detections(self, results, frame_width, frame_height):
        if self.counting_line is None:
            self.setup_counting_line(frame_width, frame_height)
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        cls_ids = results[0].boxes.cls.int().cpu().tolist()
        current_ids = set()
        line_y = self.counting_line[0][1]
        current_frame_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            current_ids.add(track_id)
            if track_id in self.prev_positions:
                prev_cy = self.prev_positions[track_id]
                if (prev_cy <= line_y) and (cy > line_y):
                    if track_id not in self.crossed_ids:
                        self.crossed_ids.add(track_id)
                        class_name = {2: "car",  3: "motorcycle",
5:"bus",7:"truck"}.get(cls_id, "unknown")
                        if class_name in self.class_counter:
                            # 更新总计统计
                            self.class_counter[class_name] += 1
                            # 记录当前帧检测到的车辆
                            current_frame_counts[class_name] += 1
            self.prev_positions[track_id] = cy
        expired_ids = set(self.prev_positions.keys()) - current_ids
        for tid in expired_ids:
            del self.prev_positions[tid]
        return self.class_counter.copy(), current_frame_counts
# 主函数
# 初始化模型
model = initialize_model()
# 初始化流量分析器
traffic_analyzer = TrafficAnalyzer()
# 初始化CSV文件
with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['时间戳', '汽车', '摩托车', '公交车', '卡车', '总计', '汇总'])
# 视频输入
video_path = "D:\python-anaconda-learn1\流量统计\视频.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))
# 视频输出
out = cv2.VideoWriter("output9.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
print("开始处理视频...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # YOLO检测跟踪
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7],
                          tracker="bytetrack.yaml", conf=0.4)
    # 处理检测结果
    total_counts, current_frame_counts=traffic_analyzer.process_detections(results, w, h)
    # 实时更新总计计数器
    for class_name in total_counts:
        if class_name in total_counter:
            total_counter[class_name] = total_counts[class_name]
    # 更新周期计数器
    for class_name in current_frame_counts:
        if class_name in period_counter:
            period_counter[class_name] += current_frame_counts[class_name]
# 每周期统计和上传逻辑
    current_time = time.time()
    time_diff = current_time - last_report_time
    if time_diff >= 10 or (not ret and sum(period_counter.values()) > 0):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        period_total = sum(period_counter.values())
        total = sum(total_counter.values())
        telemetry_data = {
            "ts": int(current_time * 1000),
            "values": {
                "car": period_counter['car'],
                "motorcycle": period_counter['motorcycle'],
                "bus": period_counter['bus'],
                "truck": period_counter['truck'],
                "total": total,  "period_total": period_total,
                "timestamp": timestamp}}
        send_telemetry(telemetry_data) # 发送到ThingsBoard
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,period_counter['car'],
                period_counter['motorcycle'],period_counter['bus'],
                period_counter['truck'],total, period_total ])
        period_counter = {k: 0 for k in period_counter}
        last_report_time = current_time if ret else current_time - time_diff
    annotated_frame = results[0].plot()
    if traffic_analyzer.counting_line:
        cv2.line(annotated_frame, traffic_analyzer.counting_line[0],
                 traffic_analyzer.counting_line[1], (0, 255, 0), 2)
    y_offset = 50
    cv2.putText(annotated_frame,f"TotalVehicles: {len(traffic_analyzer.crossed_ids)}",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for idx, (cls, count) in enumerate(total_counter.items()):
        y_offset += 40
        cv2.putText(annotated_frame, f"{cls.capitalize()}: {count}",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y_offset += 30
    cv2.putText(annotated_frame, f"10s-total: {sum(period_counter.values())}",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y_offset += 30
    cv2.putText(annotated_frame, f"Total: {sum(total_counter.values())}",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    status_color = (0, 255, 0) if mqtt_connected else (0, 0, 255)
    status_text = "Connected to ThingsBoard" if mqtt_connected else "Disconnected"
    cv2.putText(annotated_frame, status_text, (w - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    out.write(annotated_frame)
    cv2.imshow("Traffic Analytics - ThingsBoard", annotated_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        while cv2.waitKey(1) != ord(' '):
            pass
cap.release()
out.release()
cv2.destroyAllWindows()
if sum(period_counter.values()) > 0:# 发送最终数据
    final_data = {"ts": int(time.time() * 1000),
        "values": {
            "car": period_counter['car'],
            "motorcycle": period_counter['motorcycle'],
            "bus": period_counter['bus'],"truck": period_counter['truck'],
            "total": sum(total_counter.values()),
            "period_total": sum(period_counter.values()),
            "status": "processing_completed"}}
    send_telemetry(final_data)
mqtt_client.loop_stop()# 断开MQTT连接
mqtt_client.disconnect()
