# ROS bag 파일로 데이터 셋 만들기

### ROS bag 파일 내부에 있어야 할 토픽 타입
- 이미지 토픽
  - default : /image_raw/compressed
  - 현재 sensor_msgs/msg/CompressedImage 타입만 지원
  - 추후 sensor_msgs/msg/Image 지원 예정
- 라이다 토픽
  - default : /scan
  - 현재 sensor_msgs/msg/LaserScan 타입만 지원
  - 추후 sensor_msgs/msg/PointCloud2 지원 예정
- 위치 토픽(nav_msgs/msg/Odometry)
  - default : /odom
---
### 데이터 셋 만드는 법
```
python bag2csv.py --bag /path/to/ros2_ws/bag/data.bag \ 
--csvDir ../data/custom_dataset/csv \
--lidarDir ../data/custom_dataset/lidar \
--cameraDir ../data/custom_dataset/camera \
--cameraTopic /compressed/image_raw \
--lidarTopic /scan \
--odometryTopic /odom
```
---
### 순서
1. ros2 bag cli로 bag 파일 생성
2. bag2csv.py 실행
3. 아키텍쳐에 활용할 csv 파일 및 이미지, PCD 파일 생성