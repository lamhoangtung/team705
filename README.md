# team705 - HAUI.notTrashCar
Package của Team 705 - HAUI.notTrashCar cho vòng loại Cuộc đua số 2018 - 2019

## Dependency
- Python 3.6+ (recommend Python 3.6.7)
- CUDA, CuDNN, cmake, opencv-python (`pip3 install opencv-python`), ros_bridge
- Darknet binary cho model object detection: clone repo https://github.com/lamhoangtung/darknet, `Makefile` đã được chỉnh sửa sẵn, chỉ cần `make` rồi copy file `libdarknet.so` vào thư mục `/src/team705/model/` và đổi tên thành `darknet.so`. 
```
git clone https://github.com/lamhoangtung/darknet ~/darknet/
cd ~/darknet/
make
cp ./libdarknet.so ~/team705/src/team705/model/darknet.so
```
- Sau đó build lại package `team705` một lần nữa:
```
cd team705
catkin_make
```

## Cách run
Package được chạy theo hướng dẫn trong file [`team705.launch`](/src/team705/launch/team705.launch), đầu tiên chạy `ros_bridge` khởi tạo kết nối đến simulator sau đó chạy đến [`main.py`](/src/team705/src/main.py) như một node của ROS
``` 
source ./devel/setup.bash
roslaunch team705 team705.launch
```


## Liên hệ
Vui lòng liên hệ với mình nếu cần bất kì hỗ trợ gì:
- Hoàng Tùng Lâm
- 0962724967
- lamhoangtung.vz@gmail.com
- https://facebook.com/lam.hoangtung.69
