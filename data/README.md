# 데이터 다운로드

`download_data.py` 는 현재 구글 드라이브에 올라가 있는 AI Hub 데이터 셋을 다운로드 해주는 파일

### Usage

```bash
python download_data.py
```

##### result

```
data
    \_NIA
         \_ camera
                  \_ image files
         \_ csv
               \_csv files
         \_ lidar
                 \_ pcd files
```

# 데이터 train, test 분리

`train_test_split.py` 는 `ros/bag2csv.py` 파일로 생성된 `whole_synchronized_data.csv` 를 학습할 수 있게 train과 test로 나눠주는 파일

### Usage

```bash
python3 train_test_split.py --dataset iiclab --file whole_synchronized_data.csv
```

##### before

```
data
    \_iiclab
            \_ camera
                     \_ image files
            \_ csv
            	  \_whole_synchronized_data.csv
            \_ lidar
            	    \_ pcd files
```

##### after

```
data
    \_iiclab
            \_ camera
                     \_ image files
            \_ csv
            	  \_whole_synchronized_data.csv
                  \_train_db_data.csv
		  \_train_query_data.csv
		  \_test_db_data.csv
		  \_test_query_data.csv
            \_ lidar
                    \_ pcd files
```
