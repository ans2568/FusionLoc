import os
import gdown
import zipfile

def download_Zip(data_path, output, quiet=False):
    if os.path.exists(output):
        print(output + " data already exist!")
        return
    gdown.download(data_path, output=output, quiet=quiet)

def extract_Zip(zip_path, output_path):
    print("Start extract " + zip_path)
    with zipfile.ZipFile(zip_path) as file:
        if os.path.exists(output_path) and os.path.isdir(output_path):
            sub_dirs = [subDir for subDir in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, subDir))]
            dir_name = zip_path.split('/')[-1].split('.')[0]
            for sub_dir in sub_dirs:
                if sub_dir == dir_name:
                    print(dir_name + " directory already exist")
                    return
        file.extractall(path=output_path)
        print("Successful extraction " + zip_path + " data")

def makeDirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print('already exist ' + dir_path + ' directory')

if __name__ == "__main__":
    google_path = 'https://drive.google.com/uc?id='
    data_folder = "./data/NIA/"
    makeDirs(data_folder)

    camera_zip_id = '1wMMvq1SGyt_gEvlfA9u9eTpHiz_VmIru'
    camera_zip = 'camera.zip'

    lidar_zip_id = '1IYUkvp6hworm33YVQfYPlRhf2G8hirnu'
    lidar_zip = 'lidar.zip'

    csv_zip_id = '1HDFKOohrLOkRLGCBqxwe4EFtTZECASO9'
    csv_zip = 'csv.zip'

    download_Zip(google_path+camera_zip_id, data_folder+camera_zip)
    download_Zip(google_path+lidar_zip_id, data_folder+lidar_zip)
    download_Zip(google_path+csv_zip_id, data_folder+csv_zip)
    
    extract_Zip(data_folder+camera_zip, data_folder)
    extract_Zip(data_folder+lidar_zip, data_folder)
    extract_Zip(data_folder+csv_zip, data_folder)