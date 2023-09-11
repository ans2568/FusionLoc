import cv2
import json
import base64
import numpy as np
from flask import Flask, request
from flask_restx import Resource, Api, Namespace
from rest_pose_estimation import initial_pose_estimation

InitialPoseEstimation = Namespace(name='Initial Pose Estimation')

@InitialPoseEstimation.route('')
class IPE(Resource):
    def get(self):
        try:
            req = request.get_json()
            data_decoded = base64.b64decode(req['image'])
            jpg_arr = np.frombuffer(data_decoded, np.uint8)
            image = cv2.imdecode(jpg_arr, cv2.IMREAD_COLOR)
            resume = req['resume']
            lidar = req['lidar']
            x, y, theta = initial_pose_estimation(image=image, lidar=lidar, resume=str(resume))
            json_val = json.dumps({'x':x, 'y':y, 'theta':theta}, cls=NumpyEncoder)
            result = json.loads(json_val)
            return result
        except Exception as e:
            print(str(e))
            return str(e)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask('Initial Pose Estimation')
api = Api(
    app,
    version='0.1',
    title='Initial Pose Estimation',
    description='This is API Server',
    terms_url='/',
    contact='ans2568@kangwon.ac.kr',
)
api.add_namespace(InitialPoseEstimation, '/poseestimation')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)