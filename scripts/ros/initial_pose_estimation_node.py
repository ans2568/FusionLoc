import rclpy
from rclpy.node import Node
from io import BytesIO
from submodules.pose_estimation import initial_pose_estimation
from img_scan.srv import ImgScan

class InitialPoseEstimation(Node):
    def __init__(self):
        super().__init__('ipe')
        self.srv = self.create_service(ImgScan, 'initial_pose_estimation', self.callback)
        print('start server')

    def callback(self, request, response):
        image = request.image
        scan = request.scan
        img_stream = BytesIO(image.data)
        response.x, response.y, response.theta = initial_pose_estimation(img_stream, scan)
        print("x : ", str(response.x))
        print("y : ", str(response.y))
        print("theta : ", str(response.theta))
        return response

def main(args=None):
    rclpy.init(args=args)
    server = InitialPoseEstimation()
    rclpy.spin(server)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
