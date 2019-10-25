#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;
class Object
{
public:
	Object();
	Object(int index, float confidence, String name, Rect rect);
	~Object();
public:
	int index;
	String name;
	float confidence;
	Rect rect;
private:
};
Object::Object() {
}
Object::Object(int index, float confidence, String name, Rect rect) {
	this->index = index;
	this->confidence = confidence;
	this->name = name;
	this->rect = rect;
}
Object::~Object() {
}
//----------------------------全局常量----------------------------------
//配置好protxt文件，网络结构描述文件
//配置好caffemodel文件，训练好的网络权重
const String PROTOTX_FILE = "../MobileNetSSD_deploy.prototxt";
const String CAFFE_MODEL_FILE = "../mobilenet_iter_73000.caffemodel";
const String classNames[] = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };
const float CONF_THRESH = 0.7f;
int main() {
	//------------------------实例化网络----------------------------
	Net mobileNetSSD = readNetFromCaffe(PROTOTX_FILE, CAFFE_MODEL_FILE);
	if (mobileNetSSD.empty()) {
		cerr << "加载网络失败！" << endl;
		return -1;
	}
	TickMeter t;
	//----------------------设置网络输入-----------------------
	string imgsrc = { "../000067.jpg", "../000067.jpg" };
	Mat srcImg = imread("../000067.jpg");
	//将二维图像转换为CNN输入的张量Tensor,作为网络的输入
	mobileNetSSD.setInput(blobFromImage(srcImg, 1.0 / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false));
	t.start();
	//--------------------CNN网络前向计算----------------------
	Mat netOut = mobileNetSSD.forward();
	t.stop();
	cout << "检测时间=" << t.getTimeMilli() << "ms" << endl;

	TickMeter s;
	mobileNetSSD.setInput(blobFromImage(srcImg, 1.0 / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), true, false));
	s.start();
	//--------------------CNN网络前向计算----------------------
	Mat asd = mobileNetSSD.forward();
	s.stop();
	cout << "检测时间=" << s.getTimeMilli() << "ms" << endl;


	//----------------------解析计算结果-----------------------
	vector<Object> detectObjects;
	Mat detectionResult(netOut.size[2], netOut.size[3], CV_32F, netOut.ptr<float>());
	for (int i = 0; i < detectionResult.rows; i++) {
		//目标类别的索引
		int objectIndex = detectionResult.at<float>(i, 1);
		//检测结果置信度
		float confidence = detectionResult.at<float>(i, 2);
		//根据置信度阈值过滤掉置信度较小的目标
		if (confidence < CONF_THRESH) {
			continue;
		}
		//反归一化，得到图像坐标
		int xLeftUp = static_cast<int>(detectionResult.at<float>(i, 3)*srcImg.cols);
		int yLeftUp = static_cast<int>(detectionResult.at<float>(i, 4)*srcImg.rows);
		int xRightBottom = static_cast<int>(detectionResult.at<float>(i, 5)*srcImg.cols);
		int yRightBottom = static_cast<int>(detectionResult.at<float>(i, 6)*srcImg.rows);
		//矩形框
		Rect rect(Point{ xLeftUp,yLeftUp }, Point{ xRightBottom,yRightBottom });
		//保存结果
		detectObjects.push_back(Object{ objectIndex,confidence,classNames[objectIndex],rect });
	}
	//------------------------显示结果-----------------------------------
	int count = 0;
	for (auto& i : detectObjects) {
		rectangle(srcImg, i.rect, Scalar(0, 255, 255), 2);
		putText(srcImg, i.name, i.rect.tl(), 1, 1.8, Scalar(255, 0, 0), 2);
		cout << "第" << count << "个目标：" << i.name << "\t" << i.rect << "\t" << i.confidence << endl;
		count++;
	}
	imshow("MobileNet-SSD", srcImg);

	waitKey(0);
}
