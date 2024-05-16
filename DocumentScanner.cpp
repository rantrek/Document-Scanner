

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//Function Prototypes

Mat loadImage(string);
Mat processImage(Mat);
vector<vector<Point>> detectContours(Mat, Mat);
void orderPoints(vector<Point>,vector<Point>&);
double distancePoints(Point, Point);
void fourPointTransform(Mat, Mat&, vector<vector<Point>>);

int main()
{    
	string imagePath("document.jpg");
	Mat image, original, processed, transformed;
	vector<vector<Point>> cpts;
	image = loadImage(imagePath);
	double ratio = image.rows / 500.0;
	original = image.clone();
	resize(image, image, Size(500,500), INTER_LINEAR);
	processed = processImage(image);
	cpts = detectContours(image,processed);
	fourPointTransform(image, transformed, cpts);
	return 0;
}

Mat loadImage(string path) {

	Mat img = imread(path, IMREAD_COLOR);


	if (img.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return Mat();
	}
	return img;
}

Mat processImage(Mat img) {

	Mat grey, edged;
	cvtColor(img, grey, COLOR_BGR2GRAY);
	GaussianBlur(grey, grey, Size(5, 5), 0);
	Canny(grey,edged, 75, 200);

	imshow("Edged", edged);
	waitKey(0);
	return edged;
}

vector<vector<Point>> detectContours(Mat img,Mat edge) {
	
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(edge.clone(), contours,hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	Mat icopy = img.clone();
	sort(contours.begin(), contours.end(), 
		[](const vector<Point>& a, const vector<Point>& b) { 
		  return contourArea(a)>contourArea(b); });
	vector<vector<Point>> cnts(contours.begin(), contours.begin() + 5);
	double peri, epsilon;
	vector<Point> approx;
	vector<vector<Point>> screencnt;
	//Loop over the contours
	for (auto &c : cnts) {
		//approximate the contour
		peri = arcLength(c, true);
		epsilon = double(0.02) * peri;
		approxPolyDP(c, approx,epsilon, true);
//if our approximated contour has four points, then add it to the vector screencnt
		if (approx.size() == 4) {
			screencnt.push_back(approx);
			break;
		}
	}

	drawContours(icopy, screencnt, -1, Scalar(0, 255, 0), 2);
	imshow("Contours", icopy);
	waitKey(0);

	return screencnt;
}

void orderPoints(vector<Point> inpts, vector<Point>& ordered)
{
	sort(inpts.begin(), inpts.end(), [](Point &a, Point &b) {return a.x< b.x; });
	vector<Point> lm(inpts.begin(), inpts.begin() + 2);
	vector<Point> rm(inpts.end() - 2, inpts.end());

	sort(lm.begin(), lm.end(), [](Point &a, Point &b) {return a.y < b.y; });
	Point tl(lm[0]);
	Point bl(lm[1]);
	vector<pair<Point, Point>> tmp;
	for (size_t i = 0; i < rm.size(); i++)
	{
		tmp.push_back(make_pair(tl, rm[i]));
	}

	sort(tmp.begin(), tmp.end(), [](pair<Point, Point> &a, pair<Point, Point> &b)
	{return (norm(a.first - a.second) < norm(b.first - b.second)); });
	Point tr(tmp[0].second);
	Point br(tmp[1].second);

	ordered.push_back(tl);
	ordered.push_back(tr);
	ordered.push_back(br);
	ordered.push_back(bl);
}


double distancePoints(Point p1, Point p2) {
	return sqrt(((p1.x - p2.x) * (p1.x - p2.x)) +
		((p1.y - p2.y) * (p1.y - p2.y)));
}

void fourPointTransform(Mat src, Mat& dst, vector<vector<Point>> pts) {
	vector<Point> ordered_pts;
	for (auto p : pts) {
		orderPoints(p, ordered_pts);
	}
	double wa = distancePoints(ordered_pts[2], ordered_pts[3]);
	double wb = distancePoints(ordered_pts[1], ordered_pts[0]);
	double mw = max(wa, wb);

	double ha = distancePoints(ordered_pts[1], ordered_pts[2]);
	double hb = distancePoints(ordered_pts[0], ordered_pts[3]);
	double mh = max(ha, hb);

	Point2f src_[] =
		{
			Point2f(ordered_pts[0].x, ordered_pts[0].y),
			Point2f(ordered_pts[1].x, ordered_pts[1].y),
			Point2f(ordered_pts[2].x, ordered_pts[2].y),
			Point2f(ordered_pts[3].x, ordered_pts[3].y),
		};
	Point2f dst_[] =
		{
			Point2f(0, 0),
			Point2f(mw - 1, 0),
			Point2f(mw - 1, mh - 1),
			Point2f(0, mh - 1)};
	Mat m = getPerspectiveTransform(src_, dst_);
	warpPerspective(src, dst, m, Size(mw, mh));

	imshow("Warped", dst);
	waitKey(0);
}