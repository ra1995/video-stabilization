/*
Thanks Nghia Ho and Chen jia for their excellent work on video-stabilization
I modified the code to use GPU processing via opencl
As a result it can more smoothly process live video streaming 
modified by Rishabh Agrawal.
email:rg1995007@gmail.com
visit http://nghiaho.com/?p=2093 for implementation details
*/

#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;
using namespace cv::ocl;

const int HORIZONTAL_BORDER_CROP = 20;

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
	// "+"
	friend Trajectory operator+(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x+c2.x,c1.y+c2.y,c1.a+c2.a);
	}
	//"-"
	friend Trajectory operator-(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x-c2.x,c1.y-c2.y,c1.a-c2.a);
	}
	//"*"
	friend Trajectory operator*(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x*c2.x,c1.y*c2.y,c1.a*c2.a);
	}
	//"/"
	friend Trajectory operator/(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x/c2.x,c1.y/c2.y,c1.a/c2.a);
	}
	//"="
	Trajectory operator =(const Trajectory &rx){
		x = rx.x;
		y = rx.y;
		a = rx.a;
		return Trajectory(x,y,a);
	}

    double x;
    double y;
    double a; // angle
};

int main()
{
	VideoCapture cap(0);
	Mat image,prev_image;
	Mat frame_curr,frame_prev;
	oclMat dimage,dprev_image;
	oclMat dframe_curr,dframe_prev;
	
	cap>>prev_image;
	cvtColor(prev_image,frame_prev,CV_BGR2GRAY);
	dprev_image.upload(prev_image);
	dframe_prev.upload(frame_prev);
	
	namedWindow("video",CV_WINDOW_AUTOSIZE);
	namedWindow("stabilized video",CV_WINDOW_AUTOSIZE);

	double a = 0;
	double x = 0;
	double y = 0;

	Trajectory X;//posteriori state estimate
	Trajectory X_;//priori estimate
	Trajectory P;// posteriori estimate error covariance
	Trajectory P_;// priori estimate error covariance
	Trajectory K;//gain
	Trajectory z;//actual measurement
	double pstd = 4e-3;//can be changed
	double cstd = 0.25;//can be changed
	Trajectory Q(pstd,pstd,pstd);// process noise covariance
	Trajectory R(cstd,cstd,cstd);// measurement noise covariance 

	int k=1;
	int vert_border = HORIZONTAL_BORDER_CROP * prev_image.rows / prev_image.cols;
	Mat rigidtransform,last_rigidtransform;

	while(true)
	{
		cap>>image;
		if(image.empty())
			break;

		cvtColor(image,frame_curr,CV_BGR2GRAY);
		dimage.upload(image);
		dframe_curr.upload(frame_curr);

		vector<Point2f> prevpts,currpts;
		vector<Point2f> prev_corner, cur_corner;
		Mat status,err,corners;
		oclMat prev_corners,curr_corners;

		ocl::GoodFeaturesToTrackDetector_OCL(300)(dframe_prev,prev_corners);
		prev_corners.download(corners);
		corners.row(0).copyTo(prevpts);
		calcOpticalFlowPyrLK(frame_prev,frame_curr,prevpts,currpts,status,err);
		
		// weed out bad matches
		for(int i=0; i < status.rows; i++) {
			if((int)status.at<uchar>(i,0)==1) {
				prev_corner.push_back(prevpts[i]);
				cur_corner.push_back(currpts[i]);
			}
		}
		Mat rigidtrans=estimateRigidTransform(prev_corner,cur_corner,false);
		
		if(rigidtrans.empty())
		{
			rigidtrans=last_rigidtransform.clone();
		}
		last_rigidtransform=rigidtrans.clone();

			double dx = rigidtrans.at<double>(0,2);
			double dy = rigidtrans.at<double>(1,2);
			double da = atan2(rigidtrans.at<double>(1,0), rigidtrans.at<double>(0,0));

			x += dx;
			y += dy;
			a += da;

			z = Trajectory(x,y,a);
			if(k==1)
			{
				// intial guesses
				X = Trajectory(0,0,0); //Initial estimate,  set 0
				P =Trajectory(1,1,1); //set error variance,set 1
			}
			else
			{
				//time update prediction
				X_ = X; //X_(k) = X(k-1);
				P_ = P+Q; //P_(k) = P(k-1)+Q;
				// measurement update correction
				K = P_/( P_+R ); //gain;K(k) = P_(k)/( P_(k)+R );
				X = X_+K*(z-X_); //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k)); 
				P = (Trajectory(1,1,1)-K)*P_; //P(k) = (1-K(k))*P_(k);
			}

			double diff_x = X.x - x;
			double diff_y = X.y - y;
			double diff_a = X.a - a;

			dx = dx + diff_x;
			dy = dy + diff_y;
			da = da + diff_a;

			rigidtrans.at<double>(0,0) = cos(da);
			rigidtrans.at<double>(0,1) = -sin(da);
			rigidtrans.at<double>(1,0) = sin(da);
			rigidtrans.at<double>(1,1) = cos(da);

			rigidtrans.at<double>(0,2) = dx;
			rigidtrans.at<double>(1,2) = dy;

			oclMat dfinal_frame;
			Mat final_frame;
			ocl::warpAffine(dprev_image,dfinal_frame,rigidtrans,Size(640,480));
			dfinal_frame.download(final_frame);

			final_frame = final_frame(Range(vert_border, final_frame.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, final_frame.cols-HORIZONTAL_BORDER_CROP));

			// Resize cur2 back to cur size, for better side by side comparison
			resize(final_frame, final_frame, image.size());
		
		//Mat roi=final_frame(Range(50,430),Range(50,590));
		imshow("video",image);
		imshow("stabilized video",final_frame);
		
		waitKey(1);
		prev_image=image.clone();
		frame_curr.copyTo(frame_prev);
		dprev_image.upload(prev_image);
		dframe_prev.upload(frame_prev);
		k++;
	}

	return 0;
}
