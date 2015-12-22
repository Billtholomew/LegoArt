#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <omp.h>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;

#define pi 3.141592653589793238462643383279

Mat loadLookup() {
	CvMLData mldata;
	mldata.read_csv("legoColorLookup.csv");
	const CvMat* tmp = mldata.get_values();
	Mat lookupTable(tmp, true);
	lookupTable = lookupTable(Range::all(), Range(0, 3));
	lookupTable = lookupTable.reshape(3, 0);
	lookupTable.convertTo(lookupTable, CV_8U);
	cvtColor(lookupTable, lookupTable, CV_RGB2Lab);
	lookupTable = lookupTable.reshape(3, 0);
	return lookupTable;
}

double deltaE(Vec3f lab1, Vec3f lab2)
{
	double L1 = lab1[0] * 100 / 255;
	double a1 = lab1[1];
	double b1 = lab1[2];

	double L2 = lab2[0] * 100 / 255;
	double a2 = lab2[1];
	double b2 = lab2[2];

	return sqrt(pow(L1 - L2, 2) + pow(a1 - a2, 2) + pow(b1 - b2, 2));

}

double deltaE2000(Vec3f lab1, Vec3f lab2)
{
	double Lstd = lab1[0];
	double astd = lab1[1];
	double bstd = lab1[2];

	double Lsample = lab2[0];
	double asample = lab2[1];
	double bsample = lab2[2];

	double _kL = 1.0;
	double _kC = 1.0;
	double _kH = 1.0;

	double Cabstd = sqrt(astd*astd + bstd*bstd);
	double Cabsample = sqrt(asample*asample + bsample*bsample);

	double Cabarithmean = (Cabstd + Cabsample) / 2.0;

	double G = 0.5*(1.0 - sqrt(pow(Cabarithmean, 7.0) / (pow(Cabarithmean, 7.0) + pow(25.0, 7.0))));

	double apstd = (1.0 + G)*astd; // aprime in paper
	double apsample = (1.0 + G)*asample; // aprime in paper
	double Cpsample = sqrt(apsample*apsample + bsample*bsample);

	double Cpstd = sqrt(apstd*apstd + bstd*bstd);
	// Compute product of chromas
	double Cpprod = (Cpsample*Cpstd);


	// Ensure hue is between 0 and 2pi
	double hpstd = atan2(bstd, apstd);
	if (hpstd<0) hpstd += 2.0*pi;  // rollover ones that come -ve

	double hpsample = atan2(bsample, apsample);
	if (hpsample<0) hpsample += 2.0*pi;
	if ((fabs(apsample) + fabs(bsample)) == 0.0)  hpsample = 0.0;

	double dL = (Lsample - Lstd);
	double dC = (Cpsample - Cpstd);

	// Computation of hue difference
	double dhp = (hpsample - hpstd);
	if (dhp>pi)  dhp -= 2.0*pi;
	if (dhp<-pi) dhp += 2.0*pi;
	// set chroma difference to zero if the product of chromas is zero
	if (Cpprod == 0.0) dhp = 0.0;

	// Note that the defining equations actually need
	// signed Hue and chroma differences which is different
	// from prior color difference formulae

	double dH = 2.0*sqrt(Cpprod)*sin(dhp / 2.0);
	//%dH2 = 4*Cpprod.*(sin(dhp/2)).^2;

	// weighting functions
	double Lp = (Lsample + Lstd) / 2.0;
	double Cp = (Cpstd + Cpsample) / 2.0;

	// Average Hue Computation
	// This is equivalent to that in the paper but simpler programmatically.
	// Note average hue is computed in radians and converted to degrees only
	// where needed
	double hp = (hpstd + hpsample) / 2.0;
	// Identify positions for which abs hue diff exceeds 180 degrees
	if (fabs(hpstd - hpsample)  > pi) hp -= pi;
	// rollover ones that come -ve
	if (hp<0) hp += 2.0*pi;

	// Check if one of the chroma values is zero, in which case set
	// mean hue to the sum which is equivalent to other value
	if (Cpprod == 0.0) hp = hpsample + hpstd;

	double Lpm502 = (Lp - 50.0)*(Lp - 50.0);;
	double Sl = 1.0 + 0.015*Lpm502 / sqrt(20.0 + Lpm502);
	double Sc = 1.0 + 0.045*Cp;
	double T = 1.0 - 0.17*cos(hp - pi / 6.0) + 0.24*cos(2.0*hp) + 0.32*cos(3.0*hp + pi / 30.0) - 0.20*cos(4.0*hp - 63.0*pi / 180.0);
	double Sh = 1.0 + 0.015*Cp*T;
	double delthetarad = (30.0*pi / 180.0)*exp(-pow(((180.0 / pi*hp - 275.0) / 25.0), 2.0));
	double Rc = 2.0*sqrt(pow(Cp, 7.0) / (pow(Cp, 7.0) + pow(25.0, 7.0)));
	double RT = -sin(2.0*delthetarad)*Rc;

	// The CIE 00 color difference
	return sqrt(pow((dL / Sl), 2.0) + pow((dC / Sc), 2.0) + pow((dH / Sh), 2.0) + RT*(dC / Sc)*(dH / Sh));
}

Vec3b my_translate(Vec3b lab)
{
	return Vec3b(lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128);
}

Vec3f getValue(Mat lookupTable, Vec3f oldColor) 
{
	Vec3f newColor;
	/*
	Mat delta = abs(lookupTable - repeat(Mat(oldColor).reshape(0, 1), lookupTable.rows, 1));
	Mat weights = (Mat_<double>(1, 3) << 0.475/360, 0.2875, 0.2375/256);
	weights = repeat(weights, lookupTable.size[0], 1);
	multiply(delta, weights, delta, 1, CV_32F);
	multiply(delta, delta, delta);
	Mat kernel = Mat::ones(1, 3, CV_32F);
	filter2D(delta, delta, -1, kernel);
	delta = delta(Range().all(), Range(1, 2));
	multiply(delta, delta, delta);
	*/
	float minD = INFINITY;
	float cDelta;
	Vec3b tColor;
	for (int i = 0; i < lookupTable.rows; i++) {
		tColor = lookupTable.at<Vec3b>(i);
		cDelta = deltaE(oldColor, tColor);
		//Vec3b tColor2 = my_translate(tColor);
		//Vec3b oldColor2 = my_translate(oldColor);
		//std::cout << cDelta << "\t" << oldColor2 << "\t" << tColor2 << "\t" << tColor << std::endl;
		//cDelta = delta.at<float>(i);
		if (cDelta <= minD) {
			minD = cDelta;
			newColor = tColor;
		}
	}
	//std::cout << minD << "\t";
	//std::cout << "[" << newColor[0] * 100 / 255 << "," << newColor[1] - 128 << "," << newColor[2] - 128 << "]" << std::endl;
	return newColor;
}

int main(int argc, const char** argv)
{
	Mat lookupTable = loadLookup();
	Mat oim = imread("C:/Users/William/Desktop/LegoArt/pel.png");
	
	oim.convertTo(oim, CV_8U);
	cvtColor(oim, oim, CV_RGB2Lab);

	//Vec3b testColor(50*255/100,73+128,51+128);

	//getValue(lookupTable, testColor);
	
	float f;
	if (oim.rows <= oim.cols)
		f = 128 / float(oim.rows);
	else
		f = 128 / float(oim.cols);

	int Y = oim.rows*f;
	int	X = oim.cols*f;
	Y = Y / 1.2;
	resize(oim, oim, Size(X, Y));

	Mat nim;
	oim.copyTo(nim);

	for (int r = 0; r < oim.rows; r++) {
		for (int c = 0; c < oim.cols; c++) {
			if (r == 65 && c == 89)
				std::cout << "BREAK TIME!" << std::endl;
			Vec3b oldColor = oim.at<Vec3b>(Point(c, r));
			Vec3b newColor = getValue(lookupTable, oldColor);
			nim.at<Vec3b>(Point(c, r)) = newColor;
		}
	}

	cvtColor(nim, nim, CV_Lab2RGB);

	nim.convertTo(nim, CV_8U);

	resize(nim, nim, Size(0, 0), 5, 5*1.2, INTER_NEAREST);

	cvNamedWindow("LegoArt", WINDOW_AUTOSIZE);
	IplImage out = nim;
	cvShowImage("LegoArt", &out);
	waitKey(0);

	return 0;
}