//
//  metric_rectification.cpp
//  metric-rectification
//
//  Created by Patrick Skinner on 20/01/20.
//  Copyright © 2020 Patrick Skinner. All rights reserved.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>

using namespace cv;
using namespace std;

// Cotangent function
float cotan(float i) {
    if( i < CV_PI/2 + 0.001 && i > CV_PI/2 - 0.001){
        return 0;
    }
    return(1 / tan(i));
}

// Find intersection point of two circles, code from jupdike on Github
Point2f intersectTwoCircles(float x1, float y1, float r1, float x2, float y2, float r2) {
    float centerdx = x1 - x2;
    float centerdy = y1 - y2;
    float R = sqrt(centerdx * centerdx + centerdy * centerdy);
    if (!(abs(r1 - r2) <= R && R <= r1 + r2)) { // no intersection
        return Point2f(0,0);
    }
    
    float R2 = R*R;
    float R4 = R2*R2;
    float a = (r1*r1 - r2*r2) / (2 * R2);
    float r2r2 = (r1*r1 - r2*r2);
    float c =sqrt(2 * (r1*r1 + r2*r2) / R2 - (r2r2 * r2r2) / R4 - 1);
    
    float fx = (x1+x2) / 2 + a * (x2 - x1);
    float gx = c * (y2 - y1) / 2;
    float ix1 = fx + gx;
    float ix2 = fx - gx;
    
    float fy = (y1+y2) / 2 + a * (y2 - y1);
    float gy = c * (x1 - x2) / 2;
    float iy1 = fy + gy;
    float iy2 = fy - gy;
    
    // note if gy == 0 and gx == 0 then the circles are tangent and there is only one solution
    // but that one solution will just be duplicated as the code is currently written

    return Point2f(ix2, iy2);
}

int main(int argc, const char * argv[]) {

    Mat src = Mat(500, 500, CV_8UC1, Scalar(255,255,255));
    Mat srcWarped = Mat(500, 500, CV_8UC1, Scalar(255,255,255));
    Mat srcRect = Mat(500, 500, CV_8UC1, Scalar(255,255,255));
    Mat srcMetr = Mat(500, 1000, CV_8UC1, Scalar(255,255,255));
    
    // ordering : vert, vert, horiz, horiz, middle vert
    vector<Vec4f> lines {Vec4f(100,100,100,400), Vec4f(400, 100, 400, 400), Vec4f(100,100, 400, 100), Vec4f(100,400,400,400), Vec4f(300,100,300,400)};
        
    for(int i = 0; i < lines.size(); i++){
        Vec4f l = lines[i];
        line( src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 2, 5 );
    }
    
    vector<Point2f> corners {Point2f(100,100), Point2f(100,400), Point2f(400,400), Point2f(400,100)};
    vector<Point2f> warpedCorners {Point2f(140,120), Point2f(100,400), Point2f(380,345), Point2f(400,100)};
    
    Mat warp = getPerspectiveTransform(corners, warpedCorners);
    
    vector<Point2f> in;
    vector<Point2f> out;
    for( int i = 0; i < lines.size(); i++){
        in.push_back( Point2f( lines[i][0], lines[i][1]) );
        in.push_back( Point2f( lines[i][2], lines[i][3]) );
    }
    
    perspectiveTransform( in , out, warp);
    
    vector<Vec4f> warpedLines;
    for( int i = 0; i < out.size(); i += 2){
        line( srcWarped, Point(out[i].x, out[i].y), Point(out[i+1].x, out[i+1].y), Scalar(0,0,255), 2, 0);
        warpedLines.push_back( Vec4f{out[i].x, out[i].y, out[i+1].x, out[i+1].y });
    }
    
    // Convert all warped lines to homogenous form
    vector<Vec3f> hmgLines;
    for(int i = 0; i < warpedLines.size(); i++){
        hmgLines.push_back( Vec3f(warpedLines[i][0], warpedLines[i][1], 1).cross( Vec3f(warpedLines[i][2], warpedLines[i][3], 1 ) ) );
    }
    
    // Find intersections of parallel line pairs to finding vanishing points
    Vec3f intersection1 = hmgLines[0].cross(hmgLines[1]);
    Vec3f intersection2 = hmgLines[2].cross(hmgLines[3]);
    
    // Find the line at infinity between the vanishing points.
    Vec3f lineL = intersection1.cross(intersection2);
    
    line( src , Point(intersection1[0], intersection1[1]), Point(intersection2[0], intersection2[1]), Scalar(0,0,255), 2, 5 );
    
    Mat affineTransform = Mat(3, 3, CV_32F, 0.0);
    affineTransform.at<float>(0,0) = 1.0;
    affineTransform.at<float>(1,1) = 1.0;
    affineTransform.at<float>(2,0) = lineL[0] / lineL[2];
    affineTransform.at<float>(2,1) = lineL[1] / lineL[2];
    affineTransform.at<float>(2,2) = lineL[2] / lineL[2];
    
    cout << "Affine Matrix: \n" << affineTransform << endl;
    
    // Perform affine rectification
    vector<Point2f> rect;
    perspectiveTransform( out , rect, affineTransform);
    
    for( int i = 0; i < rect.size(); i += 2){
        line( srcRect, Point(rect[i].x, rect[i].y), Point(rect[i+1].x, rect[i+1].y), Scalar(0,0,255), 2, 0);
    }
    
    Mat constraints = Mat(500, 500, CV_8UC1, Scalar(255,255,255));
    
    // Generate a constraint circle from a known angle between lines
    float a = (-hmgLines[0][1]) / hmgLines[0][0]; // vert
    float b = (-hmgLines[2][1]) / hmgLines[2][0]; // hori
    float theta = CV_PI / 2;
    
    float ca = (a+b)/2 ;
    float cb = ((a-b)/2) * cotan(theta);
    float r = abs( (a-b) / (2 * sin(theta)) );
    
    circle(constraints, Point(ca+250,cb+250), r, Scalar(0,255,0));

    
    // Generate a constraint circle from a known angle between lines
    /*
    a = (-hmgLines[1][1]) / hmgLines[1][0]; // vert
    b = (-hmgLines[3][1]) / hmgLines[3][0]; // hori
    float ca2 = (a+b)/2 ;
    float cb2 = ((a-b)/2) * cotan(theta);
    float r2 = abs( (a-b) / (2 * sin(theta)) );
    
    circle(constraints, Point(ca2+250 ,cb2 + 250), r2, Scalar(0,255,0));
    */
    
    
    //Generate a constraint circle from known length ratio between two non parallel lines
    float dx1 = warpedLines[1][0] - warpedLines[1][2];
    float dy1 = warpedLines[1][1] - warpedLines[1][3];
    
    float dx2 = warpedLines[3][0] - warpedLines[3][2];
    float dy2 = warpedLines[3][1] - warpedLines[3][3];
    
    float ratio = 1; // Horizontal and vertical lines are of equal length
    
    float ca3 = ((dx1*dy1) - ((ratio*ratio)*dx2*dy2)) / ((dy1*dy1)-((ratio*ratio)*(dy2*dy2)));
    float cb3 = 0;
    float r3 = abs( (ratio*(dx2*dy1-dx1*dy2)) / ((dy1*dy1)-(ratio*ratio)*(dy2*dy2)) );

    circle(constraints, Point(ca3+250 ,cb3 + 250), r3, Scalar(0,255,0));

    // Draw axes
    line( constraints, Point(0, 250), Point(500, 250), Scalar(0,0,128));
    line( constraints, Point(250, 0), Point(250, 500), Scalar(0,0,128));
    
    // Find where constraint circles intersect
    Point2f inter = intersectTwoCircles(ca, cb, r, ca3, cb3, r3);
    
    cout << "Contraint Intersection: " << inter << endl;
    //circle(constraints, Point(inter.x, inter.y), 5, Scalar(0,0,0), FILLED);
    
    Mat metricTransform = Mat(3, 3, CV_32F, 0.0);
    metricTransform.at<float>(0,0) = 1 / inter.y;
    metricTransform.at<float>(0,1) = -(inter.x/inter.y);
    metricTransform.at<float>(1,1) = 1.0;
    metricTransform.at<float>(2,2) = 1.0;

    cout << "Metric Matrix: \n" << metricTransform << endl;
    
    //Perform metric rectification
    vector<Point2f> metr;
    perspectiveTransform( rect , metr, metricTransform);
    
    // Choose a line we want to be vertical in our image and rotate the line set accordingly.
    Vec4f vertLine = Vec4f(metr[0].x, metr[0].y, metr[1].x, metr[1].y);
    double angle = 90 - getAngle(vertLine);
    Mat rotationMat = getRotationMatrix2D(Point(250,250), angle, 1);
    for( int i = 0; i < metr.size(); i++){
        Mat point;
        Mat(metr[i], CV_64F).convertTo(point, CV_64F);
        point = point.t() * rotationMat;
        metr[i].x = point.at<double>(0);
        metr[i].y = point.at<double>(1);
    }
    
    for( int i = 0; i < metr.size(); i += 2){
        line( srcMetr, Point(metr[i].x, metr[i].y), Point(metr[i+1].x, metr[i+1].y), Scalar(0,0,255), 2, 0);
    }
    
    imshow("Original", src); // Original unwarped image.
    imshow("Unrectified", srcWarped); // Perspective warped image.
    imshow("Affine Rectified", srcRect); // Affine rectification of warped image.
    imshow("Metric Rectified", srcMetr); // Metric rectification of warped image.
    imshow("Constraints", constraints); // Visualisation of constraint circles.
    waitKey();
    
    return 0;
}
