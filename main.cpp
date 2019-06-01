#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <cmath>
#include <vector>
#include "Matrix.h"

using namespace std;

#ifndef PIXEL
	#define PIXEL(img,x,y) (((uchar *)((img)->imageData + (y)*(img)->widthStep))[(x)])
	#define FPIXEL(img,x,y) (((float *)((img)->imageData + (y)*(img)->widthStep))[(x)])

	#define PIXELB(img,x,y) (((uchar *)((img)->imageData + (y)*(img)->widthStep))[(x)*(img)->nChannels + 0])
	#define PIXELG(img,x,y) (((uchar *)((img)->imageData + (y)*(img)->widthStep))[(x)*(img)->nChannels + 1])
	#define PIXELR(img,x,y) (((uchar *)((img)->imageData + (y)*(img)->widthStep))[(x)*(img)->nChannels + 2])
#endif

void threshold(IplImage *img, int value){
  int x,y;
  for (y=0;y<img->height;y++){
    for (x=0;x<img->width;x++){
      if (PIXEL(img,x,y) < value){
	PIXEL(img,x,y) = 255;
      }else{
	PIXEL(img,x,y) = 0;
      }
    }
  }
}

void edgesSeg(IplImage *src, IplImage *des){
  cvSet(des, cvScalar(0));
  int x,y;
  for (y=1;y<src->height;y++){
    for (x=0;x<src->width;x++){
      if (PIXEL(src,x,y) != PIXEL(src,x,y-1)){
	PIXEL(des,x,y) = 255;
      }
    }
  }
  for (y=0;y<src->height;y++){
    for (x=1;x<src->width;x++){
      if (PIXEL(src,x,y) != PIXEL(src,x-1,y)){
	PIXEL(des,x,y) = 255;
      }
    }
  }
}

void get_contour_at(IplImage *src, IplImage *des, int x, int y){
  if (PIXEL(src,x,y) != 0) {
    PIXEL(des, x, y) = 255;
    PIXEL(src, x, y) = 0;
    if (x+1 < src->width) get_contour_at(src,des,x+1,y);
    if (x-1 > 0) get_contour_at(src,des,x-1,y);
    if (y+1 < src->height) get_contour_at(src,des,x,y+1);
    if (y-1 > 0) get_contour_at(src,des,x,y-1);
    if (x+1 < src->width && y+1 < src->height)
      get_contour_at(src,des,x+1,y+1);
    if (x-1 > 0 && y-1 > 0)
      get_contour_at(src,des,x-1,y-1);
    if (x+1 < src->width && y-1 > 0)
      get_contour_at(src,des,x+1,y-1);
    if (x-1 > 0 && y+1 < src->height)
      get_contour_at(src,des,x-1,y+1);
  }
}

int conture_size_recursive(IplImage *src, int x, int y){
	int sum = 0;
	if (x>src->width-1 || x<0 || y>src->height || y<0) return 0;
	if (PIXEL(src,x,y) != 0){
		PIXEL(src,x,y) = 0;
		sum += 1;
		sum += conture_size_recursive(src, x-1, y);
		sum += conture_size_recursive(src, x, y-1);
		sum += conture_size_recursive(src, x+1, y);
		sum += conture_size_recursive(src, x, y+1);
	}
	return sum;
}

int conture_size(IplImage *src, IplImage *temp, int x, int y){
	cvCopyImage(src,temp);
	return conture_size_recursive(temp, x, y);
}

int get_contour(IplImage *src, IplImage *des){
  cvSet(des, cvScalar(0));
  //IplImage *temp = cvCloneImage(src);
  int x,y,size;
  for (y=0;y<src->height;y++){
    for (x=0;x<src->width;x++){
      if (PIXEL(src,x,y) != 0){
      	//size = conture_size(src, temp, x, y);
      	//if (size > 100){
			get_contour_at(src,des,x,y);
			//cvReleaseImage(&temp);
			return 1;
		//}
      }
    }
  }
  //cvReleaseImage(&temp);
  return 0;
}

class MyLine{
public:
  float rho;
  float theta;
};

class Line{
public:
  CvPoint a,b;
  float rho, theta;
};

void transformImage(IplImage *src, IplImage *des, CvPoint *points){
    CvMat *input_mat = cvCreateMat(4,2,CV_32FC1);
    CvMat *output_mat = cvCreateMat(4,2,CV_32FC1);
    CvMat *H = cvCreateMat(3,3,CV_32FC1);
    
    cvmSet(output_mat, 0, 0, 0);
    cvmSet(output_mat, 0, 1, 0);
    cvmSet(output_mat, 1, 0, des->width);
    cvmSet(output_mat, 1, 1, 0);
    cvmSet(output_mat, 2, 0, des->width);
    cvmSet(output_mat, 2, 1, des->height);
    cvmSet(output_mat, 3, 0, 0);
    cvmSet(output_mat, 3, 1, des->height);

    cvmSet(input_mat, 0, 0, points[0].x);
    cvmSet(input_mat, 0, 1, points[0].y);
    cvmSet(input_mat, 1, 0, points[1].x);
    cvmSet(input_mat, 1, 1, points[1].y);
    cvmSet(input_mat, 2, 0, points[2].x);
    cvmSet(input_mat, 2, 1, points[2].y);
    cvmSet(input_mat, 3, 0, points[3].x);
    cvmSet(input_mat, 3, 1, points[3].y);

    cvFindHomography(input_mat, output_mat, H);
    cvWarpPerspective(src, des, H);
    
    cvReleaseMat(&input_mat);
    cvReleaseMat(&output_mat);
    cvReleaseMat(&H);
}

float getError(IplImage *temp, IplImage *obj){
	float sum = 0.0f;
	int x,y;
	float count = 0.0f;
	for (y=0;y<temp->height;y++){
		for (x=0;x<temp->width;x++){
			sum += (float)(((int)PIXEL(obj,x,y)-(int)PIXEL(temp,x,y))*((int)PIXEL(obj,x,y)-(int)PIXEL(temp,x,y)));
			count += 1.0f;
		}
	}
	return sum/count;
}

void filterImage(IplImage *src){
	int x,y;
	cvSmooth(src,src,CV_MEDIAN,5);
	int minVal = 255;
	int maxVal = 0;
	for (y=0;y<src->height;y++){
		for (x=0;x<src->width;x++){
			if (PIXEL(src,x,y) > maxVal) maxVal = PIXEL(src,x,y);
			if (PIXEL(src,x,y) < minVal) minVal = PIXEL(src,x,y);
		}
	}
	float m = 255.0f/((float)(maxVal - minVal));
	for (y=0;y<src->height;y++){
		for (x=0;x<src->width;x++){
			PIXEL(src,x,y) = ((int)(((float)(PIXEL(src,x,y)-minVal))*m));
		}
	}
}

class Rectangle{
public:
	CvPoint a,b,c,d;
	float error;
	bool findRotationByTemplate(const char *filename, IplImage *src){
		IplImage *temp = cvLoadImage(filename, 0);
		IplImage *obj = cvCloneImage(temp);
		CvPoint points[4], min_p[4], ack_p[4];
		points[0] = a; points[1] = b; points[2] = c; points[3] = d;
		ack_p[0] = a; ack_p[1] = b; ack_p[2] = c; ack_p[3] = d;
		min_p[0] = a; min_p[0] = b; min_p[0] = c; min_p[0] = d;
		transformImage(src, obj, ack_p);
		filterImage(obj);
		float min_err = getError(temp, obj), err;
		int i,j,k,l;
		for (i=0;i<4;i++){
			for (j=0;j<4;j++){
				if (i==j) continue;
				for (k=0;k<4;k++){
					if (i==k || j == k) continue;
					for (l=0;l<4;l++){
						if (l==i||l==j||l==k) continue;
						//printf("%d%d%d%d\n",i,j,k,l);
						ack_p[0] = points[i];
						ack_p[1] = points[j];
						ack_p[2] = points[k];
						ack_p[3] = points[l];
						transformImage(src, obj, ack_p);
						filterImage(obj);
						err = getError(temp, obj);
						if (err < min_err) {
							min_p[0] = ack_p[0];
							min_p[1] = ack_p[1];
							min_p[2] = ack_p[2];
							min_p[3] = ack_p[3];
							min_err = err;
						}
					}
				}
			}
		}
		transformImage(src, obj, min_p);
		filterImage(obj);
		error = getError(temp, obj);
		printf("Error: %f\n", error);
		//cvNamedWindow("transform");
		//cvShowImage("transform", obj);
		//cvWaitKey(0);
		a = min_p[0]; b = min_p[1]; c = min_p[2]; d = min_p[3];
	}
	
	void draw(IplImage *img){
		cvLine(img, a,b, cvScalar(0,255,0),2);
		cvLine(img, b,c, cvScalar(0,255,0),2);
		cvLine(img, c,d, cvScalar(0,255,0),2);
		cvLine(img, d,a, cvScalar(0,255,0),2);
		cvCircle(img, a, 5, cvScalar(255,0,0), -1);
		cvCircle(img, b, 5, cvScalar(255,255,0), -1);
		cvCircle(img, c, 5, cvScalar(0,255,255), -1);
		cvCircle(img, d, 5, cvScalar(0,0,255), -1);
	}
};

bool intersectP(CvPoint o1, CvPoint p1, CvPoint o2, CvPoint p2,
                      CvPoint &r)
{
    CvPoint x;
    x.x = o2.x - o1.x;
    x.y = o2.y - o1.y;

    CvPoint d1;
    d1.x = p1.x - o1.x;
    d1.y = p1.y - o1.y;
    CvPoint d2;
    d2.x = p2.x - o2.x;
    d2.y = p2.y - o2.y;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r.x = o1.x + d1.x * t1;
    r.y = o1.y + d1.y * t1;
    return true;
}

bool intersect(Line A, Line B, CvPoint &r){
	return intersectP(A.a,A.b,B.a,B.b,r);
}

void analyze(IplImage *img, Line *out_lines, int *size){
	    
	    CvMemStorage* storage = cvCreateMemStorage(0);
	    CvSeq* lines = 0;
	    
	    //filterImage(img);
	    cvSmooth(img, img, CV_GAUSSIAN, 5);
	    cvSaveImage("normalized.jpg", img);
	    
	    lines = cvHoughLines2( img,
                               storage,
                               CV_HOUGH_STANDARD,
                               1,
                               CV_PI/180,
                               100,
                               0,
                               0 );
	    
	    int i, out_size=0;
	    //printf("lines: %d\n", lines->total);
	    if (lines->total > 0){
	      for( i = 0; i < MIN(lines->total,100); i++ ){
		  float* line = (float*)cvGetSeqElem(lines,i);
		  
		  MyLine ml;
		  
		  ml.rho = line[0];
		  ml.theta = line[1];
		  
		  //printf("line: rho:%f, theta: %f\n", ml.rho, ml.theta);
		  CvPoint pt1, pt2;
		  double a = cos(ml.theta), b = sin(ml.theta);
		  double x0 = a*ml.rho, y0 = b*ml.rho;
		  pt1.x = cvRound(x0 + 1000*(-b));
		  pt1.y = cvRound(y0 + 1000*(a));
		  pt2.x = cvRound(x0 - 1000*(-b));
		  pt2.y = cvRound(y0 - 1000*(a));
		  cvLine( img, pt1, pt2, cvScalar(128), 1, 8);
		  out_lines[out_size].a.x = pt1.x;
		  out_lines[out_size].a.y = pt1.y;
		  out_lines[out_size].b.x = pt2.x;
		  out_lines[out_size].b.y = pt2.y;
		  out_lines[out_size].rho = ml.rho;
		  out_lines[out_size].theta = ml.theta;
		  out_size++;
	      }
	    }
	    *size = out_size;
}

int pointDistance(int x, int y, int x2, int y2){
	return (int)sqrt((float)(
		(x-x2)*(x-x2)+
		(y-y2)*(y-y2)
	));
}

void add_points(vector<CvPoint> &points, CvPoint &p, IplImage *img){
	int i;
	//printf("pridavam bod: [%d,%d]\n", p.x, p.y);
	//fflush(stdout);
	if (p.x < 0 || p.x >= img->width) return;
	if (p.y < 0 || p.y >= img->height) return;
	bool isPresent = false;
	for (i=0;i<points.size();i++){
		if (50 > pointDistance(p.x, p.y, points[i].x, points[i].y)) isPresent = true;
	}
	if (!isPresent){
		printf("point: [%d,%d]\n", p.x, p.y);
		fflush(stdout);
		points.push_back(p);
	}
}

void reduce(IplImage *out, Line *lines, Line *out_lines, int size, int *out_size, float dist = 20.0f){
  int i, j, out_count=0;
  out_lines[0].a.x = lines[0].a.x;
  out_lines[0].b.x = lines[0].b.x;
  out_lines[0].a.y = lines[0].a.y;
  out_lines[0].b.y = lines[0].b.y;
  out_lines[0].rho = lines[0].rho;
  out_lines[0].theta = lines[0].theta;
  out_count++;
  //printf("line: [%d,%d]->[%d,%d]\n", 
  //  lines[0].a.x, lines[0].a.y,
  //  lines[0].b.x, lines[0].b.y
  //);
  cvLine(out, out_lines[0].a, out_lines[0].b, cvScalar(255), 1);
  for (i=1;i<size;i++){
    int isNew = 1;
    for (j=0;j<out_count;j++){
      if (i!=j){
	float distance = sqrt(
	  (lines[i].a.x - out_lines[j].a.x)*(lines[i].a.x - out_lines[j].a.x)+
	  (lines[i].a.y - out_lines[j].a.y)*(lines[i].a.y - out_lines[j].a.y)+
	  (lines[i].b.x - out_lines[j].b.x)*(lines[i].b.x - out_lines[j].b.x)+
	  (lines[i].b.y - out_lines[j].b.y)*(lines[i].b.y - out_lines[j].b.y)
	);
	float distance2 = sqrt(
	  (lines[i].a.x - out_lines[j].b.x)*(lines[i].a.x - out_lines[j].b.x)+
	  (lines[i].a.y - out_lines[j].b.y)*(lines[i].a.y - out_lines[j].b.y)+
	  (lines[i].b.x - out_lines[j].a.x)*(lines[i].b.x - out_lines[j].a.x)+
	  (lines[i].b.y - out_lines[j].a.y)*(lines[i].b.y - out_lines[j].a.y)
	);
	if (distance2 < distance) distance = distance2;
	//printf("Dist: %f < %f\n", distance, (float)dist);
	if (distance < (float)150) isNew = 0;
      }
    }
    if (isNew){
      out_lines[out_count].a.x = lines[i].a.x;
      out_lines[out_count].b.x = lines[i].b.x;
      out_lines[out_count].a.y = lines[i].a.y;
      out_lines[out_count].b.y = lines[i].b.y;
      cvLine(out, out_lines[out_count].a, out_lines[out_count].b, cvScalar(0), 1);
      out_count++;
    }
  }
  *out_size = out_count;
}

void transformR3(Matrix *point, double alpha, double beta, double gamma, double x, double y, double z){
	Matrix *Rx = HomogenousTrans::createXRotationMatrix(alpha);
    Matrix *Ry = HomogenousTrans::createYRotationMatrix(beta);
    Matrix *Rz = HomogenousTrans::createZRotationMatrix(gamma);
    Matrix *Txyz = HomogenousTrans::createTranslationMatrix(x, y, z);
    Matrix *Final = new Matrix(4,4);
    Matrix *R1 = new Matrix(4,4);
    Matrix *R2 = new Matrix(4,4);
    Matrix *v = new Matrix(1,4);
    
    Rx->multiplyBy(Ry, R1);
    R1->multiplyBy(Rz, R2);
    R2->multiplyBy(Txyz, Final);
    point->multiplyBy(Final, v);
    
    MAT_EL(point,0,0) = MAT_EL(v,0,0);
    MAT_EL(point,0,1) = MAT_EL(v,0,1);
    MAT_EL(point,0,2) = MAT_EL(v,0,2);
    MAT_EL(point,0,3) = MAT_EL(v,0,3);
    
    delete(Rx);
    delete(Ry);
    delete(Rz);
    delete(Txyz);
    delete(Final);
    delete(R1);
    delete(R2);
    delete(v);
}

int main(int argc, char **argv){
	const char *filename = "foto/image4.jpg";

	  IplImage *img = cvLoadImage(filename, 0);
	  IplImage *imgColor = cvLoadImage(filename, 1);
	  
	  cvSmooth(img, img, CV_MEDIAN, 5);
	  
	  //filterImage(img);
	  IplImage *edges = cvCloneImage(img);
	  IplImage *countour = cvCloneImage(img);
	  IplImage *thres = cvCloneImage(img);
	  
	  threshold(thres, 70);
	  edgesSeg(thres, edges);
	  cvSaveImage("thres.jpg",thres);
  	  cvSaveImage("edges.jpg",edges);
	  
	  //cvShowImage("img", img);
	  while (get_contour(edges, countour)){
	    Line lines[100], out_lines[100];
	    int size, out_size;
	    analyze(countour, lines, &size);
	    if (size <= 0) continue;
	    //reduce(contour, lines, out_lines, size, &out_size);
	    reduce(img, lines, out_lines, size, &out_size);
	    printf("%d lines found. After reduction: %d\n", size, out_size);
	    int i,j;
	    if (out_size!=4) continue;
	    vector<CvPoint> points;
	    for (i=0;i<out_size;i++){
	      //cvLine(img, out_lines[i].a, out_lines[i].b, cvScalar(255), 1);
              for (j=0;j<out_size;j++){
		if (i==j) continue;
                CvPoint its;
                intersect(out_lines[i], out_lines[j], its);
		add_points(points, its, img);
		//cvCircle(img, its, 5, cvScalar(255));
              }
	    }
            
            printf("points: %d\n", (int)points.size());
            if ((int)points.size()!=4) continue;
	    Rectangle r;
            r.a = points[0]; r.b = points[1]; r.c = points[2]; r.d = points[3];
	    r.findRotationByTemplate("template.png", img);
	    	// ***************************************** Vratit zpatky *****************************
            if (r.error > 10000) continue;
            r.draw(imgColor);
	    //cvShowImage("contour", countour);
	    //cvWaitKey(0);
	  }
	  
	  //cvNamedWindow("img");
	  //cvShowImage("img", img);
	  
	  //cvWaitKey(0);
      
      /********************************************************************************************/
      
      double test[4] = {0,0,0};
      Matrix *test_point = new Matrix(1,4,test);
      test_point->print();
      transformR3(test_point, 0, 0, 0, 1, 2, 3);
      test_point->print();
      
      delete(test_point);
      
      /********************************************************************************************/
      
      /*double scale = min(img->width/2, img->height/2);
      Matrix *Proj = HomogenousTrans::createProjectionMatrix(50);
      Matrix *trans1 = HomogenousTrans::createTranslationMatrix(0, 0, 5);
      Matrix *Rx = HomogenousTrans::createXRotationMatrix(1.0);
      Matrix *Ry = HomogenousTrans::createXRotationMatrix(1.0);
      Matrix *Rz = HomogenousTrans::createXRotationMatrix(1.0);
      Matrix *T1 = new Matrix(4,4);

      Matrix *TR1 = new Matrix(4,4);
      Matrix *TR = new Matrix(4,4);
      
      Proj->multiplyBy(trans1,T1);
      //Scale->multiplyBy(Proj,T);
      
      double p1[4] = {-0.5,-0.5,-0.0,1};
      double p2[4] = {0.5,-0.5,-0.0,1};
      double p3[4] = {0.5,0.5,-0.0,1};
      double p4[4] = {-0.5,0.5,-0.0,1};

      Matrix *p1_in = new Matrix(1,4,p1);
      Matrix *p2_in = new Matrix(1,4,p2);
      Matrix *p3_in = new Matrix(1,4,p3);
      Matrix *p4_in = new Matrix(1,4,p4);
      
      printf("Input:\n");
      p1_in->print();
      p2_in->print();
      p3_in->print();
      p4_in->print();
      
      Matrix *p1_out = new Matrix(1,4);
      Matrix *p2_out = new Matrix(1,4);
      Matrix *p3_out = new Matrix(1,4);
      Matrix *p4_out = new Matrix(1,4);
      
      T1->multiplyBy(p1_in, p1_out);
      T1->multiplyBy(p2_in, p2_out);
      T1->multiplyBy(p3_in, p3_out);
      T1->multiplyBy(p4_in, p4_out);
      
      double w1 = MAT_EL(p1_out,0,3);
      double w2 = MAT_EL(p2_out,0,3);
      double w3 = MAT_EL(p3_out,0,3);
      double w4 = MAT_EL(p4_out,0,3);
      
      p1_out->multiplyBy(w1, p1_out); MAT_EL(p1_out, 0, 3) = 1;
      p2_out->multiplyBy(w2, p2_out); MAT_EL(p2_out, 0, 3) = 1;
      p3_out->multiplyBy(w3, p3_out); MAT_EL(p3_out, 0, 3) = 1;
      p4_out->multiplyBy(w4, p4_out); MAT_EL(p4_out, 0, 3) = 1;
      
      // po projekci vykreslit do prostoru obrazu
      Matrix *T = new Matrix(4,4);
      Matrix *Tran = HomogenousTrans::createTranslationMatrix(img->width/2, img->height/2, 0);
      Matrix *Scale = HomogenousTrans::createScaleMatrix(scale,scale,scale);
      Tran->multiplyBy(Scale,T);
            
      T->multiplyBy(p1_out, p1_in);
      T->multiplyBy(p2_out, p2_in);
      T->multiplyBy(p3_out, p3_in);
      T->multiplyBy(p4_out, p4_in);
      
      printf("Output:\n");
      p1_in->print();
      p2_in->print();
      p3_in->print();
      p4_in->print();
      
      cvLine(imgColor, cvPoint(MAT_EL(p1_in,0,0),MAT_EL(p1_in,0,1)), cvPoint(MAT_EL(p2_in,0,0),MAT_EL(p2_in,0,1)), cvScalar(0,0,255), 2);
      cvLine(imgColor, cvPoint(MAT_EL(p2_in,0,0),MAT_EL(p2_in,0,1)), cvPoint(MAT_EL(p3_in,0,0),MAT_EL(p3_in,0,1)), cvScalar(0,0,255), 2);
      cvLine(imgColor, cvPoint(MAT_EL(p3_in,0,0),MAT_EL(p3_in,0,1)), cvPoint(MAT_EL(p4_in,0,0),MAT_EL(p4_in,0,1)), cvScalar(0,0,255), 2);
      cvLine(imgColor, cvPoint(MAT_EL(p4_in,0,0),MAT_EL(p4_in,0,1)), cvPoint(MAT_EL(p1_in,0,0),MAT_EL(p1_in,0,1)), cvScalar(0,0,255), 2);
      
      printf("Projekce:\n");
      Proj->print();
      printf("Translace:\n");
      Tran->print();
      printf("Scaleing:\n");
      Scale->print();
      printf("Vysledna transformace:\n");
      T->print();
      
      delete(Proj);
      delete(Tran);
      delete(Scale);
      delete(TR);
      delete(TR1);
      delete(T1);
      delete(T);
      delete(Rx);
      delete(Ry);
      delete(Rz);      
      delete(p1_in);
      delete(p2_in);
      delete(p3_in);
      delete(p4_in);
      delete(p1_out);
      delete(p2_out);
      delete(p3_out);
      delete(p4_out);*/
      /********************************************************************************************/
      
      cvSaveImage("result2.jpg", imgColor);
      cvSaveImage("result2_out.jpg", img);
      
	  cvReleaseImage(&img);
	  cvReleaseImage(&edges);
	  cvReleaseImage(&countour);
	  cvReleaseImage(&thres);
}
