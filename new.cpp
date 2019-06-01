#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <cmath>
#include <vector>
#include <stack>
#include "genalg.h"

#ifndef PIXEL
#define PIXEL(img,x,y) (((uchar *)((img)->imageData + (y)*img->widthStep))[(x)])
#endif

float dilate(IplImage *src, IplImage *out, int r=0){
	float max = 0.0;
	int x,y,i,j;
	for (y=2;y<out->height-2;y++){
		for (x=2;x<out->width-2;x++){
				if (PIXEL(src,x,y)!=0){
					if (r==0){
						PIXEL(out,x,y) = 255;
						PIXEL(out,x+1,y) = 255;
						PIXEL(out,x-1,y) = 255;
						PIXEL(out,x,y+1) = 255;
						PIXEL(out,x,y-1) = 255;
						PIXEL(out,x+1,y+1) = 255;
						PIXEL(out,x-1,y-1) = 255;
						PIXEL(out,x-1,y+1) = 255;
						PIXEL(out,x+1,y-1) = 255;
					}else{
						cvCircle(out,cvPoint(x,y),r,cvScalar(255),-1);
					}
				}
			
		}
	}
}

struct MyPoint{
	int x,y;
};

struct MyRect{
	MyPoint points[4];
};

inline void getCorneresAndEdges(IplImage *img, IplImage *edges, std::vector<MyPoint> &points){
	printf("getting corners and edgesn\n");
	fflush(stdout);
	/*********************** GOOD FEATURES TO TRACK **************************/
	IplImage *eig = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
	IplImage *temp = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_32F, 1);
	
	CvPoint2D32f corners[10000]; // tady budou vystupni body po vypoctu
	const float quality = 0.05;
	const int min_dist = 10;
	int pocet_bodiku;
	
	cvGoodFeaturesToTrack(img, // vstupni sedobily obraz
				eig,            // output
				temp,			  // temp obraz
				corners,			  // odkaz na pole, kam se maji ulozit bodiky
				&pocet_bodiku,		  // odkaz na promennou kde se ma ulozit vysledny pocet nalezenych bodu
				quality,		  // nastaveni kvality detekce (jak kvalitne musi byt bod detekovan aby byl dan do vystupu)
				min_dist,		  // minimalni odstup od detekovanych bodu
				NULL,				  // cert vi ... proste null
				3,		  // velikost bloku pro vypocet vlastnich cisel
				false);		  // ma se pri vypoctu pouzit harris detektor?? ne!
	
	/*************************************************************************/
	
	IplImage *out = cvCloneImage(img);
	printf("running canny\n");
	fflush(stdout);
	cvCanny(img,out,80,150);
	cvSaveImage("canny.png", out);
	cvSet(edges, cvScalar(0));
	printf("dilatation\n");
	fflush(stdout);
	dilate(out, edges, 2);
	cvSaveImage("edges.png", edges);
	
	IplImage *test = cvCloneImage(edges);
	
	int i;
	for (i=0;i<pocet_bodiku;i++){
		MyPoint p;
		p.x = (int)corners[i].x;
		p.y = (int)corners[i].y;
		points.push_back(p);
		// test
		cvCircle(test, cvPoint(p.x,p.y), 3, cvScalar(128), -1);
		// end test
	}
	
	cvSaveImage("test.png", test);
	cvReleaseImage(&test);
	
	cvReleaseImage(&eig);
	cvReleaseImage(&temp);
	cvReleaseImage(&out);
}

void recursiveEdgeMover(IplImage *src, IplImage *dest, int x, int y){
	if (x<0||y<0||x>src->width-1||y>src->height-1) return;
	if (PIXEL(src,x,y) != 255) return;
	PIXEL(dest, x, y) = 255;
	PIXEL(src, x, y) = 0;
	recursiveEdgeMover(src, dest, x+1, y);
	recursiveEdgeMover(src, dest, x-1, y);
	recursiveEdgeMover(src, dest, x, y+1);
	recursiveEdgeMover(src, dest, x, y-1);
}

void nonRecursiveEdgeMover(IplImage *src, IplImage *dest, int x, int y){
	//printf("running edge remover at (%d, %d)\n", x, y);
	//fflush(stdout);
	std::stack<MyPoint> toProcess;
	MyPoint np;
	np.x = x;
	np.y = y;
	toProcess.push(np);
	while (toProcess.size() != 0){
		MyPoint ackp = toProcess.top();
		toProcess.pop();
		if (ackp.x<0||ackp.y<0||ackp.x>src->width-1||ackp.y>src->height-1) continue;
		if (PIXEL(src,ackp.x,ackp.y) != 255) continue;
		PIXEL(dest, ackp.x, ackp.y) = 255;
		PIXEL(src, ackp.x, ackp.y) = 0;
		MyPoint next;
		next.x = ackp.x + 1; next.y = ackp.y;
		toProcess.push(next);
		next.x = ackp.x - 1; next.y = ackp.y;
		toProcess.push(next);
		next.x = ackp.x + 1; next.y = ackp.y + 1;
		toProcess.push(next);
		next.x = ackp.x - 1; next.y = ackp.y - 1;
		toProcess.push(next);
		next.x = ackp.x; next.y = ackp.y + 1;
		toProcess.push(next);
		next.x = ackp.x; next.y = ackp.y - 1;
		toProcess.push(next);
		next.x = ackp.x + 1; next.y = ackp.y - 1;
		toProcess.push(next);
		next.x = ackp.x - 1; next.y = ackp.y + 1;
		toProcess.push(next);
	}
}

void getAllPointsFromPoint(IplImage *edges, MyPoint p, std::vector<MyPoint> &allPoints, std::vector<MyPoint> &outpoints){
	//printf("getting all points from point (%d,%d)\n", p.x, p.y);
	//fflush(stdout);
	IplImage *temp = cvCreateImage(cvSize(edges->width, edges->height), IPL_DEPTH_8U, 1);
	cvSet(temp, cvScalar(0));
	//recursiveEdgeMover(edges, temp, p.x, p.y);
	nonRecursiveEdgeMover(edges, temp, p.x, p.y);
	int i, size = allPoints.size();
	for (i=0; i<size; i++){
		if (PIXEL(temp, allPoints[i].x, allPoints[i].y) != 0) outpoints.push_back(allPoints[i]);
	}
	/********************** TESTOVACI *********************************/
	//size = outpoints.size();
	//for (i=0;i<size;i++){
	//	CvPoint p;
	//	p.x = outpoints[i].x;
	//	p.y = outpoints[i].y;
	//	cvCircle(temp, p, 5, cvScalar(127), -1);
	//}
	//cvNamedWindow("object", 0);
	//cvShowImage("object", temp);
	//cvWaitKey(0);
	/******************************************************************/
	cvReleaseImage(&temp);
}

void findAllRects(IplImage *edges, std::vector<MyPoint> &points, std::vector<MyRect> &rects){
	printf("looking for all rects\n");
	fflush(stdout);
	int i, size = points.size();
	for (i=0;i<size;i++){
		std::vector<MyPoint> object_points;
		getAllPointsFromPoint(edges, points[i], points, object_points);
		if (object_points.size() == 4){
			MyRect r;
			r.points[0] = object_points[0];
			r.points[1] = object_points[1];
			r.points[2] = object_points[2];
			r.points[3] = object_points[3];
			rects.push_back(r);
		}
	}
}

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

float SquareErrorSum(IplImage *img, IplImage *img2){
	int x,y;
	float sum = 0.0f;
	for (y=0;y<img->height;y++){
		for (x=0;x<img->width;x++){
			sum += (((float)PIXEL(img,x,y))-((float)PIXEL(img2,x,y)))*(((float)PIXEL(img,x,y))-((float)PIXEL(img2,x,y)));
		}
	}
	return sum;
}

float getError(MyRect *rect, IplImage *img, IplImage *pattern){
	IplImage *data = cvCloneImage(pattern);
	CvPoint points[4];
	points[0] = cvPoint(rect->points[0].x,rect->points[0].y);
	points[1] = cvPoint(rect->points[1].x,rect->points[1].y);
	points[2] = cvPoint(rect->points[2].x,rect->points[2].y);
	points[3] = cvPoint(rect->points[3].x,rect->points[3].y);
	transformImage(img, data, points);
	float err = SquareErrorSum(pattern, data);
	err /= (data->width * data->height);
	cvReleaseImage(&data);
	return err;
}

float getError2(MyRect *rect, IplImage *img, IplImage *pattern){
	IplImage *data = cvCloneImage(pattern);
	/*if (data == NULL || img == NULL){
		printf("getError2: Images problem\n");
		fflush(stdout);
	}else{
		printf("getError2: images OK\n");
		fflush(stdout);
	}*/
	CvPoint points[4];
	points[0] = cvPoint(rect->points[0].x,rect->points[0].y);
	points[1] = cvPoint(rect->points[1].x,rect->points[1].y);
	points[2] = cvPoint(rect->points[2].x,rect->points[2].y);
	points[3] = cvPoint(rect->points[3].x,rect->points[3].y);
	//printf("Transforming\n");
	//fflush(stdout);
	transformImage(img, data, points);
	//printf("Computing SQRS\n");
	//fflush(stdout);
	float minerr = SquareErrorSum(pattern, data);
	//printf("Prepareing for permutations\n");
	//fflush(stdout);
	minerr /= (data->width * data->height);
	float ackerr = minerr;
	int a,b,c,d;
	int an=0,bn=1,cn=2,dn=3;
	//printf("Permuting\n");
	//fflush(stdout);
	for (a=0;a<4;a++){
		for (b=0;b<4;b++){
			if (b==a) continue;
			for (c=0;c<4;c++){
				if (c==a || c==b) continue;
				for (d=0;d<4;d++){
					if (d==a || d==b || d==c) continue;
					//printf("Permutation: %d%d%d%d\n",a,b,c,d);
					//fflush(stdout);
					//printf("%d%d%d%d\n", a,b,c,d);
					points[0] = cvPoint(rect->points[a].x,rect->points[a].y);
					points[1] = cvPoint(rect->points[b].x,rect->points[b].y);
					points[2] = cvPoint(rect->points[c].x,rect->points[c].y);
					points[3] = cvPoint(rect->points[d].x,rect->points[d].y);
					transformImage(img, data, points);
					ackerr = SquareErrorSum(pattern, data);
					ackerr /= (data->width * data->height);
					if (ackerr < minerr){
						minerr = ackerr;
						an = a;
						bn = b;
						cn = c;
						dn = d;
					}
				}
			}
		}
	}
	//printf("Done permuting, preparing results\n");
	//printf("Minimal error: %f, Permutation: [%d,%d,%d,%d]\n", minerr, an,bn,cn,dn);
	//fflush(stdout);
	MyPoint pan,pbn,pcn,pdn;
	pan = rect->points[an];
	pbn = rect->points[bn];
	pcn = rect->points[cn];
	pdn = rect->points[dn];
	rect->points[0] = pan;
	rect->points[1] = pbn;
	rect->points[2] = pcn;
	rect->points[3] = pdn;
	cvReleaseImage(&data);
	return minerr;
}

void drawRect(IplImage *img, MyRect rect){
	cvLine(img, cvPoint(rect.points[0].x, rect.points[0].y),cvPoint(rect.points[1].x, rect.points[1].y), cvScalar(0), 10);
	cvLine(img, cvPoint(rect.points[0].x, rect.points[0].y),cvPoint(rect.points[1].x, rect.points[1].y), cvScalar(255), 4);
	cvLine(img, cvPoint(rect.points[1].x, rect.points[1].y),cvPoint(rect.points[2].x, rect.points[2].y), cvScalar(0), 10);
	cvLine(img, cvPoint(rect.points[1].x, rect.points[1].y),cvPoint(rect.points[2].x, rect.points[2].y), cvScalar(255), 4);
	cvLine(img, cvPoint(rect.points[2].x, rect.points[2].y),cvPoint(rect.points[3].x, rect.points[3].y), cvScalar(0), 10);
	cvLine(img, cvPoint(rect.points[2].x, rect.points[2].y),cvPoint(rect.points[3].x, rect.points[3].y), cvScalar(255), 4);
	cvLine(img, cvPoint(rect.points[3].x, rect.points[3].y),cvPoint(rect.points[0].x, rect.points[0].y), cvScalar(0), 10);
	cvLine(img, cvPoint(rect.points[3].x, rect.points[3].y),cvPoint(rect.points[0].x, rect.points[0].y), cvScalar(255), 4);
	cvCircle(img, cvPoint(rect.points[0].x, rect.points[0].y), 20, cvScalar(128), -1);
	cvCircle(img, cvPoint(rect.points[0].x, rect.points[0].y), 20, cvScalar(0), 3);
	cvCircle(img, cvPoint(rect.points[1].x, rect.points[1].y), 20, cvScalar(255), -1);
	cvCircle(img, cvPoint(rect.points[1].x, rect.points[1].y), 20, cvScalar(0), 3);
	cvCircle(img, cvPoint(rect.points[2].x, rect.points[2].y), 20, cvScalar(255), -1);
	cvCircle(img, cvPoint(rect.points[2].x, rect.points[2].y), 20, cvScalar(0), 3);
	cvCircle(img, cvPoint(rect.points[3].x, rect.points[3].y), 20, cvScalar(255), -1);
	cvCircle(img, cvPoint(rect.points[3].x, rect.points[3].y), 20, cvScalar(0), 3);
}

void filterRects(std::vector<MyRect> &rects, IplImage *img, IplImage *pattern){
	int i;
	float err;
	std::vector<MyRect> temp;
	printf("We have %d rects to process\n", (int)rects.size());
	fflush(stdout);
	for (i=0;i<rects.size();i++){
		err = getError2(&rects[i], img, pattern);
		printf("r: %d Error: %f\n", i, err);
		printf("a: (%d,%d), b: (%d,%d), c: (%d,%d), d: (%d,%d)\n",
			rects[i].points[0].x,rects[i].points[0].y,
			rects[i].points[1].x,rects[i].points[1].y,
			rects[i].points[2].x,rects[i].points[2].y,
			rects[i].points[3].x,rects[i].points[3].y
		);
		fflush(stdout);
		if (err>10000) continue;
		temp.push_back(rects[i]);
		//drawRect(img, rects[i]);
		//for (j=0;j<4;j++){
		//	cvCircle(img, cvPoint(rects[i].points[j].x, rects[i].points[j].y), 20, cvScalar(255), -1);
		//	cvCircle(img, cvPoint(rects[i].points[j].x, rects[i].points[j].y), 20, cvScalar(0), 3);
		//}
	}
	rects.clear();
	for (i=0;i<temp.size();i++)
		rects.push_back(temp[i]);
	printf("All rects processed\n", (int)rects.size());
	fflush(stdout);
}

/* test */

double distance3d(double x, double y, double z, double x2, double y2, double z2){
	return sqrt((x-x2)*(x-x2)+(y-y2)*(y-y2)+(z-z2)*(z-z2));
}

double getDistError(Chromosome *ch, double *data, int paramsize){
	double t[4];
	double d = data[15];
	double d2 = sqrt(d*d + d*d);
	t[0] = ch->genes[0];
	t[1] = ch->genes[1];
	t[2] = ch->genes[2];
	t[3] = ch->genes[3];
	CvScalar P[4];
	P[0].val[0] = data[0] + t[0]*data[3];
	P[0].val[1] = data[1] + t[0]*data[4];
	P[0].val[2] = data[2] + t[0]*data[5];
	
	//if (P[0].val[2] < 0.0) return -1;
	
	P[1].val[0] = data[0] + t[1]*data[6];
	P[1].val[1] = data[1] + t[1]*data[7];
	P[1].val[2] = data[2] + t[1]*data[8];

	//if (P[1].val[2] < 0.0) return -1;

	P[2].val[0] = data[0] + t[2]*data[9];
	P[2].val[1] = data[1] + t[2]*data[10];
	P[2].val[2] = data[2] + t[2]*data[11];

	//if (P[2].val[2] < 0.0) return -1;

	P[3].val[0] = data[0] + t[3]*data[12];
	P[3].val[1] = data[1] + t[3]*data[13];
	P[3].val[2] = data[2] + t[3]*data[14];
	
	//if (P[3].val[2] < 0.0) return -1;
	
	double distances[6];
	distances[0] = distance3d(P[0].val[0],P[0].val[1],P[0].val[2],P[1].val[0],P[1].val[1],P[1].val[2]);
	distances[1] = distance3d(P[1].val[0],P[1].val[1],P[1].val[2],P[2].val[0],P[2].val[1],P[2].val[2]);
	distances[2] = distance3d(P[2].val[0],P[2].val[1],P[2].val[2],P[3].val[0],P[3].val[1],P[3].val[2]);
	distances[3] = distance3d(P[3].val[0],P[3].val[1],P[3].val[2],P[0].val[0],P[0].val[1],P[0].val[2]);
	distances[4] = distance3d(P[0].val[0],P[0].val[1],P[2].val[2],P[2].val[0],P[2].val[1],P[2].val[2]);
	distances[5] = distance3d(P[1].val[0],P[1].val[1],P[1].val[2],P[3].val[0],P[3].val[1],P[3].val[2]);
	double maxerr = 0.0;
	double ackerr = 0.0;
	int i;
	for (i=0;i<4;i++){
		ackerr += (distances[i] - d)*(distances[i] - d);
		/*if (ackerr > maxerr){
			maxerr = ackerr;
		}*/
	}
	for (i=4;i<6;i++){
		ackerr += (distances[i] - d2)*(distances[i] - d2);
		/*if (ackerr > maxerr){
			maxerr = ackerr;
		}*/
	}
	return ackerr;
}

void findRect3Dposition(MyRect *rect, IplImage *image, double markSize, double focalDistance, double chipWidth, double chipHeight, FILE *file = NULL){
	int i;
	double sx = chipWidth/(double)image->width;
	double sy = chipHeight/(double)image->height;
	/* prenest obraz na stred (0,0) a zmensit na velikost cipu */ 
	for (i=0;i<4;i++){
		rect->points[i].x -= image->width/2;
		rect->points[i].y -= image->height/2;
		rect->points[i].x *= sx;
		rect->points[i].y *= sy;
	}
	/* spocitat vektory od stredu promitani na bod v promitaci plose */
	CvScalar v[4];
	for (i=0;i<4;i++){
		v[i].val[0] = rect->points[i].x;
		v[i].val[1] = rect->points[i].y;
		v[i].val[2] = focalDistance;
	}
	/* vytvorit jednotkove vektory */
	CvScalar j[4];
	float s;
	for (i=0;i<4;i++){
		s = sqrt(v[i].val[0]*v[i].val[0]+v[i].val[1]*v[i].val[1]+v[i].val[2]*v[i].val[2]);
		j[i].val[0] = v[i].val[0]/s;
		j[i].val[1] = v[i].val[1]/s;
		j[i].val[2] = v[i].val[2]/s;
	}
	double data[16];
	data[0] = 0; // projection center point
	data[1] = 0;
	data[2] = -focalDistance;
	data[3] = j[0].val[0]; // unified vectors (for all 4 points)
	data[4] = j[0].val[1];
	data[5] = j[0].val[2];
	data[6] = j[1].val[0];
	data[7] = j[1].val[1];
	data[8] = j[1].val[2];
	data[9] = j[2].val[0];
	data[10] = j[2].val[1];
	data[11] = j[2].val[2];
	data[12] = j[3].val[0];
	data[13] = j[3].val[1];
	data[14] = j[3].val[2];
	data[15] = markSize; // size of the mark
	double t[4];
	t[0] = 500;
	t[1] = 500;
	t[2] = 500;
	t[3] = 500;
	GeneticAlgorithm *ga = new GeneticAlgorithm(1000, 4, getDistError,t);
	ga->workError(0.005, 0.5, 100.0, data, 16);
	t[0] = ga->best->genes[0];
	t[1] = ga->best->genes[1];
	t[2] = ga->best->genes[2];
	t[3] = ga->best->genes[3];
	printf("best %f, %f, %f, %f: ERR: %f\n", t[0], t[1], t[2], t[3], ga->error);
	double A[3],B[3],C[3],D[3];
	A[0] = 0 + t[0]*j[0].val[0];
	A[1] = 0 + t[0]*j[0].val[1];
	A[2] = -focalDistance + t[0]*j[0].val[2];
	B[0] = 0 + t[1]*j[1].val[0];
	B[1] = 0 + t[1]*j[1].val[1];
	B[2] = -focalDistance + t[1]*j[1].val[2];
	C[0] = 0 + t[2]*j[2].val[0];
	C[1] = 0 + t[2]*j[2].val[1];
	C[2] = -focalDistance + t[2]*j[2].val[2];
	D[0] = 0 + t[3]*j[3].val[0];
	D[1] = 0 + t[3]*j[3].val[1];
	D[2] = -focalDistance + t[3]*j[3].val[2];
	printf("A: [%f, %f, %f]\n",A[0],A[1],A[2]);
	printf("B: [%f, %f, %f]\n",B[0],B[1],B[2]);
	printf("C: [%f, %f, %f]\n",C[0],C[1],C[2]);
	printf("D: [%f, %f, %f]\n",D[0],D[1],D[2]);
	if (file != NULL){
		fprintf(file, "%f;%f;%f\n",A[0],A[1],A[2]);
		fprintf(file, "%f;%f;%f\n",B[0],B[1],B[2]);
		fprintf(file, "%f;%f;%f\n",C[0],C[1],C[2]);
		fprintf(file, "%f;%f;%f\n",D[0],D[1],D[2]);
	}
	delete(ga);
}

int main(){
	IplImage *img = cvLoadImage("Foto_2/IMG_0078_test.JPG", 0);
	IplImage *pattern = cvLoadImage("template.png", 0);
	cvSmooth(img, img, CV_MEDIAN, 3);
	
	cvSaveImage("smoth.png", img);
	
	IplImage *edges = cvCloneImage(img);
	std::vector<MyPoint> points;
	
	// nalezeni rohu
	getCorneresAndEdges(img, edges, points);
	printf("bodu: %d\n", (int)points.size());
	
	// nalezeni ctvercu
	std::vector<MyRect> rects;
	findAllRects(edges, points, rects);
	filterRects(rects, img, pattern);
	printf("Rects count: %d\n", (int)rects.size());
	
	int i,j;
	float err;
	for (i=0;i<rects.size();i++){
		drawRect(img, rects[i]);
	}
	
	FILE *f;
	f = fopen("points.csv", "w");
	
	for (i=0;i<rects.size();i++){
		printf("processing rect: %d\n", i);
		findRect3Dposition(&rects[i], img, 90, 19.7, 22.3, 14.9, f);
		fflush(stdout);
	}
	
	fclose(f);
	
	cvSaveImage("zed2/out.jpg", img);
	
	cvNamedWindow("input",0);
	//cvNamedWindow("edges",0);
	
	cvShowImage("input", img);
	//cvShowImage("edges", edges);
	
	cvWaitKey(0);
	
	cvReleaseImage(&img);
	cvReleaseImage(&edges);
	cvReleaseImage(&pattern);
	return 0;
}
