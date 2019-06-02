#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <stack>
#include "genalg.h"
#include "pmfilter.h"
#include <algorithm>

//90, 19.7, 22.3, 14.9
#define MARK_SIZE 90
#define FOCAL_DISTANCE 17.2
#define CHIP_WIDTH 22.3
#define CHIP_HEIGHT 14.9

float my_dilate(const cv::Mat &src, cv::Mat &out, const int r = 0){
    for (auto y = 1; y < src.rows - 1; y++){
        for (auto x = 1; x < src.cols - 1; x++){
            auto c = src.at<uchar>(y, x);
            if (c != 0){
                c = 255;
                if (r == 0){
                    out.at<uchar>(y,x) = c;
                    out.at<uchar>(y-1,x) = c;
                    out.at<uchar>(y+1,x) = c;
                    out.at<uchar>(y,x-1) = c;
                    out.at<uchar>(y,x+1) = c;
                }else{
                    circle(out, cv::Point(x,y), r, c, -1);
                }
            }
        }
    }
    return 0.0f;
}

struct my_point{
    int x,y;
};

struct my_rect{
    my_point points[4];
};

void transpose_3d(double *points, const double tx, const double ty, const double tz){
    int i;
    float a[16];
    auto ma = cv::Mat(4, 4, CV_32FC1, a);
    setIdentity(ma);
    ma.at<float>(0,3) = tx;
    ma.at<float>(1,3) = ty;
    ma.at<float>(2,3) = tz;
    float b[4];
    auto mb = cv::Mat(4, 1, CV_32FC1, b);
    for (i = 0; i < 4; i++) mb.at<float>(i,0) = points[i];
    cv::Mat mc = ma * mb;
    for (i = 0; i < 4; i++) points[i] = mc.at<float>(i, 0);
}

void scale_3d(double *points, const double sx, const double sy, const double sz){
    points[0] *= sx;
    points[1] *= sy;
    points[2] *= sz;
}

void rotateX_3d(double *points, const double angle){
    int i;
    float a[16];
    auto ma = cv::Mat(4, 4, CV_32FC1, a);
    setIdentity(ma);
    ma.at<float>(1,1) = cos(angle);
    ma.at<float>(1,2) = -sin(angle);
    ma.at<float>(2,1) = sin(angle);
    ma.at<float>(2,2) = cos(angle);
    float b[4];
    auto mb = cv::Mat(4, 1, CV_32FC1, b);
    for (i = 0; i < 4; i++) mb.at<float>(i,0) = points[i];
    cv::Mat mc = ma * mb;
    for (i = 0; i < 4; i++) points[i] = mc.at<float>(i,0);
}

void rotateY_3d(double *points, const double angle){
    int i;
    float a[16];
    auto ma = cv::Mat(4, 4, CV_32FC1, a);
    setIdentity(ma);
    ma.at<float>(0,0) = cos(angle);
    ma.at<float>(0,2) = sin(angle);
    ma.at<float>(2,0) = sin(angle);
    ma.at<float>(2,2) = cos(angle);
    float b[4];
    auto mb = cv::Mat(4, 1, CV_32FC1, b);
    for (i=0;i<4;i++) mb.at<float>(i,0) = points[i];
    float c[4];
    cv::Mat mc = ma * mb;
    for (i = 0; i < 4; i++) points[i] = mc.at<float>(i,0);
}

void rotateZ_3d(double *points, const double angle) {
    int i;
    float a[16];
    auto ma = cv::Mat(4, 4, CV_32FC1, a);
    setIdentity(ma);
    ma.at<float>(0, 0) = cos(angle);
    ma.at<float>(0, 1) = -sin(angle);
    ma.at<float>(1, 0) = sin(angle);
    ma.at<float>(1, 1) = cos(angle);
    float b[4];
    auto mb = cv::Mat(4, 1, CV_32FC1, b);
    for (i = 0; i < 4; i++) mb.at<float>(i, 0) = points[i];
    float c[4];
    cv::Mat mc = ma * mb;
    for (i = 0; i < 4; i++) points[i] = mc.at<float>(i, 0);
}

void perspective(double *points, const double fd){
    int i,j;
    float a[16];
    auto ma = cv::Mat(4, 4, CV_32FC1, a);
    setIdentity(ma);
    ma.at<float>(3,3) = 0.0f;
    ma.at<float>(3,2) = 1.0f / fd;
    float b[4];
    auto mb = cv::Mat(4, 1, CV_32FC1, b);
    for (i=0; i<4; i++) mb.at<float>(i,0) = points[i];
    float c[4];
    cv::Mat mc = ma * mb;
    for (i = 0; i < 4; i++) points[i] = mc.at<float>(i,0);
}

void deharmonize(double *points){
    for (auto i = 0; i < 4; i++){
        points[i] /= points[3];
    }
}

void to_view_port(double *points, const double width, const double height, const double chip_w, const double chip_h){
    const auto max_size = std::max(width, height) / std::max(chip_w, chip_h);
    scale_3d(points, max_size, max_size, 1);
    transpose_3d(points, width / 2.0, height / 2.0, 0.0);
}

void transform_to_3d(double *points, const double ra, const double rb, const double rg,
                   const double tx, const double ty, const double tz, const double width,
                   const double height, const double f, const double chip_w, const double chip_h){
    rotateX_3d(points, ra);
    rotateY_3d(points, rb);
    rotateZ_3d(points, rg);
    transpose_3d(points, tx, ty, tz);
    perspective(points, f);
    deharmonize(points);
    to_view_port(points, width, height, chip_w, chip_h);
}

inline void get_corners_and_edges(cv::Mat &img, cv::Mat &edges, std::vector<my_point> &points){
	printf("getting corners and edges\n");
	fflush(stdout);
	/*********************** GOOD FEATURES TO TRACK **************************/
	printf("Looking for good features to track...");
	fflush(stdout);
	cv::Mat eig = cv::Mat(img.rows, img.cols, CV_16FC1);
    cv::Mat temp = cv::Mat(img.rows, img.cols, CV_16FC1);
	
    std::vector< cv::Point2f > corners; // tady budou vystupni body po vypoctu
	const float quality = 0.05;
	const auto min_dist = 10;
	int points_count;
	
	/*cvGoodFeaturesToTrack(img, // vstupni sedobily obraz
				eig,            // output
				temp,			  // temp obraz
				corners,			  // odkaz na pole, kam se maji ulozit bodiky
				&points_count,		  // odkaz na promennou kde se ma ulozit vysledny pocet nalezenych bodu
				quality,		  // nastaveni kvality detekce (jak kvalitne musi byt bod detekovan aby byl dan do vystupu)
				min_dist,		  // minimalni odstup od detekovanych bodu
				NULL,				  // cert vi ... proste null
				3,		  // velikost bloku pro vypocet vlastnich cisel
				false);		  // ma se pri vypoctu pouzit harris detektor?? ne!
	*/
	/*************************************************************************/

    goodFeaturesToTrack(img, corners, 1000, quality, min_dist);

	printf("DONE\n");
	fflush(stdout);
	
	cv::Mat out = img.clone();
	printf("Running canny...");
	fflush(stdout);
	Canny(img, out, 80, 150);
    imwrite("canny.png", out);
	printf("DONE\n");
	fflush(stdout);
	edges = 0;
	printf("Dilatation...");
	fflush(stdout);
	my_dilate(out, edges);
    imwrite("edges.png", edges);
	printf("DONE\n");
	fflush(stdout);

    auto test = edges.clone();

    for (auto& corner : corners)
    {
		my_point p;
		p.x = static_cast<int>(corner.x);
		p.y = static_cast<int>(corner.y);
		points.push_back(p);
		circle(test, cv::Point(p.x, p.y), 3, cv::Scalar(128), -1);
	}
	
	imwrite("test.png", test);
}

void recursive_edge_mover(cv::Mat &src, cv::Mat &dest, const int x, const int y){
	if (x < 0 || y < 0 || x > src.cols - 1 || y > src.rows - 1) return;
	if (src.at<uchar>(x,y) != 255) return;
	dest.at<uchar>(x, y) = 255;
	src.at<uchar>(x, y) = 0;
	recursive_edge_mover(src, dest, x + 1, y);
	recursive_edge_mover(src, dest, x - 1, y);
	recursive_edge_mover(src, dest, x, y + 1);
	recursive_edge_mover(src, dest, x, y - 1);
}

void non_recursive_edge_mover(cv::Mat &src, cv::Mat &dest, const int x, const int y){
	std::stack<my_point> to_process;
	my_point np;
	np.x = x;
	np.y = y;
	to_process.push(np);
	while (!to_process.empty()){
	    const auto active_point = to_process.top();
		to_process.pop();
		if (active_point.x < 0 ||
            active_point.y < 0 ||
            active_point.x > src.cols - 1 ||
            active_point.y > src.rows - 1) continue;
		if (src.at<uchar>(active_point.y,active_point.x) != 255) continue;
		dest.at<uchar>(active_point.y, active_point.x) = 255;
		src.at<uchar>(active_point.y, active_point.x) = 0;
		my_point next;
		next.x = active_point.x + 1; next.y = active_point.y;
		to_process.push(next);
		next.x = active_point.x - 1; next.y = active_point.y;
		to_process.push(next);
		next.x = active_point.x + 1; next.y = active_point.y + 1;
		to_process.push(next);
		next.x = active_point.x - 1; next.y = active_point.y - 1;
		to_process.push(next);
		next.x = active_point.x; next.y = active_point.y + 1;
		to_process.push(next);
		next.x = active_point.x; next.y = active_point.y - 1;
		to_process.push(next);
		next.x = active_point.x + 1; next.y = active_point.y - 1;
		to_process.push(next);
		next.x = active_point.x - 1; next.y = active_point.y + 1;
		to_process.push(next);
	}
}

void get_all_points_from_point(cv::Mat &edges, my_point p, std::vector<my_point> &allPoints, std::vector<my_point> &outpoints){
    auto temp = cv::Mat(edges.rows, edges.cols, CV_8UC1);
	temp = 0;
	//recursive_edge_mover(edges, temp, p.x, p.y);
	non_recursive_edge_mover(edges, temp, p.x, p.y);
    const int size = allPoints.size();
	for (auto i = 0; i<size; i++){
		if (temp.at<uchar>(allPoints[i].y, allPoints[i].x) != 0) outpoints.push_back(allPoints[i]);
	}
}

void find_all_rects(cv::Mat &edges, std::vector<my_point> &points, std::vector<my_rect> &rects){
	printf("Looking for all rects\n");
	fflush(stdout);
    const int size = points.size();
	for (auto i = 0;i < size; i++){
		std::vector<my_point> object_points;
		get_all_points_from_point(edges, points[i], points, object_points);
		if (object_points.size() == 4){
			my_rect r;
			r.points[0] = object_points[0];
			r.points[1] = object_points[1];
			r.points[2] = object_points[2];
			r.points[3] = object_points[3];
			rects.push_back(r);
		}
	}
}

void transform_image(cv::Mat &src, cv::Mat &des, cv::Point *points){
    cv::Mat input_mat = cv::Mat(4,2,CV_32FC1);
    cv::Mat output_mat = cv::Mat(4,2,CV_32FC1);
    cv::Mat H = cv::Mat(3,3,CV_32FC1);
    
    output_mat.at<float>(0, 0) = 0.0f;
    output_mat.at<float>(0, 1) = 0.0f;
    output_mat.at<float>(1, 0) = des.cols;
    output_mat.at<float>(1, 1) = 0.0f;
    output_mat.at<float>(2, 0) = des.cols;
    output_mat.at<float>(2, 1) = des.rows;
    output_mat.at<float>(3, 0) = 0.0f;
    output_mat.at<float>(3, 1) = des.rows;

    input_mat.at<float>(0, 0) = points[0].x;
    input_mat.at<float>(0, 1) = points[0].y;
    input_mat.at<float>(1, 0) = points[1].x;
    input_mat.at<float>(1, 1) = points[1].y;
    input_mat.at<float>(2, 0) = points[2].x;
    input_mat.at<float>(2, 1) = points[2].y;
    input_mat.at<float>(3, 0) = points[3].x;
    input_mat.at<float>(3, 1) = points[3].y;

    findHomography(input_mat, output_mat, H);
    warpPerspective(src, des, H, cv::Size(des.cols, des.rows));
}

float square_error_sum(cv::Mat &img, cv::Mat &img2){
    auto sum = 0.0f;
	for (auto y = 0;y<img.rows;y++){
		for (auto x = 0;x<img.cols;x++){
		    const uchar diff = img.at<uchar>(x, y) - img2.at<uchar>(x, y);
            sum += diff * diff;
		}
	}
	return sum;
}

float get_error(my_rect *rect, cv::Mat &img, cv::Mat &pattern){
    auto data = pattern.clone();
	cv::Point points[4];
	points[0] = cv::Point(rect->points[0].x,rect->points[0].y);
	points[1] = cv::Point(rect->points[1].x,rect->points[1].y);
	points[2] = cv::Point(rect->points[2].x,rect->points[2].y);
	points[3] = cv::Point(rect->points[3].x,rect->points[3].y);
	transform_image(img, data, points);
	float err = square_error_sum(pattern, data);
	err /= data.cols * data.rows;
	return err;
}

float get_error2(my_rect *rect, cv::Mat &img, cv::Mat &pattern){
	cv::Mat data = pattern.clone();
	cv::Point points[4];
	points[0] = cv::Point(rect->points[0].x,rect->points[0].y);
	points[1] = cv::Point(rect->points[1].x,rect->points[1].y);
	points[2] = cv::Point(rect->points[2].x,rect->points[2].y);
	points[3] = cv::Point(rect->points[3].x,rect->points[3].y);
	transform_image(img, data, points);
    auto min_err = square_error_sum(pattern, data);
	min_err /= data.cols * data.rows;
    auto ack_err = min_err;
    auto an = 0, bn = 1, cn = 2, dn = 3;
	for (auto a = 0;a<4;a++){
		for (auto b = 0;b<4;b++){
			if (b==a) continue;
			for (auto c = 0;c<4;c++){
				if (c==a || c==b) continue;
				for (auto d = 0;d<4;d++){
					if (d==a || d==b || d==c) continue;
					points[0] = cv::Point(rect->points[a].x,rect->points[a].y);
					points[1] = cv::Point(rect->points[b].x,rect->points[b].y);
					points[2] = cv::Point(rect->points[c].x,rect->points[c].y);
					points[3] = cv::Point(rect->points[d].x,rect->points[d].y);
					transform_image(img, data, points);
					ack_err = square_error_sum(pattern, data);
					ack_err /= (data.cols * data.rows);
					if (ack_err < min_err){
						min_err = ack_err;
						an = a;
						bn = b;
						cn = c;
						dn = d;
					}
				}
			}
		}
	}
    auto pan = rect->points[an];
    auto pbn = rect->points[bn];
    auto pcn = rect->points[cn];
    auto pdn = rect->points[dn];
	rect->points[0] = pan;
	rect->points[1] = pbn;
	rect->points[2] = pcn;
	rect->points[3] = pdn;
	return min_err;
}

void draw_rect(cv::Mat &img, my_rect rect){
    line(img, cv::Point(rect.points[0].x, rect.points[0].y), cv::Point(rect.points[1].x, rect.points[1].y), cv::Scalar(0), 10);
    line(img, cv::Point(rect.points[0].x, rect.points[0].y), cv::Point(rect.points[1].x, rect.points[1].y), cv::Scalar(255), 4);
    line(img, cv::Point(rect.points[1].x, rect.points[1].y), cv::Point(rect.points[2].x, rect.points[2].y), cv::Scalar(0), 10);
    line(img, cv::Point(rect.points[1].x, rect.points[1].y), cv::Point(rect.points[2].x, rect.points[2].y), cv::Scalar(255), 4);
    line(img, cv::Point(rect.points[2].x, rect.points[2].y), cv::Point(rect.points[3].x, rect.points[3].y), cv::Scalar(0), 10);
    line(img, cv::Point(rect.points[2].x, rect.points[2].y), cv::Point(rect.points[3].x, rect.points[3].y), cv::Scalar(255), 4);
    line(img, cv::Point(rect.points[3].x, rect.points[3].y), cv::Point(rect.points[0].x, rect.points[0].y), cv::Scalar(0), 10);
    line(img, cv::Point(rect.points[3].x, rect.points[3].y), cv::Point(rect.points[0].x, rect.points[0].y), cv::Scalar(255), 4);
    circle(img, cv::Point(rect.points[0].x, rect.points[0].y), 20, cv::Scalar(128), -1);
    circle(img, cv::Point(rect.points[0].x, rect.points[0].y), 20, cv::Scalar(0), 3);
    circle(img, cv::Point(rect.points[1].x, rect.points[1].y), 20, cv::Scalar(255), -1);
    circle(img, cv::Point(rect.points[1].x, rect.points[1].y), 20, cv::Scalar(0), 3);
    circle(img, cv::Point(rect.points[2].x, rect.points[2].y), 20, cv::Scalar(255), -1);
    circle(img, cv::Point(rect.points[2].x, rect.points[2].y), 20, cv::Scalar(0), 3);
    circle(img, cv::Point(rect.points[3].x, rect.points[3].y), 20, cv::Scalar(255), -1);
    circle(img, cv::Point(rect.points[3].x, rect.points[3].y), 20, cv::Scalar(0), 3);
}

void filter_rects(std::vector<my_rect> &rects, cv::Mat &img, cv::Mat &pattern){
	int i;
    std::vector<my_rect> temp;
	printf("We have %d rects to process\n", static_cast<int>(rects.size()));
	fflush(stdout);
	for (i=0;i<rects.size();i++){
		float err = get_error2(&rects[i], img, pattern);
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
	}
	rects.clear();
	for (i=0; i < temp.size(); i++)
		rects.push_back(temp[i]);
	printf("All rects processed: %d\n", rects.size());
	fflush(stdout);
}

/* test */

double distance_3d(const double x, const double y, const double z, const double x2, const double y2, const double z2){
	return sqrt((x-x2)*(x-x2)+(y-y2)*(y-y2)+(z-z2)*(z-z2));
}

// chromosom bude velikosti 6 -> 3 rozmery rotace a 3 rozmery posunu
// data parametu budou rozmeru 14 -> focal distance, chip width, chip height,
//             image width, image height a 4 body o 2 rozmerech, marker size
double getDistError2(chromosome *chr, double *data, int paramsize){
    int i;
    float ra = chr->genes[0];
    float rb = chr->genes[1];
    float rc = chr->genes[2];
    float tx = chr->genes[3];
    float ty = chr->genes[4];
    float tz = chr->genes[5];
    if (tz < 0) return 100000000.0;
    if (tz > 20000) return 100000000.0;
    float f = data[0]; // f,cw,ch,iw,ih,ax,ay,bx,by,cx,cy,dx,dy,msize
    float cw = data[1];
    float ch = data[2];
    float iw = data[3];
    float ih = data[4];
    float An[2];
    An[0] = data[5];
    An[1] = data[6];
    float Bn[2];
    Bn[0] = data[7];
    Bn[1] = data[8];
    float Cn[2];
    Cn[0] = data[9];
    Cn[1] = data[10];
    float Dn[2];
    Dn[0] = data[11];
    Dn[1] = data[12];
    float mSize = data[13];
    //printf("Data parameters in GA:\n");
	//for (i=0;i<14;i++){
	//    printf("\t%f\n",data[i]);
	//}
    double A[4],B[4],C[4],D[4];
    A[0] = -mSize/2.0; A[1] = -mSize/2.0; A[2] = 0.0; A[3] = 1.0;
    B[0] =  mSize/2.0; B[1] = -mSize/2.0; B[2] = 0.0; B[3] = 1.0;
    C[0] =  mSize/2.0; C[1] =  mSize/2.0; C[2] = 0.0; C[3] = 1.0;
    D[0] = -mSize/2.0; D[1] =  mSize/2.0; D[2] = 0.0; D[3] = 1.0;
    //printf("before [%f,%f][%f,%f][%f,%f][%f,%f]\n",A[0],A[1],B[0],B[1],C[0],C[1],D[0],D[1]);
    transform_to_3d(A,ra,rb,rc,tx,ty,tz,iw,ih,f,cw,ch);
	transform_to_3d(B,ra,rb,rc,tx,ty,tz,iw,ih,f,cw,ch);
	transform_to_3d(C,ra,rb,rc,tx,ty,tz,iw,ih,f,cw,ch);
	transform_to_3d(D,ra,rb,rc,tx,ty,tz,iw,ih,f,cw,ch);
	//printf("[%f,%f][%f,%f][%f,%f][%f,%f]\n",A[0],A[1],B[0],B[1],C[0],C[1],D[0],D[1]);
	//printf("[%f,%f][%f,%f][%f,%f][%f,%f]\n\n",An[0],An[1],Bn[0],Bn[1],Cn[0],Cn[1],Dn[0],Dn[1]);
	fflush(stdout);
	float err = 0.0;
	err += (A[0]-An[0])*(A[0]-An[0]);
	err += (A[1]-An[1])*(A[1]-An[1]);
	err += (B[0]-Bn[0])*(B[0]-Bn[0]);
	err += (B[1]-Bn[1])*(B[1]-Bn[1]);
	err += (C[0]-Cn[0])*(C[0]-Cn[0]);
	err += (C[1]-Cn[1])*(C[1]-Cn[1]);
	err += (D[0]-Dn[0])*(D[0]-Dn[0]);
	err += (D[1]-Dn[1])*(D[1]-Dn[1]);
	return err;
}

double getDistError(chromosome *ch, double *data, int paramsize){
	double t[4];
    auto d = data[15];
    auto d2 = sqrt(d*d + d*d);
	t[0] = ch->genes[0];
	t[1] = ch->genes[1];
	t[2] = ch->genes[2];
	t[3] = ch->genes[3];
	cv::Scalar P[4];
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
	distances[0] = distance_3d(P[0].val[0],P[0].val[1],P[0].val[2],P[1].val[0],P[1].val[1],P[1].val[2]);
	distances[1] = distance_3d(P[1].val[0],P[1].val[1],P[1].val[2],P[2].val[0],P[2].val[1],P[2].val[2]);
	distances[2] = distance_3d(P[2].val[0],P[2].val[1],P[2].val[2],P[3].val[0],P[3].val[1],P[3].val[2]);
	distances[3] = distance_3d(P[3].val[0],P[3].val[1],P[3].val[2],P[0].val[0],P[0].val[1],P[0].val[2]);
	distances[4] = distance_3d(P[0].val[0],P[0].val[1],P[2].val[2],P[2].val[0],P[2].val[1],P[2].val[2]);
	distances[5] = distance_3d(P[1].val[0],P[1].val[1],P[1].val[2],P[3].val[0],P[3].val[1],P[3].val[2]);
    auto max_error = 0.0;
    auto ack_error = 0.0;
	int i;
	for (i=0;i<4;i++){
		ack_error += (distances[i] - d)*(distances[i] - d);
	}
	for (i=4;i<6;i++){
		ack_error += (distances[i] - d2)*(distances[i] - d2);
	}
	return ack_error;
}

void find_rect_3D_position(my_rect *rect, cv::Mat &image, double markSize, double focalDistance, double chipWidth, double chipHeight, FILE *file = NULL){
	int i;
    const auto sx = chipWidth/static_cast<double>(image.cols);
    const auto sy = chipHeight/static_cast<double>(image.rows);
	/* prenest obraz na stred (0,0) a zmensit na velikost cipu */ 
	for (i=0;i<4;i++){
		rect->points[i].x -= image.cols/2;
		rect->points[i].y -= image.rows/2;
		rect->points[i].x *= sx;
		rect->points[i].y *= sy;
	}
	/* spocitat vektory od stredu promitani na bod v promitaci plose */
	cv::Scalar v[4];
	for (i=0;i<4;i++){
		v[i].val[0] = rect->points[i].x;
		v[i].val[1] = rect->points[i].y;
		v[i].val[2] = focalDistance;
	}
	/* vytvorit jednotkove vektory */
	cv::Scalar j[4];
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
	genetic_algorithm *ga = new genetic_algorithm(1000, 4, getDistError,t);
	ga->work_error(0.005, 0.5, 100.0, data, 16);
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

void find_rect_3D_position2(my_rect *rect, cv::Mat &image, double markSize, double focalDistance, double chipWidth, double chipHeight, FILE *file = NULL){
    double data[14]; // f,cw,ch,iw,ih,ax,ay,bx,by,cx,cy,dx,dy,msize
	data[0] = focalDistance;
	data[1] = chipWidth;
	data[2] = chipHeight;
	data[3] = image.cols;
	data[4] = image.rows;
	data[5] = rect->points[0].x;
	data[6] = rect->points[0].y;
	data[7] = rect->points[1].x;
	data[8] = rect->points[1].y;
	data[9] = rect->points[2].x;
	data[10] = rect->points[2].y;
	data[11] = rect->points[3].x;
	data[12] = rect->points[3].y;
	data[13] = markSize;
	printf("Data parameters for GA:\n");
	for (int i = 0; i < 14; i++){
	    printf("\t%f\n",data[i]);
	}
	double initialization_data[6];
	initialization_data[0] = 0;
	initialization_data[1] = 0;
	initialization_data[2] = 0;
	initialization_data[3] = 0;
	initialization_data[4] = 0;
	initialization_data[5] = 500;
    auto ga = genetic_algorithm(1000, 6, getDistError2, initialization_data);
	ga.work_error(1.0, 0.3, 1000.0, data, 14);
	float best[6];
	best[0] = ga.best->genes[0];
	best[1] = ga.best->genes[1];
	best[2] = ga.best->genes[2];
	best[3] = ga.best->genes[3];
	best[4] = ga.best->genes[4];
	best[5] = ga.best->genes[5];
	printf("best: %f, %f, %f, %f, %f, %f: ERR: %f\n", best[0], best[1], best[2], best[3], best[4], best[5], ga.error);
	if (file != nullptr){
		fprintf(file, "%f, %f, %f, %f, %f, %f,ERR: %f\n", best[0], best[1], best[2], best[3], best[4], best[5], ga.error);
	}
	double a[4], b[4], c[4], d[4];
    a[0] = -markSize/2.0; a[1] = -markSize/2.0; a[2] = 0.0; a[3] = 1.0;
    b[0] =  markSize/2.0; b[1] = -markSize/2.0; b[2] = 0.0; b[3] = 1.0;
    c[0] =  markSize/2.0; c[1] =  markSize/2.0; c[2] = 0.0; c[3] = 1.0;
    d[0] = -markSize/2.0; d[1] =  markSize/2.0; d[2] = 0.0; d[3] = 1.0;

    transform_to_3d(a,best[0], best[1], best[2], best[3], best[4], best[5], image.cols, image.rows, focalDistance, chipWidth, chipHeight);
	transform_to_3d(b,best[0], best[1], best[2], best[3], best[4], best[5], image.cols, image.rows, focalDistance, chipWidth, chipHeight);
	transform_to_3d(c,best[0], best[1], best[2], best[3], best[4], best[5], image.cols, image.rows, focalDistance, chipWidth, chipHeight);
	transform_to_3d(d,best[0], best[1], best[2], best[3], best[4], best[5], image.cols, image.rows, focalDistance, chipWidth, chipHeight);
	
    my_rect mr;
	mr.points[0].x = a[0];
	mr.points[0].y = a[1];
	mr.points[1].x = b[0];
	mr.points[1].y = b[1];
	mr.points[2].x = c[0];
	mr.points[2].y = c[1];
	mr.points[3].x = d[0];
	mr.points[3].y = d[1];

	draw_rect(image, mr);
}

int main(){
	auto imgRGB = cv::imread("IMG_1952.JPG");
    auto img = cv::Mat(imgRGB.cols, imgRGB.rows, CV_8UC1);

	perona_malik pm;
	imwrite("RGB_orig.png", imgRGB);
	printf("Filtering image...");
	fflush(stdout);
	pm.filter(imgRGB, 20, 1); // 20, 50
	printf("DONE\n");
    imwrite("RGB_filter.png", pm.output);
	cvtColor(pm.output, img, cv::COLOR_RGB2GRAY);
    auto pattern = cv::imread("template.png", 0);
	
	imwrite("smooth.png", img);

    auto edges = img.clone();
	std::vector<my_point> points;
	
	// nalezeni rohu
	get_corners_and_edges(img, edges, points);
	printf("bodu: %d\n", static_cast<int>(points.size()));
	
	// nalezeni ctvercu
	std::vector<my_rect> rects;
	find_all_rects(edges, points, rects);
	filter_rects(rects, img, pattern);
	printf("Rects count: %d\n", static_cast<int>(rects.size()));

    int i;
	for (i = 0; i < rects.size(); i++){
		draw_rect(img, rects[i]);
	}

    FILE* f = fopen("points.csv", "w");
	fprintf(f, "alpha, beta, gama, x, y, z\n");
	for (i = 0; i < rects.size(); i++){
		printf("processing rect: %d\n", i);
		find_rect_3D_position2(&rects[i], img, 90, 19.7, 22.3, 14.9, f);
		fflush(stdout);
	}
	
	fclose(f);
	
	imwrite("zed2/out.jpg", img);

    cv::namedWindow("input",0);
	imshow("input", img);

    cv::waitKey(0);

	return 0;
}
