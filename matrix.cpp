/*
 * Matrix.cpp
 *
 *  Created on: Apr 19, 2012
 *      Author: karel
 */

#include "matrix.h"
#include <cstdlib>
#include <cstdio>

matrix::~matrix() {
	free(this->data);
}

matrix::matrix(const int width, const int height){
	this->width = width;
	this->height = height;
	this->data = static_cast<double*>(malloc(this->width * this->height * sizeof(double)));
}

matrix::matrix(const int width, const int height, double *data){
	this->width = width;
	this->height = height;
	this->data = static_cast<double*>(malloc(this->width * this->height * sizeof(double)));
	this->set_data(data);
}

void matrix::set_data(const double *data) const
{
    for (auto y = 0; y<this->height; y++){
		for (auto x = 0; x<this->width; x++){
			MAT_EL(this,x,y) = *(data+(y*this->width)+x);
		}
	}
}

void matrix::print() const
{
    for (auto y = 0; y < this->height;y++){
		for (auto x = 0; x < this->width;x++){
			printf("%7.2f ", MAT_EL(this,x,y));
		}
		printf("\n");
	}
	printf("\n");
}

int matrix::multiply_by(matrix *m, matrix *result) const
{
    if (this->width!=m->height) return -1;
	if (result->width!=m->width||result->height!=this->height) return -1;
	for (auto row = 0; row < this->height; row++){
		for (auto column = 0; column < m->width; column++){
			double sum = 0.0;
			for (int i = 0;i<this->width;i++){
				sum += MAT_EL(this,i,row)*MAT_EL(m,column,i);
			}
			MAT_EL(result,column,row) = sum;
		}
	}
	return 0;
}

int matrix::multiply_by(const double d, matrix *result) const
{
    if (result->width != this->width || result->height != this->height) return -1;
	for (auto y = 0;y<this->height;y++){
		for (auto x = 0;x<this->width;x++){
			MAT_EL(result,x,y) = MAT_EL(this,x,y)*d;
		}
	}
	return 0;
}

int matrix::add(matrix *m, matrix *result) const
{
    if (m->width != this->width || m->height != this->height) return -1;
	if (result->width != this->width || result->height != this->height) return -1;
	for (auto y = 0; y < this->height; y++){
		for (auto x = 0; x < this->width; x++){
			MAT_EL(result,x,y) = MAT_EL(this,x,y) + MAT_EL(m,x,y);
		}
	}
	return 0;
}

int matrix::subtract(matrix *m, matrix *result) const
{
    if (m->width != this->width || m->height != this->height) return -1;
	if (result->width != this->width || result->height != this->height) return -1;
	for (auto y = 0; y < this->height; y++){
		for (auto x = 0; x < this->width; x++){
			MAT_EL(result,x,y) = MAT_EL(this,x,y) - MAT_EL(m,x,y);
		}
	}
	return 0;
}

int matrix::transpose(matrix *result) const
{
	if (result->width != this->height || result->height != this->width) return -1;
	for (auto y = 0; y < this->height; y++){
		for (auto x = 0; x < this->width; x++){
			MAT_EL(result,x,y) = MAT_EL(this,y,x);
		}
	}
	return 0;
}

matrix * HomogenousTrans::create_projection_matrix(const double f){
	double data[16] =
		{
		   1.0, 0.0,    0.0, 0.0,
		   0.0, 1.0,    0.0, 0.0,
		   0.0, 0.0,    1.0, 0.0,
		   0.0, 0.0, -1.0/f, 0.0
		};
	return new matrix(4,4,data);
}

matrix * HomogenousTrans::create_translation_matrix(const double tx, const double ty, const double tz){
	double data[16] =
		{
		   1.0, 0.0, 0.0,  tx,
		   0.0, 1.0, 0.0,  ty,
		   0.0, 0.0, 1.0,  tz,
		   0.0, 0.0, 0.0, 1.0
		};
	return new matrix(4,4,data);
}

matrix * HomogenousTrans::create_scale_matrix(const double sx, const double sy, const double sz){
	double data[16] =
			{
			    sx, 0.0, 0.0, 0.0,
			   0.0,  sy, 0.0, 0.0,
			   0.0, 0.0,  sz, 0.0,
			   0.0, 0.0, 0.0, 1.0
			};
	return new matrix(4,4,data);
}

matrix * HomogenousTrans::create_x_rotation_matrix(const double a){
	double data[16] =
		{
		   1.0,    0.0,     0.0, 0.0,
		   0.0, cos(a), -sin(a), 0.0,
		   0.0, sin(a),  cos(a), 0.0,
		   0.0,    0.0,     0.0, 1.0
		};
	return new matrix(4,4,data);
}

matrix * HomogenousTrans::create_y_rotation_matrix(const double a){
	double data[16] =
		{
		    cos(a), 0.0, sin(a), 0.0,
		       0.0, 1.0,    0.0, 0.0,
		   -sin(a), 0.0, cos(a), 0.0,
		       0.0, 0.0,    0.0, 1.0
		};
	return new matrix(4,4,data);
}

matrix * HomogenousTrans::create_z_rotation_matrix(const double a){
	double data[16] =
		{
		   cos(a), -sin(a), 0.0, 0.0,
		   sin(a),  cos(a), 0.0, 0.0,
		      0.0,     0.0, 1.0, 0.0,
		      0.0,     0.0, 0.0, 1.0
		};
	return new matrix(4,4,data);
}
