/*
 * Matrix.h
 *
 *  Created on: Apr 19, 2012
 *      Author: karel
 */

#ifndef MATRIX_H_
#define MATRIX_H_

class matrix;

#ifndef MAT_EL
#define MAT_EL(mat,x,y) (*((mat)->data+(y)*(mat)->width+(x)))
#define MAT_EL_IJ(mat,i,j) (*((mat)->data+(i)*(mat)->width+(j)))
#endif

class matrix {
public:
	double *data;
	int width;
	int height;

	matrix(int width, int height);
	matrix(int width, int height, double *data);

	int multiply_by(matrix *m, matrix *result) const;
	int multiply_by(double d, matrix *result) const;
	int add(matrix *m, matrix *result) const;
	int subtract(matrix *m, matrix *result) const;
	int transpose(matrix *result) const;

	void set_data(const double *data) const;
	void print() const;
	~matrix();
};

namespace HomogenousTrans{
	matrix * create_projection_matrix(double f);
	matrix * create_translation_matrix(double tx, double ty, double tz);
	matrix * create_scale_matrix(double sx, double sy, double sz);
	matrix * create_x_rotation_matrix(double a);
	matrix * create_y_rotation_matrix(double a);
	matrix * create_z_rotation_matrix(double a);
}

#endif /* MATRIX_H_ */
