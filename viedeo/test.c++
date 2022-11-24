#include "stdio.h"

int main()
{
	int iWidth = 960;
	int iHeight = 540;
	FILE* fp = fopen("_depth0.raw", "rb");

	float * data = new float[iWidth*iHeight];

	fread(data, sizeof(float), iWidth*iHeight, fp);

	//The depth value of pixel (i,j)
	int i = 426, j = 391;
	float z = 1.0 / data[i+j*iWidth];

	printf("z:%f\n",z);

	return 0;
}