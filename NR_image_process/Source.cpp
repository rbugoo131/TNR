#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define BLOCKSIZE 8
//#define TRAININGPATTERN
//#define TESTPATTERN
#define IMG
//#define _PSNR
//#define pastpack
//#define Deblock

void create_py(char *pyname, char *img_h, char *img_t, int FRAME, int NUMBER, int mode, int ref)
{
	Mat a;
	FILE *fin;
	int count = 0;
	int px, py, mx, my;
	FILE *fout;
	int row = 0;
	int col = 0;

	char filtname[200];
	char mvfilename[300];

	fout = fopen(pyname, "wb");
	fprintf(fout, "input  = [\n");
	for (int i = FRAME; i < FRAME + NUMBER; i++)
	{
		cout << "frame " << i << endl;

		sprintf(filtname, "%s_%d_%s", img_h, i - ref, img_t);
		sprintf(mvfilename, "H:\\DeepLearning_NR\\DataPreprocessing\\NR_image_process\\MV\\seq1\\20150916_ISO12800_seq1_%d_rgb.txt", i);
		a = imread(filtname);
		fin = fopen(mvfilename, "r");

		while (fscanf(fin, "%d %d %d %d\n", &px, &py, &mx, &my) != EOF)
		{
			row = py + my * mode;
			col = px + mx * mode;

			row = (row < 0) ? row - my * mode : (row > 1079) ? row - my * mode : row;
			col = (col < 0) ? col - mx * mode : (col > 1919) ? col - mx * mode : col;

			for (int i = 0; i < BLOCKSIZE; i++)
			{
				for (int j = 0; j < BLOCKSIZE; j++)
				{
					if (i == 0 && j == 0) fprintf(fout, "[ ");
					if (row + i > 1079) row = row - my * mode;
					if (col + j > 1919) col = col - mx * mode;
					fprintf(fout, "%3d,", (int)a.at<Vec3b>(row + i, col + j)[0]);
					fprintf(fout, "%3d,", (int)a.at<Vec3b>(row + i, col + j)[1]);
					fprintf(fout, "%3d,", (int)a.at<Vec3b>(row + i, col + j)[2]);
					if (i == BLOCKSIZE - 1 && j == BLOCKSIZE - 1) fprintf(fout, " ],\n");
				}
			}
		}
		fclose(fin);
	}
	fprintf(fout, " ]\n");
	fclose(fout);
}

static const int MAX_QUANT = 60;

const int alphas[] = {
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 4, 4,
	5, 6, 7, 8, 9, 10,
	12, 13, 15, 17, 20,
	22, 25, 28, 32, 36,
	40, 45, 50, 56, 63,
	71, 80, 90, 101, 113,
	127, 144, 162, 182,
	203, 226, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255
};

const int betas[] = {
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 2, 2,
	2, 3, 3, 3, 3, 4,
	4, 4, 6, 6,
	7, 7, 8, 8, 9, 9,
	10, 10, 11, 11, 12,
	12, 13, 13, 14, 14,
	15, 15, 16, 16, 17,
	17, 18, 18,
	19, 20, 21, 22, 23, 24, 25, 26, 27
};

const int cs[] = {
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 1, 1, 1,
	1, 1, 1, 1, 1, 1,
	1, 2, 2, 2, 2, 3,
	3, 3, 4, 4, 5, 5,
	6, 7, 8, 8, 10,
	11, 12, 13, 15, 17,
	19, 21, 23, 25, 27, 29, 31, 33, 35
};

static inline int clip(int x, int min, int max)
{
	if (x < min) {
		return min;
	}
	if (x > max) {
		return max;
	}
	return x;
}

static inline void deblock_horizontal_edge_c(uint8_t *srcp, int srcPitch, int alpha, int beta, int c0)
{
	uint8_t *sq0 = srcp;
	uint8_t *sq1 = srcp + srcPitch;
	uint8_t *sq2 = srcp + 2 * srcPitch;
	uint8_t *sp0 = srcp - srcPitch;
	uint8_t *sp1 = srcp - 2 * srcPitch;
	uint8_t *sp2 = srcp - 3 * srcPitch;

	for (int i = 0; i < 4; i++)
	{
		if ((std::abs(sp0[i] - sq0[i]) < alpha) && (std::abs(sp1[i] - sp0[i]) < beta) && (std::abs(sq0[i] - sq1[i]) < beta))
		{
			int ap = std::abs(sp2[i] - sp0[i]);
			int aq = std::abs(sq2[i] - sq0[i]);
			int c = c0;
			if (aq < beta) c++;
			if (ap < beta) c++;

			int avg0 = (sp0[i] + sq0[i] + 1) >> 1;

			int delta = clip((((sq0[i] - sp0[i]) << 2) + (sp1[i] - sq1[i]) + 4) >> 3, -c, c);
			int deltap1 = clip((sp2[i] + avg0 - (sp1[i] << 1)) >> 1, -c0, c0);
			int deltaq1 = clip((sq2[i] + avg0 - (sq1[i] << 1)) >> 1, -c0, c0);

			if (ap < beta) {
				sp1[i] = sp1[i] + deltap1;
			}

			sp0[i] = clip(sp0[i] + delta, 0, 255);
			sq0[i] = clip(sq0[i] - delta, 0, 255);

			if (aq < beta) {
				sq1[i] = sq1[i] + deltaq1;
			}
		}
	}
}

static inline void deblock_vertical_edge_c(uint8_t *srcp, int src_pitch, int alpha, int beta, int c0)
{
	//-3,0 -2,1 -1,2 0,3 1,4 2,5
	for (int i = 0; i < 4; i++)
	{
		if ((std::abs(srcp[3] - srcp[2]) < alpha) && (std::abs(srcp[4] - srcp[3]) < beta) && (std::abs(srcp[2] - srcp[1]) < beta))
		{
			int ap = std::abs(srcp[5] - srcp[3]);
			int aq = std::abs(srcp[0] - srcp[2]);
			int c = c0;
			if (aq < beta) c++;
			if (ap < beta) c++;

			int avg0 = (srcp[3] + srcp[2] + 1) >> 1;

			int delta = clip((((srcp[3] - srcp[2]) << 2) + (srcp[-2] - srcp[1]) + 4) >> 3, -c, c);
			int deltaq1 = clip((srcp[5] + avg0 - (srcp[4] << 1)) >> 1, -c0, c0);
			int deltap1 = clip((srcp[0] + avg0 - (srcp[1] << 1)) >> 1, -c0, c0);

			if (aq < beta) {
				srcp[1] = (srcp[1] + deltap1);
			}

			srcp[-1] = clip(srcp[2] + delta, 0, 255);
			srcp[0] = clip(srcp[3] - delta, 0, 255);

			if (ap < beta) {
				srcp[1] = (srcp[4] + deltaq1);
			}
		}
		srcp += src_pitch;
	}
}

static void deblock_c(uint8_t *srcp, int src_pitch, int width, int height, int alpha, int beta, int c0) {
	for (int x = 0; x < width-4; x += 4) {
		deblock_vertical_edge_c(srcp + x, src_pitch, alpha, beta, c0);
	}

	srcp += 4 * src_pitch;
	for (int y = 0; y < height-4; y += 4) {
		deblock_horizontal_edge_c(srcp, src_pitch, alpha, beta, c0);

		for (int x = 0; x < width-4; x += 4) {
			deblock_horizontal_edge_c(srcp + x, src_pitch, alpha, beta, c0);
			deblock_vertical_edge_c(srcp + x, src_pitch, alpha, beta, c0);
		}
		srcp += 4 * src_pitch;
	}
}


int main()
{
	int FRAME = 10;
	int FRAMEtesting = 11;
	int NUMBER = 1;

	#ifdef pastpack
		Mat img(1080, 1920, CV_8U);
		FILE *fin;
		fin = fopen("E:\\PROGRAMS\\FRC\\transfer\\Output_12.txt", "rb");
		for (int frame = 9; frame < 40; frame += 2)
		{
			for (int i = 0; i < 108; i++)
			{
				for (int j = 0; j < 192; j++)
				{
					float n;
					fscanf(fin, "%f", &n);
					n = 255 * n;
					for (int row = i * 10; row < i * 10 + 10; row++)
					{
						for (int col = j * 10; col < j * 10 + 10; col++)
						{
							img.at<uchar>(row, col) = (int)n;
						}

					}
				}
			}
			char filtname[200];
			sprintf(filtname, "E:\\PROGRAMS\\FRC\\transfer\\Out_%03d.bmp", frame);
			imwrite(filtname, img);
		}

	#endif

	#ifdef TRAININGPATTERN

		create_py("Training.py", "H:\\DeepLearning_NR\\SRC_image\\20150916_ISO12800_seq1", "rgb.bmp", FRAME, NUMBER, 0, 0);
		create_py("Training_mv.py", "H:\\DeepLearning_NR\\SRC_image\\20150916_ISO12800_seq1", "rgb.bmp", FRAME, NUMBER, 1, 1);
		create_py("Training_zmv.py", "H:\\DeepLearning_NR\\SRC_image\\20150916_ISO12800_seq1", "rgb.bmp", FRAME, NUMBER, 0, 1);

		create_py("Target.py", "H:\\DeepLearning_NR\\NR_image\\NR9_1920x1080_12800_seq1_r797_sc4th14", "Convert.bmp", FRAME, NUMBER, 0, 0);

		create_py("Testing.py", "H:\\DeepLearning_NR\\SRC_image\\20150916_ISO12800_seq1", "rgb.bmp", FRAMEtesting, NUMBER, 0, 0);
		create_py("Testing_mv.py", "H:\\DeepLearning_NR\\SRC_image\\20150916_ISO12800_seq1", "rgb.bmp", FRAMEtesting, NUMBER, 1, 1);
		create_py("Testing_zmv.py", "H:\\DeepLearning_NR\\SRC_image\\20150916_ISO12800_seq1", "rgb.bmp", FRAMEtesting, NUMBER, 0, 1);

	#endif

	#ifdef TESTPATTERN
		Mat a;
		Mat ref;
		char filtname[200];
		char filtname_ref[200];
		FILE *fout;
		fout = fopen("Testing.py", "wb");
		fprintf(fout, "input  = [\n");
		int count = 0;

		for (int i = 11; i < 12; i++)
		{
			cout << "frame " << i << endl;
			//sprintf(filtname, "G:\\DeepLearning_NR\\src1\\20150916_ISO12800_seq1_%d_rgb.bmp", i);
			//sprintf(filtname, "G:\\DeepLearning_NR\\src1_NR\\NR9_1920x1080_12800_seq1_r797_sc4th14_%d_Convert.bmp", i);
			a = imread("G:\\DeepLearning_NR\\src1\\20150916_ISO12800_seq1_10_rgb.bmp", CV_8U);
			ref = imread("G:\\DeepLearning_NR\\src1\\20150916_ISO12800_seq1_11_rgb.bmp", CV_8U);
			//ref = imread("G:\\DeepLearning_NR\\DataPreprocessing\\Project1\\Project1\\out21.bmp", CV_8U);
			for (int row = 0; (row + BLOCKSIZE - 1) < a.rows; row += BLOCKSIZE)
			{
				//if (count)break;
				for (int col = 0; (col + BLOCKSIZE - 1) < a.cols; col += BLOCKSIZE)
				{
					//cout << row << "," << col << endl;
					for (int i = 0; i < BLOCKSIZE; i++)
					{
						for (int j = 0; j < BLOCKSIZE; j++)
						{
							if (i == 0 && j == 0)fprintf(fout, "[ ");
							fprintf(fout, "%3d,", (int)a.at<uchar>(row + i, col + j));
							fprintf(fout, "%3d,", (int)ref.at<uchar>(row + i, col + j));
							if (i == BLOCKSIZE - 1 && j == BLOCKSIZE - 1)
								fprintf(fout, " ],\n");
						}
					}
					//if (count)break;
				}
			}
		}
		fprintf(fout, "]\n");
		fclose(fout);
	#endif

	#ifdef IMG
		/*
		int col, row, mx, my;
		char mvfilename[300];
		sprintf(mvfilename, "H:\\TNR\\DataPreprocessing\\NR_image_process\\MV\\seq1\\20150916_ISO12800_seq1_%d_rgb.txt", FRAMEtesting);
		FILE *finfin = fopen(mvfilename, "r");

		FILE *fin = fopen("H:\\TNR\\AI\\Output.txt", "r");
		Mat img(1080, 1920, CV_8UC3);
		while (fscanf(finfin, "%d %d %d %d\n", &col, &row, &mx, &my) != EOF)
		{
			//cout << row << "," << col << endl;
			for (int i = 0; i < BLOCKSIZE; i++)
			{
				for (int j = 0; j < BLOCKSIZE; j++)
				{
					float value;
					int n = 16;

					fscanf(fin, "%f", &value);
					img.at<Vec3b>(i + row, j + col)[0] = (int)value;
					fscanf(fin, "%f", &value);
					img.at<Vec3b>(i + row, j + col)[1] = (int)value;
					fscanf(fin, "%f", &value);
					img.at<Vec3b>(i + row, j + col)[2] = (int)value;

				}
			}
		}

		imwrite("out.bmp", img);
		*/
		
		FILE *fin = fopen("H:\\TNR\\AI\\Output_82.txt", "r");
		Mat img(1080, 1920, CV_8UC3);
		for (int row = 0; (row + BLOCKSIZE - 1) < 1080; row += BLOCKSIZE)
		{
			for (int col = 0; (col + BLOCKSIZE - 1) < 1920; col += BLOCKSIZE)
			{
				//cout << row << "," << col << endl;
				for (int i = 0; i < BLOCKSIZE; i++)
				{
					for (int j = 0; j < BLOCKSIZE; j++)
					{
						float value;
						int n = 16;

						fscanf(fin, "%f", &value);
						img.at<Vec3b>(i + row, j + col)[0] = (int)value;
						fscanf(fin, "%f", &value);
						img.at<Vec3b>(i + row, j + col)[1] = (int)value;
						fscanf(fin, "%f", &value);
						img.at<Vec3b>(i + row, j + col)[2] = (int)value;

					}
				}
			}

		}

		imwrite("out.bmp", img);
		
	#endif

	#ifdef _PSNR
		Mat ori_img;
		Mat rec_img;
		int col = 0, row = 0, mx = 0, my = 0;
		double MSE = 0.0;
		char filtname[200];
		int frame = 12;
		sprintf(filtname, "H:\\DeepLearning_NR\\NR_image\\NR9_1920x1080_12800_seq1_r797_sc4th14_%d_Convert.bmp", FRAMEtesting);
		//sprintf(filtname, "H:\\TNR\\SRC_image\\20150916_ISO12800_seq1_%d_rgb.bmp", frame);
		ori_img = imread(filtname, CV_8U);

		//sprintf(filtname, "H:\\TNR\\SRC_image\\20150916_ISO12800_seq1_%d_rgb.bmp", frame+1);
		rec_img = imread("out.bmp", CV_8U);
		//double MSE = 0;
		double ori_value;
		double rec_value;
		double PSNR;
		/*
		Mat img(1080, 1920, CV_8U);
		FILE *fin;
		fin = fopen("E:\\PROGRAMS\\FRC\\transfer\\Output_12.txt", "rb");
		*/
		double SAD_NOISE = 0;
		int temp = 0;
		int block = 4;
		for (int i = 0; i < 1080; i += block)
		{
			for (int j = 0; j < 1920; j += block)
			{
				
				ori_value = ori_img.at<uchar>(i + row, j + col);
				rec_value = rec_img.at<uchar>(i + row, j + col);

				MSE += (ori_value - rec_value)*(ori_value - rec_value);
				/*
				SAD_NOISE = 0;
				for (int row = 0; row < block; row++)
				{
					for (int col = 0; col < block; col++)
					{
						ori_value = ori_img.at<uchar>(i + row, j + col);
						rec_value = rec_img.at<uchar>(i + row, j + col);

						//MSE += (ori_value - rec_value)*(ori_value - rec_value);

						temp = ori_value - rec_value;
						if (temp < 0) temp *= -1;
						SAD_NOISE += temp;
					}
				}
				SAD_NOISE = SAD_NOISE / (block * block);
				*/
				/*
				double avg_org = 0.0;
				double avg_cur = 0.0;
				for (int row = 0; row < block; row++)
				{
					for (int col = 0; col < block; col++)
					{
						ori_value = ori_img.at<uchar>(i + row, j + col);
						rec_value = rec_img.at<uchar>(i + row, j + col);
						//uiSum += abs(piOrg[n] - piCur[n]);
						avg_org += ((double)(ori_value - 128) / 256.0);
						avg_cur += ((double)(rec_value - 128) / 256.0);
					}
				}

				avg_org = avg_org / (block * block);
				avg_cur = avg_cur / (block * block);

				double d_org = 0.0;
				double d_cur = 0.0;
				double d_co = 0.0;

				double avg_o = (avg_org * 256.0) + 128.0;
				double avg_c = (avg_cur * 256.0) + 128.0;

				for (int row = 0; row < block; row++)
				{
					for (int col = 0; col < block; col++)
					{
						ori_value = ori_img.at<uchar>(i + row, j + col);
						rec_value = rec_img.at<uchar>(i + row, j + col);
						//uiSum += abs(piOrg[n] - piCur[n]);
						d_org += ((double)ori_value - avg_o) * ((double)ori_value - avg_o);
						d_cur += ((double)rec_value - avg_c) * ((double)rec_value - avg_c);
						d_co += ((double)ori_value - avg_o) * ((double)rec_value - avg_c);
					}
				}

				d_org = d_org / (double)(block * block - 1);
				d_cur = d_cur / (double)(block * block - 1);
				d_co = d_co / (double)(block * block - 1);

				const double C1 = (0.01 * 255.0) * (0.01 * 255.0);
				const double C2 = (0.03 * 255.0) * (0.03 * 255.0);
				const double B = 80.0;
				const double S = 108.0;

				double L = ((avg_org + avg_cur) * (avg_org + avg_cur) + 2 * C1) / (2 * ((avg_org * avg_org + avg_cur * avg_cur) + C1));
				double T = pow((d_co + sqrt(d_org * d_cur) + C2) / (d_org + d_cur + C2), (1.0 + (d_org + d_cur) / B));

				//double new_uiSum = S * block * block * (1.0 / (L*T) - 1.0);
				int new_uiSum = (int)(L*T * 256);
				printf("%d\n", new_uiSum);

				for (int row = 0; row < block; row++)
				{
					for (int col = 0; col < block; col++)
					{
						img.at<uchar>(i + row, j + col) = (int)new_uiSum;
					}
				}
				*/
			}
		}
		/*
		char filename[200];
		sprintf(filename, "SSIM%d_%d.bmp", frame, frame+1);
		imwrite(filename, img);
		*/
		MSE = MSE / (1072.0*1920.0);
		PSNR = 10.0 * log10(255.0*255.0 / MSE);
		cout << PSNR;

	#endif

#ifdef Deblock
		Mat src;
		src = imread("H:\\DeepLearning_NR\\DataPreprocessing\\NR_image_process\\out_10000_epoch.bmp", CV_8U);

		Mat rgb[3];
		split(src, rgb);
		
		uchar *data_R = rgb[0].data;
		uchar *data_G = rgb[1].data;
		uchar *data_B = rgb[2].data;

		int quant = 32;
		int a_offset = 0;
		int b_offset = 0;

		int index_a = clip(quant + a_offset, 0, MAX_QUANT);
		int index_b = clip(quant + b_offset, 0, MAX_QUANT);
		int alpha_ = alphas[index_a];
		int beta_ = betas[index_b];
		int c0_ = cs[index_a];
		
		deblock_c(data_R, 255, 1920, 1080, alpha_, beta_, c0_);
		printf("R\n");
		//deblock_c(data_G, 255, 1920, 1080, alpha_, beta_, c0_);
		printf("G\n");
		//deblock_c(data_B, 255, 1920, 1080, alpha_, beta_, c0_);
		printf("B\n");
		
		Mat r_rec(1080, 1920, CV_8UC1, data_R);
		//Mat g_rec(1080, 1920, CV_8UC1, data_G);
		//Mat b_rec(1080, 1920, CV_8UC1, data_B);

		//rgb[0] = r_rec;
		//rgb[1] = g_rec;
		//rgb[2] = b_rec;

		//merge(rgb, 3, src);
		imwrite("out_r.bmp", rgb[0]);
#endif

	system("pause");
	return 0;
}