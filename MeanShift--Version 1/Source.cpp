#include "opencv2\opencv.hpp"
#include <iostream>
#include <cstdint>
#include <ctime>


using namespace std;
using namespace cv;


int main()
{
	srand(time(NULL));
	const float bandwidth = 0.2;

	Mat im = imread("3.jpg", CV_LOAD_IMAGE_COLOR);

	Size size(im.cols / 4, im.rows / 4);
	Size sizeoriginal(im.cols, im.rows);
	resize(im, im, size, 0, 0, CV_INTER_AREA); //for shrinking an image, CV_INTER_AREA interpolation works best.
	//resize(im, im, Size(), 0.25, 0.25, CV_INTER_AREA); same output as above

	//reshapePlusConvertingToDouble start

	int newRows = im.rows*im.cols;
	int numofRows = im.rows;

	vector <float> X(newRows * 3);
	int k = 0;
	for (int i = 0; i < im.rows; i++)
	{
		for (int j = 0; j < im.cols; j++, k++)
		{
			X[0 * newRows + k] = (float)(im.at<Vec3b>(i, j)[2]) / 255; //red
			X[1 * newRows + k] = (float)(im.at<Vec3b>(i, j)[1]) / 255; //green
			X[2 * newRows + k] = (float)(im.at<Vec3b>(i, j)[0]) / 255; //blue
		}
	}
	//reshapePlusConvertingToDoubleEnd

	//Initializations---------------------------------------

	int numDims = 3;
	int numPts = newRows;
	int numClust = 0;
	float bandSq = pow(bandwidth, 2);

	//initPtInds
	vector<int> initPtInds;

	for (int i = 0; i < numPts; i++)
	{
		initPtInds.push_back(i);
	}

	double stopThresh = 1e-3*bandwidth;


	//clustCent
	vector <float> clustCent;

	//beenVisitedFlag
	vector <uint8_t> beenVisitedFlag(numPts);

	//numInitPts
	int numInitPts = numPts;

	//clusterVotes
	vector <int> clusterVotes;


	float myMean[3];

	//EndInitializations---------------------------------------

	while (numInitPts)
	{
		float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //rand nmber generator between 0 and 1
		int tempInd = ceil((numInitPts - 1e-6) * r);
		tempInd = tempInd - 1;
		int stInd = initPtInds[tempInd];

		myMean[0] = X[0 * newRows + stInd];
		myMean[1] = X[1 * newRows + stInd];
		myMean[2] = X[2 * newRows + stInd];

		vector <int> myMembers;

		//this clusterVotes
		vector <uint16_t> thisClusterVotes(numPts);

		while (1)
		{
			vector <float> sqDistToAll(numPts);

			for (int i = 0; i < numPts; i++)
			{
				sqDistToAll[i] = pow((myMean[0] - X[0 * newRows + i]), 2) + pow((myMean[1] - X[1 * newRows + i]), 2) + pow((myMean[2] - X[2 * newRows + i]), 2);
			}

			vector<int> inInds;
			inInds.clear();

			for (int i = 0; i < numPts; i++)
			{
				if (sqDistToAll[i] < bandSq)
				{
					inInds.push_back(i);
				}
			}

			for (int i = 0; i < inInds.size(); i++)
			{
				thisClusterVotes[inInds[i]] = thisClusterVotes[inInds[i]] + 1;
			}

			float myOldMean[3];
			myOldMean[0] = myMean[0];
			myOldMean[1] = myMean[1];
			myOldMean[2] = myMean[2];

			float redSum = 0;
			float greenSum = 0;
			float blueSum = 0;


			for (int i = 0; i < inInds.size(); i++)
			{
				redSum = redSum + X[0 * newRows + inInds[i]];
				greenSum = greenSum + X[1 * newRows + inInds[i]];
				blueSum = blueSum + X[2 * newRows + inInds[i]];
				myMembers.push_back(inInds[i]);
			}

			myMean[0] = (float)redSum / (float)inInds.size();
			myMean[1] = (float)greenSum / (float)inInds.size();
			myMean[2] = (float)blueSum / (float)inInds.size();

			/*	for (int i = 0; i < inInds.size(); i++)
			{
			myMembers.push_back(inInds[i]);
			}
			*/

			for (int i = 0; i < myMembers.size(); i++)
			{
				beenVisitedFlag[myMembers[i]] = (uint8_t)1;
			}

			if ((sqrt(pow((myMean[0] - myOldMean[0]), 2) + pow((myMean[1] - myOldMean[1]), 2) + pow((myMean[2] - myOldMean[2]), 2))) < stopThresh)
			{
				//check for merge possibilities

				int mergeWith = 0;

				for (int cN = 1; cN <= numClust; cN++)

				{
					float distToOther = sqrt(pow((myMean[0] - clustCent[((cN - 1) * 3) + 0]), 2) + pow((myMean[1] - clustCent[((cN - 1) * 3) + 1]), 2) + pow((myMean[2] - clustCent[((cN - 1) * 3) + 2]), 2));
					if (distToOther < bandwidth / 2)
					{
						mergeWith = cN;
						break;
					}
				}

				if (mergeWith>0)
				{
					clustCent[((mergeWith - 1) * 3) + 0] = 0.5 * (myMean[0] + clustCent[((mergeWith - 1) * 3) + 0]);
					clustCent[((mergeWith - 1) * 3) + 1] = 0.5 * (myMean[1] + clustCent[((mergeWith - 1) * 3) + 1]);
					clustCent[((mergeWith - 1) * 3) + 2] = 0.5 * (myMean[2] + clustCent[((mergeWith - 1) * 3) + 2]);

					for (int i = 0; i < numPts; i++)
					{
						clusterVotes[(mergeWith - 1) * numPts + i] = clusterVotes[(mergeWith - 1) * numPts + i] + thisClusterVotes[i];
					}

				}

				else
				{
					numClust = numClust + 1;
					clustCent.push_back(myMean[0]);
					clustCent.push_back(myMean[1]);
					clustCent.push_back(myMean[2]);

					for (int i = 0; i < numPts; i++)
					{
						clusterVotes.push_back(thisClusterVotes[i]);
					}
				}
				break;
			}


		}

		initPtInds.clear();
		for (int i = 0; i < numPts; i++)
		{
			if (beenVisitedFlag[i] == 0)
			{
				initPtInds.push_back(i);
			}
		}

		numInitPts = initPtInds.size();

	} //end of main while

	vector <int> data2cluster(numPts);

	for (int i = 0; i < numPts; i++)
	{
		int max = 0;

		for (int j = 0; j < numClust; j++)
		{
			int max1 = clusterVotes[j * numPts + i];
			if (max1 >= max)
			{
				data2cluster[i] = j;
				max = max1;
			}
		}
	}

	vector <int> cluster2dataCell;
	vector <int> numOfColsInEachClust(numClust);

	for (int cN = 0; cN < numClust; cN++)
	{
		int sizeNum = 0;
		for (int i = 0; i < numPts; i++)
		{
			if (data2cluster[i] == cN)
			{
				cluster2dataCell.push_back(i);
				sizeNum++;
			}
		}
		numOfColsInEachClust[cN] = sizeNum;
	}

	int myindex = 0;

	for (int cN = 0; cN < numClust; cN++)
	{
		int numCols = numOfColsInEachClust[cN];

		for (int i = 0; i < numCols; i++)
		{
			int col = cluster2dataCell[i + myindex];

			X[0 * newRows + col] = clustCent[(cN * 3) + 0];
			X[1 * newRows + col] = clustCent[(cN * 3) + 1];
			X[2 * newRows + col] = clustCent[(cN * 3) + 2];
		}
		myindex += numCols;
	}



	for (int i = 0; i < newRows; i++)
	{
		X[0 * newRows + i] = ceil((float)X[0 * newRows + i] * (float)255);
		X[1 * newRows + i] = ceil((float)X[1 * newRows + i] * (float)255);
		X[2 * newRows + i] = ceil((float)X[2 * newRows + i] * (float)255);
	}


	k = 0;

	for (int i = 0; i < im.rows; i++)
	{
		for (int j = 0; j < im.cols; j++, k++)
		{
			(im.at<Vec3b>(i, j)[2]) = X[0 * newRows + k];  //red
			(im.at<Vec3b>(i, j)[1]) = X[1 * newRows + k];  //green
			(im.at<Vec3b>(i, j)[0]) = X[2 * newRows + k];  //blue
		}
	}

	data2cluster = std::vector<int>();
	cluster2dataCell = std::vector<int>();
	numOfColsInEachClust = std::vector<int>();
	clusterVotes = std::vector<int>();
	beenVisitedFlag = std::vector<uint8_t>();
	initPtInds = std::vector<int>();
	X = std::vector<float>();
	//clustCent not clear, contains means


	imshow("resized", im);
	waitKey();
	resize(im, im, sizeoriginal, 0, 0, CV_INTER_AREA);
	imshow("resized", im);
	waitKey();

	im.release();
	return 0;
}


