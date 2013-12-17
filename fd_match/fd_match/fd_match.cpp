//
//  main.cpp
//  fd_match
//
//  Created by Pham Duc Giam on 16/12/13.
//  Copyright (c) 2013 Pham Duc Giam. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <getopt.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

typedef enum {kLongOptionIndexNone, kLongOptionIndexDirectory, kLongOptionIndexDetector, kLongOptionIndexDetectorAdapter, kLongOptionIndexExtractor, kLongOptionIndexExtractorAdapter, kLongOptionIndexMatcher, kLongOptionIndexInput, kLongOptionIndexDistanceRatio, kLongOptionIndexMinimunMatchedPoints} LongOptionIndex;

static const char* DETECTOR_ALGORITHMS[] = {"SURF","FAST","STAR","SIFT","ORB","BRISK","MSER","GFTT","HARRIS","Dense","SimpleBlob"};
static const char* EXTRACTOR_ALGORITHMS[] = {"SURF","SIFT","BRIEF","BRISK","ORB","FREAK"};
static const char* MATCH_ALGORITHMS[] = {"FlannBased","BruteForce","BruteForce-L1","BruteForce-Hamming","BruteForce-Hamming(2)"};
static const char* INPUT_DEFAULT = "input.yml";
static const float DISTANCE_RATIO_DEFAULT = 0.6f;
static const int MINIMUN_MATCHED_POINTS_DEFAULT = 5;

const Ptr<FeatureDetector> getDetector(const char *detectorAdapter, const char *detectorAlgorithm);
const Ptr<DescriptorExtractor> getExtractor(const char *extractorAdapter, const char *extractorAlgorithm);
const Ptr<DescriptorMatcher> getMatcher(const char *matchAlgorithm);

int main(int argc, char * const *argv)
{
    const char *directoryName, *detectorAlgorithm, *detectorAdapter, *extractorAlgorithm, *extractorAdapter, *matchAlgorithm, *input;
    directoryName = detectorAlgorithm = detectorAdapter = extractorAlgorithm = extractorAdapter = matchAlgorithm = input = NULL;
    
    float distanceRatio = DISTANCE_RATIO_DEFAULT; //if distance from this point is less than distance from next point, that point is considered as matched point
    int minimunMatchedPoints = MINIMUN_MATCHED_POINTS_DEFAULT;   //minimum number of matched points that one image need to have to make it as matched image
    
    struct option longOptions[] = {
        {"directory", required_argument, 0, kLongOptionIndexDirectory},
        {"detector", required_argument, 0, kLongOptionIndexDetector},
        {"detector_adapter", required_argument, 0, kLongOptionIndexDetectorAdapter},
        {"extractor", required_argument, 0, kLongOptionIndexExtractor},
        {"extractor_adapter", required_argument, 0, kLongOptionIndexExtractorAdapter},
        {"matcher", required_argument, 0, kLongOptionIndexMatcher},
        {"input", required_argument, 0, kLongOptionIndexInput},
        {"distance_ratio", required_argument, 0, kLongOptionIndexDistanceRatio},
        {"min_point", required_argument, 0, kLongOptionIndexMinimunMatchedPoints},
        {0, 0, 0, 0}
    };
    
    int c, optionIndex;
    while ((c=getopt_long(argc, argv, "", longOptions, &optionIndex))!=-1) {
        switch (c) {
            case kLongOptionIndexDirectory: {
                directoryName = optarg;
                break;
            }
            case kLongOptionIndexDetector: {
                detectorAlgorithm = optarg;
                break;
            }
            case kLongOptionIndexDetectorAdapter: {
                detectorAdapter = optarg;
                break;
            }
            case kLongOptionIndexExtractor: {
                extractorAlgorithm = optarg;
                break;
            }
            case kLongOptionIndexExtractorAdapter: {
                extractorAdapter = optarg;
                break;
            }
            case kLongOptionIndexMatcher: {
                matchAlgorithm = optarg;
                break;
            }
            case kLongOptionIndexInput: {
                input = optarg;
                break;
            }
            case kLongOptionIndexDistanceRatio: {
                distanceRatio = atof(optarg);
                break;
            }
            case kLongOptionIndexMinimunMatchedPoints: {
                minimunMatchedPoints = atoi(optarg);
                break;
            }
            default:
                break;
        }
    }
    
    if (!directoryName) {
        cout << "need input directory" << endl;
        return -1;
    }
    
    if (!detectorAlgorithm) {
        cout << "use " << DETECTOR_ALGORITHMS[0] << " as detection algorithm" << endl;
        detectorAlgorithm = DETECTOR_ALGORITHMS[0];
    }
    
    if (!extractorAlgorithm) {
        cout << "use " << EXTRACTOR_ALGORITHMS[0] << " as extraction algorithm" << endl;
        extractorAlgorithm = EXTRACTOR_ALGORITHMS[0];
    }
    
    if (!input) {
        cout << "use " << INPUT_DEFAULT << " as input file" << endl;
        input = INPUT_DEFAULT;
    }
    
    if (fabs(distanceRatio)<0.01f || distanceRatio<0) {
        cout << "use " << DISTANCE_RATIO_DEFAULT << " as distance ratio" << endl;
        distanceRatio = DISTANCE_RATIO_DEFAULT;
    }
    
    if (minimunMatchedPoints<=0) {
        cout << "use " << MINIMUN_MATCHED_POINTS_DEFAULT << " as mininum number of matched points" << endl;
        minimunMatchedPoints = MINIMUN_MATCHED_POINTS_DEFAULT;
    }
    
    DIR *dir;
    dir = opendir(directoryName);
    if (!dir) {
        cout << "could not open directory " << directoryName << endl;
        return -1;
    }
    
    cout << "open input file...";
    FileStorage fsInput(input, FileStorage::READ);
    if (!fsInput.isOpened()) {
        cout << endl << "could not open input file " << input << endl;
        return -1;
    }
    
    Mat features;
    cv::vector<int> indexes;
    cv::vector<string> filenames;
    
    fsInput["features"] >> features;
    fsInput["filenames"] >> filenames;
    fsInput["indexes"] >> indexes;
    cout << "\tdone" << endl;
    fsInput.release();
    
    flann::KDTreeIndexParams indexParams(5);
    //flann::LshIndexParams indexParams(20, 15, 2);
    flann::Index kdtree(features, indexParams);
    
    struct dirent *ep;
    char *filename;
    char path[1000];
    struct stat buf;
    
    Ptr<FeatureDetector> detector = getDetector(detectorAdapter, detectorAlgorithm);
    Ptr<DescriptorExtractor> extractor = getExtractor(extractorAdapter, extractorAlgorithm);
    cv::vector<KeyPoint> keypoints;
    Mat image;
    Mat descriptors;
    Mat indices;
    Mat dists;
    
    int trueMatch = 0;
    int totalFile = 0;
    double t;
    
    cout << "matching..." << endl;
    
    t = (double)getTickCount();
    
    while ((ep = readdir(dir))) {
    	filename = ep->d_name;
        if(strlen(filename)==0 || filename[0]=='.')
        	continue;
        
        sprintf(path, "%s/%s", directoryName, filename);
        
        lstat(path, &buf);
        if (S_ISREG(buf.st_mode)) {
        	image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
        	if (image.data) {
                totalFile++;
                
                cout << "File " << filename << "...";
        		detector->detect(image, keypoints);
        		extractor->compute(image, keypoints, descriptors);
                
                kdtree.knnSearch(descriptors, indices, dists, 2, cv::flann::SearchParams(64));
                
                vector<int> matchPoints(indexes.size(), 0);
                vector<int>::iterator begin = indexes.begin();
                vector<int>::iterator end = indexes.end();
                vector<int>::iterator iter;
                
                int i,j,k;
                for (i=0; i<indices.rows; i++) {
                    if (dists.at<float>(i, 0) < distanceRatio * dists.at<float>(i, 1)) {
                        k = indices.at<int>(i, 0);
                        iter = std::upper_bound(begin, end, k);
                        if (iter!=end) {
                            j = (int)(iter - begin) - 1;
                            matchPoints[j]++;
                        }
                    }
                }
                
                k = 0;
                j = -1;
                for (i=0; i<matchPoints.size(); i++) {
                    if (matchPoints[i]>k) {
                        k = matchPoints[i];
                        j = i;
                    }
                }
                
                if (j>=0 && k>=minimunMatchedPoints) {
                    cout << "matching image: " << filenames[j] << "; number of matched points: " << k;
                    if (filenames[j].compare(filename)==0) {
                        trueMatch++;
                        cout << "...true match";
                    }
                    else {
                        cout << "...false match";
                    }
                }
                else {
                    cout << "could find matched image";
                }
                
                cout << "...done" << endl;
            }
        }
    }
    closedir(dir);
    
    t = 1000 * (((double)getTickCount() - t) / getTickFrequency());
	cout << endl << "Time passed in miliseconds: " << t << endl;
    
    cout.precision(2);
    cout << endl << "correct rate: " << (100.0 * trueMatch / totalFile) << endl;
    
    return 0;
}

const Ptr<FeatureDetector> getDetector(const char *detectorAdapter, const char *detectorAlgorithm)
{
    string detectorType = detectorAlgorithm;
    if (detectorAdapter) {
        detectorType = detectorAdapter + detectorType;
    }
    
    cout << "create feature detector with type: " << detectorType << endl;
    Ptr<FeatureDetector> result = FeatureDetector::create(detectorType);
    
    return result;
}

const Ptr<DescriptorExtractor> getExtractor(const char *extractorAdapter, const char *extractorAlgorithm)
{
    string extractorType = extractorAlgorithm;
    if (extractorAdapter) {
        extractorType = extractorAdapter + extractorType;
    }
    
    cout << "create descriptor extractor with type: " << extractorType << endl;
    Ptr<DescriptorExtractor> result = DescriptorExtractor::create(extractorType);
    
    return result;
}

const Ptr<DescriptorMatcher> getMatcher(const char *matchAlgorithm)
{
    return DescriptorMatcher::create(matchAlgorithm);
}

