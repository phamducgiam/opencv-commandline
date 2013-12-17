//
//  main.cpp
//  fd_generate
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

typedef enum {kLongOptionIndexNone, kLongOptionIndexDirectory, kLongOptionIndexDetector, kLongOptionIndexDetectorAdapter, kLongOptionIndexExtractor, kLongOptionIndexExtractorAdapter, kLongOptionIndexOutput} LongOptionIndex;

static const char* DETECTOR_ALGORITHMS[] = {"SURF","FAST","STAR","SIFT","ORB","BRISK","MSER","GFTT","HARRIS","Dense","SimpleBlob"};
static const char* EXTRACTOR_ALGORITHMS[] = {"SURF","SIFT","BRIEF","BRISK","ORB","FREAK"};
static const char* OUTPUT_DEFAULT = "output.yml";

const Ptr<FeatureDetector> getDetector(const char *detectorAdapter, const char *detectorAlgorithm);
const Ptr<DescriptorExtractor> getExtractor(const char *extractorAdapter, const char *extractorAlgorithm);

int main(int argc, char * const *argv)
{
    const char *directoryName, *detectorAlgorithm, *detectorAdapter, *extractorAlgorithm, *extractorAdapter, *output;
    directoryName = detectorAlgorithm = detectorAdapter = extractorAlgorithm = extractorAdapter = output = NULL;
    
    struct option longOptions[] = {
        {"directory", required_argument, 0, kLongOptionIndexDirectory},
        {"detector", required_argument, 0, kLongOptionIndexDetector},
        {"detector_adapter", required_argument, 0, kLongOptionIndexDetectorAdapter},
        {"extractor", required_argument, 0, kLongOptionIndexExtractor},
        {"extractor_adapter", required_argument, 0, kLongOptionIndexExtractorAdapter},
        {"output", required_argument, 0, kLongOptionIndexOutput},
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
            case kLongOptionIndexOutput: {
                output = optarg;
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
    
    if (!output) {
        cout << "use " << OUTPUT_DEFAULT << " as output file" << endl;
        output = OUTPUT_DEFAULT;
    }
    
    DIR *dir;
    dir = opendir(directoryName);
    if (!dir) {
        cout << "could not open image directory " << directoryName << endl;
        return -1;
    }
    
    struct dirent *ep;
    char *filename;
    char path[1000];
    struct stat buf;
    
    Ptr<FeatureDetector> detector = getDetector(detectorAdapter, detectorAlgorithm);
    Ptr<DescriptorExtractor> extractor = getExtractor(extractorAdapter, extractorAlgorithm);
    cv::vector<KeyPoint> keypoints;
    Mat image;
    Mat descriptors;
    Mat features;
    cv::vector<int> indexes;
    cv::vector<string> filenames;
    int k = 0;
    
    cout << "building..." << endl;
    
    double t = (double)getTickCount();
    
    while ((ep = readdir(dir))) {
    	filename = ep->d_name;
        if(strlen(filename)==0 || filename[0]=='.')
        	continue;
        
        sprintf(path, "%s/%s", directoryName, filename);
        
        lstat(path, &buf);
        if (S_ISREG(buf.st_mode)) {
        	image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
        	if (image.data) {
                cout << "File " << filename << "...";
        		detector->detect(image, keypoints);
        		extractor->compute(image, keypoints, descriptors);
             	features.push_back(descriptors);
                filenames.push_back(filename);
                indexes.push_back(k);
                k += descriptors.rows;
                cout << " done" << endl;
            }
        }
    }
    closedir(dir);
    
    t = 1000 * (((double)getTickCount() - t) / getTickFrequency());
	cout << endl << "Time passed in miliseconds: " << t << endl;
    
    cout << "write to output file " << output << "...";
    FileStorage fsOutput(output, FileStorage::WRITE);
    fsOutput << "features" << features;
    fsOutput << "filenames" << filenames;
    fsOutput << "indexes" << indexes;
    cout << "\tdone" << endl;
    fsOutput.release();
    
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
