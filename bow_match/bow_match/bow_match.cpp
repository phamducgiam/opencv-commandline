//
//  main.cpp
//  bow_match
//
//  Created by Pham Duc Giam on 11/12/13.
//  Copyright (c) 2013 Pham Duc Giam. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

typedef enum {kLongOptionIndexNone, kLongOptionIndexDirectory, kLongOptionIndexDetector, kLongOptionIndexDetectorAdapter, kLongOptionIndexExtractor, kLongOptionIndexExtractorAdapter, kLongOptionIndexMatcher, kLongOptionIndexFeaturesInput, kLongOptionIndexDescriptorsInput} LongOptionIndex;

static const char* detectorAlgorithms[] = {"SURF","FAST","STAR","SIFT","ORB","BRISK","MSER","GFTT","HARRIS","Dense","SimpleBlob"};
static const char* extractorAlgorithms[] = {"SURF","SIFT","BRIEF","BRISK","ORB","FREAK"};
static const char* matchAlgorithms[] = {"FlannBased","BruteForce","BruteForce-L1","BruteForce-Hamming","BruteForce-Hamming(2)"};
static const char* featuresInputDefault = "features.yml";
static const char* descriptorsInputDefault = "descriptors.yml";

const Ptr<FeatureDetector> getDetector(const char *detectorAdapter, const char *detectorAlgorithm);
const Ptr<DescriptorExtractor> getExtractor(const char *extractorAdapter, const char *extractorAlgorithm);
const Ptr<DescriptorMatcher> getMatcher(const char *matchAlgorithm);

int main(int argc, char * const *argv)
{
    const char *directoryName, *detectorAlgorithm, *detectorAdapter, *extractorAlgorithm, *extractorAdapter, *matchAlgorithm, *featuresInput, *descriptorsInput;
    directoryName = detectorAlgorithm = detectorAdapter = extractorAlgorithm = extractorAdapter = featuresInput = descriptorsInput = NULL;
    
    struct option longOptions[] = {
        {"directory", required_argument, 0, kLongOptionIndexDirectory},
        {"detector", required_argument, 0, kLongOptionIndexDetector},
        {"detector_adapter", required_argument, 0, kLongOptionIndexDetectorAdapter},
        {"extractor", required_argument, 0, kLongOptionIndexExtractor},
        {"extractor_adapter", required_argument, 0, kLongOptionIndexExtractorAdapter},
        {"matcher", required_argument, 0, kLongOptionIndexMatcher},
        {"features_input", required_argument, 0, kLongOptionIndexFeaturesInput},
        {"descriptors_input", required_argument, 0, kLongOptionIndexDescriptorsInput},
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
            case kLongOptionIndexFeaturesInput: {
                featuresInput = optarg;
                break;
            }
            case kLongOptionIndexDescriptorsInput: {
                descriptorsInput = optarg;
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
        cout << "use " << detectorAlgorithms[0] << " as detection algorithm" << endl;
        detectorAlgorithm = detectorAlgorithms[0];
    }
    
    if (!extractorAlgorithm) {
        cout << "use " << extractorAlgorithms[0] << " as extraction algorithm" << endl;
        extractorAlgorithm = extractorAlgorithms[0];
    }
    
    if (!matchAlgorithm) {
        cout << "use " << matchAlgorithms[0] << " as matcher algorithm" << endl;
        matchAlgorithm = matchAlgorithms[0];
    }
    
    if (!featuresInput) {
        cout << "use " << featuresInputDefault << " as input for features" << endl;
        featuresInput = featuresInputDefault;
    }
    
    if (!descriptorsInput) {
        cout << "use " << descriptorsInputDefault << " as input for descriptors" << endl;
        descriptorsInput = descriptorsInputDefault;
    }
    
    FileStorage fsFeatures(featuresInput, FileStorage::READ);
    if (!fsFeatures.isOpened()) {
        cout << "could not open features input file " << featuresInput << endl;
        return -1;
    }
    
    FileStorage fsDescriptors(descriptorsInput, FileStorage::READ);
    if (!fsDescriptors.isOpened()) {
        cout << "could not open descriptors input file " << descriptorsInput << endl;
        return -1;
    }
    
    DIR *dir;
    dir = opendir(directoryName);
    if (!dir) {
        cout << "could not open directory " << directoryName << endl;
        return -1;
    }
    
    Mat vocabulary;
    cv::vector<Mat> bowDescriptors;
    vector<string> filenames;
    
    fsFeatures["vocabulary"] >> vocabulary;
    fsFeatures.release();
    
    fsDescriptors["descriptors"] >> bowDescriptors;
    fsDescriptors["filenames"] >> filenames;
    fsDescriptors.release();
    
    struct dirent *ep;
    char *filename;
    char path[1000];
    struct stat buf;
    
    Ptr<FeatureDetector> detector = getDetector(detectorAdapter, detectorAlgorithm);
    Ptr<DescriptorMatcher> bowMatcher = getMatcher(matchAlgorithm);
    Ptr<DescriptorExtractor> extractor = getExtractor(extractorAdapter, extractorAlgorithm);
    BOWImgDescriptorExtractor bowExtractor(extractor, bowMatcher);
    bowExtractor.setVocabulary(vocabulary);
    
    Ptr<DescriptorMatcher> matcher = getMatcher(matchAlgorithm);
    matcher->add(bowDescriptors);
    matcher->train();
    
    cv::vector<KeyPoint> keypoints;
    Mat image;
    Mat bowDescriptor;
    cv::vector<DMatch> matches;
    
    cout << "Matching..." << endl;
    while ((ep = readdir(dir))) {
    	filename = ep->d_name;
        if((strchr(filename, '.')-filename)==0)
        	continue;
        
        sprintf(path, "%s/%s", directoryName, filename);
        
        lstat(path, &buf);
        if (S_ISREG(buf.st_mode)) {
        	image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
        	if (image.data) {
                cout << "File " << filename << "..." << endl;
        		detector->detect(image, keypoints);
                bowExtractor.compute(image, keypoints, bowDescriptor);
                matcher->match(bowDescriptor, matches);
                for(int i=0;i<matches.size();i++) {
                    DMatch match = matches[i];
                    cout << "\ti = " << i << "; queryIdx = " << match.queryIdx << "; trainIdx = " << match.trainIdx << "; imgIdx = " << match.imgIdx << "; distance = " << match.distance << endl;
                    int j = match.imgIdx;
                    if (j>=0 && j<filenames.size()) {
                        cout << "\tOriginal file " << filenames[j] << endl;
                    }
                }
                cout << "done" << endl;
                //break;
            }
        }
    }
    closedir(dir);
    
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
    cout << "create descriptor matcher with type: " << matchAlgorithm << endl;
    return DescriptorMatcher::create(matchAlgorithm);
}
